"""A bounded MuseTalk 1.5 renderer for one explicitly reviewed, fixed face crop.

The audio/latent/UNet operations follow the MIT-licensed upstream implementation
at 0a89dec45a0192b824e3cf4daf96c239440c5ed8. This adapter uses a supplied crop and
geometric lower-face feathering, so it needs no CUDA-only detection stack. It
does not estimate head/body motion and does not claim upstream blending parity.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .common import digest, save, wav_info

UPSTREAM_REVISION = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"


def validate_crop(crop: list[int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    if len(crop) != 4 or any(isinstance(x, bool) or not isinstance(x, int) for x in crop):
        raise ValueError("Crop requires four integer pixels")
    x1, y1, x2, y2 = crop
    if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]) or min(x2 - x1, y2 - y1) < 64:
        raise ValueError("Face crop must fit the image and be at least 64 pixels wide/high")
    return x1, y1, x2, y2


def feature_windows(features, frames: int, fps: int):
    """Map 50 Hz Whisper states to video frames using upstream 2+2 padding."""
    import torch
    if not 1 <= fps <= 60 or frames < 1 or features.ndim != 4 or features.shape[1] < 1:
        raise ValueError("Invalid audio features or frame rate")
    multiplier = 50 / fps
    pad = math.ceil(multiplier)
    left = torch.zeros((features.shape[0], pad * 2, *features.shape[2:]), dtype=features.dtype)
    right = torch.zeros((features.shape[0], pad * 6 + 10, *features.shape[2:]), dtype=features.dtype)
    padded = torch.cat([left, features, right], dim=1)
    windows = []
    for index in range(frames):
        start = math.floor(index * multiplier)
        chunk = padded[:, start:start + 10]
        if chunk.shape[1] != 10:
            raise ValueError("Audio feature coverage is shorter than the requested video")
        windows.append(chunk.reshape(1, -1, features.shape[-1]))
    return torch.cat(windows)


def positional_encoding(length: int, width: int, dtype):
    import torch
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    encoding = torch.zeros(length, width, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * frequencies)
    encoding[:, 1::2] = torch.cos(position * frequencies)
    return encoding.to(dtype)


def lower_face_mask(width: int, height: int):
    import numpy as np
    # Preserve the eyes/forehead, feather around the chin and face edges.
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = x / max(1, width - 1)
    y = y / max(1, height - 1)
    edge = np.minimum(x / 0.13, (1 - x) / 0.13)
    upper = (y - 0.45) / 0.12
    lower = (1 - y) / 0.12
    return np.clip(np.minimum(np.minimum(edge, upper), lower), 0, 1)[..., None]


def render(*, portrait: Path, audio: Path, model_lock: Path, crop: list[int], output: Path,
           fps: int = 25, threads: int = 2, dtype_name: str = "bfloat16", timeout: int = 1800,
           max_frames: int | None = None) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import WhisperFeatureExtractor, WhisperModel

    if not 1 <= threads <= 16 or fps != 25 or not 30 <= timeout <= 7200:
        raise ValueError("Use 1–16 threads, the trained 25 fps, and a 30–7200 second deadline")
    if dtype_name not in {"float32", "bfloat16"}:
        raise ValueError("CPU renderer supports float32 or bfloat16")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory to preserve prior render evidence")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    dtype = getattr(torch, dtype_name)
    info = wav_info(audio)
    if info["seconds"] > 120:
        raise ValueError("A single prepared clip is limited to 120 seconds")
    image = Image.open(portrait).convert("RGB")
    if image.width * image.height > 4096 * 4096 or image.width % 2 or image.height % 2:
        raise ValueError("Portrait must be bounded and have even dimensions")
    box = validate_crop(crop, image.size)
    lock = json.loads(model_lock.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "musetalk-model-lock.v1":
        raise ValueError("A pinned model lock is required")
    required = {"musetalk_config", "musetalk_weights", "vae_config", "vae_weights", "whisper_config", "whisper_weights", "whisper_preprocessor"}
    if set(lock.get("files", {})) != required:
        raise ValueError("Model lock is incomplete")
    paths = {}
    for name, record in lock["files"].items():
        path = Path(record["path"])
        if not path.is_absolute():
            path = model_lock.parent / path
        if digest(path) != record["sha256"]:
            raise ValueError(f"Model file checksum mismatch: {name}")
        paths[name] = path
    start = time.perf_counter()
    deadline = start + timeout
    report = {"schema_version": "musetalk-render.v1", "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(), "backend": "musetalk-1.5-fixed-crop",
        "upstream_revision": UPSTREAM_REVISION, "device": "cpu", "dtype": dtype_name,
        "threads": threads, "seed": 42, "fps": fps, "platform": platform.platform(),
        "portrait_sha256": digest(portrait), "audio_sha256": digest(audio), "audio_seconds": info["seconds"],
        "model_lock_sha256": digest(model_lock), "models": lock["models"], "crop": list(box),
        "crop_review": "assistant visually inspected the static portrait; fixed crop, no motion tracking",
        "blending": "geometric_lower_face_feather.v1", "generated_frames": 0,
        "human_visual_sync_review": "unrated", "stage_seconds": {}}
    save(output / "render.json", report)
    process = None
    try:
        stage = time.perf_counter()
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio), "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1"],
            check=True, capture_output=True, timeout=30).stdout
        samples = np.frombuffer(pcm, dtype="<f4").copy()
        total_frames = math.ceil(len(samples) / 16000 * fps)
        frames = total_frames if max_frames is None else min(total_frames, max_frames)
        if frames < 1:
            raise ValueError("Audio has no video frames")
        extractor = WhisperFeatureExtractor.from_pretrained(str(paths["whisper_config"].parent), local_files_only=True)
        whisper = WhisperModel.from_pretrained(str(paths["whisper_config"].parent), local_files_only=True, dtype=dtype, use_safetensors=True).eval()
        states = []
        with torch.inference_mode():
            for offset in range(0, len(samples), 16000 * 30):
                inputs = extractor(samples[offset:offset + 16000 * 30], sampling_rate=16000, return_tensors="pt").input_features.to(dtype)
                hidden = whisper.encoder(inputs, output_hidden_states=True).hidden_states
                states.append(torch.stack(hidden, dim=2))
            features = torch.cat(states, dim=1)[:, :math.floor(len(samples) / 16000 * 50)]
            chunks = feature_windows(features, frames, fps)
        del whisper, states, features, inputs, hidden
        gc.collect()
        report["stage_seconds"]["audio_features"] = time.perf_counter() - stage
        save(output / "render.json", report)

        stage = time.perf_counter()
        vae = AutoencoderKL.from_pretrained(str(paths["vae_config"].parent), local_files_only=True, torch_dtype=dtype, use_safetensors=True).eval()
        crop_array = np.asarray(image.crop(box).resize((256, 256), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        full = torch.from_numpy(crop_array).permute(2, 0, 1)[None].to(dtype)
        masked = full.clone(); masked[:, :, 128:, :] = 0
        with torch.inference_mode():
            masked_latents = vae.encode(masked * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
            full_latents = vae.encode(full * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
            latent = torch.cat([masked_latents, full_latents], dim=1)
        # Instantiate on meta, map only weights, and cast once: avoid two full
        # float32 copies during loading on constrained CPU machines.
        configuration = json.loads(paths["musetalk_config"].read_text())
        with torch.device("meta"):
            unet = UNet2DConditionModel(**configuration)
        weights = torch.load(paths["musetalk_weights"], map_location="cpu", mmap=True, weights_only=True)
        unet.load_state_dict(weights, assign=True, strict=True)
        unet = unet.to(dtype=dtype).eval()
        del weights
        gc.collect()
        report["stage_seconds"]["image_and_model_load"] = time.perf_counter() - stage
        report["parameter_count"] = sum(p.numel() for p in unet.parameters())
        save(output / "render.json", report)
        pe = positional_encoding(chunks.shape[1], chunks.shape[-1], dtype)
        base = np.asarray(image, dtype=np.float32)
        x1, y1, x2, y2 = box
        mask = lower_face_mask(x2 - x1, y2 - y1)
        process = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{image.width}x{image.height}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-threads", "2", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(output / "silent.mp4")],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        stage = time.perf_counter()
        with torch.inference_mode():
            for index in range(frames):
                if time.perf_counter() > deadline:
                    raise TimeoutError("MuseTalk rendering exceeded the bounded deadline")
                prediction = unet(latent, torch.tensor([0]), encoder_hidden_states=chunks[index:index + 1] + pe).sample
                decoded = vae.decode(prediction / vae.config.scaling_factor).sample
                pixels = ((decoded[0].float().permute(1, 2, 0) / 2 + 0.5).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                patch = np.asarray(Image.fromarray(pixels).resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS), dtype=np.float32)
                frame = base.copy()
                frame[y1:y2, x1:x2] = patch * mask + base[y1:y2, x1:x2] * (1 - mask)
                process.stdin.write(frame.clip(0, 255).astype(np.uint8).tobytes())
                report["generated_frames"] = index + 1
                if index % 25 == 0 or index + 1 == frames:
                    save(output / "render.json", report)
                    print(json.dumps({"stage": "video_frames", "completed": index + 1, "total": frames}), flush=True)
        process.stdin.close()
        error = process.stderr.read(8000).decode(errors="replace")
        if process.wait(timeout=60):
            raise RuntimeError("Video encoding failed: " + error)
        report["stage_seconds"]["frame_inference_and_encoding"] = time.perf_counter() - stage
        report.update(status="completed" if frames == total_frames else "partial_probe",
            requested_frames=total_frames, elapsed_seconds=time.perf_counter() - start,
            silent_video_sha256=digest(output / "silent.mp4"),
            completion_scope="Actual MuseTalk neural inference; fixed portrait and explicit geometric blending adaptation.")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - start)
        save(output / "render.json", report)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.kill(); process.wait(timeout=10)
    save(output / "render.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portrait", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--crop", type=int, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-frames", type=int, help="Explicit partial probe; never classified as a completed presentation")
    args = parser.parse_args()
    print(render(portrait=args.portrait, audio=args.audio, model_lock=args.model_lock, crop=args.crop,
        output=args.output, threads=args.threads, dtype_name=args.dtype, timeout=args.timeout, max_frames=args.max_frames)["status"])


if __name__ == "__main__":
    main()
