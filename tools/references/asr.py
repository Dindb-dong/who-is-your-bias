"""Bounded CPU ASR using a pinned local model and verified PCM input."""

from __future__ import annotations

import importlib.metadata
import os
import re
import time
import wave
from pathlib import Path

from tools.media.common import MediaError, load_manifest, sha256, timestamp, verify_artifact, write_json
from tools.media.transcribe import normalize_segments


def transcribe_reference_context(
    manifest_path: str | Path,
    *,
    model_directory: str | Path,
    model_repo: str,
    revision: str,
    output: str | Path,
    cpu_threads: int = 4,
) -> dict:
    """Run actual ASR without giving the recognizer the publisher transcript.

    NumPy PCM input uses faster-whisper's documented waveform API and avoids
    the PyAV 19 metadata_errors incompatibility observed in this environment.
    No speaker identity is inferred, and no remote model download occurs here.
    """
    if not 1 <= cpu_threads <= 4:
        raise MediaError("cpu_threads must be between 1 and 4")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_repo):
        raise MediaError("Name the model publisher/repository")
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise MediaError("Supply a pinned model revision hash")
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise MediaError("ASR output already exists; choose a new file")
    path, manifest = load_manifest(manifest_path)
    audio = verify_artifact(path, manifest, "audio")
    model_path = Path(model_directory).expanduser().resolve()
    required = [model_path / name for name in ("config.json", "model.bin", "tokenizer.json")]
    if not model_path.is_dir() or not all(file.is_file() for file in required):
        raise MediaError("A local CTranslate2 model with config, weights, and tokenizer is required")
    model_bytes = sum(file.stat().st_size for file in model_path.iterdir() if file.is_file())
    if model_bytes > 1_000_000_000:
        raise MediaError("This CPU reference workflow accepts at most 1 GB of model files")
    # ONNX Runtime can be imported by faster-whisper/VAD. Opt out before its
    # initialization, rather than calling an API after an initialization event.
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    try:
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise MediaError("Install tools/references/requirements-asr.txt in an isolated environment") from exc
    with wave.open(str(audio), "rb") as stream:
        if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getcomptype()) != (1, 16000, 2, "NONE"):
            raise MediaError("Prepare mono 16 kHz PCM16 WAV for this ASR adapter")
        frames = stream.getnframes()
        if not 0 < frames <= 120 * 16000:
            raise MediaError("ASR context must be longer than zero and at most 120 seconds")
        raw = stream.readframes(frames)
        if len(raw) != frames * 2:
            raise MediaError("ASR context WAV is truncated")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    started = time.monotonic()
    model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=cpu_threads, num_workers=1, local_files_only=True)
    segments, info = model.transcribe(samples, language="en", beam_size=5, word_timestamps=True, vad_filter=False, condition_on_previous_text=False)
    rows = []
    for segment in segments:
        rows.append({"start": segment.start, "end": segment.end, "text": segment.text, "words": [
            {"word": word.word, "start": word.start, "end": word.end, "probability": word.probability}
            for word in (segment.words or [])
        ]})
    result = {
        "schema_version": "transcript.v1", "created_at": timestamp(),
        "source_id": manifest["source_id"], "source_url": manifest["source_url"],
        "source_timeline": manifest["source_timeline"], "source_start_seconds": manifest["source_start_seconds"],
        "audio_sha256": manifest["artifacts"]["audio"]["sha256"], "language": info.language,
        "speaker_attribution": "unverified", "review_status": "draft",
        "engine": {
            "name": "faster-whisper", "version": importlib.metadata.version("faster-whisper"),
            "ctranslate2_version": importlib.metadata.version("ctranslate2"),
            "model": model_repo, "revision": revision, "model_sha256": sha256(model_path / "model.bin"),
            "device": "cpu", "compute_type": "int8", "cpu_threads": cpu_threads,
            "num_workers": 1, "beam_size": 5, "initial_prompt": None,
            "input_decoder": "stdlib_wave_pcm16_float32", "vad_filter": False,
            "onnx_telemetry_disabled_before_import": os.environ.get("ORT_DISABLE_TELEMETRY") == "1",
            "condition_on_previous_text": False, "alignment": "faster-whisper-word-timestamps",
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "segments": normalize_segments(rows, offset=manifest["source_start_seconds"], duration=manifest["clip"]["duration_seconds"]),
    }
    write_json(destination, result)
    return result
