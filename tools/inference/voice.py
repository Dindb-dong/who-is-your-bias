"""Synthesize one short utterance through the real VoiceEngine, with explicit reference provenance."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import signal
import time
import wave
from dataclasses import replace
from pathlib import Path

from services.voice.engine import VoiceEngine, VoiceSettings, load_qwen_model, read_reference

from .common import new_report, progress, safe_error, save_report


def run(args: argparse.Namespace) -> dict:
    report = new_report("local_voice")
    report.update(
        model_id=args.model, device="cpu", dtype=args.dtype, text=args.text,
        language=args.language, x_vector_only_mode=args.x_vector_only,
        max_new_tokens=args.max_new_tokens, cpu_threads=args.threads, seed=args.seed,
        voice_likeness="not_evaluated", speaker_identity="not_verified_by_this_runner",
        intelligibility="not_evaluated", streaming=False,
    )
    save_report(args.report, report)
    try:
        import numpy as np
        import torch
        from huggingface_hub import HfApi, snapshot_download

        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        transcript = args.reference_text_file.read_text(encoding="utf-8").strip() if args.reference_text_file else None
        settings = VoiceSettings(
            model_id=args.model, device="cpu", dtype=args.dtype, attention="sdpa",
            reference_wav=args.reference, reference_text=transcript,
            x_vector_only_mode=args.x_vector_only, max_new_tokens=args.max_new_tokens,
            max_audio_seconds=20,
        )
        reference, reference_rate = read_reference(settings)
        report["reference"] = {
            "sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "sample_rate": reference_rate, "seconds": len(reference) / reference_rate,
            "transcript_supplied": bool(transcript), "review": "not_supplied",
        }
        if args.reference_manifest:
            manifest = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
            report["reference"]["review"] = manifest.get("review", "not_supplied")
            report["reference"]["source_id"] = manifest.get("source_id")
            report["reference"]["source_timeline"] = manifest.get("source_timeline")
            report["reference"]["source_start_seconds"] = manifest.get("source_start_seconds")
            report["reference"]["source_end_seconds"] = manifest.get("source_end_seconds")
            expected_hash = manifest.get("artifacts", {}).get("audio", {}).get("sha256")
            if expected_hash and expected_hash != report["reference"]["sha256"]:
                raise ValueError("Reference audio does not match its preparation manifest SHA-256")
            # The review is recorded as supplied metadata, never upgraded here.
        save_report(args.report, report)

        info = HfApi(token=False).model_info(args.model, revision=args.revision, files_metadata=True, timeout=30)
        weight_bytes = sum(file.size or 0 for file in info.siblings if file.rfilename.endswith(".safetensors"))
        report.update(model_revision=info.sha, published_safetensors_bytes=weight_bytes)
        if not weight_bytes or weight_bytes > 2_600_000_000:
            raise RuntimeError("This voice smoke accepts only known safetensors weights up to 2.6 GB")
        save_report(args.report, report)
        progress("downloading_voice", weight_bytes=weight_bytes, revision=info.sha)
        download_started = time.perf_counter()
        snapshot = snapshot_download(
            args.model, revision=info.sha, cache_dir=args.cache_dir, token=False,
            allow_patterns=["*.json", "*.txt", "*.safetensors"], max_workers=2,
        )
        report["download_seconds"] = time.perf_counter() - download_started
        settings = replace(settings, model_id=snapshot)
        save_report(args.report, report)

        class MeasuredModel:
            """Record the same service calls without replacing synthesis with a fake."""

            def __init__(self, model):
                self.model = model

            def create_voice_clone_prompt(self, **kwargs):
                started = time.perf_counter()
                prompt = self.model.create_voice_clone_prompt(**kwargs)
                report["reference_encoding_seconds"] = time.perf_counter() - started
                save_report(args.report, report)
                progress("reference_encoded", seconds=report["reference_encoding_seconds"])
                return prompt

            def generate_voice_clone(self, **kwargs):
                started = time.perf_counter()

                def timeout_handler(signum, frame):
                    raise TimeoutError("Voice generation exceeded its bounded smoke deadline")

                previous_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
                try:
                    return self.model.generate_voice_clone(**kwargs)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous_handler)
                    report["generation_seconds"] = time.perf_counter() - started
                    save_report(args.report, report)

        def measured_loader(config: VoiceSettings):
            started = time.perf_counter()
            model = load_qwen_model(config)
            report["load_seconds"] = time.perf_counter() - started
            report["main_model_parameter_count"] = sum(parameter.numel() for parameter in model.model.parameters())
            save_report(args.report, report)
            progress("voice_loaded", seconds=report["load_seconds"])
            return MeasuredModel(model)

        engine = VoiceEngine(settings, model_loader=measured_loader)
        synthesize_started = time.perf_counter()
        payload = engine.synthesize(args.text, args.language)
        report["service_call_seconds"] = time.perf_counter() - synthesize_started
        with wave.open(io.BytesIO(payload), "rb") as stream:
            rate = stream.getframerate()
            count = stream.getnframes()
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            pcm = np.frombuffer(stream.readframes(count), dtype="<i2").astype(np.float32) / 32768
        if not pcm.size or not np.any(pcm):
            raise RuntimeError("Generated PCM waveform was empty or entirely silent")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
        seconds = count / rate
        report.update(
            status="completed", output=str(args.output), output_sha256=hashlib.sha256(payload).hexdigest(),
            pcm={"sample_rate": rate, "frames": count, "channels": channels, "sample_width": sample_width,
                 "seconds": seconds, "peak": float(np.abs(pcm).max()), "rms": float(np.sqrt(np.mean(pcm ** 2)))},
            generation_real_time_factor=report["generation_seconds"] / seconds,
            service_status_after=engine.status(),
            duration_limit_note="max_new_tokens bounds generation; EOS completion is not exposed by the upstream waveform API",
        )
        progress("voice_completed", seconds=seconds, generation_seconds=report["generation_seconds"])
    except Exception as error:
        report.update(status="timed_out" if isinstance(error, TimeoutError) else "failed", error=safe_error(error))
        progress("voice_failed", error=report["error"])
    save_report(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--revision", help="Optional immutable Hugging Face commit from a previous report")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--reference-text-file", type=Path)
    parser.add_argument("--x-vector-only", action="store_true", help="Explicitly use speaker embedding only, with no reference transcript")
    parser.add_argument("--text", default="This is a research simulation.")
    parser.add_argument("--language", choices=("English", "Korean"), default="English")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/local/voice-smoke.wav"))
    parser.add_argument("--report", type=Path, default=Path("reports/local/voice-smoke.json"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.x_vector_only and not args.reference_text_file:
        parser.error("Supply an accurate --reference-text-file or explicitly select --x-vector-only")
    if not 1 <= args.max_new_tokens <= 128 or not 1 <= args.timeout_seconds <= 300 or not 1 <= args.threads <= 8 or not 1 <= len(args.text.strip()) <= 200:
        parser.error("Use 1–128 output tokens, 1–300 seconds, 1–8 CPU threads, and 1–200 input characters")
    return 0 if run(args)["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
