"""Independent unprompted multilingual recognition in a separate CPU process."""

from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import time
from pathlib import Path

from .metrics import EvaluationError, lexical_metrics, wav_metrics
from .run import checkpoint, digest, read_json, synthesis_rows, timestamp, verified_audio


def transcribe_run(synthesis_path: str | Path, *, model_directory: str | Path,
    model_repo: str, revision: str, output: str | Path, cpu_threads: int = 4) -> dict:
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 4:
        raise EvaluationError("ASR CPU threads must be between 1 and 4")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_repo) or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise EvaluationError("Supply a publisher/model repository and immutable revision")
    source = Path(synthesis_path).expanduser().resolve()
    synthesis = read_json(source)
    rows = synthesis_rows(synthesis)
    selected = [row for row in rows if row.get("status") == "succeeded"]
    if not selected:
        raise EvaluationError("No successful synthesis cases to recognize")
    for row in selected:
        if row.get("asr_language") not in {"en", "ko"} or not isinstance(row.get("text"), str):
            raise EvaluationError("The declared expected text and ASR language are required")
        verified_audio(source, row)
    target = Path(output).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise EvaluationError("Choose a new ASR result file")
    model_path = Path(model_directory).expanduser().resolve()
    required = ("config.json", "model.bin", "tokenizer.json")
    if not model_path.is_dir() or any(not (model_path / name).is_file() for name in required):
        raise EvaluationError("A complete local CTranslate2 checkpoint is required")
    if sum(path.stat().st_size for path in model_path.iterdir() if path.is_file()) > 1_100_000_000:
        raise EvaluationError("The CPU evaluation accepts at most 1.1 GB of recognizer files")
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise EvaluationError("Install tools/voice_evaluation/requirements-asr.txt in a separate environment") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"schema_version": "voice_asr.v1", "created_at": timestamp(), "status": "running",
        "synthesis_sha256": digest(source), "condition": synthesis.get("condition"),
        "engine": {"name": "faster-whisper", "version": importlib.metadata.version("faster-whisper"),
            "ctranslate2_version": importlib.metadata.version("ctranslate2"), "model": model_repo,
            "revision": revision, "weights_sha256": digest(model_path / "model.bin"),
            "device": "cpu", "compute_type": "int8", "cpu_threads": cpu_threads, "num_workers": 1,
            "beam_size": 5, "initial_prompt": None, "prefix": None, "hotwords": None,
            "condition_on_previous_text": False, "vad_filter": False, "word_timestamps": True,
            "decoder": "ffmpeg_pcm16_mono_16khz_to_numpy", "telemetry_disabled_before_import": True},
        "human_listening": "not_performed", "cases": []}
    checkpoint(target, report)
    load_started = time.monotonic()
    try:
        model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=cpu_threads,
            num_workers=1, local_files_only=True)
    except Exception as exc:
        report.update(status="incomplete", error_type=type(exc).__name__, stage="model_load",
            elapsed_seconds=round(time.monotonic() - started, 6))
        checkpoint(target, report)
        return report
    report["model_load_seconds"] = round(time.monotonic() - load_started, 6)
    for row in selected:
        observed = {"id": row["id"], "audio_sha256": row["audio"]["sha256"],
            "asr_language": row["asr_language"], "status": "running"}
        report["cases"].append(observed)
        checkpoint(target, report)
        case_started = time.monotonic()
        try:
            audio = verified_audio(source, row)
            signal = wav_metrics(audio.read_bytes())
            decoded = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-protocol_whitelist", "file,pipe", "-i", str(audio), "-map", "0:a:0", "-vn",
                "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
                check=True, capture_output=True, timeout=30)
            if not decoded.stdout or len(decoded.stdout) > 180 * 16000 * 2 or len(decoded.stdout) % 2:
                raise EvaluationError("Invalid or oversized decoded ASR input")
            samples = np.frombuffer(decoded.stdout, dtype="<i2").astype(np.float32) / 32768.0
            segments, info = model.transcribe(samples, language=row["asr_language"], beam_size=5,
                word_timestamps=True, vad_filter=False, condition_on_previous_text=False,
                initial_prompt=None, prefix=None, hotwords=None)
            transcript, timings = [], []
            for segment in segments:
                transcript.append(segment.text.strip())
                timings.append({"start": segment.start, "end": segment.end, "text": segment.text,
                    "words": [{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} for word in (segment.words or [])]})
                if len(timings) > 512 or sum(map(len, transcript)) > 12000:
                    raise EvaluationError("ASR exceeded the transcript output budget")
            text = " ".join(transcript).strip()
            observed.update(status="succeeded", recognized_text=text, expected_text=row["text"],
                detected_language=info.language, segments=timings,
                elapsed_seconds=round(time.monotonic() - case_started, 6),
                audio_seconds=signal["duration_seconds"], metrics=lexical_metrics(row["text"], text, row["asr_language"]))
        except (EvaluationError, OSError, subprocess.SubprocessError, RuntimeError) as exc:
            observed.update(status="failed", error_type=type(exc).__name__, elapsed_seconds=round(time.monotonic() - case_started, 6))
        checkpoint(target, report)
    report.update(status="completed" if all(row["status"] == "succeeded" for row in report["cases"]) else "incomplete",
        completed_at=timestamp(), elapsed_seconds=round(time.monotonic() - started, 6))
    checkpoint(target, report)
    return report
