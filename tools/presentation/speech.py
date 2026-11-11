"""Synthesize each prepared segment with the existing real Qwen3-TTS engine."""
from __future__ import annotations

import argparse
import io
import platform
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

from .common import digest, read_plan, save, subtitles, wav_info


def synthesize(plan_path: Path, output: Path) -> dict:
    from services.voice.engine import VoiceEngine, VoiceSettings
    plan = read_plan(plan_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory to preserve existing media")
    output.mkdir(parents=True, exist_ok=True)
    settings = VoiceSettings.from_env()
    if not settings.revision:
        raise ValueError("Pin BIAS_VOICE_REVISION to an immutable model revision")
    report = {"schema_version": "prepared-speech.v1", "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(), "backend": "qwen3-tts-base",
        "model": settings.model_id, "revision": settings.revision, "device": settings.device,
        "dtype": settings.dtype, "threads": settings.threads, "seed": settings.seed,
        "plan_sha256": digest(plan_path), "reference_audio_sha256": digest(settings.reference_wav) if settings.reference_wav else None,
        "reference_transcript_sha256": __import__("hashlib").sha256((settings.reference_text or "").encode()).hexdigest(),
        "platform": platform.platform(), "segments": [], "human_listening_review": "unrated"}
    save(output / "speech.json", report)
    start = time.perf_counter()
    try:
        engine = VoiceEngine(settings)
        chunks, total_frames, rate = [], 0, None
        for segment in plan["segments"]:
            begin = time.perf_counter()
            payload = engine.synthesize(segment["text"], "English" if plan["language"] == "en" else "Korean")
            path = output / (segment["id"] + ".wav")
            path.write_bytes(payload)
            info = wav_info(path)
            with wave.open(io.BytesIO(payload), "rb") as stream:
                if rate is not None and stream.getframerate() != rate:
                    raise ValueError("Synthesis sample rates changed between segments")
                rate = stream.getframerate()
                pcm = stream.readframes(stream.getnframes())
            report["segments"].append({**segment, "path": path.name, **info,
                "start_seconds": total_frames / rate, "end_seconds": (total_frames + info["frames"]) / rate,
                "request_seconds": time.perf_counter() - begin,
                "backend_generation": engine.status()["last_generation"]})
            total_frames += info["frames"]
            chunks.append(pcm)
            # Keep a short exact silence between segments, outside subtitle cues.
            silence = int(rate * 0.2)
            if segment is not plan["segments"][-1]:
                chunks.append(b"\0\0" * silence)
                total_frames += silence
            save(output / "speech.json", report)
        with wave.open(str(output / "speech.wav"), "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate)
            stream.writeframes(b"".join(chunks))
        (output / "captions.srt").write_text(subtitles(report["segments"]), encoding="utf-8")
        (output / "captions.vtt").write_text(subtitles(report["segments"], vtt=True), encoding="utf-8")
        report.update(status="completed", audio=wav_info(output / "speech.wav"),
            total_seconds=time.perf_counter() - start,
            completion_scope="Waveforms generated; token-cap/EOS and transcript accuracy need separate inspection.")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(output / "speech.json", report)
        raise
    save(output / "speech.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(synthesize(args.plan, args.output)["status"])


if __name__ == "__main__":
    main()
