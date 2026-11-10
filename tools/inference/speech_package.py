"""Exercise real HTTP speech preparation, audio retrieval, and audio/timing binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from bias.lipsync import wav_duration_ms

from .common import new_report, progress, safe_error, save_report


def read_bounded(response: httpx.Response, limit: int) -> bytes:
    response.raise_for_status()
    data = bytearray()
    for part in response.iter_bytes():
        data.extend(part)
        if len(data) > limit:
            raise ValueError("HTTP response exceeded the smoke's byte bound")
    return bytes(data)


def run(args: argparse.Namespace) -> dict:
    report = new_report("speech_package_http")
    report.update(text=args.text, language=args.language, voice_likeness="not_evaluated",
                  intelligibility="not_evaluated", browser_playback="not_exercised")
    save_report(args.report, report)
    started = time.perf_counter()
    try:
        url = urlsplit(args.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}:
            raise ValueError("Use an HTTP(S) API origin without credentials, path, query, or fragment")
        # Loopback services do not need a public proxy. Other explicit origins
        # keep the normal environment proxy configuration.
        local = url.hostname in {"127.0.0.1", "localhost", "::1"}
        with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout, trust_env=not local) as client:
            with client.stream("POST", "/api/speech/prepare", json={"text": args.text, "language": args.language}) as response:
                package = json.loads(read_bounded(response, 2_000_000))
            report["prepare_seconds"] = time.perf_counter() - started
            if not isinstance(package, dict) or not re.fullmatch(r"/api/speech/clips/[a-f0-9]{32}", package.get("audio_url", "")):
                raise ValueError("The prepared package has an unexpected audio URL")
            with client.stream("GET", package["audio_url"]) as response:
                audio = read_bounded(response, 20_000_000)
                mime = response.headers.get("content-type", "").split(";", 1)[0]
        digest = hashlib.sha256(audio).hexdigest()
        if not audio or digest != package.get("audio_sha256"):
            raise ValueError("Retrieved audio does not match the prepared audio hash")
        timing = package.get("speech_timing")
        if timing is not None and (timing.get("audio_sha256") != digest or timing.get("schema_version") != "speech-timing.v1"):
            raise ValueError("Mouth cues are not bound to the retrieved audio")
        duration = wav_duration_ms(audio)
        if timing is not None and (duration is None or abs(duration - timing.get("audio_duration_ms", 0)) > 1):
            raise ValueError("Mouth-cue and decoded audio durations differ")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(audio)
        if timing is not None:
            args.output.with_suffix(".timing.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
        report.update(status="completed", http_audio_roundtrip=True, output=str(args.output), audio_sha256=digest,
                      audio_bytes=len(audio), mime_type=mime, audio_duration_ms=duration,
                      timing_status=package.get("timing_status"), timing_basis=timing.get("timing_basis") if timing else None,
                      mouth_cues=len(timing["segments"]) if timing else 0,
                      timing_provenance=timing.get("provenance") if timing else None,
                      total_seconds=time.perf_counter() - started)
        progress("speech_package_completed", seconds=report["total_seconds"], mouth_cues=report["mouth_cues"])
    except Exception as error:
        report.update(status="failed", error=safe_error(error), total_seconds=time.perf_counter() - started)
        progress("speech_package_failed", error=report["error"])
    save_report(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--text", default="This is a research simulation.")
    parser.add_argument("--language", choices=("en", "ko"), default="en")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=Path("reports/local/voice-http.wav"))
    parser.add_argument("--report", type=Path, default=Path("reports/local/voice-http.json"))
    args = parser.parse_args()
    if not 1 <= args.timeout <= 300 or not 1 <= len(args.text.strip()) <= 200:
        parser.error("Use a 1–300 second timeout and 1–200 input characters")
    return 0 if run(args)["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
