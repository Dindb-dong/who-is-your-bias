"""Optional audio-based mouth-cue analysis, with an explicit fallback state."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LipsyncResult:
    status: str
    timing: dict | None = None
    message: str | None = None


def wav_duration_ms(data: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(data), "rb") as stream:
            rate = stream.getframerate()
            frames = stream.getnframes()
            channels = stream.getnchannels()
            if stream.getcomptype() != "NONE" or stream.getsampwidth() != 2 or channels not in {1, 2} or not 8000 <= rate <= 96000:
                return None
            if not rate / 1000 <= frames <= rate * 180:
                return None
            if len(stream.readframes(frames)) != frames * channels * 2:
                return None
            return frames / rate * 1000
    except (wave.Error, EOFError, OSError):
        return None


def parse_rhubarb(document: dict, *, audio_sha256: str, duration_ms: float, version: str | None = None) -> dict:
    """Keep predicted A–H/X labels; do not present them as aligned phonemes."""
    if not re.fullmatch(r"[a-f0-9]{64}", audio_sha256) or isinstance(duration_ms, bool) or not math.isfinite(duration_ms) or not 1 <= duration_ms <= 180000:
        raise ValueError("Invalid audio identity or duration")
    if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict):
        raise ValueError("Missing mouth-cue metadata")
    declared = document["metadata"].get("duration")
    if isinstance(declared, bool) or not isinstance(declared, (int, float)) or not math.isfinite(declared) or abs(declared * 1000 - duration_ms) > 100:
        raise ValueError("Mouth cues do not match the audio duration")
    cues = document.get("mouthCues")
    if not isinstance(cues, list) or not 0 < len(cues) <= 10000:
        raise ValueError("Invalid number of mouth cues")
    segments = []
    previous_end = 0.0
    for cue in cues:
        if not isinstance(cue, dict) or cue.get("value") not in set("ABCDEFGHX"):
            raise ValueError("Unsupported mouth label")
        start, end = cue.get("start"), cue.get("end")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (start, end)):
            raise ValueError("Invalid mouth-cue time")
        start_ms, end_ms = round(start * 1000, 3), round(end * 1000, 3)
        if start_ms < previous_end or start_ms < 0 or not start_ms < end_ms or end_ms > duration_ms + 1:
            raise ValueError("Overlapping or out-of-range mouth cues")
        end_ms = min(end_ms, duration_ms)
        if start_ms >= end_ms:
            raise ValueError("Empty mouth cue")
        segments.append({"start_ms": start_ms, "end_ms": end_ms, "label": cue["value"]})
        previous_end = end_ms
    provenance = {"engine": "rhubarb-lip-sync"}
    if version:
        provenance["version"] = version
    return {"schema_version": "speech-timing.v1", "audio_sha256": audio_sha256,
            "audio_duration_ms": duration_ms, "timing_basis": "audio_recognition",
            "label_set": "rhubarb", "segments": segments, "provenance": provenance}


class LipsyncEngine:
    def __init__(self, binary: str | None = None, *, timeout: float = 30, threads: int = 2):
        if not 1 <= timeout <= 120 or not 1 <= threads <= 8:
            raise ValueError("Invalid lip-sync resource bounds")
        self.binary = binary
        self.timeout = timeout
        self.threads = threads

    @classmethod
    def from_env(cls) -> "LipsyncEngine":
        return cls(os.getenv("BIAS_LIPSYNC_BIN") or None,
                   timeout=float(os.getenv("BIAS_LIPSYNC_TIMEOUT", "30")),
                   threads=int(os.getenv("BIAS_LIPSYNC_THREADS", "2")))

    @property
    def configured(self) -> bool:
        return bool(self.binary)

    async def analyze(self, data: bytes, text: str, language: str = "en") -> LipsyncResult:
        if not self.binary:
            return LipsyncResult("not_configured", message="Audio-based lip sync is not configured; playback uses approximate mouth movement.")
        if len(data) > 20_000_000 or len(text) > 4000 or "\x00" in text:
            return LipsyncResult("unsupported_audio", message="The audio or transcript exceeds the lip-sync input bounds.")
        duration = wav_duration_ms(data)
        if duration is None:
            return LipsyncResult("unsupported_audio", message="Audio-based lip sync requires a bounded PCM16 WAV; the original audio remains playable.")
        binary = shutil.which(self.binary)
        if not binary:
            return LipsyncResult("unavailable", message="The configured lip-sync executable is unavailable.")
        process = None
        with tempfile.TemporaryDirectory(prefix="bias-lipsync-") as directory:
            root = Path(directory)
            audio_path, text_path, result_path = root / "speech.wav", root / "speech.txt", root / "mouth.json"
            audio_path.write_bytes(data)
            text_path.write_text(text, encoding="utf-8")
            arguments = [binary, "-f", "json", "-o", str(result_path), "-r", "pocketSphinx" if language == "en" else "phonetic",
                         "--threads", str(self.threads), "--quiet"]
            if language == "en":
                arguments += ["-d", str(text_path)]
            arguments.append(str(audio_path))
            try:
                process = await asyncio.create_subprocess_exec(*arguments, stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await asyncio.wait_for(process.wait(), timeout=self.timeout)
                if process.returncode != 0 or not result_path.is_file() or result_path.stat().st_size > 1_500_000:
                    return LipsyncResult("failed", message="Audio analysis did not produce usable mouth cues; playback uses approximate movement.")
                document = json.loads(result_path.read_text(encoding="utf-8"))
                timing = parse_rhubarb(document, audio_sha256=hashlib.sha256(data).hexdigest(), duration_ms=duration)
                timing["provenance"]["model"] = "pocketSphinx" if language == "en" else "phonetic"
                return LipsyncResult("available", timing)
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                return LipsyncResult("timeout", message="Audio analysis timed out; playback uses approximate mouth movement.")
            except (OSError, ValueError, TypeError, KeyError):
                return LipsyncResult("failed", message="Audio analysis failed; playback uses approximate mouth movement.")
            finally:
                if process is not None and process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
