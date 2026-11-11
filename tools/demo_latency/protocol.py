"""Strict streaming and PCM checks shared by the actual-service harness."""
from __future__ import annotations

import array
import hashlib
import io
import json
import math
import re
import sys
import wave
from collections.abc import Iterable, Iterator


def sse_events(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """Decode complete SSE frames, including comments, CRLF and multiline data."""
    event, data, length = "message", [], 0
    for raw in lines:
        line = raw.rstrip("\r\n")
        length += len(line)
        if length > 1_048_576:
            raise ValueError("SSE frame exceeds 1 MiB")
        if not line:
            if data:
                payload = json.loads("\n".join(data))
                if not isinstance(payload, dict):
                    raise ValueError("SSE data must be a JSON object")
                yield event, payload
            event, data, length = "message", [], 0
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        payload = json.loads("\n".join(data))
        if not isinstance(payload, dict):
            raise ValueError("SSE data must be a JSON object")
        yield event, payload


def speech_text(text: str) -> str:
    """Match the production browser's citation removal without changing words."""
    return re.sub(r"\[source:[A-Za-z0-9_.-]+\]", "", text).strip()


def pcm_info(data: bytes, expected_sha256: str, *, threshold: float = 0.002) -> dict:
    """Inspect a fully acquired WAV; this is not browser/acoustic playback."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError("The downloaded WAV does not match its speech package")
    if not 0 < threshold < 1:
        raise ValueError("PCM RMS threshold must be between zero and one")
    with wave.open(io.BytesIO(data), "rb") as stream:
        channels, rate, count = stream.getnchannels(), stream.getframerate(), stream.getnframes()
        if stream.getcomptype() != "NONE" or stream.getsampwidth() != 2 or channels not in {1, 2}:
            raise ValueError("Expected mono or stereo PCM16 WAV")
        if not 8000 <= rate <= 96000 or not 0 < count / rate <= 180:
            raise ValueError("WAV duration or sample rate is out of bounds")
        raw = stream.readframes(count)
        if len(raw) != count * channels * 2:
            raise ValueError("The WAV payload is truncated")
    samples = array.array("h", raw)
    if sys.byteorder != "little":
        samples.byteswap()
    frames_per_window = max(1, round(rate * 0.01))
    stride = frames_per_window * channels
    first_signal = None
    peak_rms = 0.0
    for start in range(0, len(samples), stride):
        window = samples[start:start + stride]
        rms = math.sqrt(sum(float(value) ** 2 for value in window) / len(window)) / 32768
        peak_rms = max(peak_rms, rms)
        if first_signal is None and rms > threshold:
            first_signal = start / channels / rate * 1000
    return {"sha256": digest, "bytes": len(data), "sample_rate": rate, "channels": channels,
            "frames": count, "duration_ms": count / rate * 1000,
            "offline_pcm_onset_offset_ms": first_signal, "offline_pcm_peak_window_rms": peak_rms,
            "offline_pcm_detector": {"rms_threshold": threshold, "window_frames": frames_per_window,
                "window_ms": frames_per_window / rate * 1000,
                "basis": "Decoded complete WAV, consecutive non-overlapping PCM windows; no playback."}}
