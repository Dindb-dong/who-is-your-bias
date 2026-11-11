"""Small dependency-light presentation contracts and media checks."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "presentation-plan.v1":
        raise ValueError("Unsupported presentation plan")
    if not isinstance(plan.get("title"), str) or not 1 <= len(plan["title"]) <= 120:
        raise ValueError("Presentation title is required")
    if plan.get("language") not in {"en", "ko"}:
        raise ValueError("Use en or ko for the presentation language")
    segments = plan.get("segments")
    if not isinstance(segments, list) or not 1 <= len(segments) <= 30:
        raise ValueError("Use between one and thirty segments")
    ids = []
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {"id", "text", "source_refs"}:
            raise ValueError("Every segment needs id, text, and source_refs")
        if not isinstance(segment["id"], str) or not segment["id"].isascii() or not segment["id"].replace("-", "").isalnum():
            raise ValueError("Segment ids must be ASCII letters, digits, or hyphens")
        text = segment["text"]
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 240 or any(ord(c) < 32 for c in text):
            raise ValueError("Each segment must contain 1–240 printable characters")
        if not isinstance(segment["source_refs"], list) or not segment["source_refs"] or not all(isinstance(x, str) and x for x in segment["source_refs"]):
            raise ValueError("Each segment needs its source references")
        ids.append(segment["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Segment ids must be unique")
    if plan.get("disclosure") != "AI research simulation — not the real person":
        raise ValueError("The explicit simulation disclosure is required")
    return plan


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as stream:
        if stream.getcomptype() != "NONE" or stream.getsampwidth() != 2 or stream.getnchannels() != 1:
            raise ValueError("Presentation audio must be mono PCM16 WAV")
        frames, rate = stream.getnframes(), stream.getframerate()
        if not 8000 <= rate <= 96000 or not 0 < frames / rate <= 600:
            raise ValueError("Audio duration or sample rate is out of bounds")
        payload = stream.readframes(frames)
        if len(payload) != frames * 2:
            raise ValueError("Audio is truncated")
    return {"sample_rate": rate, "frames": frames, "seconds": frames / rate, "sha256": digest(path)}


def probe(path: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def check_timing(media: dict, audio_seconds: float, fps: int) -> dict:
    streams = media.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ValueError("A presentation must have exactly one video and one audio stream")
    video, audio = videos[0], audios[0]
    def finite(value: object) -> float:
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("Media timing must be finite and nonnegative")
        return number
    vd, ad = finite(video["duration"]), finite(audio["duration"])
    vs, ass = finite(video.get("start_time", 0)), finite(audio.get("start_time", 0))
    # AAC output can include one 1024-sample packet of rounding; no claim of
    # perceptual synchronization follows from a successful container check.
    tolerance = max(1 / fps, 1024 / int(audio["sample_rate"])) + 0.005
    if abs(vd - audio_seconds) > tolerance or abs(ad - audio_seconds) > tolerance or abs(vs - ass) > tolerance:
        raise ValueError("Audio/video duration or start offset exceeds one frame/packet")
    if video.get("codec_name") != "h264" or audio.get("codec_name") != "aac":
        raise ValueError("Portable playback requires H.264 video and AAC audio")
    return {"passed": True, "video_seconds": vd, "audio_seconds": ad,
        "source_audio_seconds": audio_seconds, "start_offset_seconds": abs(vs - ass),
        "duration_difference_seconds": abs(vd - ad), "tolerance_seconds": tolerance,
        "scope": "Container/timeline integrity; visual lip-sync quality requires review."}


def timestamp(seconds: float, *, vtt: bool = False) -> str:
    milliseconds = round(seconds * 1000)
    hours, rem = divmod(milliseconds, 3600000)
    minutes, rem = divmod(rem, 60000)
    whole, fraction = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{whole:02}{'.' if vtt else ','}{fraction:03}"


def subtitles(segments: list[dict], *, vtt: bool = False) -> str:
    rows = ["WEBVTT\n"] if vtt else []
    previous_end = 0.0
    for index, segment in enumerate(segments, 1):
        start, end = segment["start_seconds"], segment["end_seconds"]
        if not math.isfinite(start) or not math.isfinite(end) or start < previous_end or end <= start:
            raise ValueError("Subtitle segments must be ordered, finite, and non-overlapping")
        text = segment["text"].replace("-->", "→").replace("<", "‹").replace(">", "›")
        rows.append(f"{index}\n{timestamp(start, vtt=vtt)} --> {timestamp(end, vtt=vtt)}\n{text}\n")
        previous_end = end
    return "\n".join(rows) + "\n"
