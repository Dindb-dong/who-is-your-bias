"""Import local, timestamped transcripts as review-required evidence drafts."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .models import Evidence, Locator

TIMING = re.compile(r"(?P<start>(?:\d{1,3}:)?\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(?P<end>(?:\d{1,3}:)?\d{2}:\d{2}[.,]\d{1,3})")


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end) or self.start < 0 or self.end <= self.start:
            raise ValueError("Transcript segments require 0 <= start < end")
        if not self.text.strip():
            raise ValueError("Transcript segments cannot contain blank text")


def timestamp(value: str) -> float:
    pieces = value.replace(",", ".").split(":")
    if len(pieces) == 2:
        hours = 0
        minutes, seconds = pieces
    elif len(pieces) == 3:
        hours, minutes, seconds = pieces
    else:
        raise ValueError(f"Invalid timestamp: {value}")
    if int(minutes) >= 60 or float(seconds) >= 60:
        raise ValueError(f"Invalid timestamp: {value}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_captions(text: str) -> list[Segment]:
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    segments: list[Segment] = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip().splitlines()
        if not lines or lines[0].startswith(("NOTE", "STYLE", "REGION")):
            continue
        index = next((i for i, line in enumerate(lines) if TIMING.search(line)), None)
        if index is None:
            continue
        match = TIMING.search(lines[index])
        assert match is not None
        raw = " ".join(lines[index + 1:])
        speaker_match = re.search(r"<v(?:\.[^ >]+)?\s+([^>]+)>", raw)
        clean = html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()
        if not clean:
            continue
        segments.append(Segment(timestamp(match["start"]), timestamp(match["end"]), clean, speaker_match[1] if speaker_match else None))
    if not segments:
        raise ValueError("No valid timestamped captions found")
    return segments


def parse_whisperx(data: dict) -> list[Segment]:
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError("Whisper/WhisperX JSON must contain a segments array")
    result = []
    for segment in data["segments"]:
        if not isinstance(segment, dict):
            raise ValueError("Each transcript segment must be an object")
        text = segment.get("text", "").strip()
        if not text:
            continue
        if segment.get("start") is None or segment.get("end") is None:
            raise ValueError("Each nonempty transcript segment needs start and end timestamps")
        result.append(Segment(float(segment["start"]), float(segment["end"]), text, segment.get("speaker")))
    if not result:
        raise ValueError("No nonempty timestamped transcript segments found")
    return result


def import_transcript(path: Path, *, source_id: str, timeline: str, source_offset: float = 0, speaker: str | None = None, manifest: Path | None = None) -> list[Evidence]:
    if path.stat().st_size > 20_000_000:
        raise ValueError("Transcript exceeds the 20 MB import limit")
    if source_offset < 0:
        raise ValueError("source_offset must be nonnegative")
    raw = path.read_bytes()
    if manifest:
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if info.get("source_id") != source_id:
            raise ValueError("Manifest source_id does not match the requested source")
        source_offset = float(info["source_start_seconds"])
        timeline = str(info["source_timeline"])
        if not math.isfinite(source_offset) or source_offset < 0:
            raise ValueError("Manifest source_start_seconds must be finite and nonnegative")
    if path.suffix.lower() == ".json":
        segments = parse_whisperx(json.loads(raw.decode("utf-8-sig")))
    elif path.suffix.lower() in {".srt", ".vtt"}:
        segments = parse_captions(raw.decode("utf-8-sig"))
    else:
        raise ValueError("Supported transcript files are .srt, .vtt, and WhisperX .json")
    digest = hashlib.sha256(raw).hexdigest()
    records = []
    for index, segment in enumerate(segments):
        if speaker is not None and segment.speaker != speaker:
            continue
        records.append(Evidence(
            id=f"{source_id}-{digest[:10]}-{index:05d}",
            source_id=source_id,
            text=segment.text,
            topics=[],
            locator=Locator(label=f"Imported transcript segment {index + 1}", start_seconds=round(source_offset + segment.start, 3), end_seconds=round(source_offset + segment.end, 3), timeline=timeline),
            kind="transcript", confidence="unverified_transcript", review_status="draft",
            behavioral_observation=None,
            transcript_speaker=segment.speaker,
            transcript_sha256=digest,
            review_required=["speaker_identity", "transcription_accuracy", "source_timeline_alignment"],
        ))
    if not records:
        raise ValueError("No transcript segments match the selected speaker")
    return records


def write_evidence(records: list[Evidence], output: Path, *, overwrite: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids silently replacing curated data.
    with output.open("w" if overwrite else "x", encoding="utf-8") as stream:
        for record in records:
            stream.write(record.model_dump_json(exclude_none=True) + "\n")
