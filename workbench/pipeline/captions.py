from __future__ import annotations

import html
import re
from pathlib import Path

from .common import PipelineError, interval, read_json

STAMP = r"(?:\d+:)?\d{2}:\d{2}[.,]\d{3}"
TIMING = re.compile(rf"^\s*({STAMP})\s*-->\s*({STAMP})")


def seconds(value):
    parts = value.replace(",", ".").split(":")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def parse(path):
    """Preserve cues and raw text. Rolling-caption duplicates remain visible for review."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = read_json(path)
        rows = value.get("segments", []) if isinstance(value, dict) else value
        if not isinstance(rows, list):
            raise PipelineError("JSON 자막은 segments 배열 또는 배열 자체여야 합니다.")
        cues = [{"start": r["start"], "end": r["end"], "text": r["text"]} for r in rows]
    elif path.suffix.lower() in {".vtt", ".srt"}:
        lines = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").splitlines()
        cues, i = [], 0
        while i < len(lines):
            match = TIMING.match(lines[i])
            if not match:
                i += 1
                continue
            start, end = (seconds(v) for v in match.groups())
            i += 1
            body = []
            while i < len(lines) and lines[i].strip() and not TIMING.match(lines[i]):
                body.append(lines[i])
                i += 1
            raw = "\n".join(body)
            cleaned = html.unescape(re.sub(r"<[^>]*>", "", raw))
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                cues.append({"start": start, "end": end, "text": cleaned, "raw_text": raw})
    else:
        raise PipelineError("자막은 SRT, VTT, JSON 중 하나여야 합니다.")
    if not cues:
        raise PipelineError("시간이 있는 자막 구간을 찾지 못했습니다.")
    result = []
    for i, cue in enumerate(cues, 1):
        start, end = interval(cue["start"], cue["end"])
        if not isinstance(cue["text"], str) or not cue["text"].strip():
            raise PipelineError("빈 자막은 사용할 수 없습니다.")
        result.append({**cue, "segment_id": f"seg-{i:05d}", "start": start, "end": end})
    return result
