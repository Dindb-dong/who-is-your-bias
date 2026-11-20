from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


class PipelineError(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def run_id(prefix):
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", str(value)):
        raise PipelineError("ID는 영문·숫자·밑줄·하이픈으로 1~80자 입력하세요.")
    return str(value)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text):
    return hashlib.sha256(re.sub(r"\W+", "", text.casefold()).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_csv(path, rows, fields):
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, "\ufeff" + stream.getvalue())


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def boolean(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def interval(start, end, duration=None):
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        raise PipelineError("시작·종료 시각은 초 단위 숫자여야 합니다.") from None
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise PipelineError(f"잘못된 시간 범위: {start}~{end}")
    if duration is not None and end > duration + 0.05:
        raise PipelineError(f"종료 시각 {end}초가 원음 길이 {duration:.2f}초를 초과합니다.")
    return start, end


def load_env(path):
    """Read simple KEY=value pairs as data; never source/execute shell text."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", key.strip()):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
