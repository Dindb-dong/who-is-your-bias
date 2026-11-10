"""Small standard-library helpers; importing media tools never loads models."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class MediaError(ValueError):
    """An actionable input, dependency, or media-processing failure."""


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_number(value: Any, name: str, low: float = 0, high: float = 1e9) -> float:
    if isinstance(value, bool):
        raise MediaError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MediaError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise MediaError(f"{name} must be between {low} and {high}")
    return number


def local_file(value: str | Path, *, max_bytes: int = 4 * 1024**3) -> Path:
    if isinstance(value, str) and "://" in value:
        raise MediaError("Provide an existing local file, not a URL")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise MediaError(f"Local file does not exist: {path.name}")
    if path.stat().st_size > max_bytes:
        raise MediaError(f"File exceeds the {max_bytes}-byte input limit")
    return path


def source_identity(source_id: str, source_url: str, timeline: str) -> None:
    if not isinstance(source_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", source_id):
        raise MediaError("Source ID must contain only lowercase letters, numbers, dots, dashes, or underscores")
    if not isinstance(source_url, str):
        raise MediaError("Source URL must be an HTTPS provenance URL")
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise MediaError("Source URL must be an HTTPS provenance URL without credentials")
    if not isinstance(timeline, str) or not timeline.strip() or len(timeline) > 160:
        raise MediaError("Name the source media timeline explicitly (up to 160 characters)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(manifest_path: Path, name: str) -> Path:
    """A manifest may only refer to files inside its own directory, including symlinks."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise MediaError("Manifest artifact names must be relative paths")
    base = manifest_path.resolve().parent
    candidate = (base / name).resolve()
    if not candidate.is_relative_to(base):
        raise MediaError("Manifest artifact path escapes the clip directory")
    return local_file(candidate)


def load_manifest(value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = local_file(value, max_bytes=1024 * 1024)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MediaError("Manifest must be UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "media.v1":
        raise MediaError("Expected a media.v1 clip manifest")
    source_identity(manifest.get("source_id", ""), manifest.get("source_url", ""), manifest.get("source_timeline", ""))
    finite_number(manifest.get("source_start_seconds"), "source_start_seconds")
    clip = manifest.get("clip")
    if not isinstance(clip, dict):
        raise MediaError("Manifest clip description is missing")
    finite_number(clip.get("duration_seconds"), "duration_seconds", 0.001, 120)
    return path, manifest


def verify_artifact(manifest_path: Path, manifest: dict[str, Any], kind: str) -> Path:
    path = artifact_path(manifest_path, manifest["clip"].get(kind, ""))
    expected = manifest.get("artifacts", {}).get(kind, {}).get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise MediaError(f"Manifest has no valid {kind} SHA-256")
    if sha256(path) != expected:
        raise MediaError(f"{kind} has changed since preparation; prepare a new manifest")
    return path


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        return
    fd, temporary = tempfile.mkstemp(prefix=".media-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_media(command: list[str], *, timeout: int = 180) -> str:
    if not shutil.which(command[0]):
        raise MediaError(f"Install {command[0]} and make it available on PATH")
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{command[0]} exceeded its {timeout}-second processing limit") from exc
    if result.returncode:
        # Local CLI diagnostics only, never returned by the voice HTTP service.
        detail = result.stderr.strip()[-1200:]
        raise MediaError(f"{command[0]} failed: {detail}")
    return result.stdout


def probe(path: Path) -> dict[str, Any]:
    output = run_media([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-show_entries",
        "format=duration:stream=codec_type,width,height,avg_frame_rate,sample_rate,channels,duration",
        "-of", "json", str(path),
    ], timeout=30)
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise MediaError("ffprobe returned invalid metadata") from exc
