"""Small report helpers which do not import optional model dependencies."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("torch", "torchaudio", "transformers", "accelerate", "qwen-tts", "huggingface-hub", "numpy", "soundfile"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def safe_error(error: Exception) -> str:
    """Keep a useful failure reason without retaining signed download queries."""
    def clean(match: re.Match[str]) -> str:
        parsed = urlsplit(match.group(0))
        return urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))
    return f"{type(error).__name__}: " + re.sub(r"https?://[^\s\"<>]+", clean, str(error))[:2000]


def new_report(kind: str) -> dict:
    return {
        "schema_version": "inference-smoke.v1",
        "kind": kind,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "versions": package_versions(),
        "scope": "Execution smoke only; not persona, voice likeness, or product-quality validation.",
    }


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def progress(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)
