"""Sequential local HTTP voice evaluation with atomic partial-run checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .metrics import EvaluationError, wav_metrics, words

DEFAULT_CASES = Path(__file__).with_name("cases.json")
ID = re.compile(r"[a-z][a-z0-9_-]{0,63}")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def read_json(path: str | Path, *, max_bytes: int = 2_000_000) -> dict:
    file = Path(path)
    if not file.is_file() or file.stat().st_size > max_bytes:
        raise EvaluationError("JSON artifact is missing or too large")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise EvaluationError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise EvaluationError("Nonfinite JSON value")
    try:
        data = json.loads(file.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvaluationError("Invalid JSON document") from exc
    if not isinstance(data, dict):
        raise EvaluationError("Expected a JSON object")
    return data


def checkpoint(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def cases_from_file(path: str | Path) -> list[dict]:
    data = read_json(path)
    cases = data.get("cases")
    if data.get("schema_version") != "voice_cases.v1" or not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise EvaluationError("Expected 1–32 declared voice_cases.v1 cases")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not ID.fullmatch(case["id"]) or case["id"] in seen:
            raise EvaluationError("Case IDs must be distinct simple identifiers")
        seen.add(case["id"])
        if (case.get("language"), case.get("asr_language")) not in {("English", "en"), ("Korean", "ko")}:
            raise EvaluationError("Declare matching English/en or Korean/ko language hints")
        if not isinstance(case.get("text"), str) or not 1 <= len(case["text"].strip()) <= 2000 or "\x00" in case["text"]:
            raise EvaluationError("Each case needs 1–2000 characters of text without NUL")
        if not words(case["text"]):
            raise EvaluationError("Each case needs scoreable text")
    return cases


def local_base_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        hostname, port = parsed.hostname, parsed.port
        loopback = hostname == "localhost" or bool(hostname and ipaddress.ip_address(hostname).is_loopback)
    except ValueError:
        loopback = False
    if not loopback or parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise EvaluationError("Use a local loopback service origin without credentials, path, query, or fragment")
    return url.rstrip("/")


def verified_audio(report_path: Path, row: dict) -> Path:
    artifact = row.get("audio")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}\.wav", artifact["path"]):
        raise EvaluationError("Invalid evaluation audio artifact")
    file = report_path.parent / artifact["path"]
    if file.is_symlink() or not file.is_file() or file.stat().st_size > 20_000_000 or digest(file) != artifact.get("sha256"):
        raise EvaluationError("Audio artifact is missing, replaced, oversized, or has a mismatched hash")
    return file


def synthesis_rows(report: dict) -> list[dict]:
    rows = report.get("cases")
    if report.get("schema_version") != "voice_synthesis.v1" or not isinstance(rows, list) or not 1 <= len(rows) <= 32:
        raise EvaluationError("Expected 1–32 voice_synthesis.v1 cases")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not ID.fullmatch(row["id"]) or row["id"] in seen:
            raise EvaluationError("Synthesis case IDs must be distinct simple identifiers")
        seen.add(row["id"])
        if (row.get("language"), row.get("asr_language")) not in {("English", "en"), ("Korean", "ko")}:
            raise EvaluationError("Synthesis cases require matching declared languages")
        if not isinstance(row.get("text"), str) or not 1 <= len(row["text"].strip()) <= 2000 or "\x00" in row["text"]:
            raise EvaluationError("Synthesis cases require bounded expected text")
    return rows


async def _health(client: httpx.AsyncClient, base: str, headers: dict, deadline: float) -> dict:
    try:
        async with asyncio.timeout(min(3.0, max(0.001, deadline - time.monotonic()))):
            async with client.stream("GET", base + "/health", headers=headers) as response:
                response.raise_for_status()
                raw = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    raw.extend(chunk)
                    if len(raw) > 16000:
                        raise EvaluationError("Health response exceeds 16 KB")
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise EvaluationError("Health response is not an object")
                return {key: data[key] for key in ("status", "reference_configured", "reference_mode", "model_loaded", "busy", "streaming", "runtime", "last_generation",
                    "persona_id", "reference_identity", "adaptation", "synthesis") if key in data}
    except (ValueError, TimeoutError, httpx.HTTPError) as exc:
        return {"unavailable": type(exc).__name__}


async def run_synthesis(*, base_url: str, output: str | Path, cases_path: str | Path = DEFAULT_CASES,
    condition: str, api_key: str | None = None, per_case_timeout: float = 300,
    total_timeout: float = 900, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    base = local_base_url(base_url)
    cases = cases_from_file(cases_path)
    if not isinstance(condition, str) or not ID.fullmatch(condition):
        raise EvaluationError("Condition needs a simple identifier, with no secrets or paths")
    for name, value, upper in (("per_case_timeout", per_case_timeout, 1800), ("total_timeout", total_timeout, 3600)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= upper:
            raise EvaluationError(f"{name} must be positive and at most {upper} seconds")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise EvaluationError("Choose a new evaluation directory; existing runs are not overwritten")
    destination.mkdir(parents=True)
    path = destination / "synthesis.json"
    report = {"schema_version": "voice_synthesis.v1", "created_at": timestamp(), "condition": condition,
        "cases_sha256": digest(Path(cases_path)), "status": "running",
        "limits": {"per_case_timeout_seconds": per_case_timeout, "total_timeout_seconds": total_timeout, "response_bytes": 20_000_000},
        "execution": "sequential_http", "streaming_audio": False, "human_listening": "not_performed",
        "limitations": ["One sample per declared case is not a population quality estimate.",
            "The backend does not expose EOS/token completion. A valid WAV is not proof that all requested speech was generated.",
            "A timed-out HTTP client does not stop server inference. Later cases are not admitted after an uncertain failure."],
        "cases": [dict(case, status="pending") for case in cases]}
    checkpoint(path, report)
    started = time.monotonic()
    deadline = started + total_timeout
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    stop = False
    async with httpx.AsyncClient(timeout=httpx.Timeout(per_case_timeout, connect=min(10, per_case_timeout)), follow_redirects=False, trust_env=False, transport=transport) as client:
        report["service_before"] = await _health(client, base, headers, deadline)
        for row in report["cases"]:
            if stop or time.monotonic() >= deadline:
                row["status"] = "not_run_after_failure" if stop else "not_run_budget_exhausted"
                continue
            row.update(status="running", started_at=timestamp())
            checkpoint(path, report)
            case_started = time.monotonic()
            try:
                async with asyncio.timeout(min(per_case_timeout, deadline - time.monotonic())):
                    async with client.stream("POST", base + "/v1/audio/speech", headers=headers,
                        json={"model": "qwen3-tts", "input": row["text"], "language": row["language"], "voice": "reference", "response_format": "wav"}) as response:
                        row["http_status"] = response.status_code
                        row["time_to_response_headers_seconds"] = round(time.monotonic() - case_started, 6)
                        response.raise_for_status()
                        if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() not in {"audio/wav", "audio/x-wav", "audio/wave"}:
                            raise EvaluationError("Service did not return WAV content type")
                        body = bytearray()
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            body.extend(chunk)
                            if len(body) > 20_000_000:
                                raise EvaluationError("Audio response exceeds 20 MB")
                        raw = bytes(body)
                wall = time.monotonic() - case_started
                signal = wav_metrics(raw)
                audio = destination / (row["id"] + ".wav")
                audio.write_bytes(raw)
                row.update(status="succeeded", request_wall_seconds=round(wall, 6), signal=signal,
                    real_time_factor=round(wall / signal["duration_seconds"], 6),
                    audio={"path": audio.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
                    termination_reason="not_exposed_by_backend")
                row["service_after"] = await _health(client, base, headers, deadline)
            except (TimeoutError, httpx.HTTPError, EvaluationError) as exc:
                row.update(status="timed_out" if isinstance(exc, (TimeoutError, httpx.TimeoutException)) else "failed",
                    error_type=type(exc).__name__, request_wall_seconds=round(time.monotonic() - case_started, 6))
                stop = True
            checkpoint(path, report)
    report.update(status="completed" if all(row["status"] == "succeeded" for row in report["cases"]) else "incomplete",
        completed_at=timestamp(), elapsed_seconds=round(time.monotonic() - started, 6))
    checkpoint(path, report)
    return report
