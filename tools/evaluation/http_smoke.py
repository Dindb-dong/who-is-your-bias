"""Bounded live HTTP/SSE integration checks against already running services."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx


def collect(args: argparse.Namespace, report: dict, client: httpx.Client) -> None:
    """Keep partial rows even when a real transport disconnects mid-response."""
    try:
        language_health = client.get(args.language_url.rstrip("/") + "/health")
        language_health.raise_for_status()
        report["language_before"] = language_health.json()
        health = client.get(args.app_url.rstrip("/") + "/api/health")
        health.raise_for_status()
        report["application_health"] = health.json()
        started = time.perf_counter()
        key = os.getenv("BIAS_LANGUAGE_API_KEY")
        response = client.post(args.language_url.rstrip("/") + "/v1/chat/completions", headers={"Authorization": f"Bearer {key}"} if key else None, json={"model": args.model, "messages": [{"role": "user", "content": "Reply with the single word Hello."}], "max_tokens": 16, "stream": False})
        report["runs"].append({"kind": "direct_json", "http_status": response.status_code, "wall_ms": (time.perf_counter() - started) * 1000, "response": response.json()})
        for label, question in [
            ("ordinary_question", "Why did Tesla start with an expensive sports car?"),
            ("rocket_reuse", "Why does reusing rockets lower launch costs?"),
        ]:
            started = time.perf_counter()
            result = {"kind": "application_sse", "label": label, "question": question, "mode": "behavior", "events": [], "raw_text": "", "first_text_ms": None}
            report["runs"].append(result)
            with client.stream("POST", args.app_url.rstrip("/") + "/api/chat/stream", json={"message": question, "language": "en", "mode": "behavior"}) as response:
                result["http_status"] = response.status_code
                event = "message"
                for line in response.iter_lines():
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        payload = json.loads(line[5:].strip())
                        result["events"].append(event)
                        if event == "token":
                            result["raw_text"] += payload["text"]
                            if result["first_text_ms"] is None:
                                result["first_text_ms"] = (time.perf_counter() - started) * 1000
                        elif event == "result":
                            result["turn"] = payload["turn"]
                        elif event == "error":
                            result["error"] = payload
            result["wall_ms"] = (time.perf_counter() - started) * 1000
            print(json.dumps({"label": label, "provider": result.get("turn", {}).get("provider"), "grounding": result.get("turn", {}).get("grounding"), "error": result.get("error"), "wall_ms": result["wall_ms"]}), flush=True)
        report["language_after"] = client.get(args.language_url.rstrip("/") + "/health").json()
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        report["failure"] = {"type": type(exc).__name__, "message": "The live HTTP/protocol check could not complete; earlier rows are preserved."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language-url", default="http://127.0.0.1:8020")
    parser.add_argument("--app-url", default="http://127.0.0.1:8021")
    parser.add_argument("--model", default="qwen3-local")
    parser.add_argument("--timeout", type=float, default=155)
    parser.add_argument("--output", type=Path, default=Path("reports/local/language-http-smoke.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The output already exists; choose a new report path.")
    if not 1 <= args.timeout <= 600:
        parser.error("The per-request timeout must be between 1 and 600 seconds.")
    report = {"schema_version": "language-http-smoke.v1", "started_at": datetime.now(timezone.utc).isoformat(), "execution": "actual_http_requests", "runs": [], "limitations": ["Execution and citation identifiers are separate checks; semantic support and persona fidelity are not automatically graded.", "No model output is patched to add a missing citation.", "This smoke does not establish browser rendering or audio playback."]}
    loopback_only = all(urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"} for url in (args.language_url, args.app_url))
    with httpx.Client(timeout=args.timeout, trust_env=not loopback_only, follow_redirects=False) as client:
        collect(args, report, client)
    execution_passed = not report.get("failure") and len(report["runs"]) == 3 and report["runs"][0]["http_status"] == 200 and all(run.get("turn", {}).get("provider") == "openai" and run.get("events", [])[-1:] == ["done"] for run in report["runs"][1:])
    report["execution_passed"] = execution_passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "execution_passed": execution_passed, "semantic_review": "ungraded"}))
    return 0 if execution_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
