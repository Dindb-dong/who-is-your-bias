"""Run a real-model, source-preview API integration check without generation.

Use the expanded-corpus environment example after materialization and snapshot
registration. This runner requires the development and hybrid dependencies.
Full returned source passages are hashed, never copied into the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from fastapi.testclient import TestClient

from bias.api import create_app
from bias.config import Settings, repository_root


def run(settings: Settings) -> dict:
    if settings.llm.configured or settings.tts.configured:
        raise ValueError("This source-preview smoke requires unconfigured LLM and TTS providers")
    if settings.retrieval_backend != "hybrid" or not settings.retrieval_prewarm or not settings.persona_snapshot:
        raise ValueError("Select hybrid retrieval, startup warmup and the exact expanded-corpus snapshot")
    started = time.perf_counter()
    root = repository_root()
    report = {"schema_version": "persona-runtime-smoke.v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "transport": "in_process_asgi_testclient", "llm_inference": False, "tts_inference": False,
              "acoustic_latency_ms": None, "runtime": {"python": platform.python_version(), "torch": version("torch"),
                "sentence_transformers": version("sentence-transformers")},
              "implementation_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [Path(__file__).resolve(), *sorted((root / "src/bias").glob("*.py"))]}, "checks": []}
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()
        if not health["retrieval"]["prewarmed"]:
            raise AssertionError("API became ready before configured retrieval warmup")
        report.update(snapshot=health["persona_snapshot"], retrieval=health["retrieval"], corpus=health["corpus"])
        session = initial = None
        for query, language, kind in [
            ("How did you first learn to write computer programs as a child?", "en", "original"),
            ("Why that approach?", "en", "followup"), ("How so?", "en", "chained_followup"),
            ("현재 2030년 화성 탐사 계획은 뭐야?", "ko", "current_future_plan"),
            ("그 방법은 왜 그래?", "ko", "current_followup"), ("How so?", "en", "current_chained_followup"),
        ]:
            before = time.perf_counter()
            response = client.post("/api/chat", json={"message": query, "language": language, "session_id": session})
            if response.status_code != 200:
                raise AssertionError(f"{kind} returned HTTP {response.status_code}")
            turn = response.json()
            session, context, ids = turn["session_id"], turn["context"], turn["grounding"]["retrieved_ids"]
            if initial is None:
                initial = turn["id"]
                if "qa26-henry-ford-2008-turn-00015" not in ids:
                    raise AssertionError("The registered original programming passage was not supplied")
            if kind in {"followup", "chained_followup"} and initial not in context["query_resolution"]["supporting_turn_ids"]:
                raise AssertionError("Chained follow-up lost the original user topic")
            if kind.startswith("current"):
                if context["temporal"]["mode"] != "current" or turn["grounding"]["status"] != "insufficient" or turn["citations"]:
                    raise AssertionError("Old evidence was promoted to a current future-target plan")
            report["checks"].append({"kind": kind, "query": query, "language": language,
                "elapsed_ms": round((time.perf_counter() - before) * 1000, 3), "provider": turn["provider"],
                "grounding_status": turn["grounding"]["status"], "temporal_mode": context["temporal"]["mode"],
                "retrieved_ids": ids, "citation_ids": [c["evidence_id"] for c in turn["citations"]],
                "resolved_query": context["query_resolution"], "response_text_sha256": hashlib.sha256(turn["text"].encode()).hexdigest()})
        state = client.get(f"/api/sessions/{session}/context").json()
        report["context_turn_count"] = state["turn_count"]
        if state["turn_count"] != 6:
            raise AssertionError("Completed source-preview turns were not retained in bounded state")
    report.update(elapsed_seconds=round(time.perf_counter() - started, 3), completed_at=datetime.now(timezone.utc).isoformat(), passed=True)
    report["limitations"] = ["Source previews only; no language generation or speech inference.",
        "In-process ASGI transport does not measure network or speaker latency.",
        "Six deterministic turns are not a fresh human persona or 20-scenario acceptance evaluation."]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; recorded runs are not overwritten")
    try:
        report = run(Settings.from_env())
    except Exception as exc:
        report = {"schema_version": "persona-runtime-smoke.v1", "passed": False,
                  "completed_at": datetime.now(timezone.utc).isoformat(), "failure_type": type(exc).__name__,
                  "failure_message": str(exc) if isinstance(exc, (AssertionError, ValueError)) else "Runtime failed; inspect the local traceback."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": True, "warmup_ms": report["retrieval"]["startup_warmup_ms"],
                      "checks": len(report["checks"]), "elapsed_seconds": report["elapsed_seconds"], "report": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
