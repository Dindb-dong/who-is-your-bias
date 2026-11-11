"""Exercise registered scenarios through the application's actual HTTP API.

The deterministic preview uses the bundled, already exposed research corpus.
An optional configured run exercises the same state contracts with the selected
retrieval/generation providers. Neither mode grades semantic persona fidelity.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from bias.api import create_app
from bias.config import Settings, repository_root


REGISTRY = Path(__file__).with_name("scenarios.v1.json")
EXPECTATIONS = {"query_contains", "query_excludes", "supports", "forbidden_supports", "temporal_mode",
                "years", "period", "preview_empty_evidence", "is_correction", "state_count", "bounded_state"}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_registry(path: Path = REGISTRY) -> dict:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "dialogue-scenarios.v1":
        raise ValueError("Unknown dialogue registry schema")
    ids = set()
    for scenario in registry["scenarios"]:
        if scenario["id"] in ids or len(scenario["steps"]) < 2:
            raise ValueError("Scenario IDs must be unique and each scenario must be multi-turn")
        ids.add(scenario["id"])
        labels = set()
        for step in scenario["steps"]:
            if step["label"] in labels or step["action"] not in {"chat", "interrupt"} or step["language"] not in {"en", "ko"}:
                raise ValueError("Invalid or duplicate scenario step")
            if not isinstance(step["message"], str) or not step["message"].strip():
                raise ValueError("Each step needs a nonempty user request")
            if set(step["expect"]) - EXPECTATIONS:
                raise ValueError("Unknown assertion: registry assertions must not be silently ignored")
            for key in ("supports", "forbidden_supports"):
                if set(step["expect"].get(key, [])) - labels:
                    raise ValueError("An assertion refers to a future or missing user turn")
            labels.add(step["label"])
    return registry


def preview_settings() -> Settings:
    # Explicit settings make a deterministic preview independent of secrets or
    # provider variables in the caller's environment.
    return Settings(persona_snapshot=repository_root() / "data/snapshots/elon-musk-2026-11-11.json")


async def interrupt_stream(app, request: dict) -> dict:
    """Cancel the ASGI request owner after a streaming response starts.

    HTTPX buffers ASGI responses and does not model client disconnects. Hold the
    first status delivery so the unfinished turn cannot race to completion.
    This tests cancelled request ownership; it does not measure a network
    disconnect, neural cancellation latency or acoustic interruption.
    """
    incoming: asyncio.Queue = asyncio.Queue()
    await incoming.put({"type": "http.request", "body": json.dumps(request).encode(), "more_body": False})
    sent, body = [], bytearray()
    interrupted = asyncio.Event()

    async def receive():
        return await incoming.get()

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body":
            body.extend(message.get("body", b""))
            if b"event: status\n" in body and not interrupted.is_set():
                interrupted.set()
                raise asyncio.CancelledError

    path = "/api/chat/stream"
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "method": "POST",
             "path": path, "raw_path": path.encode(), "query_string": b"", "headers": [(b"content-type", b"application/json")],
             "scheme": "http", "server": ("scenario", 80), "client": ("scenario", 50000), "http_version": "1.1", "root_path": ""}
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(task, 5)
    except asyncio.CancelledError:
        if not interrupted.is_set():
            raise
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    if not interrupted.is_set() or b"event: result\n" in body or b"event: done\n" in body:
        raise AssertionError("Interrupted stream completed or never reached the interruption point")
    meta = next(json.loads(block.split(b"data: ", 1)[1]) for block in bytes(body).split(b"\n\n") if block.startswith(b"event: meta\n"))
    return {"session_id": meta["session_id"], "turn_id": meta["turn_id"], "point": "retrieving_status_backpressure",
            "result_delivered": False, "transport": "asgi_request_task_cancellation"}


def check_turn(step: dict, turn: dict, state: dict, labels: dict, *, preview: bool, max_turns: int) -> list[str]:
    expect, failures = step["expect"], []
    context = turn["context"]
    resolved, temporal = context["query_resolution"], context["temporal"]
    check = lambda condition, message: failures.append(message) if not condition else None
    for value in expect.get("query_contains", []):
        check(value in resolved["query"], f"resolved query lost {value!r}")
    for value in expect.get("query_excludes", []):
        check(value not in resolved["query"], f"resolved query retained unrelated {value!r}")
    if "supports" in expect:
        expected_ids = [labels[value] for value in expect["supports"]]
        check(all(value in resolved["supporting_turn_ids"] for value in expected_ids), "missing supporting user turn")
        if not expected_ids:
            check(not resolved["supporting_turn_ids"], "independent question inherited a prior user turn")
    for label in expect.get("forbidden_supports", []):
        check(labels[label] not in resolved["supporting_turn_ids"], f"incorrect supporting user turn: {label}")
    for name, key in (("temporal_mode", "mode"), ("years", "years"), ("period", "period")):
        if name in expect:
            check(temporal and temporal[key] == expect[name], f"expected {name}={expect[name]!r}; got {temporal}")
    if preview and expect.get("preview_empty_evidence"):
        check(turn["grounding"]["status"] == "insufficient" and not turn["grounding"]["retrieved_ids"] and not turn["citations"],
              "preview supplied evidence for an unavailable position")
    if "is_correction" in expect:
        check(state["turns"][-1]["user_correction"] == expect["is_correction"], "correction provenance was lost")
    if "state_count" in expect:
        check(state["turn_count"] == expect["state_count"], "unexpected count of completed dialogue turns")
    if expect.get("bounded_state"):
        check(state["turn_count"] == max_turns, "long conversation did not retain the declared bounded state")
    check(all(not item["claims_semantically_verified"] for item in state["turns"]), "user memory was promoted to verified facts")
    check(set(citation["evidence_id"] for citation in turn["citations"]) <= set(turn["grounding"]["retrieved_ids"]),
          "a non-retrieved identifier authorized a citation")
    check(len(state["turns"]) <= max_turns, "dialogue state exceeded its configured bound")
    return failures


async def run_async(settings: Settings, *, registry_path: Path = REGISTRY, configured: bool = False, scenario_ids: set[str] | None = None) -> dict:
    registry = load_registry(registry_path)
    if not settings.persona_snapshot:
        raise ValueError("Dialogue acceptance requires an explicit corpus-bound persona snapshot")
    if not configured and (settings.llm.configured or settings.tts.configured or settings.retrieval_backend != "lexical"):
        raise ValueError("Use configured mode explicitly for external models or non-preview retrieval")
    selected = [value for value in registry["scenarios"] if scenario_ids is None or value["id"] in scenario_ids]
    if not selected or (scenario_ids and scenario_ids - {value["id"] for value in selected}):
        raise ValueError("Select existing scenario IDs")
    started = time.perf_counter()
    root = repository_root()
    report = {"schema_version": "dialogue-evaluation.v1", "registry_id": registry["id"],
              "registry_sha256": digest(registry_path.read_bytes()), "mode": "configured_runtime" if configured else "bundled_preview",
              "started_at": datetime.now(timezone.utc).isoformat(), "transport": "in_process_asgi",
              "llm_configured": settings.llm.configured, "tts_inference": False, "acoustic_latency_ms": None,
              "runtime": {"python": platform.python_version(), "platform": platform.platform()},
              "implementation_sha256": {str(path.relative_to(root)): digest(path.read_bytes()) for path in
                    [*sorted((root / "src/bias").glob("*.py")), Path(__file__).resolve()]}, "scenarios": []}
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://dialogue-scenarios") as client:
            health = (await client.get("/api/health")).json()
            report.update(snapshot=health["persona_snapshot"], retrieval=health["retrieval"], providers=health["providers"], corpus=health["corpus"])
            if not configured and health["persona_snapshot"]["corpus_sha256"] != registry["preview_corpus_sha256"]:
                raise ValueError("The preview corpus changed; register a new scenario baseline")
            for scenario in selected:
                sessions, labels, rows = {}, {}, []
                scenario_started = time.perf_counter()
                for step in scenario["steps"]:
                    before = time.perf_counter()
                    request = {"message": step["message"], "language": step["language"], "session_id": sessions.get(step["session"])}
                    row = {"label": step["label"], "session_alias": step["session"], "action": step["action"],
                           "query": step["message"], "language": step["language"], "failures": []}
                    try:
                        if step["action"] == "interrupt":
                            previous = (await client.get(f"/api/sessions/{request['session_id']}/context")).json() if request["session_id"] else None
                            receipt = await interrupt_stream(app, request)
                            labels[step["label"]] = receipt["turn_id"]
                            sessions[step["session"]] = receipt["session_id"]
                            state = (await client.get(f"/api/sessions/{receipt['session_id']}/context")).json()
                            if state["turn_count"] != (previous["turn_count"] if previous else 0) or any(t["turn_id"] == receipt["turn_id"] for t in state["turns"]):
                                row["failures"].append("interrupted generation entered dialogue memory")
                            row.update(interruption=receipt, state_turn_count=state["turn_count"])
                        else:
                            response = await client.post("/api/chat", json=request)
                            if response.status_code != 200:
                                raise AssertionError(f"Chat returned HTTP {response.status_code}")
                            turn = response.json()
                            sessions[step["session"]] = turn["session_id"]
                            state_response = await client.get(f"/api/sessions/{turn['session_id']}/context")
                            if state_response.status_code != 200:
                                raise AssertionError("Completed session context was unavailable")
                            state = state_response.json()
                            row["failures"].extend(check_turn(step, turn, state, labels, preview=not configured, max_turns=settings.max_turns))
                            labels[step["label"]] = turn["id"]
                            row.update(turn_id=turn["id"], provider=turn["provider"], context=turn["context"],
                                       state=state, grounding=turn["grounding"], citation_ids=[value["evidence_id"] for value in turn["citations"]],
                                       response_text_sha256=digest(turn["text"].encode()), response_characters=len(turn["text"]), metrics=turn["metrics"])
                    except Exception as exc:
                        row["failures"].append(str(exc) if isinstance(exc, (AssertionError, ValueError, KeyError)) else type(exc).__name__)
                    row.update(elapsed_ms=round((time.perf_counter() - before) * 1000, 3), passed=not row["failures"])
                    rows.append(row)
                    if not row["passed"] and step["label"] not in labels:
                        break  # Missing response IDs must not cause misleading later-reference failures.
                report["scenarios"].append({"id": scenario["id"], "category": scenario["category"], "steps": rows,
                                            "passed": len(rows) == len(scenario["steps"]) and all(row["passed"] for row in rows),
                                            "elapsed_ms": round((time.perf_counter() - scenario_started) * 1000, 3)})
                for session in sessions.values():
                    await client.delete(f"/api/sessions/{session}")
    report.update(passed=all(item["passed"] for item in report["scenarios"]),
                  scenario_count=len(report["scenarios"]), step_count=sum(len(item["steps"]) for item in report["scenarios"]),
                  llm_inference_turns=sum(row.get("provider") == "openai" for item in report["scenarios"] for row in item["steps"]),
                  elapsed_seconds=round(time.perf_counter() - started, 3), completed_at=datetime.now(timezone.utc).isoformat(),
                  limitations=["Registered development/runtime scenarios, not fresh independent persona efficacy or human review.",
                               "Assertions inspect conversation state, temporal policy and citation identities, not semantic entailment or stylistic quality.",
                               "In-process ASGI timing excludes network, browser playback, synthesis and acoustic output.",
                               "The interruption scenario cancels the ASGI request owner during status delivery; it is not a measured network, neural or acoustic cancellation benchmark."])
    return report


def run(settings: Settings | None = None, **kwargs) -> dict:
    return asyncio.run(run_async(settings or preview_settings(), **kwargs))
