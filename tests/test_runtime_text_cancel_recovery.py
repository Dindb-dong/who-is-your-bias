"""Transport fixtures only; these tests load no neural model."""
import asyncio
import hashlib
import json
from pathlib import Path
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import pytest

from tools.runtime_checks.text_cancel_recovery import (
    DIAGNOSTIC_SHA256, PARENT_SHA256, eligibility, exercise, load_diagnostic, stream_summary,
)
from tools.runtime_checks.text_live import StreamClient, main_server


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "reports/jensen-text-live-v1/protocol-bf16-v2.json"
DIAGNOSTIC = ROOT / "reports/jensen-text-live-v1/cancel-recovery/protocol.json"


def test_registered_diagnostic_preserves_original_prompts_and_runtime(tmp_path):
    parent = json.loads(PARENT.read_text())
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA256
    diagnostic = load_diagnostic(DIAGNOSTIC, DIAGNOSTIC_SHA256, parent)
    assert diagnostic["requests"]["recovery"]["message"] == parent["recovery_case"]["message"]
    diagnostic["policy"]["max_output_tokens"] = 200
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(diagnostic))
    with pytest.raises(ValueError, match="registered cancellation"):
        load_diagnostic(changed, hashlib.sha256(changed.read_bytes()).hexdigest(), parent)


@pytest.mark.parametrize("change", ["passed", "condition", "disconnect", "identity", "source"])
def test_diagnostic_requires_the_same_failed_original_run_before_disconnect(tmp_path, change):
    source = ROOT / "tools/runtime_checks/text_live.py"
    preflight = {"corpus_sha256": "c", "snapshot_sha256": "s", "base_identity": {}, "adapter": None}
    original = {"schema_version": "jensen-neural-text-live-result.v1", "status": "failed", "condition": "base",
        "protocol_sha256": PARENT_SHA256, "code_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checks": [{"check": "question.completed_stream", "passed": False}], "preflight": dict(preflight)}
    path = tmp_path / "original.json"
    path.write_text(json.dumps(original))
    assert eligibility(path, "base", preflight, source)["disconnect_reached"] is False
    if change == "passed":
        original["status"] = "passed"
    elif change == "condition":
        original["condition"] = "adapter"
    elif change == "disconnect":
        original["checks"].append({"check": "disconnect.active_model_token_closed", "passed": False})
    elif change == "identity":
        original["preflight"]["corpus_sha256"] = "different"
    else:
        original["code_sha256"] = "0" * 64
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="unchanged failed"):
        eligibility(path, "base", preflight, source)


def encode(event, data):
    return "event: " + event + "\ndata: " + json.dumps(data) + "\n\n"


@pytest.mark.parametrize("recovery", ["length", "http_error", "complete"])
def test_actual_tcp_probe_keeps_failure_but_still_inspects_memory_deletes_and_tears_down(recovery):
    protocol = json.loads(PARENT.read_text())
    state = {"busy": False, "turns": [], "exists": True, "calls": 0}
    app = FastAPI()

    @app.get("/api/sessions/{session}/context")
    async def context(session):
        if not state["exists"]:
            return JSONResponse({"detail": "absent"}, status_code=404)
        return {"turn_count": len(state["turns"]), "turns": list(state["turns"])}

    @app.delete("/api/sessions/{session}")
    async def delete(session):
        assert state["busy"] is False
        state["exists"] = False
        return {"deleted": True}

    @app.post("/api/chat/stream")
    async def stream(request: Request):
        body = await request.json()
        state["calls"] += 1
        call = state["calls"]
        assert body["message"] == protocol["disconnect_case" if call == 1 else "recovery_case"]["message"]
        if call == 2 and recovery == "http_error":
            return JSONResponse({"error": "fixture upstream failure"}, status_code=503)
        state["busy"] = True

        async def events():
            try:
                yield encode("meta", {"session_id": "fixture-session"})
                yield encode("token", {"text": "private generated fixture"})
                if call == 1:
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(.2)
                    if recovery == "length":
                        yield encode("error", {"code": "provider_truncated", "message": "fixture limit"})
                    else:
                        state["turns"].append({"turn_id": "completed-fixture"})
                        yield encode("result", {"turn": {"provider": "openai", "session_id": "fixture-session"}})
                        yield encode("done", {})
            finally:
                state["busy"] = False
        return StreamingResponse(events(), media_type="text/event-stream")

    class Provider:
        def request(self, *args, **kwargs):
            return 200, {"status": "ready", "model_loaded": True,
                "model": protocol["runtime"]["served_model"],
                "identity": {"schema_version": "language-model-identity.v1", **protocol["base_identity"]},
                "runtime": {**protocol["runtime"], "generation_defaults": {key: protocol["policy"][key]
                    for key in ("seed", "temperature", "top_p", "top_k")}},
                "adapter": {"configured": False, "loaded": False, "verified": False},
                "busy": state["busy"], "stage": "generating" if state["busy"] else None}, {}

    report = {"checks": [], "stages": {}, "failures": []}
    cleanup, trace = {}, []
    with main_server(app, cleanup) as port:
        exercise(StreamClient(port, trace), Provider(), protocol, "base", None, report,
                 lambda: None, time.monotonic() + 10)
    checks = {row["check"]: row["passed"] for row in report["checks"]}
    assert state["calls"] == 2
    assert checks["disconnect.active_model_token_closed"]
    assert checks["disconnect.no_partial_turn_committed"]
    assert checks["session.deleted"]
    assert cleanup["main_api_stopped"]
    assert checks["recovery.completed_neural_response"] is (recovery == "complete")
    if recovery == "length":
        assert report["stages"]["recovery"]["error_codes"] == ["provider_truncated"]
        assert checks["recovery.neural_worker_admitted"]
        assert checks["recovery.first_token_observed"]
        assert checks["recovery.failed_turn_not_committed"]
    elif recovery == "http_error":
        assert report["stages"]["recovery"]["http_status"] == 503
        assert checks["recovery.failed_turn_not_committed"]
    assert "private generated fixture" not in json.dumps(report["stages"])


def test_partial_tokens_and_length_error_never_become_a_completed_response():
    record = {"events": [{"event": "token", "data": {"text": "partial private text"}},
                         {"event": "error", "data": {"code": "provider_truncated"}}]}
    summary = stream_summary(record)
    assert summary["first_token_observed"] and summary["completed"] is False
    assert summary["token_text_sha256"] == hashlib.sha256(b"partial private text").hexdigest()
    assert "partial private text" not in json.dumps(summary)
