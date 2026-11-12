"""Harness transport/identity tests; fixtures are not neural integration evidence."""
import asyncio
from copy import deepcopy
import json
import socket
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
import pytest

from tools.runtime_checks.text_live import StreamClient, main_server, observed_identity


def identity_fixture():
    identity = {"base_model": "example/model", "base_revision": "a" * 40,
                "base_checkpoint_manifest_sha256": "b" * 64,
                "tokenizer_contract_sha256": "c" * 64,
                "chat_template_options": {"enable_thinking": False}}
    defaults = {"seed": 42, "temperature": 0, "top_p": 1, "top_k": 20}
    runtime = {"device": "cpu", "dtype": "float32", "threads": 8,
               "served_model": "functional-fixture"}
    protocol = {"base_identity": identity, "runtime": runtime, "policy": defaults}
    health = {"status": "ready", "model_loaded": True, "model": runtime["served_model"],
              "identity": deepcopy(identity), "runtime": {**runtime, "generation_defaults": defaults},
              "adapter": {"configured": False, "loaded": False, "verified": False}}
    return protocol, health


@pytest.mark.parametrize("mutation", ["unloaded", "revision", "bytes", "dtype", "decoding", "adapter"])
def test_observed_identity_rejects_a_declaration_or_wrong_live_model(mutation):
    protocol, health = identity_fixture()
    observed_identity(health, protocol, "base", None)
    if mutation == "unloaded":
        health["model_loaded"] = False
    elif mutation == "revision":
        health["identity"]["base_revision"] = "d" * 40
    elif mutation == "bytes":
        health["identity"]["base_checkpoint_manifest_sha256"] = "e" * 64
    elif mutation == "dtype":
        health["runtime"]["dtype"] = "bfloat16"
    elif mutation == "decoding":
        health["runtime"]["generation_defaults"] = {"seed": 42, "temperature": .7, "top_p": 1, "top_k": 20}
    else:
        health["adapter"]["loaded"] = True
    with pytest.raises(ValueError):
        observed_identity(health, protocol, "base", None)


def test_adapter_requires_the_exact_loaded_completed_receipt_identity():
    protocol, health = identity_fixture()
    expected = {"run_sha256": "f" * 64, "optimizer_steps": 29, "dataset_sha256": "1" * 64}
    health["adapter"] = {**expected, "configured": True, "loaded": True, "verified": True}
    observed_identity(health, protocol, "adapter", expected)
    health["adapter"]["run_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="exact completed"):
        observed_identity(health, protocol, "adapter", expected)


def encode(event, data):
    return "event: " + event + "\ndata: " + json.dumps(data) + "\n\n"


def test_actual_tcp_disconnect_waits_for_first_nonempty_token_and_closes_generator():
    app = FastAPI()
    stopped = threading.Event()

    @app.post("/api/chat/stream")
    async def endpoint():
        async def events():
            try:
                yield encode("meta", {"session_id": "fixture"})
                yield encode("token", {"text": " "})
                yield encode("token", {"text": "first"})
                await asyncio.sleep(5)
                yield encode("token", {"text": "unread"})
            finally:
                stopped.set()
        return StreamingResponse(events(), media_type="text/event-stream")

    cleanup, trace = {}, []
    observations = []

    def observed():
        observations.append("called")
        return {"busy": True, "fixture_only": True}

    with main_server(app, cleanup) as port:
        record = StreamClient(port, trace).stream({}, deadline=time.monotonic() + 3,
            disconnect=True, observe_token=observed)
        assert stopped.wait(2)
    assert observations == ["called"]
    assert record["client_closed_after_first_token"] is True
    assert record["connection_closed"] is True
    assert [row["data"].get("text") for row in record["events"]] == [None, " ", "first"]
    assert record["first_token_ms"] >= 0
    assert cleanup["main_api_stopped"] is True


def test_actual_tcp_complete_stream_and_error_response_keep_trace():
    app = FastAPI()
    calls = []

    @app.post("/api/chat/stream")
    async def endpoint():
        calls.append(True)
        if len(calls) == 2:
            return JSONResponse({"error": "fixture failure"}, status_code=503)

        async def events():
            for event, data in [("meta", {}), ("token", {"text": "fixture"}),
                                ("result", {"turn": {}}), ("done", {})]:
                yield encode(event, data)
        return StreamingResponse(events(), media_type="text/event-stream")

    cleanup, trace = {}, []
    with main_server(app, cleanup) as port:
        client = StreamClient(port, trace)
        result = client.stream({}, deadline=time.monotonic() + 3)
        assert [row["event"] for row in result["events"]] == ["meta", "token", "result", "done"]
        assert result["client_closed_after_first_token"] is False
        with pytest.raises(RuntimeError, match="503"):
            client.stream({}, deadline=time.monotonic() + 3)
    assert trace[-1]["status"] == 503
    assert trace[-1]["connection_closed"] is True
    assert "fixture failure" in trace[-1]["error_body"]


def test_main_server_records_teardown_after_body_failure():
    cleanup = {}
    port = None
    with pytest.raises(RuntimeError, match="intentional test failure"):
        with main_server(FastAPI(), cleanup) as port:
            raise RuntimeError("intentional test failure")
    assert cleanup == {"main_api_started": True, "main_api_stopped": True}
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=.2)
