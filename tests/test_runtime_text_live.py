"""Harness transport/identity tests; fixtures are not neural integration evidence."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
import pytest

from tools.runtime_checks.text_live import SEALED_PROTOCOLS, StreamClient, load_protocol, main_server, observed_identity


PROTOCOLS = Path(__file__).resolve().parents[1] / "reports/jensen-text-live-v1"


@pytest.mark.parametrize(("filename", "dtype"), [("protocol.json", "float32"), ("protocol-bf16-v2.json", "bfloat16")])
def test_both_sealed_protocols_preserve_the_original_frozen_cases_and_gates(filename, dtype):
    path = PROTOCOLS / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = load_protocol(path, digest)
    assert SEALED_PROTOCOLS[digest] == dtype
    assert protocol["runtime"]["dtype"] == dtype
    original = json.loads((PROTOCOLS / "protocol.json").read_text())
    protocol["runtime"]["dtype"] = "float32"
    assert protocol == original
    assert hashlib.sha256((PROTOCOLS / "protocol.json").read_bytes()).hexdigest() == (
        "edac6dfc7fb1545fc164fc5070ae3fdbc52a0a6dd82f7c10082fa88a58aa1210")


@pytest.mark.parametrize("changed", ["dtype", "threads", "budget", "question", "snapshot", "steps", "dataset", "event_counts"])
def test_new_caller_digest_cannot_relax_any_sealed_bf16_protocol_gate(tmp_path, changed):
    protocol = json.loads((PROTOCOLS / "protocol-bf16-v2.json").read_text())
    if changed == "dtype":
        protocol["runtime"]["dtype"] = "float16"
    elif changed == "threads":
        protocol["runtime"]["threads"] = 4
    elif changed == "budget":
        protocol["policy"]["max_output_tokens"] = 32
    elif changed == "question":
        protocol["cases"][0]["message"] = "A different question"
    elif changed == "snapshot":
        protocol["snapshots"]["adapter"]["sha256"] = "0" * 64
    elif changed == "steps":
        protocol["adapter_training"]["optimizer_steps"] = 10
    elif changed == "dataset":
        protocol["adapter_training"]["dataset_sha256"] = "1" * 64
    else:
        protocol["adapter_training"]["event_counts"]["acquired-2023"] = 15
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="registered sealed runtime"):
        load_protocol(path, hashlib.sha256(path.read_bytes()).hexdigest())


def identity_fixture():
    identity = {"base_model": "example/model", "base_revision": "a" * 40,
                "base_checkpoint_manifest_sha256": "b" * 64,
                "tokenizer_contract_sha256": "c" * 64,
                "chat_template_options": {"enable_thinking": False}}
    defaults = {"seed": 42, "temperature": 0, "top_p": 1, "top_k": 20}
    runtime = {"device": "cpu", "dtype": "float32", "threads": 8,
               "served_model": "functional-fixture"}
    protocol = {"base_identity": identity, "runtime": runtime, "policy": defaults,
                "adapter_training": {"dataset_sha256": "1" * 64,
                    "event_counts": {"fixture-event": 113}, "optimizer_steps": 29}}
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
    expected = {"run_sha256": "f" * 64, "optimizer_steps": 29, "dataset_sha256": "1" * 64,
                "training_event_families": ["fixture-event"]}
    health["adapter"] = {**expected, "configured": True, "loaded": True, "verified": True}
    observed_identity(health, protocol, "adapter", expected)
    health["adapter"]["run_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="exact completed"):
        observed_identity(health, protocol, "adapter", expected)


def test_bf16_live_identity_still_requires_the_exact_declared_precision():
    protocol, health = identity_fixture()
    protocol["runtime"]["dtype"] = "bfloat16"
    health["runtime"]["dtype"] = "bfloat16"
    observed_identity(health, protocol, "base", None)
    health["runtime"]["dtype"] = "float32"
    with pytest.raises(ValueError, match="runtime differs"):
        observed_identity(health, protocol, "base", None)


@pytest.mark.parametrize("changed", ["dataset", "family", "steps"])
def test_matching_operator_receipt_cannot_replace_frozen_subject_training(changed):
    protocol, health = identity_fixture()
    other = {"run_sha256": "f" * 64, "optimizer_steps": 29, "dataset_sha256": "1" * 64,
             "training_event_families": ["fixture-event"]}
    if changed == "dataset":
        other["dataset_sha256"] = "3" * 64
    elif changed == "family":
        other["training_event_families"] = ["other-subject-event"]
    else:
        other["optimizer_steps"] = 1
    health["adapter"] = {**other, "configured": True, "loaded": True, "verified": True}
    with pytest.raises(ValueError, match="exact completed"):
        observed_identity(health, protocol, "adapter", other)


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
