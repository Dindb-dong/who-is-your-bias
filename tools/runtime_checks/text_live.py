"""Bounded actual neural-text HTTP/session probe, separate from frozen M6.

The language child, main API and HTTP clients share this invocation's network
namespace. Raw generated text and source excerpts must stay outside Git.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from bias.config import ProviderSettings, Settings
from bias.corpus import Corpus
from bias.snapshots import PersonaSnapshot, corpus_digest
from services.language.adapters import immutable_revision, repository_id, validate_adapter_run
from tools.runtime_checks.runtime import digest, local
from tools.runtime_checks.smoke import Client, code_receipt, server, sha, write


SCHEMA = "jensen-neural-text-live-protocol.v1"
SHA256 = re.compile(r"[a-f0-9]{64}")
SEALED_PROTOCOLS = {
    "edac6dfc7fb1545fc164fc5070ae3fdbc52a0a6dd82f7c10082fa88a58aa1210": "float32",
    "6c699aaa81a5f90919be7c78579f400c58590db5924abd045e6c76bcf9cd331e": "bfloat16",
}


def load_protocol(path: Path, expected_sha256: str) -> dict:
    """Accept only the two sealed runtimes, with every other protocol byte pinned."""
    actual = digest(path)
    if expected_sha256 != actual or actual not in SEALED_PROTOCOLS:
        raise ValueError("Supply the exact frozen --protocol-sha256 of a registered sealed runtime")
    protocol = json.loads(path.read_text())
    identity = protocol.get("base_identity", {})
    if (protocol.get("schema_version") != SCHEMA or protocol.get("persona_id") != "jensen-huang"
            or protocol.get("conditions") != ["base", "adapter"]
            or not protocol.get("frozen_before_execution")
            or not repository_id(identity.get("base_model"))
            or not immutable_revision(identity.get("base_revision"))
            or identity.get("chat_template_options") != {"enable_thinking": False}
            or any(not SHA256.fullmatch(identity.get(key, "")) for key in
                   ("base_checkpoint_manifest_sha256", "tokenizer_contract_sha256"))):
        raise ValueError("Expected an explicitly frozen Jensen neural functional protocol")
    runtime = protocol["runtime"]
    policy = protocol["policy"]
    if (runtime != {"device": "cpu", "dtype": SEALED_PROTOCOLS[actual], "threads": 8,
                    "served_model": "jensen-neural-functional", "retrieval_backend": "jensen-lexical",
                    "style_examples": False, "voice_enabled": False, "motion_enabled": False}
            or policy != {"context_window": 4096, "max_output_tokens": 160, "max_response_chars": 3000,
                          "request_seconds": 240, "startup_seconds": 30, "recovery_seconds": 60,
                          "total_seconds": 1800, "temperature": 0, "top_p": 1, "top_k": 20,
                          "seed": 42, "warmup_max_tokens": 12}
            or [case.get("id") for case in protocol["cases"]] !=
               ["question", "followup", "correction", "topic_switch"]):
        raise ValueError("This wrapper requires the predeclared bounded runtime and four-turn protocol")
    return protocol


def prepare(args):
    """Validate local identities without importing or loading neural weights."""
    if not args.allow_candidates:
        raise ValueError("This optional functional configuration requires --allow-candidates")
    repo = args.repository.resolve()
    path = (args.protocol or repo / "reports/jensen-text-live-v1/protocol.json").resolve()
    protocol = load_protocol(path, args.protocol_sha256)
    identity = protocol["base_identity"]
    corpus_path = local(args.artifacts.resolve(), protocol["corpus"]["artifact_path"])
    corpus = Corpus.load(corpus_path)
    if (corpus.persona.id != "jensen-huang" or corpus_digest(corpus) != protocol["corpus"]["sha256"]
            or not corpus.sources or any(item.split != "train" for item in corpus.sources.values())):
        raise ValueError("The exact training-only original-dialogue-v4 corpus is required")
    selected = protocol["snapshots"][args.condition]
    snapshot_path = local(path.parent, selected["path"])
    if digest(snapshot_path) != selected["sha256"]:
        raise ValueError("The separately frozen functional snapshot changed")
    snapshot = PersonaSnapshot.load(snapshot_path, corpus)
    cache = args.cache_dir.expanduser().resolve()
    base_path = cache / ("models--" + identity["base_model"].replace("/", "--")) / "snapshots" / identity["base_revision"]
    if not (base_path / "tokenizer_config.json").is_file():
        raise ValueError("The pinned base must already exist in --cache-dir; this probe never downloads it")
    adapter = None
    if args.condition == "adapter":
        if args.adapter_run is None or not SHA256.fullmatch(args.adapter_run_sha256 or ""):
            raise ValueError("Adapter mode requires an exact completed --adapter-run and --adapter-run-sha256")
        verified = validate_adapter_run(args.adapter_run, base_model=identity["base_model"],
            base_revision=identity["base_revision"], expected_run_sha256=args.adapter_run_sha256)
        adapter = verified.public_identity()
        training = protocol["adapter_training"]
        dataset = verified.report["dataset_manifest"]
        if (any(dataset.get(key) != training[key] for key in ("dataset_sha256", "samples", "event_counts"))
                or adapter["optimizer_steps"] != training["optimizer_steps"]):
            raise ValueError("The completed adapter must match the frozen Jensen dataset, event counts and optimizer work")
        if any(adapter[key] != identity[key] for key in identity if key != "chat_template_options"):
            raise ValueError("The completed adapter differs from the frozen base/tokenizer identity")
        tokenizer_path = verified.directory / "tokenizer"
    else:
        if args.adapter_run is not None or args.adapter_run_sha256 is not None:
            raise ValueError("Base mode must not configure an adapter")
        tokenizer_path = base_path
    # Do not resolve virtualenv interpreter symlinks into the system interpreter.
    executable = args.inference_python.expanduser().absolute()
    probe = subprocess.run([str(executable), "-c",
        "import importlib.util,json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
        "'modules':{n:importlib.util.find_spec(n) is not None for n in ['torch','transformers','peft','uvicorn']}}))"],
        capture_output=True, text=True, timeout=15)
    if probe.returncode:
        raise ValueError("The declared inference interpreter failed dependency preflight")
    interpreter = json.loads(probe.stdout)
    if not all(interpreter["modules"].values()):
        raise ValueError("The declared inference interpreter lacks a required dependency")
    preflight = {"protocol_sha256": digest(path), "condition": args.condition,
        "corpus_sha256": corpus_digest(corpus), "snapshot_sha256": digest(snapshot_path),
        "snapshot": snapshot.summary(), "base_identity": identity, "adapter": adapter,
        "declared_runtime": protocol["runtime"], "interpreter": interpreter,
        "weights_loaded": False, "voice_enabled": False}
    return protocol, corpus, snapshot_path, tokenizer_path, preflight


def observed_identity(health: dict, protocol: dict, condition: str, adapter: dict | None) -> dict:
    runtime = protocol["runtime"]
    # The protocol pins the identity payload; the actual service adds its
    # versioned envelope. Require both, including the exact supported schema.
    expected_identity = {"schema_version": "language-model-identity.v1", **protocol["base_identity"]}
    if (health.get("model_loaded") is not True or health.get("status") != "ready"
            or health.get("model") != runtime["served_model"]
            or health.get("identity") != expected_identity
            or any(health.get("runtime", {}).get(key) != runtime[key] for key in ("device", "dtype", "threads"))
            or health.get("runtime", {}).get("generation_defaults") !=
               {key: protocol["policy"][key] for key in ("seed", "temperature", "top_p", "top_k")}):
        raise ValueError("Actual loaded neural model/tokenizer/runtime differs from the frozen protocol")
    actual = health.get("adapter", {})
    if condition == "adapter":
        training = protocol["adapter_training"]
        if (not all(actual.get(key) is True for key in ("configured", "loaded", "verified"))
                or any(actual.get(key) != value for key, value in adapter.items())
                or actual.get("dataset_sha256") != training["dataset_sha256"]
                or actual.get("training_event_families") != sorted(training["event_counts"])
                or actual.get("optimizer_steps") != training["optimizer_steps"]):
            raise ValueError("The live provider did not load the exact completed candidate adapter")
    elif any(actual.get(key) for key in ("configured", "loaded", "verified")):
        raise ValueError("The base condition unexpectedly loaded an adapter")
    return {key: health[key] for key in ("identity", "adapter", "runtime")}


class StreamClient(Client):
    """Read actual API SSE, with a hard wall deadline and explicit TCP close."""
    def stream(self, body, *, deadline, disconnect=False, observe_token=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port,
            timeout=max(.01, deadline - time.monotonic()))
        started = time.monotonic()
        record = {"method": "POST", "path": "/api/chat/stream", "request": body,
                  "events": [], "client_closed_after_first_token": False}
        self.trace.append(record)
        response = None
        try:
            connection.request("POST", "/api/chat/stream", json.dumps(body).encode(),
                {"Content-Type": "application/json", "Accept": "text/event-stream"})
            response = connection.getresponse()
            record.update(status=response.status, headers=dict(response.getheaders()))
            if response.status != 200:
                record["error_body"] = response.read(32_000).decode("utf-8", errors="replace")
                raise RuntimeError(f"Main API SSE returned HTTP {response.status}")
            name, lines = "", []
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Main API SSE exceeded its absolute deadline")
                if connection.sock:
                    connection.sock.settimeout(remaining)
                line = response.readline(65_537)
                if len(line) > 65_536:
                    raise ValueError("Main API SSE line exceeds the bounded trace budget")
                if not line:
                    break
                value = line.decode("utf-8").rstrip("\r\n")
                if value.startswith("event:"):
                    name = value[6:].strip()
                elif value.startswith("data:"):
                    lines.append(value[5:].lstrip())
                elif not value and lines:
                    event = {"event": name, "data": json.loads("\n".join(lines)),
                             "elapsed_ms": round((time.monotonic() - started) * 1000, 3)}
                    record["events"].append(event)
                    if len(record["events"]) > 2048:
                        raise ValueError("Main API SSE exceeds the bounded event budget")
                    if name == "token" and event["data"].get("text", "").strip():
                        record.setdefault("first_token_ms", event["elapsed_ms"])
                        if disconnect:
                            if observe_token:
                                record["provider_at_first_token"] = observe_token()
                            record["client_closed_after_first_token"] = True
                            break
                    if name in {"done", "error"}:
                        break
                    name, lines = "", []
        except BaseException as exc:
            record["failure_type"] = type(exc).__name__
            raise
        finally:
            if connection.sock:
                try:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if response is not None:
                response.close()
            connection.close()
            record["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            record["connection_closed"] = True
        return record


@contextmanager
def language_child(args, protocol, output, report):
    """Own the neural child and its log throughout success, failure and cancel."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    identity, runtime, policy = protocol["base_identity"], protocol["runtime"], protocol["policy"]
    declared = {"MODEL": identity["base_model"], "REVISION": identity["base_revision"],
        "SERVED_MODEL": runtime["served_model"], "CACHE_DIR": str(args.cache_dir.expanduser().resolve()),
        "DTYPE": runtime["dtype"], "DEVICE": runtime["device"], "THREADS": str(runtime["threads"]),
        "LOCAL_FILES_ONLY": "true", "GENERATION_TIMEOUT": str(policy["request_seconds"]),
        "MAX_INPUT_TOKENS": str(policy["context_window"]), "MAX_OUTPUT_TOKENS": str(policy["max_output_tokens"]),
        "MAX_OUTPUT_CHARS": str(policy["max_response_chars"]),
        **{key.upper(): str(policy[key]) for key in ("temperature", "top_p", "top_k", "seed")}}
    if args.condition == "adapter":
        declared.update(ADAPTER_RUN=str(args.adapter_run.resolve()), ADAPTER_RUN_SHA256=args.adapter_run_sha256)
    declared = {"BIAS_LANGUAGE_" + key: value for key, value in declared.items()}
    repo = args.repository.resolve()
    declared.update(PYTHONPATH=str(repo / "src") + os.pathsep + str(repo),
        OMP_NUM_THREADS=str(runtime["threads"]), MKL_NUM_THREADS=str(runtime["threads"]),
        OPENBLAS_NUM_THREADS=str(runtime["threads"]), TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env = {key: value for key, value in os.environ.items() if not key.startswith("BIAS_LANGUAGE_")}
    env.update(declared)
    command = [str(args.inference_python.expanduser().absolute()), "-m", "uvicorn",
        "services.language.app:app", "--host", "127.0.0.1", "--port", str(port)]
    write(output / "launcher.json", {"command": command, "declared_environment": declared,
        "inherited_environment_distributed": False, "network_scope": "same managed invocation"})
    process = None
    try:
        with (output / "language-service.log").open("w") as log:
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            report["cleanup"]["language_pid"] = process.pid
            provider = Client(port, report["provider_trace"])
            deadline = time.monotonic() + policy["startup_seconds"]
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Language service exited before readiness; inspect its private log")
                try:
                    status, health, _ = provider.request("GET", "/health", timeout=1)
                    if status == 200:
                        report["health_before_load"] = health
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            else:
                raise TimeoutError("Language service readiness timed out")
            yield provider, port
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    report["cleanup"]["language_forced_kill"] = True
                    process.kill()
                    process.wait(timeout=5)
            report["cleanup"].update(language_stopped=process.poll() is not None,
                                     language_returncode=process.returncode)


def public_report(report):
    """Remove utterances, source excerpts, local paths and raw response payloads."""
    keys = ("schema_version", "status", "condition", "started_at", "finished_at", "protocol_sha256",
            "transport", "voice_enabled", "human_review", "quality_claim", "checks", "cases", "cleanup",
            "observed_before", "observed_after", "failure_type", "elapsed_seconds", "timing_scope")
    value = {key: report[key] for key in keys if key in report}
    value["preflight"] = {key: report["preflight"][key] for key in
        ("corpus_sha256", "snapshot_sha256", "snapshot", "base_identity", "adapter")}
    value["code_sha256"] = report["code_sha256"]
    value["api_module_sha256"] = report["api_code"]["modules_sha256"]
    value["supporting_code_sha256"] = report["supporting_code_sha256"]
    value["provider_resources"] = report.get("provider_resources")
    value["raw_trace_sha256"] = report.get("raw_trace_sha256")
    value["passed_checks"] = sum(row["passed"] for row in report["checks"])
    value["total_checks"] = len(report["checks"])
    return value


@contextmanager
def main_server(app, cleanup):
    """Record successful teardown even when a request inside the server fails."""
    manager = server(app)
    port = manager.__enter__()
    cleanup["main_api_started"] = True
    try:
        yield port
    finally:
        manager.__exit__(*sys.exc_info())
        cleanup["main_api_stopped"] = True


def run(args, protocol, corpus, snapshot_path, tokenizer_path, preflight):
    output = args.output.expanduser().resolve()
    if output.is_relative_to(args.repository.resolve()) or output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Raw model outputs and source excerpts must be outside both Git checkouts")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    policy = protocol["policy"]
    deadline = started + policy["total_seconds"]
    trace, checks = [], []
    report = {"schema_version": "jensen-neural-text-live-result.v1", "status": "starting",
        "condition": args.condition, "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": preflight["protocol_sha256"], "preflight": preflight,
        "transport": "actual loopback TCP: HTTP client -> main API -> local neural provider",
        "voice_enabled": False, "human_review": "ungraded", "quality_claim": False,
        "timing_scope": "one sequential functional run; warmup excluded; no benchmark or condition speed comparison",
        "checks": checks, "cases": [], "provider_trace": [], "api_code": code_receipt(),
        "code_sha256": digest(Path(__file__)),
        "supporting_code_sha256": {name: digest(args.repository.resolve() / name) for name in
            ("tools/runtime_checks/smoke.py", "tools/runtime_checks/runtime.py", "services/language/app.py",
             "services/language/engine.py", "services/language/adapters.py", "services/language/schemas.py")},
        "cleanup": {"main_api_stopped": False, "language_stopped": False, "language_forced_kill": False}}

    def save():
        write(output / "http-trace.json", trace)
        report["raw_trace_sha256"] = digest(output / "http-trace.json")
        write(output / "report.json", report)
        write(output / "public-summary.json", public_report(report))

    def check(name, passed, **details):
        checks.append({"check": name, "passed": bool(passed), **details})

    def request_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("Functional run exceeded its total time budget")
        return min(deadline, time.monotonic() + policy["request_seconds"])

    def inspect(case, record, session):
        events = record["events"]
        results = [event["data"]["turn"] for event in events if event["event"] == "result"]
        complete = (len(results) == 1 and events[-1]["event"] == "done"
                    and not any(event["event"] == "error" for event in events))
        check(case["id"] + ".completed_stream", complete, event_count=len(events))
        if not complete:
            raise RuntimeError("A functional turn did not complete; partial outputs remain in the raw trace")
        turn = results[0]
        check(case["id"] + ".neural_provider", turn.get("provider") == "openai")
        check(case["id"] + ".first_token_observed", record.get("first_token_ms") is not None)
        check(case["id"] + ".same_session", session is None or turn["session_id"] == session)
        if "expected_resolution" in case:
            method = turn.get("context", {}).get("query_resolution", {}).get("method")
            check(case["id"] + ".resolution", method == case["expected_resolution"], observed=method)
        citations = turn.get("citations", [])
        valid = True
        for row in citations:
            evidence = corpus.evidence.get(row["evidence_id"])
            source = corpus.sources.get(row["source_id"])
            valid = valid and bool(evidence and source and evidence.source_id == source.id
                and row["text"] == evidence.text and row["url"] == str(source.url)
                and source.split == "train" and row["locator"] == evidence.locator.model_dump(mode="json"))
        check(case["id"] + ".citation_source_binding", valid, citation_count=len(citations))
        report["cases"].append({"id": case["id"], "provider": turn["provider"],
            "response_sha256": sha(turn["text"].encode()), "response_characters": len(turn["text"]),
            "first_token_ms": record.get("first_token_ms"), "http_ms": record["elapsed_ms"],
            "api_metrics": turn.get("metrics"), "grounding": turn.get("grounding"),
            "citations": [{"evidence_id": row["evidence_id"], "source_id": row["source_id"],
                "url": row["url"], "text_sha256": sha(row["text"].encode())} for row in citations]})
        return turn["session_id"]

    def body(case, session=None):
        value = {"message": case["message"], "language": case["language"],
                 "mode": "rag", "motion_mode": "disabled"}
        if session:
            value["session_id"] = session
        return value

    save()
    try:
        from bias.api import create_app
        with language_child(args, protocol, output, report) as (provider, language_port):
            report["status"] = "warming"
            save()
            status, warmup, _ = provider.request("POST", "/v1/chat/completions",
                {"model": protocol["runtime"]["served_model"], "messages": [{"role": "user", "content": "Reply with one word: Ready"}],
                 "stream": False, "max_tokens": policy["warmup_max_tokens"], "temperature": 0, "seed": 42},
                timeout=max(.01, request_deadline() - time.monotonic()))
            write(output / "warmup.json", {"status": status, "response": warmup, "excluded_from_case_metrics": True})
            if status != 200 or warmup.get("model") != protocol["runtime"]["served_model"]:
                raise RuntimeError("Actual model warmup failed; no main-API success is inferred")
            _, health, _ = provider.request("GET", "/health")
            report["observed_before"] = observed_identity(health, protocol, args.condition, preflight["adapter"])
            report["loaded_health"] = health
            report["provider_resources"] = {"load_ms": health.get("load_ms"),
                "peak_rss_bytes_after_warmup": health.get("process_peak_rss_bytes")}
            check("loaded_model_identity", True)
            settings = Settings(corpus_dir=local(args.artifacts.resolve(), protocol["corpus"]["artifact_path"]),
                persona_snapshot=snapshot_path, retrieval_backend=protocol["runtime"]["retrieval_backend"],
                style_examples=False, llm=ProviderSettings(f"http://127.0.0.1:{language_port}/v1", protocol["runtime"]["served_model"]),
                tts=ProviderSettings(), asr=ProviderSettings(), motion_catalog=None,
                request_timeout=policy["request_seconds"], llm_context_window=policy["context_window"],
                llm_max_tokens=policy["max_output_tokens"], llm_tokenizer_path=tokenizer_path,
                max_response_chars=policy["max_response_chars"], web_dir=args.repository.resolve() / "apps/web/dist")
            app = create_app(settings, corpus=corpus)
            with main_server(app, report["cleanup"]) as main_port:
                report["status"] = "running"
                save()
                client = StreamClient(main_port, trace)
                status, api_health, _ = client.request("GET", "/api/health")
                check("main_api.neural_configuration", status == 200 and api_health.get("provider") == "openai"
                      and api_health.get("capabilities", {}).get("llm") is True)
                check("main_api.voice_disabled", not api_health.get("capabilities", {}).get("speech")
                      and not api_health.get("capabilities", {}).get("transcription"))
                check("main_api.frozen_snapshot", api_health.get("persona_snapshot") == preflight["snapshot"])
                status, sources, _ = client.request("GET", "/api/sources")
                expected = {key: source.model_dump(mode="json") for key, source in corpus.sources.items()}
                check("main_api.source_inspection", status == 200 and
                    {source["id"]: source for source in sources.get("sources", [])} == expected,
                    source_count=len(sources.get("sources", [])))
                session = None
                for case in protocol["cases"]:
                    record = client.stream(body(case, session), deadline=request_deadline())
                    session = inspect(case, record, session)
                    save()
                status, before, _ = client.request("GET", f"/api/sessions/{session}/context")
                rows = before.get("turns", [])
                check("session.four_completed_turns", status == 200 and before.get("turn_count") == 4)
                check("session.correction_and_topic_provenance", len(rows) == 4 and rows[2].get("user_correction")
                    and rows[1].get("supporting_user_turn_ids") == [rows[0]["turn_id"]]
                    and not rows[3].get("supporting_user_turn_ids")
                    and all(not row.get("claims_semantically_verified") for row in rows))

                def token_health():
                    status, current, _ = provider.request("GET", "/health", timeout=5)
                    observed_identity(current, protocol, args.condition, preflight["adapter"])
                    return {"http_status": status, "model_loaded": current["model_loaded"],
                            "busy": current["busy"], "stage": current["stage"]}

                interrupted = client.stream(body(protocol["disconnect_case"], session),
                    deadline=request_deadline(), disconnect=True, observe_token=token_health)
                check("disconnect.after_actual_active_model_token", interrupted["client_closed_after_first_token"]
                    and interrupted.get("provider_at_first_token", {}).get("busy") is True
                    and sum(event["event"] == "token" and bool(event["data"].get("text", "").strip())
                            for event in interrupted["events"]) == 1
                    and not any(event["event"] in {"result", "done"} for event in interrupted["events"]))
                save()
                cancel_started = time.monotonic()
                recover_deadline = min(deadline, time.monotonic() + policy["recovery_seconds"])
                idle = False
                while time.monotonic() < recover_deadline:
                    status, current, _ = provider.request("GET", "/health", timeout=5)
                    if status == 200 and current.get("busy") is False:
                        idle = True
                        break
                    time.sleep(.1)
                check("disconnect.provider_became_idle", idle,
                      observed_ms=round((time.monotonic() - cancel_started) * 1000, 3))
                if not idle:
                    raise TimeoutError("Neural provider did not become idle after client cancellation")
                status, after, _ = client.request("GET", f"/api/sessions/{session}/context")
                check("disconnect.no_partial_turn_committed", status == 200 and before == after)
                recovery = protocol["recovery_case"]
                record = client.stream(body(recovery, session), deadline=request_deadline())
                inspect(recovery, record, session)
                status, after_recovery, _ = client.request("GET", f"/api/sessions/{session}/context")
                check("recovery.same_session_fifth_completed_turn", status == 200 and after_recovery.get("turn_count") == 5
                    and [row["turn_id"] for row in after_recovery.get("turns", [])[:4]] == [row["turn_id"] for row in rows])
                status, deleted, _ = client.request("DELETE", f"/api/sessions/{session}")
                absent, _, _ = client.request("GET", f"/api/sessions/{session}/context")
                check("session.deleted", status == 200 and deleted.get("deleted") and absent == 404)
                _, final_health, _ = provider.request("GET", "/health")
                report["observed_after"] = observed_identity(final_health, protocol, args.condition, preflight["adapter"])
                report["provider_resources"]["peak_rss_bytes_after_recovery"] = final_health.get("process_peak_rss_bytes")
                check("loaded_identity_unchanged", report["observed_before"] == report["observed_after"])
                check("provider_idle_after_recovery", final_health.get("busy") is False)
                save()
        check("cleanup.main_api_stopped", report["cleanup"]["main_api_stopped"])
        check("cleanup.language_stopped", report["cleanup"]["language_stopped"])
        check("cleanup.no_forced_kill", not report["cleanup"]["language_forced_kill"])
        report["status"] = "passed" if all(row["passed"] for row in checks) else "failed_checks"
    except BaseException as exc:
        report.update(status="failed", failure_type=type(exc).__name__)
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        save()
    return public_report(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--condition", choices=("base", "adapter"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--inference-python", type=Path, required=True)
    parser.add_argument("--adapter-run", type=Path)
    parser.add_argument("--adapter-run-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-candidates", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    prepared = prepare(args)
    if args.verify_only:
        print(json.dumps(prepared[-1], indent=2, ensure_ascii=False))
        return 0
    if args.output is None:
        parser.error("Execution needs a new private --output directory")
    result = run(args, *prepared)
    print(json.dumps({key: result[key] for key in ("status", "condition", "passed_checks", "total_checks", "cleanup")}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
