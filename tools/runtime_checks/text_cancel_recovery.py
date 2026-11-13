"""Separate actual cancellation probe; never changes or resumes the frozen M7 run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from bias.config import ProviderSettings, Settings
from tools.runtime_checks.runtime import digest, local
from tools.runtime_checks.smoke import code_receipt, sha, write
from tools.runtime_checks.text_live import (
    StreamClient, language_child, main_server, observed_identity, prepare,
)


DIAGNOSTIC_SHA256 = "bd0fd90916bfc07e52600aef0479960a1e0e2b562a46c7ff8e74403e82c03c7a"
PARENT_SHA256 = "6c699aaa81a5f90919be7c78579f400c58590db5924abd045e6c76bcf9cd331e"


def load_diagnostic(path, expected, parent):
    if digest(path) != expected or expected != DIAGNOSTIC_SHA256:
        raise ValueError("Supply the exact registered cancellation diagnostic seal")
    value = json.loads(path.read_text())
    requests = {name: {key: parent[name + "_case"][key] for key in ("id", "message", "language")}
                for name in ("disconnect", "recovery")}
    if (value["parent_protocol_sha256"] != PARENT_SHA256 or value["requests"] != requests
            or value["runtime"] != parent["runtime"] or value["policy"] != parent["policy"]):
        raise ValueError("Diagnostic prompts, identities and runtime must retain the original M7 contract")
    return value


def eligibility(path, condition, preflight, parent_source):
    value = json.loads(path.read_text())
    if (value.get("schema_version") != "jensen-neural-text-live-result.v1"
            or value.get("status") not in {"failed", "failed_checks"}
            or value.get("condition") != condition or value.get("protocol_sha256") != PARENT_SHA256
            or value.get("code_sha256") != digest(parent_source)
            or any(row.get("check", "").startswith("disconnect.") for row in value.get("checks", []))
            or any(value.get("preflight", {}).get(key) != preflight[key]
                   for key in ("corpus_sha256", "snapshot_sha256", "base_identity", "adapter"))):
        raise ValueError("Require the unchanged failed same-condition M7 report before any disconnect stage")
    return {"summary_sha256": digest(path), "status": value["status"],
            "condition": condition, "parent_code_sha256": value["code_sha256"],
            "completed_cases": [case["id"] for case in value.get("cases", [])],
            "disconnect_reached": False}


def stream_summary(record):
    events = record.get("events", [])
    results = [row["data"].get("turn", {}) for row in events if row["event"] == "result"]
    tokens = [row["data"].get("text", "") for row in events if row["event"] == "token"]
    errors = [row["data"].get("code") for row in events if row["event"] == "error"]
    complete = bool(len(results) == 1 and events[-1]["event"] == "done" and not errors)
    return {"http_status": record.get("status"), "http_ms": record.get("elapsed_ms"),
            "first_token_ms": record.get("first_token_ms"),
            "first_token_observed": any(text.strip() for text in tokens),
            "nonempty_token_events": sum(bool(text.strip()) for text in tokens),
            "token_text_sha256": sha("".join(tokens).encode()),
            "response_text_sha256": sha(results[0]["text"].encode())
                if complete and isinstance(results[0].get("text"), str) else None,
            "token_characters": sum(map(len, tokens)), "completed": complete,
            "completed_provider": results[0].get("provider") if complete else None,
            "error_codes": errors, "failure_type": record.get("failure_type"),
            "client_connection_closed": record.get("connection_closed", False)}


def session_from(record):
    ids = [row["data"].get("session_id") for row in record.get("events", []) if row["event"] == "meta"]
    return ids[0] if len(ids) == 1 and isinstance(ids[0], str) and ids[0] else None


def recovery_stream(client, provider, body, deadline, protocol, condition, adapter):
    """Reuse the original TCP client and independently observe busy admission."""
    stop = threading.Event()
    admission = {"observed": False}
    started = time.monotonic()

    def observe():
        while not stop.wait(.05) and time.monotonic() < deadline:
            try:
                status, health, _ = provider.request("GET", "/health", timeout=.5)
                observed_identity(health, protocol, condition, adapter)
                if status == 200 and health.get("busy") is True:
                    admission.update(observed=True, busy=True, stage=health.get("stage"),
                                     observed_ms=round((time.monotonic() - started) * 1000, 3))
                    return
            except (OSError, ValueError) as exc:
                admission["observation_error_type"] = type(exc).__name__
                return

    thread = threading.Thread(target=observe, daemon=True)
    index = len(client.trace)
    thread.start()
    try:
        record = client.stream(body, deadline=deadline)
    except Exception as exc:
        record = client.trace[index] if len(client.trace) > index else {"events": []}
        record["failure_type"] = type(exc).__name__
    finally:
        stop.set()
        thread.join(timeout=2)
        admission["observer_stopped"] = not thread.is_alive()
    return record, admission


def exercise(client, provider, protocol, condition, adapter, report, save, deadline):
    """Run at most two requests; a failed recovery still reaches memory/delete checks."""
    session, before = None, None
    checks = report["checks"]

    def check(name, passed, **details):
        checks.append({"check": name, "passed": bool(passed), **details})

    def failure(stage, exc):
        report["failures"].append({"stage": stage, "type": type(exc).__name__})

    def context():
        status, value, _ = client.request("GET", f"/api/sessions/{session}/context", timeout=5)
        return value if status == 200 else None

    def idle(stage):
        started = time.monotonic()
        limit = min(deadline, started + protocol["policy"]["recovery_seconds"])
        while time.monotonic() < limit:
            status, health, _ = provider.request("GET", "/health", timeout=5)
            observed_identity(health, protocol, condition, adapter)
            if status == 200 and health.get("busy") is False:
                check(stage + ".provider_idle", True, observed_ms=round((time.monotonic() - started) * 1000, 3))
                return True
            time.sleep(.1)
        check(stage + ".provider_idle", False)
        return False

    def body(case):
        value = {"message": case["message"], "language": case["language"], "mode": "rag", "motion_mode": "disabled"}
        if session:
            value["session_id"] = session
        return value

    def token_health():
        nonlocal session, before
        # Capture the stream record before auxiliary HTTP requests append traces.
        session = session_from(client.trace[-1])
        if not session:
            raise ValueError("Disconnect stream supplied no unique session identity")
        status, health, _ = provider.request("GET", "/health", timeout=5)
        observed_identity(health, protocol, condition, adapter)
        before = context()
        report["context_before_disconnect"] = before
        return {"http_status": status, "busy": health.get("busy"), "stage": health.get("stage"),
                "empty_context_observed": before is not None and before.get("turn_count") == 0}

    try:
        index = len(client.trace)
        try:
            record = client.stream(body(protocol["disconnect_case"]),
                deadline=min(deadline, time.monotonic() + protocol["policy"]["request_seconds"]),
                disconnect=True, observe_token=token_health)
        except Exception as exc:
            failure("disconnect", exc)
            record = client.trace[index] if len(client.trace) > index else {"events": []}
        session = session or session_from(record)
        summary = stream_summary(record)
        summary["provider_at_first_token"] = record.get("provider_at_first_token")
        report["stages"]["disconnect"] = summary
        observed = record.get("provider_at_first_token", {})
        check("disconnect.active_model_token_closed", record.get("client_closed_after_first_token")
              and observed.get("busy") is True and summary["nonempty_token_events"] == 1
              and not summary["completed"] and not summary["error_codes"])
        check("disconnect.initial_empty_context", observed.get("empty_context_observed") is True)
        save()
        available = idle("disconnect")
        after = context() if session else None
        report["context_after_disconnect"] = after
        check("disconnect.no_partial_turn_committed", before is not None and after == before)
        save()
        if available and session:
            record, admission = recovery_stream(client, provider, body(protocol["recovery_case"]),
                min(deadline, time.monotonic() + protocol["policy"]["request_seconds"]),
                protocol, condition, adapter)
            summary = stream_summary(record)
            summary["admission"] = admission
            report["stages"]["recovery"] = summary
            check("recovery.neural_worker_admitted", admission["observed"])
            check("recovery.admission_observer_stopped", admission["observer_stopped"])
            check("recovery.first_token_observed", summary["first_token_observed"])
            check("recovery.completed_neural_response", summary["completed"] and summary["completed_provider"] == "openai")
            check("recovery.same_session", session_from(record) == session)
            save()
            idle("recovery")
            final_context = context()
            report["context_after_recovery"] = final_context
            if summary["completed"]:
                check("recovery.one_completed_turn_committed", final_context is not None
                      and final_context.get("turn_count") == 1)
            else:
                check("recovery.failed_turn_not_committed", after is not None and final_context == after)
        else:
            report["stages"]["recovery"] = {"attempted": False, "reason": "no_session_or_provider_not_idle", "completed": False}
            check("recovery.completed_neural_response", False)
    except Exception as exc:
        failure("lifecycle", exc)
    finally:
        # This runs after a truncated/error recovery; it is not a success shortcut.
        if session:
            try:
                status, deleted, _ = client.request("DELETE", f"/api/sessions/{session}", timeout=5)
                absent, _, _ = client.request("GET", f"/api/sessions/{session}/context", timeout=5)
                check("session.deleted", status == 200 and deleted.get("deleted") is True and absent == 404)
            except Exception as exc:
                failure("session_delete", exc)
                check("session.deleted", False)
        else:
            check("session.deleted", False, reason="no_session_identity_observed")
        save()


def public_report(report):
    keys = ("schema_version", "status", "condition", "started_at", "finished_at", "elapsed_seconds",
            "protocol_sha256", "diagnostic_protocol_sha256", "eligibility", "code_sha256", "supporting_code_sha256",
            "checks", "stages", "failures", "cleanup", "observed_before", "observed_after", "raw_trace_sha256",
            "quality_claim", "human_review", "voice_enabled", "timing_scope")
    value = {key: report[key] for key in keys if key in report}
    value["preflight"] = {key: report["preflight"][key] for key in
        ("corpus_sha256", "snapshot_sha256", "base_identity", "adapter")}
    value["api_module_sha256"] = report["api_code"]["modules_sha256"]
    value["passed_checks"] = sum(row["passed"] for row in report["checks"])
    value["total_checks"] = len(report["checks"])
    return value


def run(args, protocol, corpus, snapshot_path, tokenizer_path, preflight, eligible):
    output = args.output.expanduser().resolve()
    if output.is_relative_to(args.repository.resolve()) or output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Raw diagnostic data must remain outside Git")
    output.mkdir(parents=True, exist_ok=False)
    claim = args.artifacts.resolve() / "runtime-checks" / f"cancel-recovery-{args.condition}-v1.attempt.json"
    claim.parent.mkdir(parents=True, exist_ok=True)
    with claim.open("x") as stream:
        json.dump({"condition": args.condition, "diagnostic_protocol_sha256": DIAGNOSTIC_SHA256,
                   "eligibility_summary_sha256": eligible["summary_sha256"], "output": str(output)}, stream)
    started = time.monotonic()
    deadline = started + protocol["policy"]["total_seconds"]
    trace = []
    report = {"schema_version": "jensen-neural-cancel-recovery-result.v1", "status": "starting",
        "condition": args.condition, "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": PARENT_SHA256, "diagnostic_protocol_sha256": DIAGNOSTIC_SHA256,
        "eligibility": eligible, "preflight": preflight, "checks": [], "stages": {}, "failures": [],
        "provider_trace": [], "quality_claim": False, "human_review": "ungraded", "voice_enabled": False,
        "timing_scope": "one separate sequential diagnostic; warmup excluded; no benchmark or M6/M7 score replacement",
        "code_sha256": digest(Path(__file__)), "api_code": code_receipt(),
        "supporting_code_sha256": {name: digest(args.repository / name) for name in
            ("tools/runtime_checks/text_live.py", "tools/runtime_checks/smoke.py", "services/language/app.py",
             "services/language/engine.py", "services/language/adapters.py")},
        "cleanup": {"main_api_stopped": False, "language_stopped": False, "language_forced_kill": False}}

    def save():
        write(output / "http-trace.json", trace)
        report["raw_trace_sha256"] = digest(output / "http-trace.json")
        write(output / "report.json", report)
        write(output / "public-summary.json", public_report(report))

    save()
    try:
        from bias.api import create_app
        with language_child(args, protocol, output, report) as (provider, port):
            status, warmup, _ = provider.request("POST", "/v1/chat/completions",
                {"model": protocol["runtime"]["served_model"], "messages": [{"role": "user", "content": "Reply with one word: Ready"}],
                 "stream": False, "max_tokens": protocol["policy"]["warmup_max_tokens"], "temperature": 0, "seed": 42},
                timeout=protocol["policy"]["request_seconds"])
            write(output / "warmup.json", {"status": status, "response": warmup, "excluded_from_case_metrics": True})
            if status != 200 or warmup.get("model") != protocol["runtime"]["served_model"]:
                raise RuntimeError("Actual neural warmup failed")
            _, health, _ = provider.request("GET", "/health")
            report["observed_before"] = observed_identity(health, protocol, args.condition, preflight["adapter"])
            report["checks"].append({"check": "loaded_model_identity", "passed": True})
            policy = protocol["policy"]
            settings = Settings(corpus_dir=local(args.artifacts.resolve(), protocol["corpus"]["artifact_path"]),
                persona_snapshot=snapshot_path, retrieval_backend=protocol["runtime"]["retrieval_backend"],
                style_examples=False, llm=ProviderSettings(f"http://127.0.0.1:{port}/v1", protocol["runtime"]["served_model"]),
                tts=ProviderSettings(), asr=ProviderSettings(), motion_catalog=None,
                request_timeout=policy["request_seconds"], llm_context_window=policy["context_window"],
                llm_max_tokens=policy["max_output_tokens"], llm_tokenizer_path=tokenizer_path,
                max_response_chars=policy["max_response_chars"], web_dir=args.repository.resolve() / "apps/web/dist")
            with main_server(create_app(settings, corpus=corpus), report["cleanup"]) as main_port:
                report["status"] = "running"
                client = StreamClient(main_port, trace)
                status, api_health, _ = client.request("GET", "/api/health")
                report["checks"].append({"check": "main_api.exact_neural_configuration", "passed": bool(
                    status == 200 and api_health.get("provider") == "openai"
                    and api_health.get("persona_snapshot") == preflight["snapshot"]
                    and not api_health.get("capabilities", {}).get("speech")
                    and not api_health.get("capabilities", {}).get("transcription"))})
                exercise(client, provider, protocol, args.condition, preflight["adapter"], report, save, deadline)
                _, health, _ = provider.request("GET", "/health")
                report["observed_after"] = observed_identity(health, protocol, args.condition, preflight["adapter"])
                report["checks"].append({"check": "loaded_identity_unchanged", "passed": report["observed_after"] == report["observed_before"]})
        report["status"] = "failed_checks" if report["failures"] or not all(row["passed"] for row in report["checks"]) else "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["failures"].append({"stage": "invocation", "type": type(exc).__name__})
        if not isinstance(exc, Exception):
            raise
    finally:
        for key in ("main_api_stopped", "language_stopped"):
            report["checks"].append({"check": "cleanup." + key, "passed": report["cleanup"][key]})
        report["checks"].append({"check": "cleanup.no_forced_kill", "passed": not report["cleanup"]["language_forced_kill"]})
        if report["status"] == "passed" and not all(row["passed"] for row in report["checks"]):
            report["status"] = "failed_checks"
        report.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.monotonic() - started, 3))
        save()
    return public_report(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-protocol-sha256", required=True)
    parser.add_argument("--condition", choices=("base", "adapter"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--inference-python", type=Path, required=True)
    parser.add_argument("--adapter-run", type=Path)
    parser.add_argument("--adapter-run-sha256")
    parser.add_argument("--eligibility-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-candidates", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    prepared = prepare(args)
    if prepared[-1]["protocol_sha256"] != PARENT_SHA256:
        parser.error("This diagnostic requires the unchanged BF16 parent protocol")
    load_diagnostic(args.diagnostic_protocol, args.diagnostic_protocol_sha256, prepared[0])
    eligible = eligibility(args.eligibility_report, args.condition, prepared[-1], args.repository / "tools/runtime_checks/text_live.py") if args.eligibility_report else None
    if args.verify_only:
        print(json.dumps({"preflight": prepared[-1], "diagnostic_protocol_sha256": DIAGNOSTIC_SHA256,
                          "eligibility": eligible, "weights_loaded": False, "generation_executed": False}, indent=2))
        return 0
    if args.output is None or eligible is None:
        parser.error("Actual execution needs a new private --output and preserved --eligibility-report")
    result = run(args, *prepared, eligible)
    print(json.dumps({key: result[key] for key in ("status", "condition", "passed_checks", "total_checks", "cleanup")}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
