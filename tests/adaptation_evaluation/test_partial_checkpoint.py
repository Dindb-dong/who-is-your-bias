"""Synthetic concurrent-writer fixtures; no models, real outputs or network."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.adaptation_evaluation import partial_checkpoint as partial
from tools.adaptation_evaluation.local_condition import SUPERVISOR_SCHEMA, _identity_hashes, make_execution_contract
from tools.adaptation_evaluation.protocol import CONDITIONS, PREFIX, SCHEMA, digest, implementation_hashes
from tools.adaptation_evaluation.run import RATINGS, _expected_request


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def seal(value, key):
    value.pop(key, None)
    value[key] = digest(value)
    return value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(tmp_path, condition="facts"):
    identity = {"base_model": "synthetic/base", "base_revision": "b" * 40,
        "base_checkpoint_manifest_sha256": "c" * 64, "tokenizer_contract_sha256": "d" * 64,
        "chat_template_options": {"enable_thinking": False}}
    cases = []
    for index in range(1, 4):
        facts = [{"id": f"synthetic-{index}", "citation": f"[source:synthetic-{index}]", "text": "Synthetic evidence only."}]
        styles = [{"example_id": "synthetic-style", "answer": "Synthetic wording only."}]
        contexts = {"facts": {"factual_evidence": facts, "style_examples": []},
                    "style": {"factual_evidence": facts, "style_examples": styles}}
        question = f"Synthetic question {index}?"
        messages = {name: [{"role": "system", "content": "Use supplied evidence."},
            {"role": "user", "content": PREFIX + json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))},
            {"role": "user", "content": question}] for name, ctx in contexts.items()}
        messages["adapter"] = copy.deepcopy(messages["style"])
        cases.append({"id": f"case-{index}", "question": question, "language": "en",
            "reference_answer": "Private synthetic reference.", "reference_sha256": digest("Private synthetic reference."),
            "factual_ids": [facts[0]["id"]], "style_ids": [styles[0]["example_id"]], "factual_sha256": digest(facts),
            "messages": messages, "prompt_sha256": {key: digest(value) for key, value in messages.items()}})
    plan = seal({"schema_version": SCHEMA, "id": "synthetic-partial", "exposure": "development",
        "implementation_sha256": implementation_hashes(), "base_identity": identity,
        "policy": {"conditions": list(CONDITIONS), "context_window": 4096, "max_tokens": 160, "seed": 42},
        "cases": cases}, "plan_sha256")
    contract = make_execution_contract([plan["plan_sha256"]], identity)
    plan_path, contract_path = tmp_path / "plan.json", tmp_path / "contract.json"
    write(plan_path, plan)
    write(contract_path, contract)
    model = {"identity": identity, "adapter": {**identity, "configured": condition == "adapter",
        "loaded": condition == "adapter", "verified": condition == "adapter", "run_sha256": "a" * 64,
        "artifact_manifest_sha256": "e" * 64, "dataset_sha256": "f" * 64,
        "optimizer_steps": 1, "training_event_families": ["synthetic-train"]},
        "runtime": {"device": "cpu", "dtype": "float32", "threads": 8, "backend": "synthetic",
            "context_window": 4096, "attention": "synthetic", "base_parameters": 1000,
            "versions": {"torch": "synthetic", "transformers": "synthetic"},
            "generation_defaults": {"seed": 42, "top_k": 20, "temperature": 0.0, "top_p": 1.0}}}
    rows = []
    for index, case in enumerate(cases):
        cut = index == 1
        raw = {"text": (f"Synthetic output {index}: 한글\n" * 30), "completed": not cut,
            "error": "provider_finish_length" if cut else None, "finish_reason": "length" if cut else "stop",
            "response_model": "synthetic-local", "wall_ms": 2,
            "request_sha256": digest(_expected_request(plan, condition, "synthetic-local", case))}
        generated = copy.deepcopy(raw)
        if index == 2:
            generated.update(completed=False, error="post_generation_identity_pending")
        rows.append({"case_id": case["id"], "condition": condition, "prompt_sha256": case["prompt_sha256"][condition],
            "factual_sha256": case["factual_sha256"], "factual_ids": case["factual_ids"],
            "style_ids": [] if condition == "facts" else case["style_ids"], "provider_generation": raw,
            "generation": generated, "identity_check": {"before": "verified", "after": "pending" if index == 2 else "verified"},
            "validation_errors": [], "judgments": {name: None for name in RATINGS}})
    report = seal({"schema_version": "text-adaptation-condition-run.v1", "condition": condition,
        "plan_sha256": plan["plan_sha256"], "policy": plan["policy"], "planned_case_ids": [case["id"] for case in cases],
        "model": model, "served_model": "synthetic-local", "requested_adapter_run_sha256": "a" * 64 if condition == "adapter" else None,
        "status": "running", "started_at": "2026-01-01T00:01:00Z", "elapsed_seconds": 2,
        "human_review": "ungraded", "efficacy_claim": False, "runs": rows}, "run_sha256")
    source = tmp_path / "active"
    write(source / "condition/run.json", report)
    metadata = {"launcher.json": {"command": ["synthetic"], "declared_environment": {}},
        "warmup.json": {"status_code": 200, "response": {"model": "synthetic-local", "choices": [1]}, "excluded_from_condition_metrics": True},
        "observed-before.json": model}
    for name, value in metadata.items():
        write(source / name, value)
    supervisor = seal({"schema_version": SUPERVISOR_SCHEMA, "condition": condition, "status": "running",
        "plan_sha256": plan["plan_sha256"], "plan_file_sha256": sha(plan_path),
        "contract_sha256": contract["contract_sha256"], "contract_file_sha256": sha(contract_path),
        "wrapper_sha256": contract["wrapper_sha256"], "launcher_sha256": digest(metadata["launcher.json"]),
        "observed_before": _identity_hashes(model), "server_stopped": False}, "supervisor_sha256")
    write(source / "supervisor.json", supervisor)
    (source / "server.log").write_bytes(b"synthetic log\n\xff\x00partial")
    result = SimpleNamespace(plan_path=plan_path, contract_path=contract_path, source=source, report=report, supervisor=supervisor)
    refresh(result)
    return result


def refresh(run):
    write(run.source / "condition/run.json", seal(run.report, "run_sha256"))
    rows = run.report["runs"]
    finished = [row for row in rows if row["identity_check"]["after"] != "pending"]
    (run.source / "condition/attempts.jsonl").write_bytes(b"".join(partial._json_bytes(row).replace(b"\n", b" ") + b"\n" for row in finished))
    (run.source / "condition/raw-generations.jsonl").write_bytes(b"".join((json.dumps({"case_id": row["case_id"],
        "condition": row["condition"], "provider_generation": row["provider_generation"]}, ensure_ascii=False) + "\n").encode() for row in rows))


def capture(run, output, **kwargs):
    return partial.snapshot_condition(run.plan_path, run.contract_path, run.source, output, read_attempts=1, **kwargs)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_finalized_prefix_preserves_failures_raw_bytes_and_metadata(tmp_path, condition):
    run = make_run(tmp_path, condition)
    with (run.source / "condition/attempts.jsonl").open("ab") as stream:
        stream.write(b'{"unfinished":')
    before = {path: path.read_bytes() for path in run.source.rglob("*") if path.is_file()}
    output = tmp_path / "snapshot"
    result = capture(run, output, include_server_log=True)
    assert result["captured_attempts"] == 2 and result["captured_completed"] == 1 and result["captured_failed"] == 1
    assert result["case_ids"] == ["case-1", "case-2"]
    assert result["collectible"] is False and result["remote_persistence_performed"] is False and result["remote_readback_verified"] is False
    assert not (output / "condition/run.json").exists() and not (output / "supervisor.json").exists()
    prefix = json.loads((output / "partial-run.json").read_bytes())
    partial._check_seal(prefix, "partial_sha256")
    partial._check_seal(result, "manifest_sha256")
    assert prefix["runs"] == run.report["runs"][:2]
    assert prefix["runs"][1]["generation"]["error"] == "provider_finish_length"
    assert prefix["server_log_observation"]["final_log"] is False
    assert (output / "server.log.snapshot").read_bytes() == before[run.source / "server.log"]
    for name, identity in result["files"].items():
        raw = (output / name).read_bytes()
        assert {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} == identity
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert "Synthetic output 2" not in (output / "partial-run.json").read_text()


def test_explicit_prefix_stays_bounded(tmp_path):
    run = make_run(tmp_path)
    result = capture(run, tmp_path / "one", attempts=1)
    assert result["captured_attempts"] == result["captured_completed"] == 1
    assert result["captured_failed"] == 0 and "server.log.snapshot" not in result["files"]
    with pytest.raises(ValueError, match="finalized"):
        capture(run, tmp_path / "too-many", attempts=3)


@pytest.mark.parametrize("change", ["seal", "case_order", "condition", "attempt", "raw", "short_line", "warmup", "supervisor", "false_success"])
def test_inconsistent_observations_never_publish_a_snapshot(tmp_path, change):
    run = make_run(tmp_path)
    if change in {"seal", "case_order", "condition", "false_success"}:
        if change == "case_order":
            run.report["runs"][0]["case_id"] = "case-2"
        elif change == "condition":
            run.report["runs"][0]["condition"] = "adapter"
        elif change == "false_success":
            run.report["runs"][1]["generation"]["completed"] = True
        else:
            run.report["elapsed_seconds"] = 3
        write(run.source / "condition/run.json", run.report if change == "seal" else seal(run.report, "run_sha256"))
    elif change in {"attempt", "raw", "short_line"}:
        path = run.source / "condition" / ("raw-generations.jsonl" if change == "raw" else "attempts.jsonl")
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b"Synthetic output 0", b"changed output 0") if change != "short_line" else raw[:50])
    elif change == "warmup":
        path = run.source / "warmup.json"
        value = json.loads(path.read_text()); value["excluded_from_condition_metrics"] = False; write(path, value)
    else:
        run.supervisor["condition"] = "adapter"
        write(run.source / "supervisor.json", seal(run.supervisor, "supervisor_sha256"))
    output = tmp_path / "rejected"
    with pytest.raises((ValueError, KeyError)):
        capture(run, output)
    assert not output.exists()


def test_report_may_advance_without_changing_the_captured_prefix(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    original = partial._read_json
    reads = 0

    def advancing(path):
        nonlocal reads
        if path == run.source / "condition/run.json":
            reads += 1
            if reads == 2:
                run.report["runs"][2]["identity_check"]["after"] = "verified"
                run.report["runs"][2]["generation"].update(completed=True, error=None)
                refresh(run)
        return original(path)

    monkeypatch.setattr(partial, "_read_json", advancing)
    result = capture(run, tmp_path / "snapshot")
    assert result["captured_attempts"] == 2
    prefix = json.loads((tmp_path / "snapshot/partial-run.json").read_text())
    assert prefix["source_report"]["file_sha256"] != prefix["source_report"]["recheck_file_sha256"]


def test_changed_finalized_prefix_is_rejected_during_recheck(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    original = partial._read_json
    reads = 0

    def changing(path):
        nonlocal reads
        if path == run.source / "condition/run.json":
            reads += 1
            if reads == 2:
                for key in ("generation", "provider_generation"):
                    run.report["runs"][0][key]["text"] = "changed after first read"
                refresh(run)
        return original(path)

    monkeypatch.setattr(partial, "_read_json", changing)
    with pytest.raises(ValueError, match="changed during"):
        capture(run, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_transient_torn_report_is_retried_without_source_writes(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    original = partial._read_json
    first = True

    def torn(path):
        nonlocal first
        if path == run.source / "condition/run.json" and first:
            first = False
            raise json.JSONDecodeError("synthetic concurrent write", "", 0)
        return original(path)

    monkeypatch.setattr(partial, "_read_json", torn)
    result = partial.snapshot_condition(run.plan_path, run.contract_path, run.source, tmp_path / "snapshot", read_attempts=2)
    assert result["captured_attempts"] == 2


def test_no_overwrite_active_directory_or_premature_manifest(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        capture(run, run.source / "snapshot")
    output = tmp_path / "snapshot"
    capture(run, output)
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="new output"):
        capture(run, output)
    assert (output / "manifest.json").read_bytes() == before
    original = Path.open

    def fail_write(path, *args, **kwargs):
        if path == tmp_path / "interrupted/attempts.prefix.jsonl" and args and args[0] == "xb":
            raise OSError("synthetic write failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_write)
    with pytest.raises(OSError):
        capture(run, tmp_path / "interrupted")
    assert not (tmp_path / "interrupted/manifest.json").exists()
