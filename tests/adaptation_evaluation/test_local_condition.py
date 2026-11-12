"""Orchestration checks use fake processes/HTTP; they do not load a model."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from bias.config import Settings
from bias.corpus import Corpus
from bias.engine import Engine
from services.language.adapters import validate_adapter_run
from tools.adaptation_evaluation import local_condition as local
from tools.adaptation_evaluation.protocol import digest, freeze_plan
from tools.adaptation_evaluation.run import read_identity, run_condition

ROOT = Path(__file__).resolve().parents[2]
REVISION, ADAPTER = "b" * 40, "a" * 64
IDENTITY = {"base_model": "test/base", "base_revision": REVISION,
    "base_checkpoint_manifest_sha256": "d" * 64, "tokenizer_contract_sha256": "e" * 64,
    "chat_template_options": {"enable_thinking": False}}


@pytest.fixture
def registered(tmp_path):
    corpus = Corpus.load(ROOT / "data/qna/elon-musk-v1")
    records = dict(corpus.evidence)
    ids = [e.id for e in corpus.training_evidence]
    qa = records[ids[1]].question_answer.model_copy(update={"completeness": "complete_exchange", "excerpt_note": None})
    records[ids[1]] = records[ids[1]].model_copy(update={"question_answer": qa})
    engine = Engine(Settings(llm_context_window=20000, llm_max_tokens=128), Corpus(corpus.persona, corpus.sources, records))
    plan = freeze_plan(engine, [{"id": "case-1", "question": "Explain electric car efficiency.", "language": "en",
        "reference_answer": "A private reference never supplied to the generating model."}],
        plan_id="single-condition-test", exposure="development",
        selections={"case-1": {"factual_ids": [ids[0]], "style_ids": [ids[1]]}},
        base_identity={"base_model": "test/base", "base_revision": REVISION})
    contract = local.make_execution_contract([plan["plan_sha256"]], copy.deepcopy(IDENTITY))
    local._write(tmp_path / "plan.json", plan)
    local._write(tmp_path / "contract.json", contract)
    return SimpleNamespace(plan=plan, contract=contract, plan_path=tmp_path / "plan.json", contract_path=tmp_path / "contract.json")


class Process:
    pid = 12345

    def __init__(self, *, ignores_terminate=False):
        self.returncode = None
        self.ignores_terminate = ignores_terminate
        self.terminated = self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.ignores_terminate:
            self.returncode = -15

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-server", timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


@pytest.fixture
def server(monkeypatch):
    state = SimpleNamespace(processes=[], launches=[], warmups=[], condition_calls=[], adapter_checks=[],
        adapter=False, warmup_status=200, ignores_terminate=False, changed_identity=None,
        abort_after_report=False, truncate_adapter=False)

    def launch(command, *, env, stdout, stderr):
        state.launches.append({"command": command, "env": env})
        state.adapter = "BIAS_LANGUAGE_ADAPTER_RUN" in env
        process = Process(ignores_terminate=state.ignores_terminate)
        state.processes.append(process)
        return process

    def warmup(url, *, json, **kwargs):
        state.warmups.append(json)
        return httpx.Response(state.warmup_status, json={"model": "comparison-local",
            "choices": [{"message": {"role": "assistant", "content": "Ready"}, "finish_reason": "stop"}]})

    async def handler(request):
        if request.url.path == "/health":
            identity = copy.deepcopy(IDENTITY)
            if state.changed_identity:
                identity.update(state.changed_identity)
            return httpx.Response(200, json={"status": "ready", "model_loaded": True, "model": "comparison-local",
                "identity": {"schema_version": "language-model-identity.v1", **identity},
                "adapter": {"configured": state.adapter, "loaded": state.adapter, "verified": state.adapter,
                    "run_sha256": ADAPTER, "base_model": "test/base", "base_revision": REVISION,
                    "base_checkpoint_manifest_sha256": "d" * 64, "tokenizer_contract_sha256": "e" * 64,
                    "artifact_manifest_sha256": "f" * 64, "dataset_sha256": "1" * 64,
                    "optimizer_steps": 4, "training_event_families": ["train-event"]},
                "runtime": {"device": "cpu", "dtype": "float32", "backend": "transformers", "context_window": 32768,
                    "threads": 8, "attention": "sdpa", "base_parameters": 1000,
                    "versions": {"torch": "test", "transformers": "test"},
                    "generation_defaults": {"seed": 42, "top_k": 20, "temperature": 0.7, "top_p": 0.8}}})
        payload = json.loads(request.content)
        assert "private reference" not in json.dumps(payload)
        events = [{"model": "comparison-local", "choices": [{"delta": {"content": "An electrical motor can be efficient. [source:qa-ted-2013-efficiency]"}, "finish_reason": None}]},
                  {"model": "comparison-local", "choices": [{"delta": {}, "finish_reason": "length" if state.adapter and state.truncate_adapter else "stop"}]}]
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
            text="".join("data: " + json.dumps(row) + "\n\n" for row in events) + "data: [DONE]\n\n")

    transport = httpx.MockTransport(handler)

    async def identity(provider, **kwargs):
        return await read_identity(provider, transport=transport, **kwargs)

    async def execute(plan, condition, provider, output, **kwargs):
        state.condition_calls.append({"plan": plan, "condition": condition, "kwargs": kwargs})
        report = await run_condition(plan, condition, provider, output, transport=transport, **kwargs)
        if state.abort_after_report:
            raise KeyboardInterrupt("test interruption after durable local report")
        return report

    def adapter(path, **kwargs):
        state.adapter_checks.append({"path": path, **kwargs})
        return SimpleNamespace(run_sha256=ADAPTER)

    monkeypatch.setattr(local, "subprocess", SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT,
                                                           TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(local.httpx, "get", lambda *args, **kwargs: httpx.Response(200))
    monkeypatch.setattr(local.httpx, "post", warmup)
    monkeypatch.setattr(local, "read_identity", identity)
    monkeypatch.setattr(local, "run_condition", execute)
    monkeypatch.setattr(local, "validate_adapter_run", adapter)
    return state


def execute(registered, condition, output, **kwargs):
    if condition == "adapter":
        kwargs.setdefault("adapter_path", output.parent / "adapter-run")
        kwargs.setdefault("adapter_sha256", ADAPTER)
    return local.run_local_condition(registered.plan_path, registered.contract_path, condition, output, **kwargs)


def check_supervisor(directory):
    report = json.loads((directory / "supervisor.json").read_text())
    unsigned = dict(report)
    assert unsigned.pop("supervisor_sha256") == digest(unsigned)
    assert report["files_sha256"] == {str(p.relative_to(directory)): local._sha(p)
        for p in directory.rglob("*") if p.is_file() and p.name != "supervisor.json"}
    return report


@pytest.mark.parametrize("condition", ["facts", "style", "adapter"])
def test_one_condition_retains_actual_runner_output_and_stops(registered, server, tmp_path, monkeypatch, condition):
    monkeypatch.setenv("BIAS_LANGUAGE_ADAPTER_RUN", "poisoned-inherited-adapter")
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(local.sys.executable)
    monkeypatch.setattr(local.sys, "executable", str(executable))
    output = tmp_path / condition
    result = execute(registered, condition, output)
    assert result == check_supervisor(output)
    assert result["status"] == "completed" and result["completed"] == result["attempted"] == 1
    assert result["server_stopped"] and server.processes[0].terminated
    assert len(server.condition_calls) == len(server.launches) == len(server.warmups) == 1
    assert server.condition_calls[0]["condition"] == condition
    assert server.condition_calls[0]["plan"] == registered.plan
    assert server.warmups[0]["max_tokens"] == 12
    assert result["warmup_included_in_condition_metrics"] is False
    assert json.loads((output / "warmup.json").read_text())["excluded_from_condition_metrics"]
    assert server.launches[0]["command"][0] == str(executable.absolute())
    assert server.launches[0]["command"][0] != str(executable.resolve())
    assert server.launches[0]["env"]["BIAS_LANGUAGE_THREADS"] == "8"
    assert server.launches[0]["env"]["BIAS_LANGUAGE_REVISION"] == REVISION
    assert result["observed_before"] == result["observed_after"]
    assert result["observed_before"]["identity_sha256"] == digest(IDENTITY)
    review = json.loads((output / "condition-review.json").read_text())
    assert review["human_review"] == "ungraded" and not review["blind"]
    assert "private reference" in review["samples"][0]["reference_answer"]
    if condition == "adapter":
        assert server.adapter_checks == [{"path": tmp_path / "adapter-run", "base_model": "test/base",
            "base_revision": REVISION, "expected_run_sha256": ADAPTER}]
        assert server.launches[0]["env"]["BIAS_LANGUAGE_ADAPTER_RUN_SHA256"] == ADAPTER
    else:
        assert not server.adapter_checks
        assert "BIAS_LANGUAGE_ADAPTER_RUN" not in server.launches[0]["env"]


@pytest.mark.parametrize("field", ["base_checkpoint_manifest_sha256", "tokenizer_contract_sha256"])
def test_wrong_actual_base_bytes_stop_before_cases(registered, server, tmp_path, field):
    server.changed_identity = {field: "0" * 64}
    with pytest.raises(ValueError, match="Observed base bytes"):
        execute(registered, "facts", tmp_path / "failed")
    report = check_supervisor(tmp_path / "failed")
    assert report["status"] == "failed" and report["server_stopped"]
    assert not server.condition_calls


@pytest.mark.parametrize("change", ["wrapper", "plan", "runtime"])
def test_contract_or_runtime_drift_rejected_without_server(registered, server, tmp_path, change):
    kwargs = {}
    if change == "runtime":
        kwargs["threads"] = 4
    else:
        registered.contract["wrapper_sha256" if change == "wrapper" else "plan_sha256"] = "0" * 64 if change == "wrapper" else ["0" * 64]
        registered.contract.pop("contract_sha256")
        registered.contract["contract_sha256"] = digest(registered.contract)
        local._write(registered.contract_path, registered.contract)
    with pytest.raises(ValueError):
        execute(registered, "facts", tmp_path / "rejected", **kwargs)
    assert not server.launches and not (tmp_path / "rejected").exists()


def test_incomplete_adapter_is_not_accepted_as_base_manifest(registered, server, tmp_path, monkeypatch):
    directory = tmp_path / "adapter-run"
    directory.mkdir()
    local._write(directory / "run.json", {"schema_version": "text-adaptation-run.v2", "status": "running"})
    monkeypatch.setattr(local, "validate_adapter_run", validate_adapter_run)
    with pytest.raises(ValueError, match="Only a completed"):
        execute(registered, "adapter", tmp_path / "rejected", adapter_sha256=local._sha(directory / "run.json"))
    assert not server.launches


def test_interruption_preserves_raw_attempt_and_kills_unresponsive_server(registered, server, tmp_path):
    server.abort_after_report = server.ignores_terminate = True
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        execute(registered, "facts", output)
    report = check_supervisor(output)
    assert report["status"] == "failed" and report["failure_type"] == "KeyboardInterrupt"
    assert report["server_stopped"] and server.processes[0].killed
    assert report["attempted"] == 1
    assert (output / "condition/raw-generations.jsonl").is_file()
    assert (output / "condition-review.json").is_file()


def test_failed_warmup_is_saved_and_stops_before_cases(registered, server, tmp_path):
    server.warmup_status = 503
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="warmup failed"):
        execute(registered, "style", output)
    assert check_supervisor(output)["server_stopped"]
    assert json.loads((output / "warmup.json").read_text())["status_code"] == 503
    assert not server.condition_calls


def persisted_conditions(registered, server, tmp_path):
    server.truncate_adapter = True
    directories, entries = [], []
    for condition in ("facts", "style", "adapter"):
        directory = tmp_path / condition
        execute(registered, condition, directory)
        directories.append(directory)
        entries.append({"condition": condition, "plan_sha256": registered.plan["plan_sha256"],
            "status": "uploaded_and_fresh_readback_verified",
            "files_sha256": {str(p.relative_to(directory)): local._sha(p) for p in directory.rglob("*") if p.is_file()},
            "archive": {"bytes": 123, "sha256": "9" * 64, "file_id": "test-" + condition, "library_file_id": "test-lib-" + condition}})
    receipts = tmp_path / "receipts.json"
    local._write(receipts, {"schema_version": local.DURABILITY_SCHEMA, "conditions": entries})
    return directories, receipts


def test_collection_retains_failure_denominator_and_exports_blind_packet(registered, server, tmp_path):
    directories, receipts = persisted_conditions(registered, server, tmp_path)
    summary = local.collect_conditions(registered.plan_path, registered.contract_path, directories, receipts, tmp_path / "collection")
    assert summary["conditions"]["adapter"]["attempted"] == 1
    assert summary["conditions"]["adapter"]["completed"] == 0
    assert summary["human_review"] == "ungraded" and not summary["efficacy_claim"]
    assert summary["local_execution"]["conditions_by_start_time"] == ["facts", "style", "adapter"]
    assert summary["local_execution"]["warmup_excluded_per_condition"]
    review = json.loads((tmp_path / "collection/review-summary.json").read_text())
    assert review["samples"] == 2 and review["failed_attempts"] == 1
    blind = (tmp_path / "collection/blind-review/reviewer/samples.json").read_text()
    assert '"condition"' not in blind
    assert len(server.launches) == 3  # Collection does not start a fourth server.


@pytest.mark.parametrize("change", ["pending", "receipt_hash", "raw_file", "missing_condition"])
def test_collection_rejects_unverified_or_changed_artifacts(registered, server, tmp_path, change):
    directories, receipts_path = persisted_conditions(registered, server, tmp_path)
    receipts = json.loads(receipts_path.read_text())
    if change == "pending":
        receipts["conditions"][0]["status"] = "uploaded_not_read_back"
    elif change == "receipt_hash":
        receipts["conditions"][0]["files_sha256"]["supervisor.json"] = "0" * 64
    elif change == "raw_file":
        with (directories[0] / "condition/raw-generations.jsonl").open("a") as stream:
            stream.write("changed after upload\n")
    else:
        receipts["conditions"].pop()
    local._write(receipts_path, receipts)
    with pytest.raises(ValueError, match="Persistence receipts|fresh-readback persistence"):
        local.collect_conditions(registered.plan_path, registered.contract_path, directories, receipts_path, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
