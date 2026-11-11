from __future__ import annotations

import asyncio
import copy
import json
import time
from pathlib import Path

import httpx
import pytest

from bias.config import ProviderSettings, Settings
from bias.corpus import Corpus
from bias.engine import Engine
from tools.adaptation_evaluation.protocol import _context, digest, freeze_plan, public_summary, validate_plan
from tools.adaptation_evaluation.run import compare_reports, export_review, read_identity, run_condition

ROOT = Path(__file__).resolve().parents[2]
REVISION = "b" * 40
ADAPTER = "a" * 64


@pytest.fixture
def inputs():
    corpus = Corpus.load(ROOT / "data/qna/elon-musk-v1")
    records = dict(corpus.evidence)
    ids = [e.id for e in corpus.training_evidence]
    qa = records[ids[1]].question_answer.model_copy(update={"completeness": "complete_exchange", "excerpt_note": None})
    records[ids[1]] = records[ids[1]].model_copy(update={"question_answer": qa})
    corpus = Corpus(corpus.persona, corpus.sources, records)
    engine = Engine(Settings(llm_context_window=20000, llm_max_tokens=128), corpus)
    cases = [{"id": "case-1", "question": "How do you explain electric car efficiency?", "language": "en",
              "reference_answer": "A private unseen reference target never supplied to the generating model."}]
    selections = {"case-1": {"factual_ids": [ids[0]], "style_ids": [ids[1]]}}
    return engine, cases, selections


def freeze(inputs):
    engine, cases, selections = inputs
    return freeze_plan(engine, cases, plan_id="controlled-test", exposure="development", selections=selections,
                       base_identity={"base_model": "test/base", "base_revision": REVISION})


def reseal(value, key):
    value.pop(key, None)
    value[key] = digest(value)


def test_only_style_context_changes_and_reference_is_private(inputs):
    plan = freeze(inputs)
    case = plan["cases"][0]
    assert case["messages"]["style"] == case["messages"]["adapter"]
    base, style = map(_context, [case["messages"]["facts"], case["messages"]["style"]])
    assert base["factual_evidence"] == style["factual_evidence"]
    assert not base["style_examples"] and style["style_examples"]
    assert inputs[1][0]["reference_answer"] not in json.dumps(case["messages"])
    assert inputs[1][0]["reference_answer"] not in json.dumps(public_summary(plan))
    assert "reference_answer" not in public_summary(plan)["cases"][0]
    assert case["style_ids"][0] not in case["factual_ids"]


def test_heldout_selection_cannot_enter_facts_or_style(inputs):
    engine, cases, selection = inputs
    test_id = next(e.id for e in engine.corpus.evidence.values() if engine.corpus.sources[e.source_id].split == "test")
    selection["case-1"]["factual_ids"] = [test_id]
    with pytest.raises(ValueError, match="eligible training"):
        freeze(inputs)


def test_duplicate_selection_rejected(inputs):
    inputs[2]["case-1"]["factual_ids"] *= 2
    with pytest.raises(ValueError, match="unique eligible"):
        freeze(inputs)


def test_changed_facts_rejected_even_with_new_outer_seal(inputs):
    plan = freeze(inputs)
    case = plan["cases"][0]
    case["messages"]["facts"][0]["content"] += " Extra instructions for one condition."
    case["prompt_sha256"]["facts"] = digest(case["messages"]["facts"])
    reseal(plan, "plan_sha256")
    with pytest.raises(ValueError, match="Only wording examples"):
        validate_plan(plan)


def test_cannot_label_unchecked_references_as_fresh(inputs):
    engine, cases, selection = inputs
    with pytest.raises(ValueError, match="validated evaluation-only"):
        freeze_plan(engine, cases, plan_id="invalid", exposure="heldout", selections=selection)
    with pytest.raises(ValueError, match="exact retrieval corpus"):
        freeze_plan(engine, cases, plan_id="invalid", exposure="heldout", selections=selection,
                    input_identity={"purpose": "evaluation_only", "leakage_policy": {"checked_corpora": []}})


def transport(condition, *, changed_revision=False, wrong_adapter=False, truncate=False,
              health_edit=None, post_health=None, response_model="local", base_parameters=1000):
    health_calls = 0

    async def handler(request):
        nonlocal health_calls
        if request.url.path == "/health":
            health_calls += 1
            health = {"status": "ready", "model_loaded": True, "model": "local",
                "identity": {"schema_version": "language-model-identity.v1", "base_model": "test/base",
                    "base_revision": "c" * 40 if changed_revision else REVISION,
                    "base_checkpoint_manifest_sha256": "d" * 64, "tokenizer_contract_sha256": "e" * 64,
                    "chat_template_options": {"enable_thinking": False}},
                "adapter": {"configured": condition == "adapter", "loaded": condition == "adapter",
                    "verified": condition == "adapter", "run_sha256": "f" * 64 if wrong_adapter else ADAPTER,
                    "base_model": "test/base", "base_revision": REVISION, "tokenizer_contract_sha256": "e" * 64,
                    "base_checkpoint_manifest_sha256": "d" * 64, "artifact_manifest_sha256": "f" * 64,
                    "dataset_sha256": "1" * 64, "optimizer_steps": 4, "training_event_families": ["train-event"]},
                "runtime": {"device": "cpu", "dtype": "float32", "backend": "transformers", "context_window": 8192,
                    "threads": 2, "attention": "sdpa", "base_parameters": base_parameters,
                    "versions": {"torch": "test", "transformers": "test"},
                    "generation_defaults": {"seed": 42, "top_k": 20, "temperature": 0.7, "top_p": 0.8}}}
            if health_edit:
                health_edit(health)
            if health_calls >= 3:
                if post_health == "timeout":
                    await asyncio.sleep(1)
                elif post_health == "unavailable":
                    return httpx.Response(503)
                elif post_health == "changed":
                    health["runtime"]["dtype"] = "bfloat16"
                elif post_health == "cancelled":
                    raise asyncio.CancelledError
            return httpx.Response(200, json=health)
        payload = json.loads(request.content)
        assert "private unseen reference" not in json.dumps(payload)
        events = [{"model": response_model, "choices": [{"delta": {"content": "An electrical motor can be efficient. [source:qa-ted-2013-efficiency]"}, "finish_reason": None}]},
                  {"model": response_model, "choices": [{"delta": {}, "finish_reason": "length" if truncate else "stop"}]}]
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n")
    return httpx.MockTransport(handler)


def execute(plan, condition, path, *, total_seconds=3600, **kwargs):
    return asyncio.run(run_condition(plan, condition, ProviderSettings("http://test/v1", "local"), path,
                       adapter_sha256=ADAPTER if condition == "adapter" else None,
                       total_seconds=total_seconds, transport=transport(condition, **kwargs)))


def test_actual_condition_controls_and_blind_export(inputs, tmp_path):
    plan = freeze(inputs)
    reports = [execute(plan, condition, tmp_path / condition) for condition in ("facts", "style", "adapter")]
    summary = compare_reports(plan, reports)
    assert all(r["completed"] == 1 for r in summary["conditions"].values())
    assert summary["cases_with_style_contrast"] == 1
    review = export_review(plan, reports, tmp_path / "review")
    assert review["samples"] == 3 and review["human_review"] == "ungraded"
    samples = (tmp_path / "review/reviewer/samples.json").read_text()
    assert '"condition"' not in samples and '"adapter"' not in samples
    assert all(all(v is None for v in r["judgments"].values()) for report in reports for r in report["runs"])


def test_wrong_base_and_wrong_adapter_stop_before_execution(inputs, tmp_path):
    plan = freeze(inputs)
    with pytest.raises(ValueError, match="preregistered"):
        execute(plan, "facts", tmp_path / "wrong-base", changed_revision=True)
    with pytest.raises(ValueError, match="exact verified"):
        execute(plan, "adapter", tmp_path / "wrong-adapter", wrong_adapter=True)
    assert not (tmp_path / "wrong-base").exists()


def test_truncation_remains_failure_and_is_not_a_blind_complete_sample(inputs, tmp_path):
    plan = freeze(inputs)
    reports = [execute(plan, c, tmp_path / c, truncate=c == "adapter") for c in ("facts", "style", "adapter")]
    summary = compare_reports(plan, reports)
    assert summary["conditions"]["adapter"]["completed"] == 0
    assert export_review(plan, reports, tmp_path / "review")["failed_attempts"] == 1


def test_mismatched_runtime_or_tampered_report_rejected(inputs, tmp_path):
    plan = freeze(inputs)
    reports = [execute(plan, c, tmp_path / c) for c in ("facts", "style", "adapter")]
    reports[-1]["model"]["runtime"]["dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="report changed"):
        compare_reports(plan, reports)
    reseal(reports[-1], "run_sha256")
    with pytest.raises(ValueError, match="precision and runtime"):
        compare_reports(plan, reports)


@pytest.mark.parametrize("failure", ["timeout", "unavailable", "changed"])
def test_post_health_failure_preserves_the_attempt_and_invalidates_completion(inputs, tmp_path, failure):
    plan = freeze(inputs)
    start = time.monotonic()
    failed = execute(plan, "adapter", tmp_path / "adapter", post_health=failure,
                     total_seconds=0.06 if failure == "timeout" else 10)
    if failure == "timeout":
        assert time.monotonic() - start < 0.75  # The 1-second probe cannot outlive the total budget.
        assert failed["status"] == "deadline_exhausted"
    else:
        assert failed["status"] == "failed"
    assert len(failed["runs"]) == 1 and failed["skipped_case_ids"] == []
    row = failed["runs"][0]
    assert row["provider_generation"]["completed"] is True
    assert row["provider_generation"]["error"] is None
    assert "electrical motor" in row["provider_generation"]["text"]
    assert row["generation"]["text"] == row["provider_generation"]["text"]
    assert row["generation"]["completed"] is False
    assert row["generation"]["error"].startswith("post_generation_identity_")
    raw = json.loads((tmp_path / "adapter/raw-generations.jsonl").read_text())
    assert raw["provider_generation"] == row["provider_generation"]
    assert json.loads((tmp_path / "adapter/attempts.jsonl").read_text()) == row
    persisted = json.loads((tmp_path / "adapter/run.json").read_text())
    assert persisted == failed
    reports = [execute(plan, c, tmp_path / c) for c in ("facts", "style")] + [failed]
    summary = compare_reports(plan, reports)
    assert summary["conditions"]["adapter"]["attempted"] == 1
    assert summary["conditions"]["adapter"]["completed"] == 0
    assert export_review(plan, reports, tmp_path / "review")["failed_attempts"] == 1
    # An outer checksum cannot turn unknown post-generation identity into success.
    row["generation"].update(completed=True, error=None)
    reseal(failed, "run_sha256")
    with pytest.raises(ValueError, match="verified request and model identity"):
        compare_reports(plan, reports)


def test_post_health_error_preserves_original_provider_truncation(inputs, tmp_path):
    report = execute(freeze(inputs), "facts", tmp_path / "failed", truncate=True, post_health="unavailable")
    row = report["runs"][0]
    assert row["provider_generation"]["completed"] is False
    assert row["provider_generation"]["error"] == "provider_finish_length"
    assert row["generation"]["error"] == "post_generation_identity_unverified"
    assert row["generation"]["finish_reason"] == "length"


def test_cancellation_during_post_health_keeps_received_output(inputs, tmp_path):
    with pytest.raises(asyncio.CancelledError):
        execute(freeze(inputs), "facts", tmp_path / "cancelled", post_health="cancelled")
    report = json.loads((tmp_path / "cancelled/run.json").read_text())
    assert report["status"] == "failed" and report["failure_type"] == "CancelledError"
    assert report["runs"][0]["provider_generation"]["completed"] is True
    assert report["runs"][0]["generation"]["completed"] is False
    assert (tmp_path / "cancelled/raw-generations.jsonl").is_file()


@pytest.mark.parametrize("response_model", [None, "another-model"])
def test_wrong_or_unknown_response_model_cannot_count_as_success(inputs, tmp_path, response_model):
    plan = freeze(inputs)
    reports = [execute(plan, c, tmp_path / c, response_model=response_model if c == "adapter" else "local")
               for c in ("facts", "style", "adapter")]
    row = reports[-1]["runs"][0]
    assert row["provider_generation"]["completed"] is True
    assert row["generation"]["completed"] is False
    assert row["generation"]["error"] == "provider_model_identity_changed"
    assert compare_reports(plan, reports)["conditions"]["adapter"]["completed"] == 0


def test_initial_health_probe_obeys_the_absolute_deadline(inputs, tmp_path):
    async def hanging_health(request):
        assert request.url.path == "/health"
        await asyncio.sleep(1)
        raise AssertionError("The deadline should cancel this probe")

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(run_condition(freeze(inputs), "facts", ProviderSettings("http://test/v1", "local"),
                    tmp_path / "not-started", total_seconds=0.03, transport=httpx.MockTransport(hanging_health)))
    assert time.monotonic() - start < 0.75
    assert not (tmp_path / "not-started").exists()


def test_actual_request_digest_is_checked_after_report_reseal(inputs, tmp_path):
    plan = freeze(inputs)
    reports = [execute(plan, c, tmp_path / c) for c in ("facts", "style", "adapter")]
    reports[-1]["runs"][0]["generation"]["request_sha256"] = "0" * 64
    reseal(reports[-1], "run_sha256")
    with pytest.raises(ValueError, match="Actual generation request"):
        compare_reports(plan, reports)


@pytest.mark.parametrize("swap", ["adapter_into_base", "base_into_adapter", "different_pin"])
def test_saved_condition_cannot_swap_adapter_identity_after_reseal(inputs, tmp_path, swap):
    plan = freeze(inputs)
    reports = [execute(plan, c, tmp_path / c) for c in ("facts", "style", "adapter")]
    if swap == "adapter_into_base":
        reports[0]["model"]["adapter"] = copy.deepcopy(reports[-1]["model"]["adapter"])
        reseal(reports[0], "run_sha256")
    elif swap == "base_into_adapter":
        reports[-1]["model"]["adapter"] = copy.deepcopy(reports[0]["model"]["adapter"])
        reseal(reports[-1], "run_sha256")
    else:
        reports[-1]["requested_adapter_run_sha256"] = "0" * 64
        reseal(reports[-1], "run_sha256")
    with pytest.raises(ValueError, match="cannot have an adapter|exact verified"):
        compare_reports(plan, reports)


def test_larger_control_requires_measured_greater_base_parameter_count(inputs, tmp_path):
    plan = freeze(inputs)
    reports = [execute(plan, c, tmp_path / c) for c in ("facts", "style", "adapter")]

    def different_model(health):
        health["identity"].update(base_model="test/larger-looking-name", base_revision="f" * 40,
                                  base_checkpoint_manifest_sha256="2" * 64)

    same_size = execute(plan, "larger", tmp_path / "same-size", health_edit=different_model, base_parameters=1000)
    with pytest.raises(ValueError, match="measured larger parameter count"):
        compare_reports(plan, [*reports, same_size])
    larger = execute(plan, "larger", tmp_path / "actually-larger", health_edit=different_model, base_parameters=2000)
    assert compare_reports(plan, [*reports, larger])["larger_model_compared"] is True


@pytest.mark.parametrize("field", ["torch", "transformers", "generation_defaults", "base_parameters"])
def test_mandatory_runtime_versions_and_measurements_cannot_be_unknown(inputs, tmp_path, field):
    def missing(health):
        if field in {"torch", "transformers"}:
            health["runtime"]["versions"][field] = None
        else:
            health["runtime"][field] = None

    with pytest.raises(ValueError, match="versions|generation defaults|measured size"):
        execute(freeze(inputs), "facts", tmp_path / "missing", health_edit=missing)
    assert not (tmp_path / "missing").exists()


def test_persisted_runtime_versions_are_revalidated_after_reseal(inputs, tmp_path):
    plan = freeze(inputs)
    reports = [execute(plan, c, tmp_path / c) for c in ("facts", "style", "adapter")]
    for report in reports:
        report["model"]["runtime"]["versions"] = {"torch": None, "transformers": None}
        reseal(report, "run_sha256")
    with pytest.raises(ValueError, match="versions are required"):
        compare_reports(plan, reports)
