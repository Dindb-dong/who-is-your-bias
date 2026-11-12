"""Default Engine prompt capture with synthetic references; no model execution."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from bias.config import ProviderSettings, Settings
from bias.corpus import Corpus
from bias.engine import Engine
from bias.retrieval import Retriever
from bias.snapshots import register_snapshot
from tools.adaptation_evaluation.protocol import _context, digest, freeze_plan
from tools.adaptation_evaluation.rag_diagnostic import capture_runtime_prompt, freeze_rag_plan, public_diagnostic, validate_rag_plan

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def inputs():
    path = ROOT / "data/qna/elon-musk-v1"
    corpus = Corpus.load(path)
    engine = Engine(Settings(corpus_dir=path, llm=ProviderSettings("http://127.0.0.1:9/v1", "never-contacted"),
        llm_max_tokens=160, llm_context_window=4096, retrieval_backend="lexical", style_examples=False), corpus)
    now = datetime(2026, 11, 12, tzinfo=timezone.utc)
    engine.snapshot = register_snapshot(corpus, identifier="test-rag", as_of=now, ingested_at=now,
                                        current_evidence_since=now.date())
    cases = [{"id": "case-supported", "question": "How do you explain electric car efficiency?", "language": "en",
        "event_family_id": "reserved-test-only", "context_before": "", "context": [],
        "reference_answer": "Synthetic isolated reference alpha never supplied to the model."},
        {"id": "case-unsupported", "question": "What is his private mobile number?", "language": "en",
        "event_family_id": "reserved-test-only", "context_before": "", "context": [],
        "reference_answer": "Synthetic isolated reference beta never supplied to the model."}]
    labels = [{"id": row["id"], "answerability": "answerable" if index == 0 else "unanswerable",
        "reason": "Synthetic fixture classification, not an actual human evaluation.",
        "question_sha256": digest(row["question"]), "reference_sha256": digest(row["reference_answer"])}
        for index, row in enumerate(cases)]
    manifest = {"purpose": "evaluation_only", "leakage_policy": {"checked_corpora": [{"files": {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in ("profile.json", "sources.json", "evidence.jsonl")}}]},
        "generation_subset": {"answerability_review_sha256": digest(labels)}}
    selection = {row["id"]: {"factual_ids": [corpus.training_evidence[0].id] if index == 0 else [], "style_ids": []}
                 for index, row in enumerate(cases)}
    prior = freeze_plan(engine, cases, plan_id="prior-fixed-test", exposure="heldout", input_identity=manifest,
        selections=selection, base_identity={"base_model": "test/base", "base_revision": "b" * 40})
    return engine, cases, prior, labels


def test_actual_engine_capture_preserves_default_rank_and_whole_passages(inputs):
    engine, cases, _, _ = inputs
    original_llm, original_retriever = engine.llm, engine.retriever
    captured = asyncio.run(capture_runtime_prompt(engine, cases[0]))
    assert captured["status"] == "prompt_captured_no_generation" and captured["model_generations"] == 0
    assert engine.llm is original_llm and engine.retriever is original_retriever
    search = captured["searches"][0]
    assert search["arguments"]["limit"] == 4 and search["arguments"]["behavior"] is False
    assert search["arguments"]["allow_live"] is True
    actual = engine.retriever.search(search["query"], **search["arguments"])
    assert [row["id"] for row in search["hits"]] == [hit.evidence.id for hit in actual.hits]
    context = _context(captured["messages"])
    assert not context["style_examples"]
    assert captured["messages"][-1]["content"] == cases[0]["question"]
    assert all(row["text"] == engine.corpus.evidence[row["id"]].text for row in context["factual_evidence"])
    assert captured["budget"]["input_units"] <= 4096 - 160
    assert not any(row["reference_answer"] in json.dumps(captured["messages"]) for row in cases)


def test_runtime_refusal_is_retained_without_synthetic_prompt(inputs):
    engine, cases, _, _ = inputs
    captured = asyncio.run(capture_runtime_prompt(engine, cases[1]))
    assert captured["status"] == "runtime_did_not_generate" and captured["model_generations"] == 0
    assert "messages" not in captured
    assert captured["runtime_result"]["provider"] == "evidence"
    assert captured["runtime_result"]["grounding"]["status"] == "insufficient"


def test_single_base_freeze_keeps_all_registered_cases_and_prior_labels(inputs):
    engine, cases, prior, labels = inputs
    plan, traces = asyncio.run(freeze_rag_plan(engine, cases, prior, labels, identifier="rag-unit-test"))
    validate_rag_plan(plan)
    assert plan["rag_diagnostic"]["registered_case_ids"] == [row["id"] for row in cases]
    assert plan["rag_diagnostic"]["execution_conditions"] == ["facts"]
    assert plan["rag_diagnostic"]["answerability"] == labels
    assert plan["rag_diagnostic"]["source_fixed_evidence_plan_sha256"] == prior["plan_sha256"]
    assert len(plan["cases"]) == 1
    case = plan["cases"][0]
    assert case["messages"]["facts"] == case["messages"]["style"] == case["messages"]["adapter"]
    assert case["style_ids"] == [] and not case["style_contrast_present"]
    summary = public_diagnostic(plan, traces)
    assert summary["registered_cases"] == 2 and summary["planned_generations"] == 1
    assert summary["attempted_generations"] == 0
    assert summary["answerability_counts"] == {"answerable": 1, "unanswerable": 1}
    assert "private mobile number" not in json.dumps(summary)
    assert "Synthetic isolated reference" not in json.dumps(summary)
    assert not summary["efficacy_claim"]


@pytest.mark.parametrize("change", ["question", "reference", "labels", "context", "snapshot"])
def test_source_or_label_changes_rejected_before_capture(inputs, change):
    engine, cases, prior, labels = inputs
    if change == "question":
        cases[0]["question"] += " Changed."
    elif change == "reference":
        cases[0]["reference_answer"] += " Changed."
    elif change == "labels":
        labels[0]["answerability"] = "unanswerable"
    elif change == "snapshot":
        engine.snapshot = engine.snapshot.model_copy(update={"id": "changed-snapshot"})
    else:
        cases[0]["context_before"] = "Hidden reference-derived context."
    with pytest.raises(ValueError):
        asyncio.run(freeze_rag_plan(engine, cases, prior, labels, identifier="rejected"))
    assert not engine.sessions.sessions


def test_opt_in_retriever_cannot_claim_default_lexical(inputs):
    engine, cases, _, _ = inputs

    class DifferentRetriever(Retriever):
        pass

    engine.retriever = DifferentRetriever(engine.corpus)
    with pytest.raises(ValueError, match="default lexical"):
        asyncio.run(capture_runtime_prompt(engine, cases[0]))


def test_changed_extra_runtime_implementation_rejected(inputs, monkeypatch):
    engine, cases, prior, labels = inputs
    plan, _ = asyncio.run(freeze_rag_plan(engine, cases, prior, labels, identifier="rag-unit-test"))
    from tools.adaptation_evaluation import rag_diagnostic
    monkeypatch.setattr(rag_diagnostic, "rag_implementation_hashes", lambda: {"changed": "0" * 64})
    with pytest.raises(ValueError, match="implementation or prompts changed"):
        validate_rag_plan(plan)
