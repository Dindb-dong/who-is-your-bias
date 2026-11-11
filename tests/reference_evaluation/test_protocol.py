import asyncio
import json
from dataclasses import replace
from datetime import date

import httpx
import pytest

from bias.config import ProviderSettings
from bias.corpus import Corpus
from bias.models import Evidence, Locator, Persona, QuestionAnswer, Source
from bias.retrieval import SearchResult
from tools.reference_evaluation.protocol import build_condition, digest, freeze_plan, qa_data, validate_plan
from tools.reference_evaluation.review import DIMENSIONS, export_bundle, ingest_review
from tools.reference_evaluation.run import generate, run


@pytest.fixture
def qa_corpus():
    sources, records = {}, {}
    for number, split in ((1, "train"), (2, "train"), (3, "test"), (4, "test")):
        source_id = f"event-{number}"
        sources[source_id] = Source(id=source_id, title=source_id, url=f"https://example.org/{source_id}", publisher="Synthetic test publisher", published_at=date(2020 + number, 1, 1), source_type="interview", split=split, source_event_id=source_id)
        question = "Why are reusable rockets worth the investment?" if split == "test" else "How can reusable rockets reduce costs?"
        answer = f"UNSEEN_REFERENCE_{number}" if split == "test" else f"Reusable rockets can reduce the cost of repeated launches, example {number}."
        qa = QuestionAnswer(question=question, answer=answer, original_language="en", questioner="Interviewer", respondent="Subject", source_event_id=source_id, completeness="complete_exchange", timing_basis="unavailable", transcript_locator={"question_cue_ids": [f"q{number}"], "answer_cue_ids": [f"a{number}"], "transcript_sha256": "a" * 64}, review={"reviewer_type": "agent", "reviewer": "Synthetic test", "reviewed_at": "2026-11-11", "basis": "Synthetic fixture, not real research data", "speaker_status": "publisher_labels_checked"}, annotation_basis="Synthetic test")
        record = Evidence(id=f"qa-{number}", source_id=source_id, text=answer, topics=["reusable", "rockets", "investment", "costs"], situations=["explanation" if number != 2 else "challenge"], locator=Locator(label="Synthetic exchange"), kind="question_answer", review_status="reviewed", question_answer=qa)
        records[record.id] = record
    return Corpus(Persona(id="subject", name="Test Subject", tagline="Synthetic", description="A fictional engineering interview subject.", qa_corpus_id="synthetic-test"), sources, records)


def mock_model(calls, *, finish="stop", text="Reusable rockets can reduce costs. [source:qa-1]"):
    def handler(request):
        calls.append(json.loads(request.content))
        events = [{"model": "fixed-model", "choices": [{"delta": {"content": text}, "finish_reason": None}]}, {"model": "fixed-model", "choices": [{"delta": {}, "finish_reason": finish}], "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content="".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n")
    return httpx.MockTransport(handler)


def test_frozen_plan_and_prompts_never_transmit_reference_answers(qa_corpus):
    manifest = {"events": {key: source.split for key, source in qa_corpus.sources.items()}}
    plan = freeze_plan(qa_corpus, manifest, plan_id="synthetic-v1")
    validate_plan(plan, qa_corpus, manifest)
    built = [build_condition(qa_corpus, plan.cases[0], condition) for condition in "ABC"]
    assert len({messages[0]["content"] for messages, _ in built}) == 1
    assert len({audit["profile_sha256"] for _, audit in built}) == 1
    assert not built[0][1]["selected"]
    for messages, audit in built:
        serialized = json.dumps(messages)
        assert "UNSEEN_REFERENCE" not in serialized
        assert all(item["source_event_id"] in {"event-1", "event-2"} for item in audit["selected"])
    changed = replace(qa_corpus, sources=qa_corpus.sources | {"event-3": qa_corpus.sources["event-3"].model_copy(update={"split": "train"})})
    with pytest.raises(ValueError, match="Frozen"):
        validate_plan(plan, changed, manifest)
    with pytest.raises(ValueError, match="Frozen"):
        validate_plan(plan, qa_corpus, {"changed": True})


def test_same_model_is_called_for_all_conditions_even_when_retrieval_is_empty(qa_corpus, monkeypatch):
    monkeypatch.setattr("tools.reference_evaluation.protocol.Retriever.search", lambda *args, **kwargs: SearchResult((), 0, "no_match"))
    plan = freeze_plan(qa_corpus, {}, plan_id="synthetic-v1")
    calls = []
    result = asyncio.run(run(plan, qa_corpus, {}, ProviderSettings(base_url="http://model.test/v1", model="fixed-model"), model_identity={"revision": "pinned", "runtime": "test"}, transport_factory=lambda: mock_model(calls)))
    assert len(calls) == len(plan.cases) * 3
    assert all(call["model"] == "fixed-model" and call["temperature"] == 0 and call["seed"] == 42 for call in calls)
    assert [row["condition"] for row in result["runs"]] == ["A", "B", "C", "B", "C", "A"]
    assert result["summary"]["conditions"]["B"]["no_examples"] == len(plan.cases)
    assert result["configuration"]["transport"] == "test_injected"
    assert result["summary"]["human_review"] == "ungraded"


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls"])
def test_truncated_or_unsupported_generation_preserves_raw_output_without_quality_claim(finish):
    row = asyncio.run(generate(ProviderSettings(base_url="http://model.test/v1", model="fixed-model"), [{"role": "user", "content": "question"}], max_tokens=20, timeout=1, seed=42, transport=mock_model([], finish=finish)))
    assert row["text"] and not row["completed"]
    assert row["error"] == "provider_finish_" + finish
    assert row["usage"]["completion_tokens"] == 10


def test_bundle_blinds_conditions_and_requires_real_review_provenance(qa_corpus, tmp_path):
    plan = freeze_plan(qa_corpus, {}, plan_id="synthetic-v1")
    report = asyncio.run(run(plan, qa_corpus, {}, ProviderSettings(base_url="http://model.test/v1", model="fixed-model"), model_identity={"revision": "pinned", "runtime": "test"}, transport_factory=lambda: mock_model([], text="<script>alert(1)</script> [source:qa-1]")))
    export_bundle(report, qa_corpus, tmp_path / "bundle")
    root = tmp_path / "bundle"
    bundle = json.loads((root / "reviewer/samples.json").read_text())
    key = json.loads((root / "private/answer-key.json").read_text())
    review = json.loads((root / "reviewer/review-template.json").read_text())
    assert all("condition" not in sample and "[source:" not in sample["candidate"] for sample in bundle["samples"])
    assert "<script>alert" not in (root / "reviewer/index.html").read_text()
    with pytest.raises(ValueError, match="actual reviewer"):
        ingest_review(review, bundle, key)
    review.update(reviewer_id="Test agent", reviewer_kind="model_assisted", method="Synthetic test fixture only")
    for judgment in review["judgments"]:
        judgment.update({dimension: 3 for dimension in DIMENSIONS}, note="Synthetic test; all conditions tied.")
    summary = ingest_review(review, bundle, key)
    assert not summary["human_review_complete"]
    assert summary["dimensions"]["content_viewpoint"]["event_balanced_C_minus_B"] == 0
    assert summary["dimensions"]["content_viewpoint"]["independent_events"] == 2
    sample_id = next(iter(key["key"]))
    modified_key = json.loads(json.dumps(key))
    modified_key["key"][sample_id]["condition"] = "C" if key["key"][sample_id]["condition"] != "C" else "A"
    with pytest.raises(ValueError, match="commitment"):
        ingest_review(review, bundle, modified_key)
    assert all("context_kind" in sample for sample in bundle["samples"])
    assert "Adjacent context" in (root / "reviewer/index.html").read_text()
    duplicated = {**report, "runs": report["runs"] + [report["runs"][0]]}
    with pytest.raises(ValueError, match="at most once"):
        export_bundle(duplicated, qa_corpus, tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()
    review["judgments"][0]["content_viewpoint"] = True
    with pytest.raises(ValueError, match="integers"):
        ingest_review(review, bundle, key)


def test_summary_records_missing_review_samples_instead_of_filling_zero(qa_corpus, tmp_path):
    plan = freeze_plan(qa_corpus, {}, plan_id="synthetic-v1")
    report = asyncio.run(run(plan, qa_corpus, {}, ProviderSettings(base_url="http://model.test/v1", model="fixed-model"), model_identity={"revision": "pinned", "runtime": "test"}, transport_factory=lambda: mock_model([])))
    export_bundle(report, qa_corpus, tmp_path / "bundle")
    root = tmp_path / "bundle"
    bundle = json.loads((root / "reviewer/samples.json").read_text())
    key = json.loads((root / "private/answer-key.json").read_text())
    review = {"schema_version": "reference-review.v1", "bundle_sha256": digest(bundle), "reviewer_id": "Human fixture", "reviewer_kind": "human", "method": "Synthetic test", "judgments": []}
    summary = ingest_review(review, bundle, key)
    assert len(summary["missing_samples"]) == 6 and summary["graded_samples"] == 0
    assert summary["dimensions"]["content_viewpoint"]["event_balanced_C_minus_B"] is None
    assert not summary["human_review_complete"]
