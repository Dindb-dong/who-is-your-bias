"""Rendered source descriptions must follow the passages that fit the prompt."""
import json
from datetime import date
from types import SimpleNamespace

import pytest

from bias.context_budget import ByteBudgetCounter, build_snapshot_messages
from bias.corpus import Corpus
from bias.models import ChatRequest, Evidence, Persona, Source


def record(identifier, kind, *, answer="Synthetic source wording for this regression."):
    answer = answer.strip()
    value = {"id": identifier, "source_id": "fixture", "kind": "stated_view",
             "text": answer, "locator": {"label": "Authored test fixture"},
             "review_status": "reviewed", "confidence": "source_grounded"}
    if kind == "editorial":
        value["text_provenance"] = "editorial_summary"
    elif kind in {"original", "excerpt"}:
        value.update(kind="question_answer", question_answer={
            "question": "What is the reasoning?", "answer": answer, "context_kind": "none",
            "original_language": "en", "questioner": "Fixture interviewer", "respondent": "Fixture subject",
            "source_event_id": "fixture", "completeness": "excerpt" if kind == "excerpt" else "complete_exchange",
            "excerpt_note": "A bounded synthetic excerpt." if kind == "excerpt" else None,
            "timing_basis": "unavailable", "transcript_locator": {"question_cue_ids": ["q1"],
                "answer_cue_ids": ["a1"], "transcript_sha256": "a" * 64},
            "review": {"reviewer_type": "agent", "reviewer": "Test fixture", "reviewed_at": "2026-11-13",
                "basis": "Authored test data, not an actual interview.", "speaker_status": "publisher_labels_checked"},
            "annotation_basis": "Authored test data."})
    return Evidence.model_validate(value)


def render(records, *, context_window=16000):
    source = Source(id="fixture", title="Authored fixture", url="https://example.org/fixture",
                    publisher="Fixture", published_at=date(2024, 1, 1), source_type="interview")
    corpus = Corpus(Persona(id="jensen-huang", name="Jensen Huang", tagline="Fixture", description="Synthetic fixture"),
                    {source.id: source}, {r.id: r for r in records})
    result = SimpleNamespace(hits=[SimpleNamespace(evidence=r) for r in records], style_hits=[])
    return build_snapshot_messages(corpus, ChatRequest(message="판단의 근거는 무엇인가요?", language="ko", mode="rag"),
                                   result, [], None, None, context_window=context_window,
                                   output_tokens=128, counter=ByteBudgetCounter())


@pytest.mark.parametrize("kinds,original,editorial", [
    (["editorial"], False, True),
    (["original"], True, False),
    (["excerpt"], True, False),
    (["original", "editorial"], True, True),
    (["excerpt", "editorial"], True, True),
])
def test_rendered_korean_source_type_preserves_original_and_editorial_distinction(kinds, original, editorial):
    rows = [record(f"fixture-{i}", kind) for i, kind in enumerate(kinds)]
    packed = render(rows)
    system = packed.messages[0]["content"]
    assert ("원문 문답·발췌" in system) is original
    assert ("편집 요약" in system) is editorial
    assert "한국어 한 문장으로 짧게 답하고" in system
    assert "그의 입장을 1인칭으로 쓰거나" in system
    context = json.loads(packed.messages[1]["content"].split("\n", 1)[1])
    assert [(r["id"], r["text"], r["citation"]) for r in context["factual_evidence"]] == [
        (r.id, r.text, f"[source:{r.id}]") for r in rows]
    for row in context["factual_evidence"]:
        if rows[int(row["id"].rsplit("-", 1)[1])].question_answer:
            assert row["completeness"] in {"complete_exchange", "excerpt"}


@pytest.mark.parametrize("kept,dropped", [("original", "editorial"), ("editorial", "original")])
def test_budget_dropped_candidate_does_not_change_rendered_source_type(kept, dropped):
    packed = render([record("short", kept), record("oversized", dropped, answer="Long authored fixture. " * 400)],
                    context_window=5000)
    assert packed.supplied_ids == ("short",)
    assert packed.budget["dropped_evidence_ids"] == ["oversized"]
    system = packed.messages[0]["content"]
    assert ("원문 문답·발췌" in system) is (kept == "original")
    assert ("편집 요약" in system) is (kept == "editorial")
