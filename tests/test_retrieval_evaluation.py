import copy
import hashlib
from pathlib import Path

import pytest

from tools.retrieval_evaluation.__main__ import capture_inputs, input_integrity, summarize, validate_cases


def example(corpus):
    record = corpus.evidence["rocket-reuse"]
    return {"schema_version": "retrieval-development-cases.v1", "split": "development", "cases": [
        {"id": "positive", "query": "Why reuse rockets?", "language": "en", "answerability": "answerable",
         "expected_evidence_ids": [record.id], "expected_source_ids": [record.source_id],
         "support_text_sha256": hashlib.sha256(record.text.encode()).hexdigest(), "review_basis": "Synthetic contract fixture"},
        {"id": "absent", "query": "What is your password?", "language": "en", "answerability": "absent",
         "expected_evidence_ids": [], "review_basis": "The fixture has no password evidence"},
    ]}


def test_evaluation_cannot_relabel_exposed_development_as_unseen(corpus):
    payload = example(corpus)
    assert len(validate_cases(payload, corpus)) == 2
    payload["split"] = "test"
    with pytest.raises(ValueError, match="development"):
        validate_cases(payload, corpus)


def test_frozen_support_and_source_identity_must_match(corpus):
    payload = example(corpus)
    payload["cases"][0]["support_text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="text changed"):
        validate_cases(payload, corpus)
    payload = example(corpus)
    payload["cases"][0]["expected_source_ids"] = ["other-source"]
    with pytest.raises(ValueError, match="source IDs"):
        validate_cases(payload, corpus)


def test_heldout_reference_cannot_become_retrieval_support(corpus):
    payload = example(corpus)
    payload["cases"][0]["expected_evidence_ids"] = ["secret-kiwi"]
    with pytest.raises(ValueError, match="training corpus"):
        validate_cases(payload, corpus)


def test_false_support_and_missing_evidence_have_distinct_metrics(corpus):
    payload = example(corpus)
    cases = copy.deepcopy(payload["cases"])
    cases[0]["conditions"] = {"hybrid": {"ids": []}}
    cases[1]["conditions"] = {"hybrid": {"ids": ["rocket-reuse"]}}
    metrics = summarize(cases, "hybrid")
    assert metrics["retrieval_hits"] == 0
    assert metrics["hit_rate_at_k"] == 0
    assert metrics["false_support_candidate_rate"] == 1
    assert metrics["failures"] == 0


def test_provider_failures_are_reported_instead_of_scored_as_abstentions(corpus):
    cases = copy.deepcopy(example(corpus)["cases"])
    for case in cases:
        case["conditions"] = {"hybrid": {"ids": [], "error": "EmbeddingError"}}
    metrics = summarize(cases, "hybrid")
    assert metrics["failures"] == 2
    assert metrics["hit_rate_at_k"] is None
    assert metrics["false_support_candidate_rate"] is None


def test_concurrent_input_edits_and_deletions_are_detected(tmp_path):
    source = tmp_path / "sources.json"
    source.write_text("first metadata")
    paths = {"sources": source, "optional": tmp_path / "missing.json"}
    first = capture_inputs(paths)
    assert input_integrity(first, capture_inputs(paths))["stable"]
    source.write_text("corrected metadata")
    report = input_integrity(first, capture_inputs(paths))
    assert not report["stable"] and report["changed_inputs"] == ["sources"]
    source.unlink()
    report = input_integrity(first, capture_inputs(paths))
    assert not report["stable"] and report["after"]["sources"] is None
