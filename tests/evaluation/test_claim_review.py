import asyncio
from copy import deepcopy

import httpx
import pytest

from bias.config import ProviderSettings, Settings
from tools.evaluation.cases import GenerationCase, GenerationSuite
from tools.evaluation.review import digest, make_publication, prepare_packet, summarize_review
from tools.evaluation.run import run_suite


def report_for(corpus, text="Repeated flights can lower launch costs. [source:rocket-reuse]"):
    case = GenerationCase(id="review-test", question="Why reuse rockets?", category="answerable", expected_action="answer", expected_evidence_ids=["rocket-reuse"], review_note="Inspect the repeated-flight cost explanation.")
    suite = GenerationSuite(schema_version="generation-evaluation.v1", id="review-protocol-test", declaration="Synthetic protocol output; not a model-quality result.", development_only=True, cases=[case])
    import json
    payload = "data: " + json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": None}]}) + '\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    factory = lambda: httpx.MockTransport(lambda request: httpx.Response(200, text=payload))
    return asyncio.run(run_suite(Settings(llm=ProviderSettings("http://test/v1", "synthetic-test")), corpus, suite, modes=("rag",), transport_factory=factory))


def annotate_supported(annotations):
    annotated = deepcopy(annotations)
    annotated["reviewer"] = {"name": "Protocol-test reviewer", "kind": "model_assisted", "description": "Synthetic annotations used only to test report validation.", "independent": False, "blinded": False}
    annotated["runs"][0]["units"][0].update(evidence_support="supported", citation_support="supported", supporting_evidence_ids=["rocket-reuse"], rationale="The cited test record explicitly connects repeated flights with lower launch costs.")
    annotated["runs"][0]["answer"].update(relevance="answers", language_quality="fluent", rationale="The sentence directly addresses reuse and is grammatical English.")
    return annotated


def test_prepare_keeps_exact_source_text_and_leaves_all_judgments_ungraded(corpus):
    report = report_for(corpus)
    packet, annotations = prepare_packet(report, corpus, report_sha256=digest(report))
    entry = packet["runs"][0]
    assert entry["raw_response"] == report["runs"][0]["raw_response"]
    assert entry["prompt_evidence"][0]["text"] == corpus.evidence["rocket-reuse"].text
    assert entry["prompt_evidence"][0]["published_at"] == "2020-01-01"
    assert entry["units"][0]["citation_ids"] == ["rocket-reuse"]
    summary = summarize_review(packet, annotations)
    assert summary["status"] == "ungraded"
    assert summary["reviewer"] is None
    assert summary["persona_fidelity"] == "ungraded"
    assert summary["populations"]["model"]["evidence_support"] == {"ungraded": 1}
    assert report["runs"][0]["semantic_review"]["status"] == "ungraded"


def test_explicit_nonblind_review_provenance_is_preserved_separately_from_generation(corpus):
    report = report_for(corpus)
    packet, annotations = prepare_packet(report, corpus, report_sha256=digest(report))
    summary = summarize_review(packet, annotate_supported(annotations))
    assert summary["status"] == "reviewed_subset"
    assert summary["reviewer"]["kind"] == "model_assisted"
    assert summary["reviewer"]["blinded"] is False
    assert summary["populations"]["model"]["citation_support"] == {"supported": 1}
    assert report["runs"][0]["turn"]["grounding"]["status"] == "unverified"


@pytest.mark.parametrize("mutation,error", [
    (lambda data: data.update(reviewer=None), "named reviewer"),
    (lambda data: data["runs"][0]["units"][0].update(rationale=None), "rationale"),
    (lambda data: data["runs"][0]["units"][0].update(supporting_evidence_ids=["secret-kiwi"]), "actual prompt evidence"),
    (lambda data: data["runs"][0]["units"][0].update(supporting_evidence_ids=["invented"]), "actual prompt evidence"),
    (lambda data: data["runs"][0]["units"].clear(), "original response unit"),
    (lambda data: data["runs"].append(deepcopy(data["runs"][0])), "each packet run"),
])
def test_unsupported_review_provenance_and_cherry_picked_units_are_rejected(corpus, mutation, error):
    report = report_for(corpus)
    packet, annotations = prepare_packet(report, corpus, report_sha256=digest(report))
    annotated = annotate_supported(annotations)
    mutation(annotated)
    with pytest.raises(ValueError, match=error):
        summarize_review(packet, annotated)


def test_uncited_statement_cannot_get_a_positive_citation_support_label(corpus):
    report = report_for(corpus, "Repeated flights can lower launch costs.")
    packet, annotations = prepare_packet(report, corpus, report_sha256=digest(report))
    annotated = annotate_supported(annotations)
    with pytest.raises(ValueError, match="actually cited"):
        summarize_review(packet, annotated)
    annotated["runs"][0]["units"][0]["citation_support"] = "not_applicable"
    summary = summarize_review(packet, annotated)
    assert summary["populations"]["model"]["evidence_support"] == {"supported": 1}
    assert summary["populations"]["model"]["citation_support"] == {"not_applicable": 1}


def test_citation_only_wrapper_can_be_reviewed_as_nonfactual_without_claiming_support(corpus):
    report = report_for(corpus, "A claim. (citation: [source:rocket-reuse])")
    packet, annotations = prepare_packet(report, corpus, report_sha256=digest(report))
    annotations["reviewer"] = {"name": "Protocol-test reviewer", "kind": "model_assisted", "description": "Synthetic annotation of a nonfactual citation wrapper.", "independent": False, "blinded": False}
    annotations["runs"][0]["units"][1].update(evidence_support="not_factual", citation_support="not_applicable", rationale="This unit is only a citation wrapper; it contains no factual proposition.")
    summary = summarize_review(packet, annotations)
    assert summary["populations"]["model"]["evidence_support"] == {"ungraded": 1, "not_factual": 1}


def test_response_or_source_tampering_breaks_packet_binding(corpus):
    report = report_for(corpus)
    packet, annotations = prepare_packet(report, corpus, report_sha256=digest(report))
    packet["runs"][0]["raw_response"] = "A different claim. [source:rocket-reuse]"
    with pytest.raises(ValueError, match="exact packet"):
        summarize_review(packet, annotations)
    corpus.evidence["rocket-reuse"] = corpus.evidence["rocket-reuse"].model_copy(update={"text": "Changed source text."})
    with pytest.raises(ValueError, match="exact corpus"):
        prepare_packet(report, corpus, report_sha256=digest(report))


def test_summary_only_or_incomplete_generation_cannot_become_a_semantic_pass(corpus):
    report = report_for(corpus)
    report["runs"][0]["execution"]["completed"] = False
    packet, annotations = prepare_packet(report, corpus, report_sha256=digest(report))
    with pytest.raises(ValueError, match="Incomplete generations"):
        summarize_review(packet, annotate_supported(annotations))
    assert summarize_review(packet, annotations)["populations"]["model"]["completed_runs"] == 0
    del report["runs"][0]["raw_response"]
    with pytest.raises(ValueError, match="raw responses"):
        prepare_packet(report, corpus, report_sha256=digest(report))


def test_publication_preserves_negative_examples_and_review_bindings_without_source_dumps(corpus):
    report = report_for(corpus, "Every launch costs exactly 1 dollar. [source:rocket-reuse]")
    packet, annotations = prepare_packet(report, corpus, report_sha256=digest(report))
    annotations = annotate_supported(annotations)
    annotations["runs"][0]["units"][0].update(evidence_support="unsupported", citation_support="unsupported", supporting_evidence_ids=[], rationale="The exact one-dollar price is absent from the source; the valid identifier does not support it.")
    runtime = {"model": "explicit-synthetic-test-model", "revision": "test-revision", "transport": "test_injected"}
    publication = make_publication([(packet, annotations)], study_labels=["Synthetic protocol test"], runtime_metadata=[runtime])
    study = publication["studies"][0]
    assert study["runs"][0]["raw_response"] == report["runs"][0]["raw_response"]
    assert study["packet_sha256"] == digest(packet)
    assert study["label"] == "Synthetic protocol test"
    assert study["runtime_metadata_sha256"] == digest(runtime)
    assert study["runs"][0]["prompt_evidence_ids"] == ["rocket-reuse"]
    assert "prompt_evidence" not in study["runs"][0]
    assert study["condition_reviews"]["B"]["citation_coverage"] == {"complete": 1}
    assert study["condition_reviews"]["B"]["answers_with_all_factual_units_supported"] == 0
    assert study["runs"][0]["units"][0]["review"]["evidence_support"] == "unsupported"
    assert publication["persona_fidelity"] == "ungraded"
    with pytest.raises(ValueError, match="same reviewed packet"):
        make_publication([(packet, annotations), (packet, annotations)])
