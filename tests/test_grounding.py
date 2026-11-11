import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from bias.config import ProviderSettings, Settings
from bias.engine import Engine
from bias.grounding import audit_citations
from bias.legacy_prompts import build_legacy_messages
from bias.models import ChatRequest
from bias.prompts import PROMPT_BUILDERS, build_compact_messages, build_grounded_messages, build_messages
from bias.retrieval import Retriever, SearchResult
from tools.evaluation.cases import read_suite
from tools.evaluation.run import fingerprint_corpus


@pytest.mark.parametrize("text,units,cited,status", [
    ("A claim. [source:a] Second claim. [source:b]", 2, 2, "complete"),
    ("A claim [source:a]. Second claim.", 2, 1, "partial"),
    ("A claim. Another claim. [source:a]", 2, 1, "partial"),
    ("[source:a] An uncited claim.", 1, 0, "missing"),
    ("A claim [source:a] with an uncited extension.", 1, 0, "missing"),
    ("[source:a]", 0, 0, "missing"),
    ("No evidence.", 1, 0, "missing"),
    ("A claim.\n[source:a]\nSecond claim [source:b].", 2, 2, "complete"),
    ("Dr. Musk described U.S. launches costing 3.5 million. [source:a]", 1, 1, "complete"),
    ("1. A claim. [source:a]\n2. Second claim [source:b].", 2, 2, "complete"),
    ("근거가 있어요. [source:a] 다음 문장입니다. [source:b]", 2, 2, "complete"),
    ("Claim. [source:a] [source:b]", 1, 1, "complete"),
    ("Claim. [source:invented]", 1, 0, "invalid"),
])
def test_sentence_marker_coverage_does_not_reduce_to_any_known_identifier(text, units, cited, status):
    audit = audit_citations(text, {"a", "b"}, required=True)
    assert len(audit.units) == units
    assert audit.cited_unit_count == cited
    assert audit.coverage_status == status
    assert audit.as_dict()["semantic_support"] == "ungraded"


@pytest.mark.parametrize("marker", ["[source:wrong id]", "[SOURCE:a]", "[source:a", "[ source:a]", "[source :a]"])
def test_malformed_marker_attempts_are_reported(marker):
    audit = audit_citations("Claim. " + marker, {"a"}, required=True)
    assert audit.coverage_status == "invalid"
    assert audit.malformed_markers == (marker,)


def test_one_malformed_marker_does_not_swallow_the_next_valid_marker():
    audit = audit_citations("Claim [source:unfinished [source:a]", {"a"}, required=True)
    assert audit.valid_ids == ("a",)
    assert audit.malformed_markers == ("[source:unfinished ",)
    assert audit.coverage_status == "invalid"


def test_false_numeric_claim_with_valid_citation_is_still_unverified(corpus):
    class FalseLLM:
        async def stream(self, messages):
            yield "Every launch costs exactly 1 dollar. [source:rocket-reuse]"

    engine = Engine(Settings(llm=ProviderSettings("http://test/v1", "synthetic-test")), corpus, llm=FalseLLM())
    turn = asyncio.run(engine.chat(ChatRequest(message="Why reuse rockets?", mode="rag")))
    assert turn.grounding.citation_status == "complete"
    assert turn.grounding.citation_unit_count == turn.grounding.cited_unit_count == 1
    assert turn.grounding.status == "unverified"
    assert turn.grounding.semantic_status == "unreviewed"
    assert turn.text == "Every launch costs exactly 1 dollar."
    assert turn.citations[0].evidence_id == "rocket-reuse"


def test_model_output_is_never_repaired_or_given_synthetic_citations(corpus):
    calls = []

    class MissingCitationLLM:
        async def stream(self, messages):
            calls.append(messages)
            yield "Reusable rockets can lower launch costs."

    engine = Engine(Settings(llm=ProviderSettings("http://test/v1", "synthetic-test")), corpus, llm=MissingCitationLLM())
    turn = asyncio.run(engine.chat(ChatRequest(message="Why reuse rockets?", mode="rag")))
    assert len(calls) == 1
    assert turn.text == "Reusable rockets can lower launch costs."
    assert turn.citations == []
    assert turn.grounding.citation_status == "missing"
    assert turn.grounding.status == "unverified"


def test_compact_prompt_has_exact_markers_dated_evidence_and_profile_no_citations(corpus):
    result = Retriever(corpus).search("rocket reuse")
    messages = build_messages(corpus.persona, ChatRequest(message="rocket reuse", mode="rag"), result, [], corpus.sources)
    context = json.loads(messages[1]["content"].split("\n", 1)[1])
    record = context["evidence_records"][0]
    assert record["citation"] == "[source:rocket-reuse]"
    assert record["published_at"] == "2020-01-01"
    assert record["text"] == corpus.evidence["rocket-reuse"].text
    assert "locator" not in record
    assert "RECORD_ID" not in json.dumps(messages)
    assert "Reusable rockets" not in messages[0]["content"]
    profile = build_messages(corpus.persona, ChatRequest(message="rocket reuse", mode="profile"), result, [], corpus.sources)
    assert "do not emit source citations" in profile[0]["content"]
    assert "[source:" not in json.dumps(profile)


def test_compact_history_keeps_only_bounded_complete_exchanges(corpus):
    result = Retriever(corpus).search("rocket reuse")
    history = [{"role": "user", "content": "x" * 12_000}, {"role": "assistant", "content": "large reply"}, {"role": "user", "content": "recent question"}, {"role": "assistant", "content": "recent reply"}]
    messages = build_messages(corpus.persona, ChatRequest(message="rocket reuse"), result, history, corpus.sources)
    assert messages[2:4] == history[2:]
    assert len(messages) == 5


def test_grounded_candidate_separates_presentation_from_source_evidence(corpus):
    result = Retriever(corpus).search("rocket reuse")
    request = ChatRequest(message="Why reuse rockets?", mode="rag")
    compact = build_compact_messages(corpus.persona, request, result, [], corpus.sources)
    grounded = build_grounded_messages(corpus.persona, request, result, [], corpus.sources)
    before = json.loads(compact[1]["content"].split("\n", 1)[1])
    after = json.loads(grounded[1]["content"].split("\n", 1)[1])
    assert before["evidence_records"] == after["evidence_records"]
    assert "profile" not in after
    assert after["presentation"]["provenance"] == "editorial_design"
    assert corpus.evidence["rocket-reuse"].text not in grounded[0]["content"]
    assert grounded[-1] == compact[-1]
    profile = request.model_copy(update={"mode": "profile"})
    assert build_grounded_messages(corpus.persona, profile, result, [], corpus.sources) == build_compact_messages(corpus.persona, profile, result, [], corpus.sources)


def test_legacy_prompts_reproduce_all_prior_measured_prompt_hashes(actual_corpus):
    root = Path(__file__).resolve().parents[1]
    measured = json.loads((root / "tools/evaluation/results/qwen3-06b-cpu.json").read_text())
    if fingerprint_corpus(actual_corpus) != measured["corpus_sha256"]:
        pytest.skip("The corpus has changed since the archived measurement; restore its corpus snapshot for exact historical reproduction.")
    cases = {case.id: case for case in read_suite(root / "tools/evaluation/cases.json").cases}
    retriever = Retriever(actual_corpus)
    checked = 0
    for row in measured["runs"]:
        if not row["prompt_audit"]["sha256"]:
            continue
        case = cases[row["case_id"]]
        mode = row["mode"]
        request = ChatRequest(message=case.question, language=case.language, mode=mode)
        result = SearchResult((), 0, "") if mode == "profile" else retriever.search(case.question, limit=measured["configuration"]["retrieval_limit"], behavior=mode == "behavior")
        messages = build_legacy_messages(actual_corpus.persona, request, result, [], actual_corpus.sources)
        digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        assert digest == row["prompt_audit"]["sha256"], (case.id, mode)
        checked += 1
    assert checked == 20


def test_versioned_prompts_reproduce_all_published_comparison_hashes(actual_corpus):
    root = Path(__file__).resolve().parents[1]
    published = json.loads((root / "tools/evaluation/results/grounding-quality-review.json").read_text())
    corpus_hash = fingerprint_corpus(actual_corpus)
    if any(study["binding"]["corpus_sha256"] != corpus_hash for study in published["studies"]):
        pytest.skip("Restore the measured corpus snapshot to reproduce these historical prompts.")
    retriever = Retriever(actual_corpus)
    checked = 0
    for study in published["studies"]:
        builder = PROMPT_BUILDERS[study["configuration"]["prompt_version"]]
        for row in study["runs"]:
            if row["population"] != "model":
                continue
            mode = row["mode"]
            request = ChatRequest(message=row["question"], language=row["language"], mode=mode)
            result = SearchResult((), 0, "") if mode == "profile" else retriever.search(
                row["question"], limit=study["configuration"]["retrieval_limit"], behavior=mode == "behavior"
            )
            messages = builder(actual_corpus.persona, request, result, [], actual_corpus.sources)
            observed = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            assert observed == row["prompt_sha256"], (study["label"], row["run_key"])
            assert hashlib.sha256(row["raw_response"].encode("utf-8")).hexdigest() == row["raw_response_sha256"]
            checked += 1
    assert checked == 100
