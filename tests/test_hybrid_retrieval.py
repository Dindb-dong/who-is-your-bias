"""Contract tests use declared synthetic vectors, never a quality benchmark."""

import json
from pathlib import Path

import httpx
import pytest

from bias.corpus import Corpus
from bias.hybrid_retrieval import EmbeddingError, HybridPolicy, HybridRetriever, OpenAIEmbeddingProvider
from bias.retrieval import Retriever


class TableEmbeddings:
    max_tokens = 4096

    def __init__(self, entries, *, fallback=(0.0, 0.0, 1.0)):
        self.entries, self.fallback, self.calls = entries, fallback, []

    @property
    def identity(self):
        return {"provider": "synthetic-test-table", "model": "not-a-semantic-benchmark", "revision": "1"}

    def count_tokens(self, text, *, kind):
        return len(text) + 8

    def encode(self, texts, *, kind):
        self.calls.append((kind, list(texts)))
        return [list(self.entries.get(text, self.fallback)) for text in texts]


@pytest.fixture
def embedder(corpus):
    return TableEmbeddings({corpus.evidence["rocket-reuse"].text: (1.0, 0.0, 0.0),
                            corpus.evidence["car-sequence"].text: (0.0, 1.0, 0.0),
                            "회수한 발사체를 다시 운용하는 까닭은?": (1.0, 0.0, 0.0),
                            "Why re-fly a booster?": (1.0, 0.0, 0.0)})


@pytest.mark.parametrize("query", ["회수한 발사체를 다시 운용하는 까닭은?", "Why re-fly a booster?"])
def test_semantic_candidate_can_pass_without_the_legacy_lexical_gate(corpus, embedder, query):
    assert not Retriever(corpus).search(query).hits
    result = HybridRetriever(corpus, embedder).search(query)
    assert [hit.evidence.id for hit in result.hits] == ["rocket-reuse"]
    assert result.trace["factual_ids"] == ["rocket-reuse"]
    assert result.trace["semantic_status"] == "unreviewed_candidates"
    assert result.hits[0].matched_terms == ()


def test_absent_evidence_remains_empty_instead_of_forcing_top_rank(corpus, embedder):
    result = HybridRetriever(corpus, embedder).search("A recipe for saffron risotto", behavior=True)
    assert not result.hits and result.confidence == 0
    assert {row["filter_reason"] for row in result.trace["candidates"]} == {"below_topical_relevance_threshold"}


def test_startup_warmup_builds_once_and_preserves_search_results(corpus, embedder):
    class Scorer:
        identity = {"model": "synthetic-startup-test"}

        def __init__(self):
            self.calls = []

        def score(self, query, passages):
            self.calls.append((query, list(passages)))
            return [1.0] * len(passages)

    scorer = Scorer()
    retriever = HybridRetriever(corpus, embedder, reranker=scorer)
    retriever.warm_up()
    retriever.warm_up()
    assert len(scorer.calls) == 1
    assert len([call for call in embedder.calls if call[0] == "passage"]) == 1
    warmed = retriever.search("Why re-fly a booster?")
    cold = HybridRetriever(corpus, embedder, reranker=Scorer()).search("Why re-fly a booster?")
    assert warmed == cold


def test_startup_warmup_propagates_provider_failure_before_readiness(corpus, embedder):
    class FailedScorer:
        identity = {"model": "synthetic-startup-failure"}

        def score(self, query, passages):
            raise EmbeddingError("Configured reranker is unavailable")

    retriever = HybridRetriever(corpus, embedder, reranker=FailedScorer())
    with pytest.raises(EmbeddingError, match="reranker is unavailable"):
        retriever.warm_up()
    assert not retriever._warmed


def test_only_reviewed_training_records_are_ever_sent_to_embedding_provider(corpus, embedder):
    draft = corpus.evidence["rocket-reuse"].model_copy(update={"id": "draft", "review_status": "draft", "text": "Unreviewed original utterance."})
    corpus.evidence[draft.id] = draft
    result = HybridRetriever(corpus, embedder).search("Why re-fly a booster?", eligible_ids=set(corpus.evidence))
    sent = [text for kind, texts in embedder.calls if kind == "passage" for text in texts]
    assert corpus.evidence["secret-kiwi"].text not in sent and draft.text not in sent
    assert result.trace["training_records"] == 2


def test_temporal_gate_precedes_situation_and_recency_reranking(corpus, embedder):
    result = HybridRetriever(corpus, embedder).search("Why re-fly a booster?", behavior=True,
             eligible_ids={"car-sequence"}, temporal_scores={"rocket-reuse": 1.0},
             exclusion_reasons={"rocket-reuse": "published_after_snapshot"})
    assert not result.hits
    rocket = next(row for row in result.trace["candidates"] if row["evidence_id"] == "rocket-reuse")
    assert not rocket["accepted_topically"]
    assert rocket["filter_reason"] == "published_after_snapshot"


def test_original_and_resolved_question_remain_separately_inspectable(corpus, embedder):
    result = HybridRetriever(corpus, embedder).search("Why re-fly a booster?", original_query="Why do that again?")
    assert result.trace["original_query"] == "Why do that again?"
    assert result.trace["search_query"] == "Why re-fly a booster?"
    assert not any(corpus.evidence["rocket-reuse"].text in str(row) for row in result.trace["candidates"])


def test_current_requests_and_unmatched_historical_years_do_not_bypass_policy(corpus, embedder):
    retriever = HybridRetriever(corpus, embedder)
    assert not retriever.search("Why reuse rockets today?").hits
    assert not embedder.calls  # no external embedding request for blocked live questions
    query = "Why re-fly a booster in 2038?"
    embedder.entries[query] = (1.0, 0.0, 0.0)
    result = retriever.search(query)
    assert not result.hits
    assert all(row["filter_reason"] == "requested_year_not_supported" for row in result.trace["candidates"])


def test_korean_year_suffix_and_current_markers_do_not_bypass_new_retriever(corpus, embedder):
    retriever = HybridRetriever(corpus, embedder)
    assert not retriever.search("지금 테슬라 주가는 얼마인가요?").hits
    assert not embedder.calls
    query = "2050년 로켓의 정확한 가격은?"
    embedder.entries[query] = (1.0, 0.0, 0.0)
    result = retriever.search(query)
    assert not result.hits
    assert all(row["filter_reason"] == "requested_year_not_supported" for row in result.trace["candidates"])


def test_fusion_deduplicates_parent_records_and_caps_event_dominance(corpus, embedder):
    first = corpus.evidence["rocket-reuse"]
    duplicate = first.model_copy(update={"id": "repeat"})
    second = first.model_copy(update={"id": "second", "text": "Repeat flight has an additional maintenance cost."})
    third = first.model_copy(update={"id": "third", "text": "Launch vehicles require inspection between flights."})
    for record in [duplicate, second, third]:
        corpus.evidence[record.id] = record
        embedder.entries[record.text] = (1.0, 0.0, 0.0)
    result = HybridRetriever(corpus, embedder, policy=HybridPolicy(max_per_event=2)).search("Why re-fly a booster?", limit=4)
    assert len(result.hits) == 2
    reasons = [row["filter_reason"] for row in result.trace["candidates"]]
    assert "duplicate_answer_content" in reasons and "event_diversity_cap" in reasons


def test_chunked_search_returns_the_complete_parent_and_contiguous_index_offsets(corpus):
    text = "Reusable rockets reduce discarded hardware. " * 12
    record = corpus.evidence["rocket-reuse"].model_copy(update={"text": text})
    corpus.evidence[record.id] = record
    provider = TableEmbeddings({}, fallback=(1.0, 0.0, 0.0))
    retriever = HybridRetriever(corpus, provider, policy=HybridPolicy(max_chunk_tokens=80))
    result = retriever.search("Why reuse rockets?", limit=1)
    assert result.hits[0].evidence.text == text
    units = [unit for unit in retriever._units if unit.evidence_id == record.id and unit.view == "answer"]
    assert len(units) > 1
    assert "".join(unit.text for unit in units) == text
    assert all(left.char_end == right.char_start for left, right in zip(units, units[1:]))
    assert all(provider.count_tokens(unit.text, kind="passage") <= 80 for unit in units)


def test_reranker_can_reject_a_topical_candidate_and_situation_cannot_restore_it(corpus, embedder):
    class Reject:
        identity = {"model": "synthetic-relevance-test"}

        def score(self, query, passages):
            return [-3.0] * len(passages)

    result = HybridRetriever(corpus, embedder, policy=HybridPolicy(reranker_min_score=0), reranker=Reject()).search("Why re-fly a booster?", behavior=True)
    assert not result.hits
    assert any(row["filter_reason"] == "below_reranker_relevance_threshold" for row in result.trace["candidates"])


def test_style_examples_are_separate_opt_in_and_cannot_be_factual_citations():
    corpus = Corpus.load(Path(__file__).resolve().parents[1] / "data/qna/elon-musk-v1")
    # Synthetic situation assignment supplies a second eligible example; this
    # unit test is not a new annotation of the original interview.
    for identifier in ["qa-ted-2013-efficiency", "qa-ted-2013-solar-how"]:
        record = corpus.evidence[identifier]
        qa = record.question_answer.model_copy(update={"completeness": "complete_exchange", "excerpt_note": None})
        corpus.evidence[identifier] = record.model_copy(update={"situations": ["explanation"], "question_answer": qa})
    provider = TableEmbeddings({}, fallback=(1.0, 0.0, 0.0))
    retriever = HybridRetriever(corpus, provider)
    query = "Explain the reasoning"
    default = retriever.search(query, limit=1, behavior=True)
    explicit = retriever.search(query, limit=1, behavior=True, include_style=True)
    assert not default.style_hits
    assert explicit.style_hits
    assert not set(explicit.trace["factual_ids"]).intersection(explicit.trace["style_ids"])
    assert all(row["selection"] == "style_only_not_factual_support" for row in explicit.trace["style_candidates"] if row["evidence_id"] in explicit.trace["style_ids"])


def test_style_only_projection_can_never_enter_factual_ranks_or_citations():
    corpus = Corpus.load(Path(__file__).resolve().parents[1] / "data/qna/elon-musk-v1")
    identifier = "qa-ted-2013-efficiency"
    record = corpus.evidence[identifier]
    qa = record.question_answer.model_copy(update={"completeness": "complete_exchange", "excerpt_note": None})
    corpus.evidence[identifier] = record.model_copy(update={"question_answer": qa, "projection_eligibility":
        {"factual": False, "text_style": True, "aligned_speech": False, "visible_behavior": False}})
    provider = TableEmbeddings({}, fallback=(1.0, 0.0, 0.0))
    retriever = HybridRetriever(corpus, provider)
    result = retriever.search("Explain the reasoning", limit=1, include_style=True)
    assert identifier not in {hit.evidence.id for hit in result.hits}
    assert identifier not in {row["evidence_id"] for row in result.trace["candidates"]}
    assert identifier in {hit.evidence.id for hit in result.style_hits}
    denied = retriever.search("Explain the reasoning", limit=1, include_style=True,
                              eligible_ids=set(corpus.evidence) - {identifier})
    assert not denied.style_hits
    allowed = retriever.search("Explain the reasoning", limit=1, include_style=True,
                               eligible_ids=set(corpus.evidence) - {identifier}, eligible_style_ids={identifier})
    assert [hit.evidence.id for hit in allowed.style_hits] == [identifier]


def test_year_filter_does_not_mutate_the_shared_factual_index(corpus, embedder):
    retriever = HybridRetriever(corpus, embedder)
    query = "Why re-fly a booster in 2038?"
    embedder.entries[query] = (1.0, 0.0, 0.0)
    assert not retriever.search(query).hits
    assert [hit.evidence.id for hit in retriever.search("Why re-fly a booster?").hits] == ["rocket-reuse"]


@pytest.mark.parametrize("values", [[[0, 0, 0]], [[float("nan"), 0, 0]], [[1, 0], [1, 2, 3]]])
def test_invalid_vectors_fail_visibly(corpus, values):
    class Invalid(TableEmbeddings):
        def encode(self, texts, *, kind):
            return values
    with pytest.raises(EmbeddingError):
        HybridRetriever(corpus, Invalid({})).search("Why reuse rockets?")


def test_endpoint_preserves_batch_order_and_applies_explicit_prefixes():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 0]}]})

    provider = OpenAIEmbeddingProvider("http://localhost:8080/v1", "e5", api_key="private-test-key", transport=httpx.MockTransport(handler))
    assert provider.encode(["first", "두 번째"], kind="passage") == ((1.0, 0.0), (0.0, 1.0))
    assert seen[0]["input"] == ["passage: first", "passage: 두 번째"]
    assert "private-test-key" not in json.dumps(provider.identity)


@pytest.mark.parametrize("data", [[], [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [1, 0]}], [{"index": 0, "embedding": [0, 0]}, {"index": 1, "embedding": [1, 0]}]])
def test_endpoint_rejects_incomplete_duplicate_or_zero_vectors(data):
    provider = OpenAIEmbeddingProvider("http://localhost/v1", "model", transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": data})))
    with pytest.raises(EmbeddingError):
        provider.encode(["one", "two"], kind="query")


def test_provider_error_is_not_a_silent_lexical_fallback(corpus):
    provider = OpenAIEmbeddingProvider("http://localhost/v1", "model", transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    with pytest.raises(EmbeddingError, match="endpoint failed"):
        HybridRetriever(corpus, provider).search("Why reuse rockets?")


def test_historical_retriever_is_unchanged_after_constructing_hybrid(corpus, embedder):
    before = Retriever(corpus).search("Why reuse rockets?", behavior=True)
    HybridRetriever(corpus, embedder).search("Why re-fly a booster?")
    assert Retriever(corpus).search("Why reuse rockets?", behavior=True) == before


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1, True])
def test_invalid_temporal_scores_are_rejected(corpus, embedder, value):
    with pytest.raises(ValueError):
        HybridRetriever(corpus, embedder).search("Why reuse rockets?", temporal_scores={"rocket-reuse": value})


def test_overlong_query_is_rejected_without_embedding_truncation(corpus, embedder):
    with pytest.raises(EmbeddingError, match="query exceeds"):
        HybridRetriever(corpus, embedder).search("r" * 5000)
