from dataclasses import replace

import pytest

from bias.config import Settings
from bias.corpus import Corpus
from bias.engine import Engine
from bias.jensen_retrieval import JensenConceptRetriever, concept_tokens
from bias.retrieval import Retriever


@pytest.fixture
def jensen_fixture(corpus):
    persona = corpus.persona.model_copy(update={"id": "jensen-huang"})
    records = {
        "contract": corpus.evidence["rocket-reuse"].model_copy(update={
            "id": "contract", "text": "The Sega contract required abandoning a wrong graphics architecture.",
            "topics": ["company founding"]}),
        "fab": corpus.evidence["car-sequence"].model_copy(update={
            "id": "fab", "text": "TSMC coordinates manufacturing across the semiconductor supply chain.",
            "topics": ["manufacturing"]}),
        "heldout": corpus.evidence["secret-kiwi"].model_copy(update={
            "id": "heldout", "text": "Sega TSMC CUDA secret kiwi findings.", "topics": ["cuda"]}),
    }
    return Corpus(persona, corpus.sources, records)


def test_korean_concepts_retrieve_source_text_without_promoting_heldout(jensen_fixture):
    retriever = JensenConceptRetriever(jensen_fixture)
    result = retriever.search("세가 게임기 계약에서 그래픽 아키텍처를 포기한 이유는?")
    assert result.hits[0].evidence.id == "contract"
    assert all(hit.evidence.id != "heldout" for hit in result.hits)
    assert "sega" in result.hits[0].matched_terms


@pytest.mark.parametrize("query", [
    "젠슨 황에 대해서 말해 줘.",
    "Did Jensen approve our CUDA chatbot?",
    "Give me his private email address about TSMC.",
    "What exactly did Jensen say about TSMC in 2035?",
    "What is the latest TSMC manufacturing strategy?",
    "TSMC quasar nebula viscosity topological marmalade xylophone",
])
def test_names_unavailable_relationships_and_unknown_content_cannot_support_answers(jensen_fixture, query):
    assert not JensenConceptRetriever(jensen_fixture).search(query).hits


def test_tokenizer_keeps_untranslated_content_and_does_not_confuse_short_words():
    terms = concept_tokens("정책과 책임, 미지의도메인, 세가가 CUDA를 제1원리로 판단")
    assert "정책" in terms and "미지의도메인" in terms
    assert {"responsibility", "sega", "cuda", "first", "principle"} <= set(terms)
    assert "book" not in terms


def test_filtered_runtime_keeps_candidate_and_subject_binding(jensen_fixture, corpus):
    engine = Engine(Settings(retrieval_backend="jensen-lexical"), jensen_fixture)
    filtered = replace(jensen_fixture, evidence={"fab": jensen_fixture.evidence["fab"]})
    scoped = engine.retriever.with_corpus(filtered)
    assert isinstance(scoped, JensenConceptRetriever)
    assert scoped.search("TSMC 공급망 제조").hits[0].evidence.id == "fab"
    assert not scoped.search("sega").hits
    with pytest.raises(ValueError, match="Jensen corpus"):
        JensenConceptRetriever(corpus)


def test_default_remains_archived_lexical(jensen_fixture):
    assert Settings().retrieval_backend == "lexical"
    assert type(Engine(Settings(), jensen_fixture).retriever) is Retriever
