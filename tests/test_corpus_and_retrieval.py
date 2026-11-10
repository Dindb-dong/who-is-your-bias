import json
from datetime import date

import pytest
from pydantic import ValidationError

from bias.corpus import Corpus, CorpusError
from bias.models import Evidence, Locator
from bias.retrieval import Retriever, tokenize


def write_corpus(directory, corpus):
    (directory / "sources.json").write_text(json.dumps([record.model_dump(mode="json") for record in corpus.sources.values()]))
    (directory / "profile.json").write_text(corpus.persona.model_dump_json())
    (directory / "evidence.jsonl").write_text("\n".join(record.model_dump_json() for record in corpus.evidence.values()))


def test_source_disjoint_index_and_draft_isolation(corpus):
    draft = corpus.evidence["rocket-reuse"].model_copy(update={"id": "draft", "review_status": "draft"})
    corpus.evidence[draft.id] = draft
    result = Retriever(corpus)
    assert {record.id for record in result.records} == {"rocket-reuse", "car-sequence"}
    assert not result.search("kiwi birds").hits


def test_duplicate_and_dangling_provenance_rejected(tmp_path, corpus):
    write_corpus(tmp_path, corpus)
    path = tmp_path / "evidence.jsonl"
    path.write_text(path.read_text() + "\n" + next(iter(corpus.evidence.values())).model_dump_json())
    with pytest.raises(CorpusError, match="Duplicate evidence"):
        Corpus.load(tmp_path)
    item = corpus.evidence["rocket-reuse"].model_dump()
    item["source_id"] = "unknown"
    path.write_text(json.dumps(item))
    with pytest.raises(CorpusError, match="Unknown source"):
        Corpus.load(tmp_path)


def test_timed_locator_requires_real_timeline():
    with pytest.raises(ValidationError, match="timeline"):
        Locator(label="Clip", start_seconds=1)
    with pytest.raises(ValidationError, match="earlier"):
        Locator(label="Clip", start_seconds=2, end_seconds=1, timeline="video")
    assert Locator(label="Clip", start_seconds=0, end_seconds=2, timeline="video").end_seconds == 2


def test_reviewed_corpus_roundtrip(tmp_path, corpus):
    write_corpus(tmp_path, corpus)
    loaded = Corpus.load(tmp_path)
    assert loaded.stats() == {"sources": 3, "evidence": 2, "held_out_sources": 1, "total_evidence": 3}
    assert loaded.citation("rocket-reuse").url == "https://example.org/rocket"


@pytest.mark.parametrize("question", ["Why reuse rockets?", "재사용 로켓을 설명해줘"])
def test_english_and_korean_retrieval(corpus, question):
    assert Retriever(corpus).search(question).hits[0].evidence.id == "rocket-reuse"


@pytest.mark.parametrize("question", ["What is Tesla's stock price today?", "What did you have for breakfast this morning?", "Explain quantum unicorn soup", "Which product will you announce next week?"])
def test_unsupported_or_live_questions_abstain(actual_corpus, question):
    assert not Retriever(actual_corpus).search(question).hits


def test_recording_year_is_distinct_from_publication_year(corpus):
    source = corpus.sources["rocket-interview"].model_copy(update={"recorded_at": date(2018, 1, 1)})
    corpus.sources[source.id] = source
    assert Retriever(corpus).search("Why reuse rockets in the 2018 interview?").hits
    assert not Retriever(corpus).search("Why reuse rockets in 2024?").hits


def test_possessives_hyphens_and_plural_forms():
    assert tokenize("Tesla's first-principles") == ["tesla", "first", "principle"]
    assert "game" in tokenize("Games")


def test_bundled_corpus_has_no_person_specific_behavior_claims(actual_corpus):
    assert len(actual_corpus.training_evidence) == 24
    assert all(record.behavioral_observation is None for record in actual_corpus.evidence.values())
    assert len({source.id for source in actual_corpus.sources.values() if source.split == "test"}) == 1
