import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import HttpUrl, ValidationError

from bias.corpus import Corpus, CorpusError, canonical_source_url, source_event_id
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


@pytest.mark.parametrize("check_splits", [True, False])
def test_duplicate_and_dangling_provenance_rejected(tmp_path, corpus, check_splits):
    write_corpus(tmp_path, corpus)
    path = tmp_path / "evidence.jsonl"
    path.write_text(path.read_text() + "\n" + next(iter(corpus.evidence.values())).model_dump_json())
    with pytest.raises(CorpusError, match="Duplicate evidence"):
        Corpus.load(tmp_path, check_splits=check_splits)
    item = corpus.evidence["rocket-reuse"].model_dump()
    item["source_id"] = "unknown"
    path.write_text(json.dumps(item))
    with pytest.raises(CorpusError, match="Unknown source"):
        Corpus.load(tmp_path, check_splits=check_splits)


@pytest.mark.parametrize("confidence", [None, "source_grounded", "unverified_transcript"])
def test_unknown_speaker_transcript_needs_explicit_review_before_indexing(tmp_path, corpus, confidence):
    # Simulate a manually copied transcript row, bypassing the importer. Even
    # an asserted source_grounded flag cannot replace an explicit review.
    unreviewed = {
        "id": "unattributed-transcript",
        "source_id": "rocket-interview",
        "text": "Otters forage in kelp forests.",
        "topics": ["otter", "kelp", "forest"],
        "locator": {"label": "Unattributed transcript segment"},
        "kind": "transcript",
        "speaker": None,
    }
    if confidence is not None:
        unreviewed["confidence"] = confidence
    write_corpus(tmp_path, corpus)
    path = tmp_path / "evidence.jsonl"
    original = path.read_text()
    path.write_text(original + "\n" + json.dumps(unreviewed))
    loaded = Corpus.load(tmp_path)
    assert loaded.evidence[unreviewed["id"]].review_status == "draft"
    assert loaded.evidence[unreviewed["id"]].model_extra["speaker"] is None
    assert unreviewed["id"] not in {record.id for record in loaded.training_evidence}
    assert not Retriever(loaded).search("Otters forage in kelp forests").hits

    if confidence == "unverified_transcript":
        # The review flag and source-grounding flag are independent gates.
        path.write_text(original + "\n" + json.dumps(dict(unreviewed, review_status="reviewed")))
        assert not Retriever(Corpus.load(tmp_path)).search("Otters forage in kelp forests").hits

    # This explicit edit represents a reviewer checking the text and speaker;
    # the loader does not perform or claim to perform that attribution review.
    promoted = dict(unreviewed, review_status="reviewed", confidence="source_grounded", speaker="Reviewed test speaker")
    path.write_text(original + "\n" + json.dumps(promoted))
    reviewed = Corpus.load(tmp_path)
    assert unreviewed["id"] in {record.id for record in reviewed.training_evidence}
    assert Retriever(reviewed).search("Otters forage in kelp forests").hits[0].evidence.id == unreviewed["id"]


@pytest.mark.parametrize("batch", [None, "batch-a", "batch-b"])
def test_curated_records_explicitly_declare_review_status(batch):
    root = Path(__file__).resolve().parents[1]
    records = [json.loads(line) for line in (root / "data/personas/elon-musk/evidence.jsonl").read_text().splitlines() if line.strip()]
    if batch:
        index = json.loads((root / "data/expansion" / batch / "index.json").read_text())
        selected = set(index["evidence_ids"])
        records = [record for record in records if record["id"] in selected]
        assert len(records) == len(selected) == index["evidence_count"]
        assert {record["source_id"] for record in records} == set(index["source_ids"])
        assert len(index["source_ids"]) == index["source_count"]
    assert records
    assert all(record.get("review_status") == "reviewed" and record.get("confidence") == "source_grounded" for record in records)


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
    expected = {record.id for record in actual_corpus.evidence.values() if actual_corpus.sources[record.source_id].split == "train" and record.review_status == "reviewed" and record.confidence == "source_grounded"}
    assert {record.id for record in actual_corpus.training_evidence} == expected
    assert all(record.behavioral_observation is None for record in actual_corpus.evidence.values())
    assert actual_corpus.sources["lex-438-2024"].split == "test"
    assert not any(record.source_id == "lex-438-2024" for record in actual_corpus.training_evidence)
    quality = actual_corpus.quality_report()
    assert not quality["sources_without_event_ids"]
    assert not quality["source_event_split_leaks"]
    assert not quality["repeated_event_families"]
    assert not quality["duplicate_urls"]
    assert not quality["duplicate_evidence"]


def test_event_family_split_leak_is_rejected_on_load_and_direct_index(tmp_path, corpus):
    corpus.sources["rocket-interview"] = corpus.sources["rocket-interview"].model_copy(update={"source_event_id": "one-recording"})
    corpus.sources["heldout"] = corpus.sources["heldout"].model_copy(update={"source_event_id": "one-recording"})
    write_corpus(tmp_path, corpus)
    with pytest.raises(CorpusError, match="Source event spans multiple splits"):
        Corpus.load(tmp_path)
    with pytest.raises(CorpusError, match="one-recording"):
        Retriever(corpus)
    # Inspection must remain possible so the invalid grouping can be diagnosed.
    assert corpus.quality_report()["source_event_split_leaks"][0]["sources"] == ["heldout", "rocket-interview"]
    assert corpus.stats()["sources"] == len(corpus.sources)
    inspected = Corpus.load(tmp_path, check_splits=False)
    assert inspected.quality_report()["source_event_split_leaks"][0]["source_event_id"] == "one-recording"
    with pytest.raises(CorpusError, match="one-recording"):
        _ = inspected.training_evidence
    with pytest.raises(CorpusError, match="one-recording"):
        Retriever(inspected)


def test_same_event_excerpts_share_a_split_and_are_reported(tmp_path, corpus):
    duplicate = corpus.sources["rocket-interview"].model_copy(update={"id": "rocket-excerpt", "url": HttpUrl("https://example.org/excerpt"), "source_event_id": "rocket-event"})
    corpus.sources["rocket-interview"] = corpus.sources["rocket-interview"].model_copy(update={"source_event_id": "rocket-event"})
    corpus.sources[duplicate.id] = duplicate
    write_corpus(tmp_path, corpus)
    loaded = Corpus.load(tmp_path)
    report = loaded.quality_report()
    assert report["repeated_event_families"][0]["source_event_id"] == "rocket-event"
    assert not report["source_event_split_leaks"]
    assert len(Retriever(loaded).records) == len(loaded.training_evidence)


def test_url_alias_cannot_cross_splits_without_an_event_tag(tmp_path, corpus):
    corpus.sources["heldout"] = corpus.sources["heldout"].model_copy(update={"url": HttpUrl("https://example.org/rocket/?utm_source=feed#chapter")})
    write_corpus(tmp_path, corpus)
    with pytest.raises(CorpusError, match="Canonical source URL spans multiple splits"):
        Corpus.load(tmp_path)
    inspected = Corpus.load(tmp_path, check_splits=False)
    assert inspected.quality_report()["url_split_leaks"][0]["sources"] == ["heldout", "rocket-interview"]
    with pytest.raises(CorpusError, match="Canonical source URL spans multiple splits"):
        Retriever(inspected)
    assert canonical_source_url("https://example.org/watch?v=a&utm_source=x") != canonical_source_url("https://example.org/watch?v=b&utm_source=x")


@pytest.mark.parametrize("event_id", ["", None, ["event"], "../event"])
def test_invalid_event_identity_is_not_silently_stringified(tmp_path, corpus, event_id):
    source = corpus.sources["rocket-interview"].model_copy(update={"source_event_id": event_id})
    with pytest.raises(CorpusError, match="Invalid source_event_id"):
        source_event_id(source)
    corpus.sources[source.id] = source
    write_corpus(tmp_path, corpus)
    with pytest.raises(CorpusError, match="Invalid source_event_id"):
        Corpus.load(tmp_path, check_splits=False)


def test_duplicate_text_is_reported_without_confusing_independent_events(corpus):
    duplicate = corpus.evidence["rocket-reuse"].model_copy(update={"id": "same-words", "source_id": "car-plan", "text": "REUSABLE rockets can lower launch costs through repeated flights!"})
    corpus.evidence[duplicate.id] = duplicate
    quality = corpus.quality_report()
    assert quality["duplicate_evidence"][0]["evidence_ids"] == ["rocket-reuse", "same-words"]
    assert not quality["source_event_split_leaks"]


def test_korean_aliases_do_not_treat_policy_or_responsibility_as_books():
    assert "book" not in tokenize("정책과 책임을 설명해줘")
    assert "book" in tokenize("책과 영화")
    assert "reliability" in tokenize("신뢰성")
    assert "trustworthiness" not in tokenize("신뢰성")


@pytest.mark.parametrize("question", ["오늘 아침에 뭘 먹었어?", "용의 마법 주문은 뭐야?"])
def test_korean_unsupported_queries_still_abstain(actual_corpus, question):
    assert not Retriever(actual_corpus).search(question).hits
