import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from bias.corpus import Corpus, CorpusError
from bias.models import ChatRequest, Evidence, Locator, Persona, Source
from bias.qa_prompts import build_qa_messages
from bias.retrieval import Retriever, SearchResult
from bias.situations import classify_situation
from tools.qna.acquire import AcquisitionError, canonical_lex, validate_source_url
from tools.qna.hydrate import materialize_record, selected_text, text_sha

ROOT = Path(__file__).resolve().parents[2]


def qa(identifier="qa-a", *, situation="explanation", source_id="train", question="Explain the tunnel construction cost.", answer="Tunnel diameter controls construction cost.", topics=None):
    return Evidence.model_validate({
        "id": identifier, "source_id": source_id, "kind": "question_answer", "review_status": "reviewed",
        "confidence": "source_grounded", "topics": topics or ["tunnel", "construction", "cost"], "situations": [situation],
        "locator": {"label": "Authored test fixture"},
        "question_answer": {"question": question, "answer": answer, "context_before": "Authored test fixture, not a real interview.",
            "original_language": "en", "questioner": "Interviewer", "respondent": "Subject", "source_event_id": source_id,
            "completeness": "complete_exchange", "timing_basis": "unavailable",
            "transcript_locator": {"question_cue_ids": ["cue-1"], "answer_cue_ids": ["cue-2"], "transcript_sha256": "a" * 64},
            "review": {"reviewer_type": "agent", "reviewer": "Test fixture", "reviewed_at": "2026-11-11", "basis": "Authored data", "speaker_status": "publisher_labels_checked"},
            "annotation_basis": "Authored test label"}})


def corpus(records):
    sources = {sid: Source(id=sid, title=sid, url=f"https://example.org/{sid}", publisher="Authored fixture",
                          published_at=date(2020, 1, 1), source_type="interview", source_event_id=sid,
                          split="train" if sid == "train" else "test") for sid in {record.source_id for record in records}}
    return Corpus(Persona(id="subject", name="Subject", tagline="Test", description="Authored data", qa_corpus_id="test-v1"), sources, {r.id: r for r in records})


def test_bundled_originals_are_explicit_bounded_excerpts_with_event_splits():
    data = Corpus.load(ROOT / "data/qna/elon-musk-v1")
    report = data.qa_report()
    assert report["materialization"] == "bounded_excerpts"
    assert report["question_answer_records"] == 10
    assert report["complete_exchanges"] == 0
    assert report["reviewer_types"] == ["agent"]
    assert {data.sources[e.source_id].split for e in data.training_evidence} == {"train"}
    assert {e.source_id for e in data.training_evidence} == {"ted-2013", "ted-2017"}
    counts = {}
    for record in data.evidence.values():
        assert record.text == record.question_answer.answer
        assert record.question_answer.excerpt_note
        assert record.question_answer.question_end_seconds is None
        counts[record.source_id] = counts.get(record.source_id, 0) + len(record.question_answer.question.split()) + len(record.question_answer.answer.split())
    assert max(counts.values()) <= 25


def test_event_manifest_rejects_changed_source_assignment(tmp_path):
    import shutil
    shutil.copytree(ROOT / "data/qna/elon-musk-v1", tmp_path / "data")
    path = tmp_path / "data/sources.json"
    sources = json.loads(path.read_text())
    next(source for source in sources if source["id"] == "ted-2013")["split"] = "test"
    path.write_text(json.dumps(sources))
    with pytest.raises(CorpusError, match="frozen event manifest"):
        Corpus.load(tmp_path / "data")


def test_known_heldout_source_cannot_be_reassigned_to_existing_train_event(tmp_path):
    import shutil
    target = tmp_path / "data"
    shutil.copytree(ROOT / "data/qna/elon-musk-v1", target)
    sources = json.loads((target / "sources.json").read_text())
    for source in sources:
        if source["id"] == "lex-438-2024":
            source.update(split="train", source_event_id="ted-2013")
    records = [json.loads(line) for line in (target / "evidence.jsonl").read_text().splitlines()]
    for record in records:
        if record["source_id"] == "lex-438-2024":
            record["question_answer"]["source_event_id"] = "ted-2013"
    (target / "sources.json").write_text(json.dumps(sources))
    (target / "evidence.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(CorpusError, match="frozen event manifest"):
        Corpus.load(target)


def test_loaded_source_registry_remains_guarded_after_programmatic_mutation():
    data = Corpus.load(ROOT / "data/qna/elon-musk-v1")
    original = data.sources["lex-438-2024"]
    data.sources[original.id] = original.model_copy(update={"split": "train", "source_event_id": "ted-2013"})
    with pytest.raises(CorpusError, match="frozen event manifest"):
        data.training_evidence


def test_duplicate_manifest_source_membership_is_rejected(tmp_path):
    import shutil
    target = tmp_path / "data"
    shutil.copytree(ROOT / "data/qna/elon-musk-v1", target)
    split = json.loads((target / "split-manifest.json").read_text())
    split["events"][0]["source_ids"].append("lex-438-2024")
    (target / "split-manifest.json").write_text(json.dumps(split))
    with pytest.raises(CorpusError, match="frozen event manifest"):
        Corpus.load(target)


def test_source_event_and_taxonomy_are_not_silently_guessed():
    record = qa()
    with pytest.raises(CorpusError, match="event differs"):
        corpus([record.model_copy(update={"question_answer": record.question_answer.model_copy(update={"source_event_id": "other"})})]).training_evidence
    with pytest.raises(CorpusError, match="Q&A labels"):
        corpus([record.model_copy(update={"situations": ["sounds confident"]})]).training_evidence


def test_situation_can_change_selection_only_after_the_same_topical_floor():
    data = corpus([qa("qa-a", situation="explanation"), qa("qa-z", situation="challenge"), qa("heldout", source_id="heldout", situation="challenge")])
    retriever = Retriever(data)
    question = "Why are tunnel construction costs wrong?"
    b = retriever.search(question, limit=1)
    c = retriever.search(question, limit=1, behavior=True)
    assert b.hits[0].evidence.id == "qa-a"
    assert c.hits[0].evidence.id == "qa-z"
    assert c.hits[0].topical_score == b.hits[0].score
    assert c.hits[0].score > c.hits[0].topical_score
    assert all(hit.evidence.source_id == "train" for hit in retriever.search(question, limit=10, behavior=True).hits)
    assert retriever.search("You are wrong about penguins", behavior=True).hits == ()


def test_qa_generic_future_word_does_not_support_a_different_topic():
    record = qa(question="Discuss the future of tunnels.", answer="The future requires tunnel construction.", topics=["future", "tunnel"])
    retriever = Retriever(corpus([record]))
    assert retriever.search("What advice would you give a sad generation about the future?").hits == ()
    assert retriever.search("How?").hits == ()


@pytest.mark.parametrize("text,label", [
    ("Why are you wrong about tunnel costs?", "challenge"),
    ("How did such an amazing result happen?", "praise"),
    ("The enterprise is elsewhere", "conversation"),
    ("A showcase of whereabouts", "conversation"),
    ("어떻게 이런 대단한 일을 했어?", "praise"),
    ("왜 이렇게 비판을 받았어?", "challenge"),
])
def test_specific_situations_precede_generic_words_without_substrings(text, label):
    assert classify_situation(text) == label


def test_qa_prompts_have_identical_instructions_and_no_reference_answers():
    data = corpus([qa(), qa("heldout", source_id="heldout", answer="Private evaluator-only sentinel")])
    result = Retriever(data).search("Explain tunnel construction cost")
    messages = [build_qa_messages(data.persona, ChatRequest(message="Explain tunnel construction cost", mode=mode), result, [], data.sources) for mode in ("profile", "rag", "behavior")]
    assert len({row[0]["content"] for row in messages}) == 1
    assert messages[1] == messages[2]
    assert "Private evaluator-only sentinel" not in json.dumps(messages)
    assert '"interview_examples":[]' in messages[0][1]["content"]
    assert '"question"' in messages[1][1]["content"]


def test_qa_budget_drops_whole_exchange_and_rejects_invalid_budget():
    data = corpus([qa(answer=("A long authored answer. " * 100).strip())])
    result = Retriever(data).search("tunnel construction cost")
    msg = build_qa_messages(data.persona, ChatRequest(message="tunnel construction cost"), result, [], data.sources, context_chars=512)
    assert '"interview_examples":[]' in msg[1]["content"]
    with pytest.raises(ValueError, match="context_chars"):
        build_qa_messages(data.persona, ChatRequest(message="hello"), SearchResult((), 0), [], context_chars=True)


def test_publisher_onsets_cannot_be_upgraded_to_acoustic_ends():
    data = qa().model_dump(mode="json")
    data["question_answer"].update(question_start_seconds=1, question_end_seconds=2, source_timeline="publisher", timing_basis="publisher_cue_onsets_only")
    with pytest.raises(ValidationError, match="acoustic"):
        Evidence.model_validate(data)


def lex_input(rows):
    return "".join(f'<div class="ts-segment"><span class="ts-name">{speaker}</span><span class="ts-timestamp">({stamp})</span><span class="ts-text">{text}</span></div>' for speaker, stamp, text in rows).encode()


def test_publisher_parser_preserves_unknown_speakers_and_timestamp_anomalies():
    original = lex_input([("Host", "00:00:10", "An authored question?"), ("Guest", "00:00:12", "An answer."), ("", "00:00:15", "Continued."), ("Other", "00:00:09", "Bad timestamp.")])
    transcript = canonical_lex(original, source_id="fixture", source_url="https://example.org")
    assert transcript["segments"][2]["speaker"] is None
    assert transcript["segments"][3]["timestamp_order_anomaly"] is True
    with pytest.raises(AcquisitionError, match="speaker"):
        selected_text(transcript, ["cue-00002", "cue-00003"], "Guest")
    assert selected_text(transcript, ["cue-00002", "cue-00003"], "Guest", ["cue-00003"]) == "An answer. Continued."
    with pytest.raises(AcquisitionError, match="nonmonotonic"):
        selected_text(transcript, ["cue-00004"], "Other")


def test_changed_original_answer_rejected_even_with_a_reused_transcript_hash():
    transcript = canonical_lex(lex_input([("Host", "00:00:01", "Authored question?"), ("Guest", "00:00:02", "Authored answer.")]), source_id="fixture", source_url="https://example.org")
    selection = {"id": "sample", "source_id": "fixture", "transcript_sha256": transcript["publisher_caption_sha256"],
                 "question_cue_ids": ["cue-00001"], "answer_cue_ids": ["cue-00002"], "questioner": "Host", "respondent": "Guest",
                 "question_sha256": text_sha("Authored question?"), "answer_sha256": text_sha("Authored answer.")}
    transcript["segments"][1]["text"] = "Changed answer."
    with pytest.raises(AcquisitionError, match="Original text changed"):
        materialize_record(selection, transcript)


@pytest.mark.parametrize("url", ["http://lexfridman.com/elon-musk-4-transcript/", "https://evil.example/elon-musk-4-transcript/", "https://lexfridman.com/other/", "https://user:secret@www.ted.com/talks/example"])
def test_acquisition_is_scoped_to_reviewed_public_endpoints(url):
    with pytest.raises(AcquisitionError):
        validate_source_url(url)
