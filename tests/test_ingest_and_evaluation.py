import json
from pathlib import Path

import pytest
from pydantic import HttpUrl, ValidationError

from bias.evaluate import evaluate_retrieval, read_cases
from bias.ingest import import_transcript, parse_captions, parse_whisperx, write_evidence
from bias.models import EvaluationCase


def test_vtt_speaker_time_and_markup():
    segments = parse_captions("WEBVTT\n\n00:01.500 --> 00:03.000 align:start\n<v Speaker One>Hello &amp; welcome.</v>\n\nNOTE editor metadata\nnot speech\n")
    assert len(segments) == 1
    assert segments[0].start == 1.5
    assert segments[0].text == "Hello & welcome."
    assert segments[0].speaker == "Speaker One"


def test_srt_and_whisperx_preserve_offsets_and_speakers(tmp_path):
    source = tmp_path / "clip.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:02,500\nHello there.\n")
    records = import_transcript(source, source_id="interview", timeline="published_video", source_offset=120)
    assert records[0].locator.start_seconds == 121
    assert records[0].review_status == "draft"
    assert records[0].confidence == "unverified_transcript"
    assert records[0].behavioral_observation is None
    output = tmp_path / "draft.jsonl"
    write_evidence(records, output)
    with pytest.raises(FileExistsError):
        write_evidence(records, output)
    assert parse_whisperx({"segments": [{"start": 1, "end": 2, "text": "Test", "speaker": "SPEAKER_01"}]})[0].speaker == "SPEAKER_01"


def test_manifest_alignment_must_match_source(tmp_path):
    transcript = tmp_path / "clip.json"
    transcript.write_text(json.dumps({"segments": [{"start": 1, "end": 2, "text": "Test"}]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source_id": "interview", "source_start_seconds": 30, "source_timeline": "source_video"}))
    records = import_transcript(transcript, source_id="interview", timeline="ignored", manifest=manifest)
    assert records[0].locator.start_seconds == 31
    with pytest.raises(ValueError, match="source_id"):
        import_transcript(transcript, source_id="other", timeline="video", manifest=manifest)


def test_import_does_not_accept_path_traversal_as_source_identity(tmp_path):
    transcript = tmp_path / "clip.vtt"
    transcript.write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nHi\n")
    with pytest.raises(ValidationError):
        import_transcript(transcript, source_id="../../outside", timeline="video")


def test_nonfinite_transcript_timing_is_rejected():
    with pytest.raises(ValueError):
        parse_whisperx({"segments": [{"start": float("nan"), "end": 2, "text": "Invalid"}]})


@pytest.mark.parametrize("data", [{}, {"segments": [{"text": "Missing timestamps"}]}, {"segments": [{"start": 2, "end": 1, "text": "Reversed"}]}])
def test_malformed_transcripts_are_rejected(data):
    with pytest.raises(ValueError):
        parse_whisperx(data)


def test_evaluation_does_not_claim_unmeasured_persona_quality(actual_corpus):
    paths = sorted((Path(__file__).resolve().parents[1] / "data/evaluation").glob("*.json"))
    report = evaluate_retrieval(actual_corpus, read_cases(paths))
    assert report["summary"]["split_leaks"] == 0
    assert report["summary"]["failed"] == 0
    assert report["summary"]["ungraded"] == sum(case.category == "injection" for case in read_cases(paths))
    assert all(row["passed"] is None for row in report["cases"] if row["category"] == "injection")
    assert not report["quality"]["source_event_split_leaks"]
    assert report["summary"]["answer_cases"] == report["summary"]["answer_hits"]
    assert report["summary"]["abstention_cases"] == report["summary"]["correct_abstentions"]


def test_held_out_topic_may_be_answered_from_an_independent_train_event(corpus):
    source = corpus.sources["rocket-interview"].model_copy(update={"id": "bird-interview", "url": HttpUrl("https://example.org/birds"), "source_event_id": "separate-bird-event"})
    corpus.sources[source.id] = source
    evidence = corpus.evidence["secret-kiwi"].model_copy(update={"id": "independent-bird-answer", "source_id": source.id, "text": "In a separate interview, the kiwi was described as a bird that cannot fly."})
    corpus.evidence[evidence.id] = evidence
    case = EvaluationCase(id="independent-answer", category="heldout", question="Can kiwi birds fly?", expected_evidence_ids=[evidence.id], expected_source_ids=[source.id], reference_evidence_ids=["secret-kiwi"], reference_source_ids=["heldout"], forbidden_source_ids=["heldout"], expected_action="answer")
    report = evaluate_retrieval(corpus, [case])
    assert report["summary"]["passed"] == 1
    assert "secret-kiwi" not in report["cases"][0]["retrieved_ids"]
    assert report["summary"]["split_leaks"] == 0


def test_evaluation_rejects_heldout_positive_targets_and_unknown_sources(corpus):
    case = EvaluationCase(id="bad-target", category="retrieval", question="kiwi birds", expected_evidence_ids=["secret-kiwi"], expected_action="answer")
    with pytest.raises(ValueError, match="not reviewed train evidence"):
        evaluate_retrieval(corpus, [case])
    unknown = EvaluationCase(id="unknown-source", category="abstention", question="kiwi birds", forbidden_source_ids=["misspelled-source"], expected_action="abstain")
    with pytest.raises(ValueError, match="Unknown evaluation source"):
        evaluate_retrieval(corpus, [unknown])
