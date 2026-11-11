from __future__ import annotations

import json
import wave
from copy import deepcopy
from pathlib import Path

import pytest

from bias.models import Evidence
from tools.adaptation.datasets import export_text, export_voice, file_hash, validate_dataset
from tools.adaptation.train import encode_example
from tools.persona_data.parsers import _dwarkesh, _ford_pdf, _lex_pdf, _recode, _wired, _yc, digest, sha_text
from tools.persona_data.pipeline import bind_projection_eligibility, extract_exchanges, load_registry, project_records, runtime_source, write_json, write_jsonl


def source():
    return {"id": "interview-one", "source_event_id": "event-one", "event_family_id": "event-one", "split": "train",
            "url": "https://example.org/interview", "published_at": "2026-01-01"}


def transcript(*, uncertain=False, answer="This is a complete original answer with its qualifications preserved."):
    turns = [
        {"id": "turn-1", "cue_ids": ["turn-1"], "speaker": "Interviewer", "speaker_basis": "explicit_label", "text": "How would you explain the original approach?", "start": None},
        {"id": "turn-2", "cue_ids": ["turn-2"], "speaker": "Elon Musk", "speaker_basis": "explicit_label", "text": answer, "start": None},
    ]
    if uncertain:
        turns.append({"id": "turn-3", "cue_ids": ["turn-3"], "speaker": None, "speaker_basis": "unknown", "text": "Unattributed continuation", "start": None})
    return {"turns": turns, "source_timeline": None, "input_sha256": "a"*64, "transcript_sha256": "b"*64}


def record(**kwargs):
    rows, failures = extract_exchanges(source(), transcript(**kwargs), acquired_at="2026-11-11T01:00:00Z")
    assert not failures
    return rows[0]


def test_speaker_unknown_never_becomes_complete_style_example():
    item = record(uncertain=True)
    assert item["question_answer"]["completeness"] == "excerpt"
    assert item["question_answer"]["excerpt_note"]
    projections, decisions = project_records([item], {source()["id"]: source()})
    assert len(projections["factual"]) == 1
    assert projections["text_style"] == []
    assert not decisions[0]["eligibility"]["aligned_speech"]["eligible"]


def test_source_text_cannot_qualify_audio_or_visual():
    projections, decisions = project_records([record()], {source()["id"]: source()})
    assert len(projections["text_style"]) == 1
    assert projections["aligned_speech"] == []
    assert projections["visible_behavior"] == []
    assert decisions[0]["eligibility"]["aligned_speech"]["reason"] == "no_reviewed_speech_alignment"


def test_runtime_projection_contract_preserves_style_without_factual_eligibility():
    item = record(answer="Yes, exactly.")
    _, decisions = project_records([item], {source()["id"]: source()})
    projected = bind_projection_eligibility([item], decisions)[0]
    serialized = Evidence.model_validate(projected).model_dump(mode="json")
    assert serialized["projection_eligibility"] == {
        "factual": False, "text_style": True, "aligned_speech": False, "visible_behavior": False}
    assert "projection_eligibility" not in item
    with pytest.raises(ValueError, match="cover each runtime record"):
        bind_projection_eligibility([item], [])
    decisions[0]["eligibility"]["factual"]["eligible"] = "false"
    with pytest.raises(ValueError, match="explicit eligibility boolean"):
        bind_projection_eligibility([item], decisions)


def test_original_materialization_does_not_reuse_summary_metadata_or_invent_event_date():
    original = dict(source(), id="dwarkesh-2026", curation="short_original_paraphrases", raw_material_included=False)
    projected = runtime_source(original, transcript(), acquired_at="2026-11-11T00:00:00Z")
    assert projected["curation"] == "original_publisher_qa_structural_agent_review"
    assert projected["raw_material_included"] is True
    assert projected["raw_recordings_included"] is False
    assert projected["recorded_at"] is None
    assert original["curation"] == "short_original_paraphrases"


def test_all_projections_exclude_development_and_test():
    for split in ("validation", "test"):
        value = dict(source(), split=split)
        projections, _ = project_records([record()], {value["id"]: value})
        assert all(not rows for rows in projections.values())


def test_complete_turns_are_rejected_whole_when_too_long():
    rows, failures = extract_exchanges(source(), transcript(answer="x"*12001), acquired_at="2026-11-11T00:00:00Z")
    assert rows == []
    assert failures[0]["reason"] == "whole_exchange_exceeds_runtime_contract"


def test_wired_full_name_anchors_required_and_pullquotes_stay_unknown():
    with pytest.raises(ValueError, match="anchor"):
        _wired(b"<p>Anderson: Question?</p><p>Musk: Answer.</p>")
    rows = _wired(b"<p>Chris Anderson: Question?</p><p>Elon Musk: Original answer.</p><p>A repeated pull quote.</p><p>Anderson: Next?</p><p>Musk: Yes.</p>")
    assert rows[2]["speaker"] is None
    assert rows[1]["text"] == "Original answer."


def test_dwarkesh_heading_attribution_resets_at_chapter_boundary():
    rows = _dwarkesh(b"<h2>Transcript</h2><p><strong>Dwarkesh Patel</strong></p><p>Question?</p><p><strong>Elon Musk</strong></p><p>First answer.</p><p>Continuation.</p><h2>New chapter</h2><p>Unassigned text.</p><p><strong>John Collison</strong></p><p>Next question?</p>")
    assert rows[1]["text"] == "First answer. Continuation."
    assert "Unassigned text." not in " ".join(row["text"] for row in rows if row["speaker"])


def test_recode_boundaries_and_typography_are_required():
    with pytest.raises(ValueError, match="boundary"):
        _recode(b"<p><strong>Question?</strong></p><p>Answer.</p>")
    rows = _recode(b"<p><strong>Kara Swisher: Question?</strong></p><p><strong>Elon Musk:</strong> Answer.</p><p><strong>Next question?</strong></p><p>Next answer.</p><p>This article originally appeared on Recode.net.</p><p>Footer must not enter the answer.</p>")
    assert [row["speaker"] for row in rows] == ["Kara Swisher", "Elon Musk", "Kara Swisher", "Elon Musk"]
    assert rows[-1]["text"] == "Next answer."


def test_yc_preserves_actual_publisher_clock_only():
    rows = _yc("<p>Sam Altman [01:03] – A question?</p><p>Elon Musk [01:10] – An answer.</p>".encode())
    assert rows[0]["start"] == 63
    assert rows[1]["start"] == 70
    assert rows[1]["end"] is None


def test_pdf_headers_do_not_enter_dialogue_and_smpte_is_not_audio_time():
    lex = _lex_pdf("This transcript was exported on a date.\nLex Fridman: 01:02 Question?\nElon Musk: 01:10 Original\nanswer.\nTranscript by Rev.com\nEpisode (Completed 11/12/19) Page 1 of 14\n")
    assert lex[1]["text"] == "Original answer."
    ford = _ford_pdf("MEDIA ID: MUSK1.MP3\nBARRY HURD:\n03:01:00;22 Question?\nPG.2\nELON MUSK:\n03:01:10;23 Original answer.\n")
    assert ford[1]["start"] is None
    assert ford[1]["publisher_locator"] == "03:01:10;23"
    assert ford[1]["text"] == "Original answer."


def registry_files(tmp_path, *, exposed=False, crossed=False):
    item = source()
    item["prior_exposure"] = {"original_qa_split": "test" if exposed else None}
    registry = {"schema_version": "persona-source-families.v1", "sources": [item]}
    events = [{"source_event_id": item["source_event_id"], "event_family_id": item["event_family_id"], "source_ids": [item["id"]], "split": "train"}]
    if crossed:
        events.append({"source_event_id": "event-two", "event_family_id": item["event_family_id"], "source_ids": ["duplicate-upload"], "split": "test"})
    write_json(tmp_path / "source-families.json", registry)
    write_json(tmp_path / "split-registration.json", {"schema_version": "persona-split-registration.v1", "events": events})


def test_exposed_pilot_cannot_be_relabeled_as_fresh_train(tmp_path):
    registry_files(tmp_path, exposed=True)
    with pytest.raises(ValueError, match="development-only"):
        load_registry(tmp_path)


def test_reposted_recording_family_cannot_cross_splits(tmp_path):
    registry_files(tmp_path, crossed=True)
    with pytest.raises(ValueError, match="family crosses"):
        load_registry(tmp_path)


def accepted_media(tmp_path, item):
    wav = tmp_path / "source.wav"
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(8000); out.writeframes(b"\x00\x00"*16000)
    return {"record_id": item["id"], "answer_sha256": sha_text(item["question_answer"]["answer"]), "source_id": item["source_id"],
            "source_timeline": "verified_clip_clock", "transcript_sha256": "b"*64, "source_artifact_sha256": "a"*64,
            "recording_id": "recording-one", "source_clock_mapping_reviewed": True, "speaker": "Elon Musk", "review_status": "accepted",
            "reviewer": "synthetic fixture reviewer", "reviewer_type": "human", "reviewed_at": "2026-11-11", "start_seconds": 0.0,
            "end_seconds": 1.5, "alignment_basis": "acoustically_reviewed", "file": wav.name, "sha256": file_hash(wav),
            "overlap": False, "cut": False, "transcript_exact": True, "noise_accepted": True}


def test_media_requires_same_text_hash_mapping_and_real_interval(tmp_path):
    item = record(); media = accepted_media(tmp_path, item)
    for field, value in (("answer_sha256", "c"*64), ("source_clock_mapping_reviewed", False), ("end_seconds", 10), ("overlap", True)):
        changed = dict(media, **{field: value})
        projections, _ = project_records([item], {source()["id"]: source()}, media_reviews={item["id"]: {"speech": changed}}, asset_root=tmp_path)
        assert projections["aligned_speech"] == []
    projections, _ = project_records([item], {source()["id"]: source()}, media_reviews={item["id"]: {"speech": media}}, asset_root=tmp_path)
    assert len(projections["aligned_speech"]) == 1
    assert projections["visible_behavior"] == []


def test_text_export_is_event_capped_and_tampering_fails(tmp_path):
    projections, _ = project_records([record()], {source()["id"]: source()})
    row = projections["text_style"][0]
    rows = [dict(row, segment_id=f"example-{i}") for i in range(5)]
    projection = tmp_path / "projection.jsonl"; write_jsonl(projection, rows)
    manifest = export_text(projection, tmp_path / "training", max_per_event=2)
    assert manifest["samples"] == 2
    assert manifest["test_or_validation_samples"] == 0
    _, examples = validate_dataset(tmp_path / "training", "text")
    assert examples[0]["messages"][-1]["content"] == row["text"]
    with (tmp_path / "training" / "train.jsonl").open("a") as stream: stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_dataset(tmp_path / "training", "text")


def test_empty_voice_data_is_reported_unready_and_cannot_train(tmp_path):
    projection = tmp_path / "empty.jsonl"; projection.write_text("")
    manifest = export_voice(projection, tmp_path / "voice", asset_root=tmp_path)
    assert manifest["ready"] is False
    with pytest.raises(ValueError, match="nonempty"):
        validate_dataset(tmp_path / "voice", "voice")


def test_voice_export_uses_identical_real_reference_and_hashes(tmp_path):
    item = record(); media = accepted_media(tmp_path, item)
    projections, _ = project_records([item], {source()["id"]: source()}, media_reviews={item["id"]: {"speech": media}}, asset_root=tmp_path)
    projection = tmp_path / "speech.jsonl"; write_jsonl(projection, projections["aligned_speech"])
    report = export_voice(projection, tmp_path / "voice", asset_root=tmp_path, reference_id=item["id"])
    assert report["ready"]
    _, rows = validate_dataset(tmp_path / "voice", "voice")
    assert rows[0]["audio"] == rows[0]["ref_audio"]
    with wave.open(rows[0]["audio"], "rb") as wav: assert wav.getnframes() == 12000


class ToyTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]


def test_assistant_only_loss_and_no_silent_truncation():
    encoded = encode_example(ToyTokenizer(), [{"role": "user"}, {"role": "assistant"}], 10)
    assert encoded["labels"] == [-100, -100, -100, 4, 5]
    assert encode_example(ToyTokenizer(), [], 4) is None


def test_mismatched_chat_template_prefix_is_rejected():
    class Wrong(ToyTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [9, 9] if add_generation_prompt else [1, 2, 3]
    with pytest.raises(ValueError, match="loss masking"):
        encode_example(Wrong(), [], 10)
