from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from tools.media.common import MediaError, sha256
from tools.observations.features import aligned_exchange, merge_speech_runs, speech_units, visual_features
from tools.observations.review import CHECKS, import_review, make_review_bundle
from tools.observations.run import validate_splits, within
from tools.references.workflow import align_transcripts


def speech_fixture():
    # Caption and audio clocks intentionally differ by many seconds. Matching
    # text, not subtraction of caption time, establishes candidate boundaries.
    publisher = {"schema_version": "publisher_transcript.v1", "source_id": "interview", "source_timeline": "caption_clock", "segments": [
        {"id": "q", "start": 100.0, "end": 105.0, "speaker": "Interviewer", "speaker_basis": "explicit_publisher_label", "speaker_spans": [{"kind": "caption_text", "speaker": "Interviewer", "text": "Why electric cars"}]},
        {"id": "a", "start": 105.0, "end": 120.0, "speaker": "Respondent", "speaker_basis": "explicit_publisher_label", "speaker_spans": [{"kind": "caption_text", "speaker": "Respondent", "text": "Because they use energy much more efficiently"}]},
    ]}
    q_words, a_words = "Why electric cars".split(), "Because they use energy much more efficiently".split()
    words = [{"word": word, "start": index * 0.5, "end": (index + 1) * 0.5, "confidence": 0.99} for index, word in enumerate(q_words)]
    words += [{"word": word, "start": 2.5 + index * 0.5, "end": 3 + index * 0.5, "confidence": 0.99} for index, word in enumerate(a_words)]
    asr = {"schema_version": "transcript.v1", "source_id": "interview", "source_timeline": "video_clock", "source_start_seconds": 10, "segments": [{"words": words}]}
    return publisher, asr


def measure_speech(publisher, asr):
    return aligned_exchange(publisher, asr, align_transcripts(publisher, asr), question_cues=["q"], answer_cues=["a"], questioner="Interviewer", respondent="Respondent", language="en")


def test_source_clock_not_subtitle_clock_for_turn_boundaries():
    publisher, asr = speech_fixture()
    result = measure_speech(publisher, asr)
    assert result["question_end_seconds_candidate"] == 11.5
    assert result["answer_start_seconds_candidate"] == 12.5
    assert result["pre_answer_pause"]["candidate_gap_seconds"] == 1
    assert result["pre_answer_pause"]["seconds"] is None
    assert result["speech_rate"]["value"] == 2
    assert result["speech_rate"]["unit"] == "orthographic_words_per_second"


def test_truncated_answer_does_not_invent_onset():
    publisher, asr = speech_fixture()
    del asr["segments"][0]["words"][3]
    result = measure_speech(publisher, asr)
    assert result["answer_start_seconds_candidate"] is None
    assert result["answer_boundary_coverage"] == {"first_token": False, "last_token": True}
    assert result["pre_answer_pause"]["candidate_gap_seconds"] is None
    assert result["speech_rate"]["scope"] == "matched_source_window_including_internal_pauses"


def test_unmatched_words_split_speech_rate_windows():
    publisher, asr = speech_fixture()
    asr["segments"][0]["words"][6]["word"] = "hallucinated"
    result = measure_speech(publisher, asr)
    assert result["speech_rate"]["value"] is None
    assert result["speech_rate"]["status"] == "intervening_unmatched_or_other_speaker_words"
    assert len(result["speech_rate_windows"]) == 2
    assert all(window["unit_count"] == 3 for window in result["speech_rate_windows"])


def test_speaker_label_mismatch_rejects_attribution():
    publisher, asr = speech_fixture()
    publisher["segments"][1]["speaker"] = "Another speaker"
    with pytest.raises(MediaError, match="speaker"):
        measure_speech(publisher, asr)


def test_no_caption_laughter_becomes_audible_laughter():
    publisher, asr = speech_fixture()
    publisher["segments"].append({"id": "stage", "start": 120, "end": 121, "speaker": None, "speaker_basis": None, "speaker_spans": [{"kind": "stage_direction", "text": "Laughter", "speaker": None}]})
    result = measure_speech(publisher, asr)
    assert result["matched_answer_words"] == 7
    assert "audible_laughter" not in result


def test_unknown_clock_cannot_be_merged():
    publisher, asr = speech_fixture()
    result = measure_speech(publisher, asr)
    other = dict(result, source_timeline="another_clock")
    with pytest.raises(MediaError, match="clocks"):
        merge_speech_runs([result, other])


def test_language_units_are_explicit():
    assert speech_units("A useful well-made thing", "en") == (5, "orthographic_words")
    assert speech_units("안녕하세요. 반갑습니다.", "ko") == (10, "hangul_syllable_blocks")
    with pytest.raises(MediaError, match="mixed-script"):
        speech_units("저는 3D 기술을 씁니다", "ko")
    with pytest.raises(MediaError, match="unit"):
        speech_units("bonjour", "fr")


def visual_fixture():
    rows, frames = [], []
    for index in range(8):
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0, "visibility": 0.9, "presence": 0.9} for _ in range(33)]
        landmarks[16]["x"] = 1.2  # right wrist outside the image
        angle = index * 0.05
        frames.append({"time_ms": index * 125, "bones": {"head": [0, math.sin(angle / 2), 0, math.cos(angle / 2)]}})
        rows.append({"source_timestamp_seconds": 20 + index / 8, "faces": [{"blendshapes": {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.7}}], "poses": [{"landmarks": landmarks}]})
    geometry = {"source": {"source_start_seconds": 20, "source_timeline": "video"}, "frames": frames}
    return rows, geometry


def measure_visual(rows, geometry, exclusions=None):
    return visual_features(rows, geometry, start=20, end=21, speech_runs=[{"answer_window_seconds": [19, 22], "question_window_seconds": [17, 18]}], exclusions=exclusions or [])


def test_smile_is_not_laughter_or_emotion():
    result = measure_visual(*visual_fixture())
    assert result["visible_smile"]["intervals"] == [{"start_seconds": 20, "end_seconds": 21, "phase": "speaking_candidate", "status": "geometry_candidate"}]
    assert result["audible_laughter"]["intervals"] is None
    assert result["head_movement"]["semantic_gesture"] is None
    assert result["speaker_association_reviewed"] is False
    assert "emotion" not in result


def test_offscreen_hand_is_unobservable_not_stationary():
    result = measure_visual(*visual_fixture())
    assert result["hands"]["right"]["status"] == "unobservable"
    assert result["hands"]["right"]["wrist_path_in_normalized_image_units"] is None
    assert result["hands"]["right"]["frame_states"] == {"off_screen": 8}
    assert result["hands"]["left"]["status"] == "relative_geometry_candidate"


def test_camera_exclusion_breaks_smile_and_rotation_path():
    rows, geometry = visual_fixture()
    result = measure_visual(rows, geometry, [(20.5, 21)])
    assert result["excluded_frames"] == 4
    assert result["visible_smile"]["intervals"][0]["end_seconds"] == 20.5
    assert result["head_movement"]["rotation_path_radians"] == pytest.approx(0.15)
    assert result["hands"]["left"]["frame_states"]["excluded_camera_interval"] == 4


def test_missing_face_does_not_prove_no_smile():
    rows, geometry = visual_fixture()
    for row in rows:
        row["faces"] = []
    result = measure_visual(rows, geometry)
    assert result["visible_smile"]["status"] == "unobservable"
    assert result["visible_smile"]["mean_bilateral_score"] is None


def test_smile_tail_cannot_extend_beyond_clip_to_meet_minimum():
    rows, geometry = visual_fixture()
    for row in rows[:-2]:
        row["faces"] = []
    result = visual_features(rows, geometry, start=20, end=20.99, speech_runs=[], exclusions=[])
    assert result["visible_smile"]["observed_frames"] == 2
    assert result["visible_smile"]["intervals"] == []


def test_low_visibility_hand_is_explicit():
    rows, geometry = visual_fixture()
    for row in rows:
        row["poses"][0]["landmarks"][15]["visibility"] = 0.2
    result = measure_visual(rows, geometry)
    assert result["hands"]["left"]["frame_states"] == {"occluded_or_low_confidence": 8}


def test_duplicate_or_out_of_order_frames_rejected():
    rows, geometry = visual_fixture()
    rows[1]["source_timestamp_seconds"] = rows[0]["source_timestamp_seconds"]
    with pytest.raises(MediaError, match="ordered"):
        measure_visual(rows, geometry)


def test_source_event_mirrors_cannot_cross_splits():
    inventory = {"events": [{"source_event_id": "e1", "split": "train", "source_ids": ["a", "b"]}]}
    sources = [{"source_id": "a", "source_event_id": "e1", "observation_split": "train"}, {"source_id": "b", "source_event_id": "e1", "observation_split": "validation"}]
    with pytest.raises(MediaError, match="Mirror"):
        validate_splits(sources, inventory)


def test_text_holdout_cannot_fit_motion():
    inventory = {"events": [{"source_event_id": "e1", "split": "test", "source_ids": ["a"]}]}
    with pytest.raises(MediaError, match="held-out"):
        validate_splits([{"source_id": "a", "source_event_id": "e1", "observation_split": "train"}], inventory)


def test_artifact_path_does_not_escape(tmp_path):
    with pytest.raises(MediaError, match="escapes"):
        within(tmp_path, "../elsewhere.json")


def review_fixture(tmp_path):
    publisher, asr = speech_fixture()
    speech = measure_speech(publisher, asr)
    row = {"exchange_id": "q1", "exchange_sha256": "a" * 64, "source_id": "interview", "source_event_id": "event", "source_url": "https://example.org/interview", "text_split": "train", "observation_split": "train", "situations": ["explanation"], "speech": {"runs": [speech]}, "visual_windows": [{"window_start_seconds": 10, "window_end_seconds": 16, "exclusions": []}]}
    report = {"schema_version": "exchange-observations.v1", "entries": [row]}
    file = tmp_path / "report.json"
    file.write_text(json.dumps(report))
    make_review_bundle(file, tmp_path / "bundle")
    review = json.loads((tmp_path / "bundle/review-template.json").read_text())
    return file, report, review


def approve(review, *, kind="visible_smile", actor="human"):
    review.update(reviewer="Example reviewer", reviewer_type=actor, reviewed_at="2026-11-11")
    item = review["entries"][0]
    item.update(decision="accept", evidence_note="Watched and checked the exact local source-clock clip.", checks={key: True for key in CHECKS}, annotations=[{"kind": kind, "start_seconds": 13, "end_seconds": 13.5, "observable": True, "phase": "speaking"}])
    return review


def import_payload(tmp_path, file, review):
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps(review))
    return import_review(file, annotation, tmp_path / "reviewed.json")


def test_review_template_does_not_create_human_scores(tmp_path):
    file, _, review = review_fixture(tmp_path)
    result = import_payload(tmp_path, file, review)
    assert result["runtime_index"] == []
    assert result["summary"]["human_reviewed"] == 0
    assert result["summary"]["unreviewed"] == 1


def test_review_bound_to_immutable_report(tmp_path):
    file, report, review = review_fixture(tmp_path)
    report["entries"][0]["situations"] = ["challenge"]
    file.write_text(json.dumps(report))
    with pytest.raises(MediaError, match="immutable"):
        import_payload(tmp_path, file, approve(review))


def test_agent_review_cannot_enter_runtime(tmp_path):
    file, _, review = review_fixture(tmp_path)
    result = import_payload(tmp_path, file, approve(review, actor="agent"))
    assert result["summary"]["agent_reviewed"] == 1
    assert result["summary"]["human_reviewed"] == 0
    assert result["runtime_index"] == []


def test_reviewed_validation_event_stays_out_of_runtime(tmp_path):
    file, report, review = review_fixture(tmp_path)
    report["entries"][0]["observation_split"] = "validation"
    file.write_text(json.dumps(report))
    review["report_sha256"] = sha256(file)
    result = import_payload(tmp_path, file, approve(review))
    assert len(result["accepted"]) == 1
    assert result["runtime_index"] == []


def test_human_train_observation_is_eligible(tmp_path):
    file, _, review = review_fixture(tmp_path)
    result = import_payload(tmp_path, file, approve(review))
    assert len(result["runtime_index"]) == 1
    assert result["accepted"][0]["pre_answer_pause_seconds"] is None


def test_laughter_requires_listening(tmp_path):
    file, _, review = review_fixture(tmp_path)
    approve(review, kind="audible_laughter")
    review["entries"][0]["checks"]["listened_to_source"] = False
    with pytest.raises(MediaError, match="listening"):
        import_payload(tmp_path, file, review)


def test_nod_requires_camera_review(tmp_path):
    file, _, review = review_fixture(tmp_path)
    approve(review, kind="head_nod")
    review["entries"][0]["checks"]["camera_motion_and_cuts_reviewed"] = False
    with pytest.raises(MediaError, match="camera-motion"):
        import_payload(tmp_path, file, review)


def test_pose_annotation_cannot_cross_cut(tmp_path):
    file, report, review = review_fixture(tmp_path)
    report["entries"][0]["visual_windows"][0]["exclusions"] = [{"start_seconds": 13.25, "end_seconds": 14}]
    file.write_text(json.dumps(report))
    review["report_sha256"] = sha256(file)
    with pytest.raises(MediaError, match="camera interval"):
        import_payload(tmp_path, file, approve(review))


def test_pause_remains_unknown_without_continuity(tmp_path):
    file, _, review = review_fixture(tmp_path)
    approve(review)
    review["entries"][0]["acoustic_boundaries"] = {"question_end_seconds": 11.5, "answer_start_seconds": 12.5, "answer_end_seconds": 16}
    review["entries"][0]["checks"]["recording_continuity_verified"] = False
    with pytest.raises(MediaError, match="continuity"):
        import_payload(tmp_path, file, review)


def test_verified_boundaries_produce_pause_separate_from_candidates(tmp_path):
    file, _, review = review_fixture(tmp_path)
    approve(review)
    review["entries"][0]["acoustic_boundaries"] = {"question_end_seconds": 11.5, "answer_start_seconds": 12.5, "answer_end_seconds": 16}
    result = import_payload(tmp_path, file, review)
    assert result["accepted"][0]["pre_answer_pause_seconds"] == 1


def test_overlapping_speech_not_positive_pause(tmp_path):
    file, _, review = review_fixture(tmp_path)
    approve(review)
    review["entries"][0]["acoustic_boundaries"] = {"question_end_seconds": 13, "answer_start_seconds": 12.5, "answer_end_seconds": 16}
    with pytest.raises(MediaError, match="Overlapping"):
        import_payload(tmp_path, file, review)
