import json

import pytest

from tools.media.common import MediaError
from tools.references.untimed import candidate_intervals, publisher_from_turns


def fixture(tmp_path):
    text = "Alpha bravo cedar delta ember forest. Garden harbor island jasmine kettle lemon."
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps({"schema_version": "persona-original-turns.v1", "source_id": "synthetic",
        "source_timeline": None, "input_sha256": "a"*64, "turns": [
            {"id": "turn-1", "speaker": "Jensen Huang", "speaker_basis": "explicit_pdf_speaker_label",
             "start": None, "end": None, "text": text}]}))
    publisher = publisher_from_turns(path, source_url="https://example.org/synthetic")
    asr = {"schema_version": "transcript.v1", "source_id": "synthetic", "source_start_seconds": 100,
        "source_timeline": "synthetic-media", "segments": [{"words": [
            {"word": word, "start": 1+index*0.8, "end": 1.65+index*0.8, "confidence": 0.99}
            for index, word in enumerate(text.split())]}]}
    return path, publisher, asr, text


def test_sentence_mapping_keeps_original_characters_and_uses_only_acoustic_times(tmp_path):
    _, publisher, asr, text = fixture(tmp_path)
    assert publisher["timing_kind"] == "untimed_text"
    assert publisher["source_timeline"] is None
    for cue in publisher["segments"]:
        first, last = cue["source_character_span"]
        assert text[first:last] == cue["text"]
        assert cue["start"] is None
    intervals = candidate_intervals(publisher, asr, speaker="Jensen Huang")
    assert len(intervals) == 1
    assert intervals[0]["words"] == 12
    assert intervals[0]["start_seconds"] == 101
    assert intervals[0]["duration_seconds"] == pytest.approx(9.45)


def test_candidate_cannot_skip_disagreeing_words_or_speaker_turn(tmp_path):
    _, publisher, asr, _ = fixture(tmp_path)
    asr["segments"][0]["words"][6]["word"] = "Different"
    assert candidate_intervals(publisher, asr, speaker="Jensen Huang") == []
    _, publisher, asr, _ = fixture(tmp_path)
    publisher["segments"][1]["speaker"] = "Interviewer"
    assert candidate_intervals(publisher, asr, speaker="Jensen Huang") == []


def test_untimed_converter_must_not_drop_known_timestamps(tmp_path):
    path, _, _, _ = fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["turns"][0]["start"] = 90
    path.write_text(json.dumps(payload))
    with pytest.raises(MediaError, match="discard existing timing"):
        publisher_from_turns(path, source_url="https://example.org/synthetic")
