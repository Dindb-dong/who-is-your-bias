"""Portable candidate packages retain exclusions and require explicit opt-in."""

import json
import shutil

import pytest

from tools.adaptation.voice_data import export_references, load_dataset
from tools.adaptation.datasets import file_hash
from tests.media.test_voice_subject_binding import reference_bundle


def selection(tmp_path, *, split="train"):
    _, manifest = reference_bundle(tmp_path / "reference")
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"schema_version": "voice-reference-selection.v1", "persona_id": "jensen-huang",
        "event_splits": {"event-one": split, "unseen-event": "test"}, "references": [
            {"reference_manifest": "reference/reference.json", "manifest_sha256": file_hash(manifest),
             "event_family_id": "event-one"}]}))
    return path


def test_candidate_export_and_training_each_require_explicit_opt_in(tmp_path):
    plan = selection(tmp_path)
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="opt-in"):
        export_references(plan, output, persona_id="jensen-huang", speaker="Jensen Huang")
    assert not output.exists()
    result = export_references(plan, output, persona_id="jensen-huang", speaker="Jensen Huang", allow_candidates=True)
    assert result["experimental_candidates"] is True
    assert result["human_listening_completed"] is False
    with pytest.raises(ValueError, match="opt-in"):
        load_dataset(output, persona_id="jensen-huang")
    _, rows = load_dataset(output, persona_id="jensen-huang", allow_candidates=True)
    assert rows[0]["audio"] == rows[0]["ref_audio"]


def test_voice_dataset_is_relocatable_and_rechecks_audio(tmp_path):
    plan = selection(tmp_path)
    original = tmp_path / "dataset"
    export_references(plan, original, persona_id="jensen-huang", speaker="Jensen Huang", allow_candidates=True)
    moved = tmp_path / "retrieved"
    shutil.move(original, moved)
    shutil.rmtree(tmp_path / "reference")
    _, rows = load_dataset(moved, persona_id="jensen-huang", allow_candidates=True)
    assert str(moved) in rows[0]["audio"]
    (moved / "audio-0000.wav").write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="audio changed"):
        load_dataset(moved, persona_id="jensen-huang", allow_candidates=True)


def test_heldout_event_cannot_enter_candidate_training(tmp_path):
    plan = selection(tmp_path, split="test")
    with pytest.raises(ValueError, match="Held-out"):
        export_references(plan, tmp_path / "dataset", persona_id="jensen-huang", speaker="Jensen Huang", allow_candidates=True)


def test_subject_and_reference_manifest_integrity_are_checked(tmp_path):
    plan = selection(tmp_path)
    for speaker in ("Elon Musk",):
        with pytest.raises(ValueError, match="another publisher speaker"):
            export_references(plan, tmp_path / "dataset", persona_id="jensen-huang", speaker=speaker, allow_candidates=True)
    path = tmp_path / "reference/reference.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="manifest changed"):
        export_references(plan, tmp_path / "dataset", persona_id="jensen-huang", speaker="Jensen Huang", allow_candidates=True)
