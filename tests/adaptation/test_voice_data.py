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


def test_authoritative_registration_prevents_split_or_source_relabeling(tmp_path):
    plan = selection(tmp_path)
    document = json.loads(plan.read_text())
    reference = json.loads((tmp_path / "reference/reference.json").read_text())
    registration = tmp_path / "registered.json"
    registration.write_text(json.dumps({"schema_version": "persona-split-registration.v1", "events": [
        {"event_family_id": "event-one", "source_ids": [reference["source_id"]], "split": "train"},
        {"event_family_id": "unseen-event", "source_ids": ["heldout-source"], "split": "test"}]}))
    document["split_registration"] = {"path": registration.name, "sha256": file_hash(registration)}
    document["event_splits"]["unseen-event"] = "train"
    plan.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="registered event families"):
        export_references(plan, tmp_path / "bad", persona_id="jensen-huang", speaker="Jensen Huang", allow_candidates=True)
    document["event_splits"]["unseen-event"] = "test"
    plan.write_text(json.dumps(document))
    output = tmp_path / "dataset"
    export_references(plan, output, persona_id="jensen-huang", speaker="Jensen Huang", allow_candidates=True)
    registration.unlink()
    load_dataset(output, persona_id="jensen-huang", allow_candidates=True)
    metadata = output / "manifest.json"
    manifest = json.loads(metadata.read_text())
    manifest["lineage"][0]["source_id"] = "heldout-source"
    metadata.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="another registered event family"):
        load_dataset(output, persona_id="jensen-huang", allow_candidates=True)
