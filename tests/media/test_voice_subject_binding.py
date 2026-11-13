"""Synthetic bytes test provenance gates; they are not Jensen recordings."""

import json
import math
import struct
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.voice.app import create_app
from services.voice.engine import VoiceSettings
from tools.adaptation.datasets import export_voice, file_hash, validate_dataset
from tools.persona_data.parsers import sha_text
from tools.references.workflow import CHECKS, _fingerprint, _ledger_entry


def reference_bundle(root: Path, *, label="Jensen Huang") -> tuple[Path, Path]:
    root.mkdir()
    audio = root / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000)
        stream.writeframes(b"".join(struct.pack("<h", int(1000 * math.sin(i / 15))) for i in range(24000)))
    text = "Synthetic fixture words."
    files = {"audio": audio}
    for name in ("reference_text", "publisher_context", "asr_context", "alignment", "media_manifest"):
        path = root / (name + ".txt")
        path.write_text(text if name == "reference_text" else "synthetic fixture")
        files[name] = path
    manifest = {"schema_version": "voice_reference.v1", "source_id": "synthetic-test",
                "source_url": "https://example.org/synthetic-test", "source_timeline": "fixture-native",
                "source_start_seconds": 0, "source_end_seconds": 1,
                "automatic_checks": {"exact_normalized_text_agreement": True, "word_boundaries_not_cut": True},
                "publisher_attribution": {"speaker_label": label},
                "artifacts": {name: {"path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size}
                              for name, path in files.items()}}
    checks = {name: {"status": "unknown", "grounds": "Synthetic test only."} for name in CHECKS}
    entry = _ledger_entry(None, 1, _fingerprint(manifest), "synthetic-test", "candidate", checks, "No real speaker.")
    (root / "review-ledger.jsonl").write_text(json.dumps(entry) + "\n")
    manifest["review"] = {"status": "candidate", "checks": checks, "listening_review_completed": False}
    manifest["review_ledger"] = {"path": "review-ledger.jsonl", "entries": 1, "head_sha256": entry["sha256"]}
    path = root / "reference.json"
    path.write_text(json.dumps(manifest))
    return audio, path


class FakeQwen:
    def create_voice_clone_prompt(self, **kwargs):
        return "synthetic-prompt"

    def generate_voice_clone(self, **kwargs):
        return [[0.1] * 240], 24000


def settings(audio, manifest, **overrides):
    return VoiceSettings(**dict({"reference_wav": audio, "reference_text": "Synthetic fixture words.",
        "reference_manifest": manifest, "persona_id": "jensen-huang", "speaker_label": "Jensen Huang",
        "allow_candidate_reference": True, "device": "cpu"}, **overrides))


def test_named_reference_preserves_candidate_status_and_detects_changed_audio(tmp_path):
    audio, manifest = reference_bundle(tmp_path / "bundle")
    client = TestClient(create_app(settings(audio, manifest), model_loader=lambda _: FakeQwen()))
    assert client.post("/v1/audio/speech", json={"input": "Test"}).status_code == 200
    identity = client.get("/health").json()["reference_identity"]
    assert identity["persona_id"] == "jensen-huang"
    assert identity["review_status"] == "candidate"
    assert identity["weight_adaptation"] is False
    assert identity["speaker_similarity_evaluated"] is False
    audio.write_bytes(audio.read_bytes() + b"tampering")
    assert client.post("/v1/audio/speech", json={"input": "Test"}).status_code == 503


@pytest.mark.parametrize("overrides", [
    {"speaker_label": "Elon Musk"}, {"persona_id": "elon-musk"},
    {"allow_candidate_reference": False}, {"reference_manifest": None},
    {"reference_text": "Changed text."},
])
def test_named_reference_rejects_incompatible_subject_or_unreviewed_inputs(tmp_path, overrides):
    audio, manifest = reference_bundle(tmp_path / "bundle")
    called = []
    client = TestClient(create_app(settings(audio, manifest, **overrides), model_loader=lambda _: called.append(1)))
    response = client.post("/v1/audio/speech", json={"input": "Test"})
    assert response.status_code == 503
    assert str(tmp_path) not in response.text
    assert called == []


def projection(audio, *, speaker="Jensen Huang", segment_id="speech-1"):
    text = "Synthetic test dialogue."
    return {"segment_id": segment_id, "source_id": "synthetic-test", "event_family_id": "event-1",
            "persona_id": "jensen-huang", "split": "train", "completeness": "complete_exchange",
            "text": text, "answer_sha256": sha_text(text),
            "media": {"speaker": speaker, "review_status": "accepted", "alignment_basis": "acoustically_reviewed",
                "file": "bundle/audio.wav", "sha256": file_hash(audio), "source_timeline": "synthetic-native",
                "start_seconds": 0, "end_seconds": 0.5}}


def test_jensen_voice_export_keeps_subject_event_and_interval_lineage(tmp_path):
    audio, _ = reference_bundle(tmp_path / "bundle")
    source = tmp_path / "projection.jsonl"
    source.write_text(json.dumps(projection(audio)) + "\n")
    output = tmp_path / "export"
    manifest = export_voice(source, output, asset_root=tmp_path, reference_id="speech-1",
                            speaker="Jensen Huang", persona_id="jensen-huang")
    assert manifest["speaker"] == "Jensen Huang"
    assert manifest["event_counts"] == {"event-1": 1}
    assert manifest["source_intervals"][0]["source_timeline"] == "synthetic-native"
    _, rows = validate_dataset(output, "voice")
    assert rows[0]["audio"] == rows[0]["ref_audio"]


def test_voice_export_cannot_mix_subjects_or_escape_via_segment_ids(tmp_path):
    audio, _ = reference_bundle(tmp_path / "bundle")
    source = tmp_path / "projection.jsonl"
    for rows in ([projection(audio), projection(audio, speaker="Elon Musk", segment_id="speech-2")],
                 [projection(audio, segment_id="../outside")]):
        source.write_text("\n".join(json.dumps(row) for row in rows))
        with pytest.raises(ValueError, match="speaker|filenames"):
            export_voice(source, tmp_path / "export", asset_root=tmp_path, reference_id=rows[0]["segment_id"])
        assert not (tmp_path / "export").exists()
