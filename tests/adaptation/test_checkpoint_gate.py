import hashlib
import json

import pytest

from tools.adaptation.checkpoint_gate import checkpoint_identity, wait_for_verification
from tools.adaptation.checkpoints import SCHEMA


def prepared(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    path = checkpoint / "checkpoint.json"
    path.write_text(json.dumps({"schema_version": SCHEMA, "optimizer_steps": 8}))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    handoff = tmp_path / "handoff"
    directory = handoff / f"step-00000008-{sha[:12]}"
    directory.mkdir(parents=True)
    receipt = {"status": "verified", "verification_status": "fresh_materialization_and_checkpoint_load_verified",
        "optimizer_steps": 8, "checkpoint_metadata_sha256": sha, "persistent_locator": "libfile_example",
        "file_id": "file_example", "archive_sha256": "a" * 64, "archive_bytes": 100,
        "public_checkpoint_sha": "b" * 40}
    return checkpoint, handoff, directory, receipt


def test_matching_external_receipt_releases_exact_checkpoint(tmp_path):
    checkpoint, handoff, directory, receipt = prepared(tmp_path)
    (directory / "verified.json").write_text(json.dumps(receipt))
    assert wait_for_verification(checkpoint, handoff, timeout=.1, poll_interval=.01) == receipt
    request = json.loads((directory / "request.json").read_text())
    assert request["checkpoint_metadata_sha256"] == checkpoint_identity(checkpoint)[1]
    assert "does not contact remote storage" in request["verifier"]


@pytest.mark.parametrize(("field", "value"), [
    ("status", "uploaded"), ("verification_status", "local_hash_only"),
    ("optimizer_steps", 7), ("checkpoint_metadata_sha256", "c" * 64),
    ("persistent_locator", ""), ("file_id", ""), ("archive_sha256", "short"),
    ("archive_bytes", 0), ("archive_bytes", True), ("public_checkpoint_sha", "pending"),
])
def test_incomplete_or_wrong_identity_receipt_stops_training(tmp_path, field, value):
    checkpoint, handoff, directory, receipt = prepared(tmp_path)
    receipt[field] = value
    (directory / "verified.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="exact checkpoint"):
        wait_for_verification(checkpoint, handoff, timeout=.1, poll_interval=.01)


def test_timeout_leaves_request_and_checkpoint_for_resume(tmp_path):
    checkpoint, handoff, directory, _ = prepared(tmp_path)
    with pytest.raises(TimeoutError, match="stop training"):
        wait_for_verification(checkpoint, handoff, timeout=.01, poll_interval=.005)
    assert (directory / "request.json").is_file()
    assert (checkpoint / "checkpoint.json").is_file()


def test_checkpoint_mutation_after_request_does_not_release(tmp_path, monkeypatch):
    checkpoint, handoff, directory, receipt = prepared(tmp_path)
    original = checkpoint_identity(checkpoint)
    real = checkpoint_identity
    calls = []
    def changed(path):
        calls.append(path)
        return real(path) if len(calls) == 1 else (original[0], "d" * 64)
    monkeypatch.setattr("tools.adaptation.checkpoint_gate.checkpoint_identity", changed)
    (directory / "verified.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="changed while waiting"):
        wait_for_verification(checkpoint, handoff, timeout=.1, poll_interval=.01)
