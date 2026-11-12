"""Bounded archive/proof/gate delivery; Torch fixtures never load a model."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

from services.language.adapters import json_digest
from tools.adaptation import checkpoint_delivery as delivery
from tools.adaptation.checkpoints import SCHEMA


def save_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def prepared(tmp_path):
    checkpoint = tmp_path / "source" / "step-00000002"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter").mkdir()
    (checkpoint / "adapter" / "weights.safetensors").write_bytes(b"fixture weights")
    (checkpoint / "training-state.pt").write_bytes(b"fixture state; not a load proof")
    binding = {"dataset_sha256": "a" * 64, "seed": 42}
    metadata = {"schema_version": SCHEMA, "optimizer_steps": 2,
                "binding": binding, "binding_sha256": json_digest(binding),
                "report": {"optimizer_steps": 2, "elapsed_seconds": 17.5, "status": "running"},
                "files": {p.relative_to(checkpoint).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in checkpoint.rglob("*") if p.is_file()}}
    save_json(checkpoint / "checkpoint.json", metadata)
    return checkpoint, metadata, tmp_path / "source.tar.gz", tmp_path / "package.json"


def test_package_keeps_exact_closed_inventory_and_refuses_overwrite(tmp_path):
    checkpoint, metadata, archive, receipt = prepared(tmp_path)
    result = delivery.package_checkpoint(checkpoint, archive, receipt)
    assert result["optimizer_steps"] == 2
    assert result["remote_storage_checked"] is False
    assert result["archive"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert json.loads(receipt.read_text()) == result
    with tarfile.open(archive, "r:gz") as tar:
        assert {member.name for member in tar} == {
            f"{checkpoint.name}/{name}" for name in (*metadata["files"], "checkpoint.json")}
        for member in tar.getmembers():
            assert member.isfile()
            assert tar.extractfile(member).read() == (checkpoint / member.name.split("/", 1)[1]).read_bytes()
    original = archive.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        delivery.package_checkpoint(checkpoint, archive, tmp_path / "new.json")
    with pytest.raises(ValueError, match="overwrite"):
        delivery.package_checkpoint(checkpoint, tmp_path / "new.tar.gz", receipt)
    assert archive.read_bytes() == original
    assert not (tmp_path / "new.tar.gz").exists()


@pytest.mark.parametrize("mutation", ["extra-file", "extra-directory", "changed", "symlink", "hardlink", "binding", "progress"])
def test_package_rejects_changed_unregistered_or_linked_checkpoint(tmp_path, mutation):
    checkpoint, metadata, archive, receipt = prepared(tmp_path)
    payload = checkpoint / "training-state.pt"
    if mutation == "extra-file":
        (checkpoint / "extra.txt").write_text("unregistered")
    elif mutation == "extra-directory":
        (checkpoint / "empty").mkdir()
    elif mutation == "changed":
        payload.write_bytes(b"changed")
    elif mutation in {"symlink", "hardlink"}:
        external = tmp_path / "external.pt"
        external.write_bytes(payload.read_bytes())
        payload.unlink()
        if mutation == "symlink":
            payload.symlink_to(external)
        else:
            os.link(external, payload)
    elif mutation == "binding":
        metadata["binding"]["seed"] = 43
        save_json(checkpoint / "checkpoint.json", metadata)
    else:
        metadata["report"]["optimizer_steps"] = 1
        save_json(checkpoint / "checkpoint.json", metadata)
    with pytest.raises(ValueError):
        delivery.package_checkpoint(checkpoint, archive, receipt)
    assert not archive.exists()
    assert not receipt.exists()


@pytest.mark.parametrize("unsafe", ["../escaped", "/absolute", "a/../b", "a//b", "a\\b", ".", "C:/file"])
def test_package_rejects_unsafe_manifest_names(tmp_path, unsafe):
    checkpoint, metadata, archive, receipt = prepared(tmp_path)
    metadata["files"][unsafe] = "c" * 64
    save_json(checkpoint / "checkpoint.json", metadata)
    with pytest.raises(ValueError, match="Unsafe"):
        delivery.package_checkpoint(checkpoint, archive, receipt)
    assert not archive.exists()


def test_package_refuses_checkpoint_output_and_linked_parent(tmp_path):
    checkpoint, _, _, receipt = prepared(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        delivery.package_checkpoint(checkpoint, checkpoint / "bad.tar.gz", receipt)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "source", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        delivery.package_checkpoint(checkpoint, alias / "bad.tar.gz", receipt)


def test_package_detects_source_mutation_during_tar_write(tmp_path, monkeypatch):
    checkpoint, _, archive, receipt = prepared(tmp_path)
    real = tarfile.TarFile.addfile

    def changed(self, info, stream=None):
        if info.name.endswith("training-state.pt"):
            return real(self, info, io.BytesIO(b"X" * info.size))
        return real(self, info, stream)

    monkeypatch.setattr(tarfile.TarFile, "addfile", changed)
    with pytest.raises(ValueError, match="payload hash"):
        delivery.package_checkpoint(checkpoint, archive, receipt)
    assert not archive.exists()
    assert not receipt.exists()
