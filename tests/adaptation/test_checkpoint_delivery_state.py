"""Real tiny AdamW/RNG state loading and its binding to the unchanged gate."""
import hashlib
import json
import random
import shutil

import pytest

torch = pytest.importorskip("torch")

from services.language.adapters import json_digest
from tools.adaptation import checkpoint_delivery as delivery
from tools.adaptation.checkpoint_gate import wait_for_verification
from tools.adaptation.checkpoints import SCHEMA, capture_state


def save_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def tiny_checkpoint(tmp_path, mutation=None):
    # A scalar and its optimizer are enough: no nn.Module, pretrained model,
    # tokenizer, network request or download is initialized by these tests.
    parameter = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        parameter.square().backward()
        optimizer.step()
    state = capture_state(optimizer, cursor={"epoch": 0, "next_index": 8}, optimizer_steps=2)
    if mutation == "optimizer":
        state["optimizer"]["state"] = {}
    elif mutation == "cursor":
        state["cursor"]["next_index"] = True
    elif mutation == "python_rng":
        state["python_rng"] = ("invalid",)
    elif mutation == "torch_rng":
        state["torch_rng"] = torch.ones(1, dtype=torch.uint8)
    checkpoint = tmp_path / "source" / "step-00000002"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "adapter" / "fixture.bin").write_bytes(b"bounded adapter payload; not initialized")
    torch.save(state, checkpoint / "training-state.pt")
    binding = {"seed": 42, "dataset_sha256": "a" * 64}
    metadata = {"schema_version": SCHEMA, "optimizer_steps": 2,
        "binding": binding, "binding_sha256": json_digest(binding),
        "report": {"optimizer_steps": 2, "elapsed_seconds": 17.5, "status": "running"},
        "files": {p.relative_to(checkpoint).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in checkpoint.rglob("*") if p.is_file()}}
    save_json(checkpoint / "checkpoint.json", metadata)
    archive, package = tmp_path / "packaged.tar.gz", tmp_path / "package.json"
    delivery.package_checkpoint(checkpoint, archive, package)
    fresh = tmp_path / "fresh.tar.gz"
    shutil.copyfile(archive, fresh)
    return checkpoint, fresh, package, tmp_path / "restore", tmp_path / "proof.json"


def verified(tmp_path):
    checkpoint, archive, package, output, proof = tiny_checkpoint(tmp_path)
    result = delivery.verify_checkpoint_archive(archive, package, output, proof)
    handoff = tmp_path / "gates"
    # Ask the real gate to write its own request, then retain the timed-out request.
    with pytest.raises(TimeoutError):
        wait_for_verification(checkpoint, handoff, timeout=.002, poll_interval=.001)
    request = next(handoff.glob("*/request.json"))
    arguments = {"proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(), "request": request,
        "artifact_id": "fixture-checkpoint-2", "file_id": "file_fixture", "library_file_id": "libfile_fixture",
        "externally_verified_public_commit": "b" * 40,
        "materialization_record": "fixture-transfer-receipt", "git_readback_record": "fixture-git-readback-receipt"}
    return checkpoint, archive, output, proof, result, arguments


def test_actual_loader_preserves_rng_and_emits_only_local_evidence(tmp_path):
    _, archive, package, output, proof = tiny_checkpoint(tmp_path)
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    result = delivery.verify_checkpoint_archive(archive, package, output, proof)
    assert result["verification"]["actual_load_checkpoint"] is True
    assert result["loaded_state"]["optimizer_state_entries"] == 1
    assert result["loaded_state"]["cursor"] == {"epoch": 0, "next_index": 8}
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert result["remote_storage_checked"] is False
    assert result["remote_materialization_origin_established"] is False
    assert result["git_commit_checked"] is False
    assert result["model_initialized"] is False
    assert result["training_resumed"] is False
    assert result["reported_elapsed_seconds"] == 17.5
    assert "checkpoint waits" in result["elapsed_interpretation"]
    assert "elapsed_active_seconds" not in result
    assert json.loads(proof.read_text()) == result


@pytest.mark.parametrize("mutation", ["optimizer", "cursor", "python_rng", "torch_rng"])
def test_actual_loader_requires_usable_optimizer_rng_cursor(tmp_path, mutation):
    _, archive, package, output, proof = tiny_checkpoint(tmp_path, mutation)
    with pytest.raises(ValueError):
        delivery.verify_checkpoint_archive(archive, package, output, proof)
    assert not output.exists()
    assert not proof.exists()


def test_proof_acknowledgement_releases_only_exact_existing_gate(tmp_path):
    checkpoint, _, _, proof, _, arguments = verified(tmp_path)
    request = arguments["request"]
    preview = delivery.build_acknowledgement(proof, **arguments)
    assert not request.with_name("verified.json").exists()
    result = delivery.acknowledge_checkpoint(proof, **arguments)
    assert result == preview
    assert result["external_verification"]["remote_storage_checked_by_module"] is False
    assert result["external_verification"]["git_checked_by_module"] is False
    assert wait_for_verification(checkpoint, request.parent.parent, timeout=.01, poll_interval=.001) == result
    saved = request.with_name("verified.json").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        delivery.acknowledge_checkpoint(proof, **arguments)
    assert request.with_name("verified.json").read_bytes() == saved


@pytest.mark.parametrize(("field", "value"), [
    ("proof_sha256", "d" * 64), ("artifact_id", ""), ("file_id", "pending"),
    ("library_file_id", "pending"), ("externally_verified_public_commit", "HEAD"),
    ("materialization_record", ""), ("git_readback_record", ""),
])
def test_acknowledgement_requires_proof_identity_and_explicit_external_evidence(tmp_path, field, value):
    _, _, _, proof, _, arguments = verified(tmp_path)
    arguments[field] = value
    with pytest.raises(ValueError):
        delivery.acknowledge_checkpoint(proof, **arguments)
    assert not arguments["request"].with_name("verified.json").exists()


@pytest.mark.parametrize("mutation", ["proof", "archive", "restored", "source", "request"])
def test_changed_verified_inputs_never_release_gate(tmp_path, mutation):
    checkpoint, archive, output, proof, result, arguments = verified(tmp_path)
    if mutation == "proof":
        result["verification"]["actual_load_checkpoint"] = False
        save_json(proof, result)
        arguments["proof_sha256"] = hashlib.sha256(proof.read_bytes()).hexdigest()
    elif mutation == "archive":
        archive.write_bytes(archive.read_bytes() + b"changed")
    elif mutation in {"restored", "source"}:
        target = output / checkpoint.name if mutation == "restored" else checkpoint
        (target / "training-state.pt").write_bytes(b"changed")
    else:
        request = json.loads(arguments["request"].read_text())
        request["optimizer_steps"] = 3
        save_json(arguments["request"], request)
    with pytest.raises(ValueError):
        delivery.acknowledge_checkpoint(proof, **arguments)
    assert not arguments["request"].with_name("verified.json").exists()
