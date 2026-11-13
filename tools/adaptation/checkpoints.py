"""Atomic, hash-verified optimizer checkpoints for bounded adaptation jobs.

These are trusted local training artifacts, never inference model packages.
An optional publisher runs synchronously: failure stops the job at the saved
boundary rather than accumulating additional unpersisted optimizer work.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from services.language.adapters import file_digest, json_digest

SCHEMA = "adaptation-resume-checkpoint.v1"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _files(directory: Path) -> dict[str, str]:
    result = {}
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Checkpoint must be a real directory")
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Checkpoint cannot contain links or special files")
        if path.is_file() and path != directory / "checkpoint.json":
            result[path.relative_to(directory).as_posix()] = file_digest(path)
    return result


def capture_state(optimizer, *, cursor: dict, optimizer_steps: int) -> dict:
    import torch

    state = {"optimizer": optimizer.state_dict(), "cursor": cursor,
             "optimizer_steps": optimizer_steps, "python_rng": random.getstate(),
             "torch_rng": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        state["mps_rng"] = torch.mps.get_rng_state()
    return state


def restore_state(optimizer, state: dict) -> None:
    import torch

    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if "cuda_rng" in state:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint cannot restore its RNG on this host")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if "mps_rng" in state:
        if not torch.backends.mps.is_available():
            raise ValueError("MPS checkpoint cannot restore its RNG on this host")
        torch.mps.set_rng_state(state["mps_rng"])


def save_checkpoint(root: Path, *, model, state: dict, binding: dict,
                    report: dict, publisher: list[str] | None = None) -> Path:
    import torch

    step = state["optimizer_steps"]
    if type(step) is not int or step < 1:
        raise ValueError("A checkpoint requires an actual positive optimizer step")
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step-{step:08d}"
    if final.exists():
        raise ValueError("Refuse to overwrite an existing checkpoint")
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    try:
        model.save_pretrained(temporary / "adapter", safe_serialization=True)
        torch.save(state, temporary / "training-state.pt")
        metadata = {"schema_version": SCHEMA, "optimizer_steps": step,
                    "binding": binding, "binding_sha256": json_digest(binding),
                    "report": report, "files": _files(temporary)}
        atomic_json(temporary / "checkpoint.json", metadata)
        # Flush all payload bytes before making the directory visible.
        for path in temporary.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        os.replace(temporary, final)
        atomic_json(root / "latest.json", {"checkpoint": final.name,
                    "optimizer_steps": step, "metadata_sha256": file_digest(final / "checkpoint.json")})
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if publisher:
        if not isinstance(publisher, list) or not publisher or any(not isinstance(x, str) or not x for x in publisher):
            raise ValueError("Checkpoint publisher must be a nonempty argv list")
        completed = subprocess.run([*publisher, str(final.resolve())], check=True,
                                   capture_output=True, text=True)
        receipt = json.loads(completed.stdout)
        if (receipt.get("status") != "verified" or not receipt.get("persistent_locator")
                or receipt.get("checkpoint_metadata_sha256") != file_digest(final / "checkpoint.json")):
            raise ValueError("Publisher did not return a verified receipt for the exact checkpoint")
        atomic_json(root / f"{final.name}.persistence.json", receipt)
    return final


def load_checkpoint(path: Path, *, binding: dict | None = None) -> tuple[dict, dict]:
    import torch

    path = Path(path)
    actual = _files(path)
    metadata_path = path / "checkpoint.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("Checkpoint receipt is missing")
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("schema_version") != SCHEMA or metadata.get("files") != actual
            or not actual or "training-state.pt" not in actual):
        raise ValueError("Checkpoint files are missing, extra, changed, or unregistered")
    recorded = metadata.get("binding")
    if json_digest(recorded) != metadata.get("binding_sha256"):
        raise ValueError("Checkpoint binding digest differs")
    if binding is not None and recorded != binding:
        raise ValueError("Dataset, model, tokenizer, or optimization settings changed since checkpoint")
    state = torch.load(path / "training-state.pt", map_location="cpu", weights_only=True)
    if state.get("optimizer_steps") != metadata.get("optimizer_steps"):
        raise ValueError("Checkpoint progress differs from its receipt")
    return metadata, state
