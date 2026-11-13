"""Package, inspect and acknowledge trusted adaptation checkpoints.

No command uploads, downloads, contacts Git or authenticates a remote receipt.
The coordinator owns those steps. A local verification proof records only checks
actually performed here; a gate acknowledgement separately names externally
performed materialization and public-commit checks. No model is initialized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import stat
import tarfile
import tempfile

from services.language.adapters import json_digest
from .checkpoints import SCHEMA, load_checkpoint

PACKAGE_SCHEMA = "adaptation-checkpoint-package.v1"
PROOF_SCHEMA = "adaptation-checkpoint-local-proof.v1"
SHA256 = re.compile(r"[a-f0-9]{64}")


def _sha(value: object) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("Expected a full lowercase SHA-256")
    return value


def _positive(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("Expected a positive integer")
    return value


def _member_name(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("Unsafe archive/payload path")
    path = PurePosixPath(value)
    if (not path.parts or path.is_absolute() or path.as_posix() != value
            or ".." in path.parts or ":" in path.parts[0]):
        raise ValueError("Unsafe archive/payload path")
    return value


def _path(path: Path) -> Path:
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("Paths cannot contain symlinks")
    return path.resolve()


def _file(path: Path) -> Path:
    path = _path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Expected a regular file without hard links")
    return path


def _identity(path: Path) -> dict:
    path = _file(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    value = json.loads(_file(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _new_path(path: Path) -> Path:
    path = _path(path)
    if path.exists():
        raise ValueError(f"Refuse to overwrite existing output: {path}")
    return path


def _publish_file(temporary: Path, output: Path) -> None:
    # link() atomically refuses an existing destination; replace() would clobber.
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(temporary, output)
    temporary.unlink()
    descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_json(path: Path, value: dict) -> None:
    path = _new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _path(path)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        _publish_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint(path: Path) -> tuple[dict, dict]:
    path = _path(path)
    if not path.is_dir():
        raise ValueError("Expected an atomic checkpoint directory")
    metadata = _json(path / "checkpoint.json")
    step = _positive(metadata.get("optimizer_steps"))
    if metadata.get("schema_version") != SCHEMA or path.name != f"step-{step:08d}":
        raise ValueError("Expected the published atomic checkpoint name and schema")
    if (not isinstance(metadata.get("binding"), dict)
            or json_digest(metadata["binding"]) != _sha(metadata.get("binding_sha256"))):
        raise ValueError("Checkpoint binding digest differs")
    report = metadata.get("report")
    if not isinstance(report, dict) or type(report.get("optimizer_steps")) is not int or report["optimizer_steps"] != step:
        raise ValueError("Checkpoint report progress differs")
    expected = metadata.get("files")
    if not isinstance(expected, dict) or "training-state.pt" not in expected or "checkpoint.json" in expected:
        raise ValueError("Checkpoint payload manifest is missing or invalid")
    parents = set()
    for name, digest in expected.items():
        _member_name(name)
        _sha(digest)
        parents.update(str(p) for p in PurePosixPath(name).parents if str(p) != ".")
    files = {}
    directories = set()
    for child in sorted(path.rglob("*")):
        _path(child)
        name = child.relative_to(path).as_posix()
        if child.is_dir():
            directories.add(name)
        else:
            files[name] = _identity(child)
    if directories != parents or set(files) != {"checkpoint.json", *expected}:
        raise ValueError("Checkpoint contains missing, extra or unregistered files/directories")
    if any(files[name]["sha256"] != digest for name, digest in expected.items()):
        raise ValueError("Checkpoint payload hash differs")
    return metadata, files


def _package(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != PACKAGE_SCHEMA:
        raise ValueError("Unsupported package receipt")
    step = _positive(value.get("optimizer_steps"))
    if value.get("checkpoint_directory") != f"step-{step:08d}":
        raise ValueError("Package checkpoint identity differs")
    _sha(value.get("checkpoint_metadata_sha256"))
    _sha(value.get("binding_sha256"))
    archive = value.get("archive", {})
    if not isinstance(archive, dict):
        raise ValueError("Invalid package archive identity")
    _positive(archive.get("bytes"))
    _sha(archive.get("sha256"))
    files = value.get("files")
    if not isinstance(files, dict) or not {"checkpoint.json", "training-state.pt"}.issubset(files):
        raise ValueError("Package payload manifest is missing")
    for name, identity in files.items():
        _member_name(name)
        if (not isinstance(identity, dict) or type(identity.get("bytes")) is not int
                or identity["bytes"] < 0):
            raise ValueError("Invalid package payload size")
        _sha(identity.get("sha256"))
    if files["checkpoint.json"]["sha256"] != value["checkpoint_metadata_sha256"]:
        raise ValueError("Package metadata digest differs")
    for field in ("source_archive", "source_checkpoint"):
        if not isinstance(value.get(field), str) or not Path(value[field]).is_absolute():
            raise ValueError("Package source paths must be absolute")
    return value


def _read_archive(archive: Path, package: dict, output: Path | None = None) -> None:
    """Accept exactly the regular files emitted by this module, without extractall."""
    root = package["checkpoint_directory"]
    expected = {f"{root}/{name}": identity for name, identity in package["files"].items()}
    seen = set()
    with tarfile.open(_file(archive), "r:gz", ignore_zeros=True) as tar:
        for member in tar:
            name = _member_name(member.name)
            if (not member.isfile() or member.issparse() or name not in expected or name in seen
                    or member.size != expected[name]["bytes"]):
                raise ValueError("Archive has a link, special, duplicate, missing or extra payload")
            seen.add(name)
            digest = hashlib.sha256()
            target = None
            if output is not None:
                destination = output / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                target = destination.open("xb")
            try:
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError("Archive member is unreadable")
                with stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                        if target is not None:
                            target.write(block)
                if digest.hexdigest() != expected[name]["sha256"]:
                    raise ValueError("Archive payload hash differs")
            finally:
                if target is not None:
                    target.close()
    if seen != set(expected):
        raise ValueError("Archive payload files are missing")


def package_checkpoint(checkpoint: Path, archive: Path, receipt: Path) -> dict:
    """Save a closed inventory of an existing atomic checkpoint; never load Torch."""
    checkpoint = _path(checkpoint)
    archive, receipt = _new_path(archive), _new_path(receipt)
    if not archive.name.endswith(".tar.gz") or archive == receipt:
        raise ValueError("Choose separate new .tar.gz archive and JSON receipt names")
    if archive.is_relative_to(checkpoint) or receipt.is_relative_to(checkpoint):
        raise ValueError("Delivery outputs must be outside the atomic checkpoint")
    metadata, files = _checkpoint(checkpoint)
    archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + archive.name + "-", dir=archive.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=3) as tar:
            for relative, identity in sorted(files.items()):
                info = tarfile.TarInfo(f"{checkpoint.name}/{relative}")
                info.size = identity["bytes"]
                info.mode = 0o600
                with _file(checkpoint / relative).open("rb") as stream:
                    tar.addfile(info, stream)
        result = {"schema_version": PACKAGE_SCHEMA, "optimizer_steps": metadata["optimizer_steps"],
            "checkpoint_directory": checkpoint.name, "source_checkpoint": str(checkpoint),
            "source_archive": str(archive), "archive": _identity(temporary), "files": files,
            "checkpoint_metadata_sha256": files["checkpoint.json"]["sha256"],
            "binding_sha256": metadata["binding_sha256"], "remote_storage_checked": False}
        _read_archive(temporary, _package(result))
        _publish_file(temporary, archive)
        _write_new_json(receipt, result)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def _matches(package: dict, metadata: dict, files: dict) -> None:
    if (metadata["optimizer_steps"] != package["optimizer_steps"]
            or metadata["binding_sha256"] != package["binding_sha256"]
            or files != package["files"]):
        raise ValueError("Restored checkpoint differs from the packaged identity")


def _cursor(value: object) -> dict:
    if (not isinstance(value, dict) or set(value) != {"epoch", "next_index"}
            or any(type(n) is not int or n < 0 for n in value.values())):
        raise ValueError("Invalid text checkpoint cursor")
    return value


def _loaded_state(state: dict, step: int) -> dict:
    """Check deserialized state and isolated RNG objects, without a training model."""
    import torch

    if not isinstance(state, dict) or type(state.get("optimizer_steps")) is not int or state["optimizer_steps"] != step:
        raise ValueError("Loaded state progress differs")
    cursor = _cursor(state.get("cursor"))
    optimizer = state.get("optimizer")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict) or not optimizer["state"]:
        raise ValueError("Checkpoint has no populated optimizer state")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Checkpoint optimizer parameter groups are missing")
    ids = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list) or not group["params"]:
            raise ValueError("Invalid checkpoint optimizer parameter group")
        ids.extend(group["params"])
    if (any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids)
            or any(type(i) is not int or i not in ids or not isinstance(v, dict) or not v
                   for i, v in optimizer["state"].items())):
        raise ValueError("Invalid checkpoint optimizer state entries")
    try:
        random.Random().setstate(state["python_rng"])
        torch.Generator(device="cpu").set_state(state["torch_rng"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Checkpoint Python/Torch CPU RNG state is invalid") from exc
    return {"optimizer_steps": step, "cursor": cursor,
            "optimizer_state_entries": len(optimizer["state"]), "optimizer_parameter_groups": len(groups),
            "additional_rng_states_loaded_without_device_restore": [key for key in ("cuda_rng", "mps_rng") if key in state]}


LOCAL_CHECKS = {"archive_hash_and_size": True, "closed_archive_inventory": True,
                "all_checkpoint_payload_hashes": True, "actual_load_checkpoint": True,
                "optimizer_state_loaded": True, "python_rng_validated": True,
                "torch_cpu_rng_validated": True, "cursor_validated": True}


def verify_checkpoint_archive(archive: Path, package_receipt: Path, output: Path, proof: Path) -> dict:
    """Inspect a coordinator-materialized copy and load its saved state on CPU.

    A distinct local pathname is necessary but cannot prove remote provenance.
    That provenance is deliberately not asserted in this local proof.
    """
    archive, package_receipt = _file(archive), _file(package_receipt)
    package = _package(_json(package_receipt))
    if archive == Path(package["source_archive"]).resolve():
        raise ValueError("Supply a separately materialized archive, not the packaged original")
    if _identity(archive) != package["archive"]:
        raise ValueError("Materialized archive size/hash differs from the package receipt")
    output, proof = _new_path(output), _new_path(proof)
    source_checkpoint = Path(package["source_checkpoint"])
    if proof.is_relative_to(output) or any(path.is_relative_to(source_checkpoint) for path in (output, proof)):
        raise ValueError("Use separate new restore/proof outputs outside the source checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    try:
        _read_archive(archive, package, output)
        checkpoint = output / package["checkpoint_directory"]
        metadata, files = _checkpoint(checkpoint)
        _matches(package, metadata, files)
        loaded_metadata, state = load_checkpoint(checkpoint, binding=metadata["binding"])
        if loaded_metadata != metadata:
            raise ValueError("Checkpoint changed during its actual state load")
        loaded = _loaded_state(state, package["optimizer_steps"])
        elapsed = metadata["report"].get("elapsed_seconds")
        if elapsed is not None and (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                                    or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("Invalid reported elapsed time")
        result = {"schema_version": PROOF_SCHEMA, "verified_at": datetime.now(timezone.utc).isoformat(),
            "package_receipt_sha256": _identity(package_receipt)["sha256"], "package": package,
            "materialized_archive": str(archive), "restored_checkpoint": str(checkpoint),
            "verification": dict(LOCAL_CHECKS), "loaded_state": loaded,
            "reported_elapsed_seconds": elapsed,
            "elapsed_interpretation": "Trainer-reported cumulative elapsed includes in-invocation checkpoint waits; not pure active compute.",
            "remote_storage_checked": False, "remote_materialization_origin_established": False,
            "git_commit_checked": False, "model_initialized": False, "training_resumed": False,
            "limitations": "Coordinator performs remote transfer and Git readback separately. This unsigned local proof is not remote authentication or a model quality result."}
        _write_new_json(proof, result)
        return result
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def _reference(value: object, label: str) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > 2048
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f"Supply an explicit {label} evidence reference")
    return value


def build_acknowledgement(proof: Path, *, proof_sha256: str, request: Path,
                          artifact_id: str, file_id: str, library_file_id: str,
                          externally_verified_public_commit: str,
                          materialization_record: str, git_readback_record: str) -> dict:
    """Build a legacy-gate receipt after local proof and explicit external checks.

    The caller must already have performed the two named external checks. Their
    references are attestations, not something this module independently checks.
    """
    proof, request = _file(proof), _file(request)
    if _identity(proof)["sha256"] != _sha(proof_sha256):
        raise ValueError("Local proof bytes differ from the expected proof SHA-256")
    value = _json(proof)
    if (value.get("schema_version") != PROOF_SCHEMA or value.get("verification") != LOCAL_CHECKS
            or any(value.get(key) is not False for key in ("remote_storage_checked",
                "remote_materialization_origin_established", "git_commit_checked", "model_initialized", "training_resumed"))):
        raise ValueError("A complete local checkpoint-loader proof is required")
    _sha(value.get("package_receipt_sha256"))
    package = _package(value.get("package", {}))
    loaded = value.get("loaded_state", {})
    if (not isinstance(loaded, dict) or type(loaded.get("optimizer_steps")) is not int
            or loaded["optimizer_steps"] != package["optimizer_steps"]):
        raise ValueError("Local proof loaded progress differs")
    _positive(loaded.get("optimizer_state_entries"))
    _positive(loaded.get("optimizer_parameter_groups"))
    _cursor(loaded.get("cursor"))
    materialized = _file(Path(value["materialized_archive"]))
    if materialized == Path(package["source_archive"]).resolve() or _identity(materialized) != package["archive"]:
        raise ValueError("Verified materialized archive changed")
    checkpoint = _path(Path(value["restored_checkpoint"]))
    _matches(package, *_checkpoint(checkpoint))
    pending = _json(request)
    step, metadata_sha = package["optimizer_steps"], package["checkpoint_metadata_sha256"]
    if (request.name != "request.json" or request.parent.name != f"step-{step:08d}-{metadata_sha[:12]}"
            or pending.get("schema_version") != "adaptation-remote-verification-request.v1"
            or type(pending.get("optimizer_steps")) is not int or pending["optimizer_steps"] != step
            or pending.get("checkpoint_metadata_sha256") != metadata_sha
            or pending.get("checkpoint") != package["source_checkpoint"]):
        raise ValueError("Gate request does not identify the packaged checkpoint")
    _matches(package, *_checkpoint(Path(pending["checkpoint"])))
    for label, value_id, pattern in (("artifact_id", artifact_id, r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}"),
            ("file_id", file_id, r"file_[A-Za-z0-9_-]+"),
            ("library_file_id", library_file_id, r"libfile_[A-Za-z0-9_-]+"),
            ("public commit SHA", externally_verified_public_commit, r"[a-f0-9]{40}")):
        if not isinstance(value_id, str) or not re.fullmatch(pattern, value_id):
            raise ValueError(f"Invalid explicit {label}")
    return {"schema_version": "adaptation-checkpoint-gate-acknowledgement.v1", "status": "verified",
        "verification_status": "fresh_materialization_and_checkpoint_load_verified",
        "optimizer_steps": step, "checkpoint_metadata_sha256": metadata_sha,
        "artifact_id": artifact_id, "persistent_locator": library_file_id, "file_id": file_id,
        "archive_sha256": package["archive"]["sha256"], "archive_bytes": package["archive"]["bytes"],
        "public_checkpoint_sha": externally_verified_public_commit,
        "source_proof": str(proof), "source_proof_sha256": proof_sha256,
        "external_verification": {"performed_by": "coordinator; not this module",
            "materialization_record": _reference(materialization_record, "fresh materialization"),
            "git_readback_record": _reference(git_readback_record, "public commit readback"),
            "remote_storage_checked_by_module": False, "git_checked_by_module": False},
        "verification_scope": "Local bytes/state proof plus coordinator attestations; supplied IDs and SHA do not independently establish remote checking."}


def acknowledge_checkpoint(proof: Path, **kwargs) -> dict:
    """Write verified.json atomically without replacing any earlier gate decision."""
    request = _file(kwargs["request"])
    destination = _new_path(request.with_name("verified.json"))
    result = build_acknowledgement(proof, **kwargs)
    _write_new_json(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package", help="Package an atomic checkpoint; no remote actions")
    package.add_argument("--checkpoint", type=Path, required=True)
    package.add_argument("--archive", type=Path, required=True)
    package.add_argument("--receipt", type=Path, required=True)
    verify = sub.add_parser("verify", help="Validate a separately materialized archive and load checkpoint state")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--package-receipt", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--proof", type=Path, required=True)
    ack = sub.add_parser("acknowledge", help="Acknowledge only after the coordinator's remote and Git checks")
    ack.add_argument("--proof", type=Path, required=True)
    ack.add_argument("--request", type=Path, required=True)
    for name in ("proof-sha256", "artifact-id", "file-id", "library-file-id",
                 "externally-verified-public-commit", "materialization-record", "git-readback-record"):
        ack.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        values = vars(args).copy()
        command = values.pop("command")
        if command == "package":
            result = package_checkpoint(**values)
        elif command == "verify":
            result = verify_checkpoint_archive(**values)
        else:
            result = acknowledge_checkpoint(**values)
    except (ValueError, OSError, tarfile.TarError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"Checkpoint delivery failed: {exc}\n")
    print(json.dumps({"schema_version": result["schema_version"],
        "optimizer_steps": result.get("optimizer_steps", result.get("loaded_state", {}).get("optimizer_steps")),
        "proof_sha256": _identity(args.proof)["sha256"] if command == "verify" else None,
        "remote_checks_performed_by_module": False}))


if __name__ == "__main__":
    main()
