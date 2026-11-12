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
    return path


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
    path.parent.mkdir(parents=True, exist_ok=True)
    return _path(path)


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
    if value.get("schema_version") != PACKAGE_SCHEMA:
        raise ValueError("Unsupported package receipt")
    step = _positive(value.get("optimizer_steps"))
    if value.get("checkpoint_directory") != f"step-{step:08d}":
        raise ValueError("Package checkpoint identity differs")
    _sha(value.get("checkpoint_metadata_sha256"))
    _sha(value.get("binding_sha256"))
    archive = value.get("archive", {})
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package", help="Package an atomic checkpoint; no remote actions")
    package.add_argument("--checkpoint", type=Path, required=True)
    package.add_argument("--archive", type=Path, required=True)
    package.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = package_checkpoint(args.checkpoint, args.archive, args.receipt)
    except (ValueError, OSError, tarfile.TarError) as exc:
        parser.exit(2, f"Checkpoint delivery failed: {exc}\n")
    print(json.dumps({"schema_version": result["schema_version"],
                      "optimizer_steps": result["optimizer_steps"], "archive": result["archive"]}))


if __name__ == "__main__":
    main()
