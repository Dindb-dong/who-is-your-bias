"""Pause a trainer at an atomic checkpoint until its remote verifier acknowledges it.

This process does not contact storage or verify a remote upload itself. The
coordinator must freshly retrieve and load the exact checkpoint before writing
the explicit receipt. No next optimizer update runs while this process waits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

from .checkpoints import SCHEMA, atomic_json

SHA256 = re.compile(r"[a-f0-9]{64}")


def checkpoint_identity(checkpoint: Path) -> tuple[int, str]:
    path = checkpoint / "checkpoint.json"
    if checkpoint.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("An atomic checkpoint directory is required")
    payload = path.read_bytes()
    metadata = json.loads(payload)
    step = metadata.get("optimizer_steps")
    if metadata.get("schema_version") != SCHEMA or type(step) is not int or step < 1:
        raise ValueError("Checkpoint metadata has invalid progress")
    return step, hashlib.sha256(payload).hexdigest()


def wait_for_verification(checkpoint: Path, handoff: Path, *, timeout=1800., poll_interval=.25) -> dict:
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in (timeout, poll_interval)) or not 0 < poll_interval <= timeout <= 3600:
        raise ValueError("Choose a finite bounded checkpoint wait")
    step, metadata_sha = checkpoint_identity(checkpoint)
    directory = handoff / f"step-{step:08d}-{metadata_sha[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    request = {"schema_version": "adaptation-remote-verification-request.v1",
        "optimizer_steps": step, "checkpoint": str(checkpoint.absolute()),
        "checkpoint_metadata_sha256": metadata_sha,
        "required_verification": "fresh archive bytes, all payload hashes, and actual optimizer/RNG/cursor loader",
        "verifier": "external coordinator; this waiting process does not contact remote storage"}
    request_path = directory / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Checkpoint handoff request differs from the saved identity")
    atomic_json(request_path, request)
    receipt_path = directory / "verified.json"
    deadline = time.monotonic() + timeout
    while True:
        if receipt_path.exists():
            if receipt_path.is_symlink():
                raise ValueError("Checkpoint verification receipt cannot be a symlink")
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("status") != "verified"
                    or receipt.get("verification_status") != "fresh_materialization_and_checkpoint_load_verified"
                    or type(receipt.get("optimizer_steps")) is not int or receipt["optimizer_steps"] != step
                    or receipt.get("checkpoint_metadata_sha256") != metadata_sha
                    or not isinstance(receipt.get("persistent_locator"), str)
                    or not receipt["persistent_locator"].startswith("libfile_")
                    or not isinstance(receipt.get("file_id"), str) or not receipt["file_id"].startswith("file_")
                    or not isinstance(receipt.get("archive_sha256"), str)
                    or not SHA256.fullmatch(receipt["archive_sha256"])
                    or type(receipt.get("archive_bytes")) is not int or receipt["archive_bytes"] <= 0
                    or not isinstance(receipt.get("public_checkpoint_sha"), str)
                    or not re.fullmatch(r"[a-f0-9]{40}", receipt["public_checkpoint_sha"])):
                raise ValueError("Remote verifier receipt does not acknowledge this exact checkpoint")
            if checkpoint_identity(checkpoint) != (step, metadata_sha):
                raise ValueError("Checkpoint changed while waiting for remote verification")
            return receipt
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("No verified remote receipt arrived; retain this local checkpoint and stop training")
        time.sleep(min(poll_interval, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    try:
        result = wait_for_verification(args.checkpoint, args.handoff, timeout=args.timeout)
    except (ValueError, OSError, TimeoutError) as exc:
        parser.exit(2, f"Checkpoint verification gate failed: {exc}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
