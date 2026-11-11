"""Combine acquired document snapshots without mutating either input."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import load_registry, write_json


def combine_acquisitions(registry_dir: Path, inputs: list[Path], output: Path) -> dict:
    registry, _ = load_registry(registry_dir)
    sources = {row["id"]: row for row in registry["sources"]}
    if len(inputs) < 2 or len({path.resolve() for path in inputs}) != len(inputs):
        raise ValueError("Combine at least two distinct acquisition directories")
    if output.exists() or output.is_symlink():
        raise ValueError("Combined acquisition output exists; choose a new snapshot")
    seen, rows, artifacts, identities = set(), [], [], []
    for root in inputs:
        manifest = root / "acquisition-attempts.json"
        raw = manifest.read_bytes()
        attempts = json.loads(raw)
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("An acquisition input must contain a nonempty attempt list")
        identities.append({"acquisition_log_sha256": hashlib.sha256(raw).hexdigest(), "attempts": len(attempts)})
        for attempt in attempts:
            identifier = attempt.get("source_id")
            if identifier not in sources or identifier in seen or sources[identifier]["split"] == "test":
                raise ValueError("Combined sources must be unique registered non-test sources")
            seen.add(identifier)
            if attempt.get("status") not in {"acquired", "failed"}:
                raise ValueError("Unknown acquisition attempt status")
            if attempt["status"] == "acquired":
                record = attempt.get("record", {})
                name = record.get("artifact")
                if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                    raise ValueError("Acquired artifact must be a recorded file name")
                artifact = root / "raw" / identifier / name
                data = artifact.read_bytes()
                if (not artifact.resolve().is_relative_to(root.resolve())
                        or record.get("source_id") != identifier or record.get("bytes") != len(data)
                        or record.get("sha256") != hashlib.sha256(data).hexdigest()):
                    raise ValueError("Acquired source bytes or identity changed")
                artifacts.append((artifact, identifier, name, record))
            rows.append(attempt)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".acquisition-combine-", dir=output.parent))
    try:
        for artifact, identifier, name, record in artifacts:
            target = temporary / "raw" / identifier
            target.mkdir(parents=True)
            shutil.copyfile(artifact, target / name)
            if hashlib.sha256((target / name).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Acquired source changed during copying")
            write_json(target / "original-acquisition.manifest.json", record)
        write_json(temporary / "acquisition-attempts.json", rows)
        result = {"schema_version": "persona-acquisition-combination.v1", "created_at": datetime.now(timezone.utc).isoformat(),
                  "inputs": identities, "sources": [row["source_id"] for row in rows], "attempts": len(rows),
                  "acquired": len(artifacts), "failed": len(rows) - len(artifacts),
                  "acquisition_log_sha256": hashlib.sha256((temporary / "acquisition-attempts.json").read_bytes()).hexdigest(),
                  "original_acquisition_timestamps_preserved": True, "source_bytes_unchanged": True,
                  "source_wording_distributed": False}
        write_json(temporary / "combination.json", result)
        temporary.rename(output)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
