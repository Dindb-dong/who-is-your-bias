from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.persona_data.combined import combine_acquisitions
from tools.persona_data.pipeline import write_json


def setup(tmp_path, *, second_split="train"):
    registry = tmp_path / "registry"
    sources = [{"id": key, "source_event_id": key, "event_family_id": key, "split": split}
               for key, split in (("one", "train"), ("two", second_split))]
    write_json(registry / "source-families.json", {"schema_version": "persona-source-families.v1", "sources": sources})
    write_json(registry / "split-registration.json", {"schema_version": "persona-split-registration.v1", "events": [
        {"source_event_id": s["id"], "event_family_id": s["id"], "source_ids": [s["id"]], "split": s["split"]} for s in sources]})
    inputs = []
    for source in sources:
        root = tmp_path / source["id"]
        raw = root / "raw" / source["id"]
        raw.mkdir(parents=True)
        value = ("Synthetic original snapshot " + source["id"]).encode()
        (raw / "source.html").write_bytes(value)
        record = {"source_id": source["id"], "artifact": "source.html", "sha256": hashlib.sha256(value).hexdigest(),
                  "bytes": len(value), "retrieved_at": "2026-11-11T00:00:00Z"}
        write_json(root / "acquisition-attempts.json", [{"source_id": source["id"], "status": "acquired", "record": record}])
        inputs.append(root)
    return registry, inputs


def test_combination_preserves_actual_artifacts_dates_and_attempts(tmp_path):
    registry, inputs = setup(tmp_path)
    before = {p: p.read_bytes() for root in inputs for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "combined"
    result = combine_acquisitions(registry, inputs, output)
    assert result["acquired"] == 2 and result["failed"] == 0
    actual = json.loads((output / "acquisition-attempts.json").read_text())
    expected = [json.loads((root / "acquisition-attempts.json").read_text())[0] for root in inputs]
    assert actual == expected
    for p, value in before.items():
        assert p.read_bytes() == value
    assert result["acquisition_log_sha256"] == hashlib.sha256((output / "acquisition-attempts.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="output exists"):
        combine_acquisitions(registry, inputs, output)


def test_combination_rejects_changed_source_bytes_before_output(tmp_path):
    registry, inputs = setup(tmp_path)
    (inputs[1] / "raw/two/source.html").write_text("Changed snapshot")
    output = tmp_path / "combined"
    with pytest.raises(ValueError, match="bytes or identity"):
        combine_acquisitions(registry, inputs, output)
    assert not output.exists()


def test_combination_rejects_duplicate_source_and_reserved_holdout(tmp_path):
    registry, inputs = setup(tmp_path, second_split="test")
    with pytest.raises(ValueError, match="non-test"):
        combine_acquisitions(registry, inputs, tmp_path / "combined")
    with pytest.raises(ValueError, match="distinct"):
        combine_acquisitions(registry, [inputs[0], inputs[0]], tmp_path / "combined")


def test_combination_retains_failed_acquisition_as_a_failure(tmp_path):
    registry, inputs = setup(tmp_path)
    failed = {"source_id": "two", "status": "failed", "error": "Synthetic unavailable publisher"}
    write_json(inputs[1] / "acquisition-attempts.json", [failed])
    output = tmp_path / "combined"
    result = combine_acquisitions(registry, inputs, output)
    assert result["acquired"] == 1 and result["failed"] == 1
    assert json.loads((output / "acquisition-attempts.json").read_text())[1] == failed
