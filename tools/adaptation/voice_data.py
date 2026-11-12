"""Portable, event-isolated voice experiments from source reference bundles.

Candidates are an explicit experimental condition. They never become the
independently accepted speech dataset merely by being exported or tokenized.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import wave
from collections import Counter
from pathlib import Path

from tools.media.common import artifact_path, sha256
from tools.references.workflow import verify_reference

SCHEMA = "qwen3-tts-reference-dataset.v1"


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_references(selection: Path, output: Path, *, persona_id: str, speaker: str,
                      allow_candidates: bool = False) -> dict:
    """Selection rows bind a verified reference to a predeclared train event."""
    if output.exists():
        raise ValueError("Choose a new voice dataset output directory")
    plan = json.loads(selection.read_text())
    if plan.get("schema_version") != "voice-reference-selection.v1" or plan.get("persona_id") != persona_id:
        raise ValueError("Voice selection must bind the requested persona")
    events = plan.get("event_splits", {})
    selected = plan.get("references", [])
    if not selected or not isinstance(events, dict):
        raise ValueError("Voice selection needs references and frozen event splits")
    if not all(isinstance(v, str) and v in {"train", "validation", "test"} for v in events.values()):
        raise ValueError("Unknown event split in voice selection")
    checked, seen, source_families = [], set(), {}
    for row in selected:
        event = row.get("event_family_id")
        if events.get(event) != "train":
            raise ValueError("Held-out events cannot enter voice adaptation")
        manifest_path = (selection.parent / row["reference_manifest"]).resolve()
        if row.get("manifest_sha256") != sha256(manifest_path):
            raise ValueError("Selected reference manifest changed")
        manifest = verify_reference(manifest_path)
        if manifest["publisher_attribution"]["speaker_label"] != speaker:
            raise ValueError("Selected reference belongs to another publisher speaker")
        source = manifest["source_id"]
        if source_families.setdefault(source, event) != event:
            raise ValueError("One source cannot belong to multiple event families")
        if manifest.get("persona_id") not in {None, persona_id}:
            raise ValueError("Selected reference belongs to another persona")
        status = manifest["review"]["status"]
        if status != "reviewed" and not (status == "candidate" and allow_candidates):
            raise ValueError("Candidate voice data requires explicit experimental opt-in")
        checks = manifest.get("automatic_checks", {})
        if checks.get("exact_normalized_text_agreement") is not True or checks.get("word_boundaries_not_cut") is not True:
            raise ValueError("Reference lacks exact ASR/publisher agreement or aligned word boundaries")
        audio = artifact_path(manifest_path, manifest["artifacts"]["audio"]["path"])
        digest = sha256(audio)
        if digest in seen:
            raise ValueError("Duplicate audio reference bytes cannot inflate training counts")
        seen.add(digest)
        with wave.open(str(audio), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
            if wav.getframerate() != 24000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2 or not 1 <= duration <= 30:
                raise ValueError("Qwen voice experiments require 1–30s mono PCM16 at 24kHz")
        text_path = artifact_path(manifest_path, manifest["artifacts"]["reference_text"]["path"])
        text = text_path.read_text(encoding="utf-8").strip()
        if not text or len(text) > 4000 or "\x00" in text:
            raise ValueError("Voice reference text is invalid")
        start, end = manifest.get("source_start_seconds"), manifest.get("source_end_seconds")
        if (not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in (start, end))
            or not 0 <= start < end or abs((end-start)-duration) > 0.02 or not manifest.get("source_timeline")):
            raise ValueError("Reference source-clock interval differs from actual audio duration")
        checked.append((row, manifest_path, manifest, audio, text, duration))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".voice-data-", dir=output.parent))
    try:
        records, lineage = [], []
        for index, (row, path, manifest, audio, text, duration) in enumerate(checked):
            name = f"audio-{index:04d}.wav"
            shutil.copyfile(audio, staging / name)
            records.append({"audio": name, "text": text, "ref_audio": "audio-0000.wav"})
            lineage.append({"audio": name, "audio_sha256": sha256(audio), "source_id": manifest["source_id"],
                "source_url": manifest["source_url"], "event_family_id": row["event_family_id"], "split": "train",
                "source_timeline": manifest["source_timeline"], "source_start_seconds": manifest["source_start_seconds"],
                "source_end_seconds": manifest["source_end_seconds"], "seconds": duration,
                "reference_manifest_sha256": sha256(path), "review_status": manifest["review"]["status"],
                "listening_review_completed": manifest["review"].get("listening_review_completed", False)})
        train = staging / "train_raw.jsonl"
        train.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        manifest = {"schema_version": SCHEMA, "persona_id": persona_id, "speaker": speaker,
            "samples": len(records), "seconds": sum(row["seconds"] for row in lineage),
            "selection_sha256": sha256(selection), "event_splits": events,
            "event_counts": dict(Counter(row["event_family_id"] for row in lineage)),
            "dataset_sha256": sha256(train), "reference_sha256": sha256(staging / "audio-0000.wav"),
            "audio_sha256": {row["audio"]: row["audio_sha256"] for row in lineage}, "lineage": lineage,
            "experimental_candidates": any(row["review_status"] != "reviewed" for row in lineage),
            "human_listening_completed": all(row["listening_review_completed"] for row in lineage),
            "speaker_similarity_evaluated": False, "heldout_samples": 0,
            "limitations": ["ASR agreement is not human listening or a speaker-similarity score.",
                            "Candidate source intervals remain candidate after adaptation."]}
        _write(staging / "manifest.json", manifest)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def load_dataset(directory: Path, *, persona_id: str, allow_candidates: bool = False) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA or manifest.get("persona_id") != persona_id:
        raise ValueError("Voice dataset schema or subject mismatch")
    if manifest.get("experimental_candidates") is not False and not allow_candidates:
        raise ValueError("Candidate adaptation requires explicit experimental opt-in")
    train = directory / "train_raw.jsonl"
    if sha256(train) != manifest.get("dataset_sha256"):
        raise ValueError("Frozen voice dataset changed")
    rows = [json.loads(line) for line in train.read_text().splitlines() if line.strip()]
    lineage = manifest.get("lineage", [])
    if not rows or len(rows) != manifest.get("samples") or len(lineage) != len(rows) or manifest.get("heldout_samples") != 0:
        raise ValueError("Voice dataset counts are inconsistent")
    families = Counter()
    for row, provenance in zip(rows, lineage):
        if provenance.get("audio") != row["audio"] or provenance.get("split") != "train" or manifest["event_splits"].get(provenance["event_family_id"]) != "train":
            raise ValueError("Voice lineage or held-out split differs")
        families[provenance["event_family_id"]] += 1
        for key in ("audio", "ref_audio"):
            path = artifact_path(directory / "manifest.json", row[key])
            digest = sha256(path)
            if digest != manifest["audio_sha256"].get(row[key]):
                raise ValueError("Frozen voice audio changed")
            if key == "ref_audio" and digest != manifest["reference_sha256"]:
                raise ValueError("Voice reference changed between training samples")
            row[key] = str(path)
    if dict(families) != manifest.get("event_counts"):
        raise ValueError("Voice event-family counts changed")
    return manifest, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--persona-id", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--allow-candidates", action="store_true")
    args = vars(parser.parse_args(argv))
    try:
        result = export_references(**args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Voice preparation failed: {exc}\n")
    print(json.dumps({k: result[k] for k in ("samples", "seconds", "event_counts", "experimental_candidates")}, indent=2))


if __name__ == "__main__":
    main()
