"""Bind source paraphrases to actual ASR word matches without speaker inference."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from bias.motion import MotionClip, MotionError, context_digest
from tools.references.workflow import align_transcripts, asr_words, read_json
from tools.media.common import finite_number, sha256


def bind_contexts(manifest_path: str | Path, asset_root: str | Path, report_path: str | Path, *, publishers: dict[str, str | Path], alignment_directories: list[str | Path]) -> dict:
    """Update only context evidence in a motion-input.v1 manifest.

    Supplied candidate alignments are recomputed from the exact publisher and
    ASR artifacts. Full transcripts remain local. No clock offset is fitted,
    no word is force-aligned, and no image track acquires a speaker label.
    """
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    if manifest.get("schema_version") != "motion-input.v1":
        raise MotionError("Expected motion-input.v1")
    root = Path(asset_root).resolve(strict=True)
    destination = Path(report_path).resolve()
    if not destination.is_relative_to(root):
        raise MotionError("Context evidence must be inside the supplied asset root")
    runs = []
    for directory in alignment_directories:
        directory = Path(directory)
        alignment = read_json(directory / "alignment.json")
        asr_file = directory / "asr.json"
        asr = read_json(asr_file)
        media = read_json(directory / "manifest.json")
        publisher_file = Path(publishers[asr["source_id"]])
        envelope = read_json(publisher_file)
        publisher = envelope.get("transcript", envelope)
        if sha256(asr_file) != alignment["asr_artifact_sha256"] or sha256(publisher_file) != alignment["publisher_artifact_sha256"]:
            raise MotionError("ASR/publisher artifact hash differs from the alignment")
        if align_transcripts(publisher, asr)["matches"] != alignment["matches"]:
            raise MotionError("Candidate token matches could not be reproduced")
        if media["source_id"] != asr["source_id"] or media["source_timeline"] != asr["source_timeline"] or media["artifacts"]["audio"]["sha256"] != asr["audio_sha256"]:
            raise MotionError("ASR source/audio differs from its prepared media")
        if abs(finite_number(asr.get("source_start_seconds"), "ASR offset") - finite_number(media.get("source_start_seconds"), "prepared media offset")) > 0.001:
            raise MotionError("ASR and prepared media source offsets differ")
        if sha256(directory / media["artifacts"]["audio"]["path"]) != asr["audio_sha256"]:
            raise MotionError("ASR audio hash mismatch")
        runs.append({"alignment": alignment, "asr": asr, "words": asr_words(asr), "publisher": publisher, "media": media, "id": directory.name, "alignment_sha256": sha256(directory / "alignment.json")})
    entries = []
    updates = {}
    for raw_record in manifest["clips"]:
        record = MotionClip.model_validate(raw_record)
        if record.context is None:
            continue
        clip_file = (root / record.path).resolve()
        if not clip_file.is_relative_to(root) or sha256(clip_file) != record.sha256:
            raise MotionError("Motion geometry hash/path mismatch")
        clip = read_json(clip_file)
        start = clip["source"]["source_start_seconds"]
        end = start + clip["duration_ms"] / 1000
        candidates = [run for run in runs if run["asr"]["source_id"] == record.source_id and run["media"]["source_start_seconds"] <= start and run["media"]["source_end_seconds"] >= end]
        if len(candidates) != 1:
            raise MotionError("Each geometry clip needs exactly one covering ASR context")
        run = candidates[0]
        if run["media"]["input"]["sha256"] != clip["source"]["input_sha256"] or run["asr"]["source_timeline"] != clip["source"]["source_timeline"]:
            raise MotionError("Geometry and ASR were not derived from the same acquired video clock")
        if record.context.timeline != run["publisher"].get("source_timeline"):
            raise MotionError("Context and publisher transcript clocks differ")
        selected_cues = [cue for cue in run["publisher"]["segments"] if cue["id"] in record.context.cue_ids]
        if (not selected_cues or len(selected_cues) != len(record.context.cue_ids)
            or [cue["id"] for cue in selected_cues] != record.context.cue_ids
            or abs(min(cue["start"] for cue in selected_cues) - record.context.start_seconds) > 0.001
            or abs(max(cue["end"] for cue in selected_cues) - record.context.end_seconds) > 0.001):
            raise MotionError("Context cues and interval differ from the publisher declaration")
        words = {index: word for index, word in enumerate(run["words"]) if word["source_start"] < end and word["source_end"] > start}
        matched = [match for match in run["alignment"]["matches"] if match["asr_word_index"] in words]
        counts = Counter(match["asr_word_index"] for match in matched)
        complete = {index for index, word in words.items() if counts[index] == len(word["tokens"])}
        coverage = len(complete) / len(words) if words else 0
        in_context = [match for match in matched if match["cue_id"] in record.context.cue_ids]
        context_coverage = len(in_context) / len(matched) if matched else 0
        supported = len(complete) >= 4 and coverage >= 0.8 and context_coverage >= 0.8
        evidence = {"id": record.id, "source_id": record.source_id, "clip_sha256": record.sha256, "source_timeline": clip["source"]["source_timeline"], "start_seconds": start, "end_seconds": end, "asr_run_id": run["id"], "asr_artifact_sha256": run["alignment"]["asr_artifact_sha256"], "publisher_artifact_sha256": run["alignment"]["publisher_artifact_sha256"], "alignment_artifact_sha256": run["alignment_sha256"], "observed_asr_words": len(words), "fully_matched_asr_words": len(complete), "word_match_fraction": round(coverage, 6), "declared_context_match_fraction": round(context_coverage, 6), "matched_cue_ids": sorted({match["cue_id"] for match in matched}), "status": "asr_word_matches" if supported else "insufficient_word_support", "anchors": [{key: match[key] for key in ("asr_word_index", "cue_id", "source_start_seconds", "source_end_seconds", "publisher_cue_start_seconds")} for match in in_context], "speaker_association_reviewed": False, "human_listening_completed": False}
        evidence["context_sha256"] = context_digest(record.context)
        entries.append(evidence)
        updates[record.id] = supported
    report = {"schema_version": "motion-context-alignment.v1", "method": "recomputed_publisher_asr_token_sequence_matches", "thresholds": {"minimum_fully_matched_words": 4, "minimum_word_match_fraction": 0.8, "minimum_declared_context_match_fraction": 0.8}, "runs": [{"id": run["id"], "source_id": run["asr"]["source_id"], "engine": run["asr"]["engine"], "audio_sha256": run["asr"]["audio_sha256"], "asr_sha256": run["alignment"]["asr_artifact_sha256"], "publisher_sha256": run["alignment"]["publisher_artifact_sha256"], "alignment_sha256": run["alignment_sha256"], "token_coverage": run["alignment"]["token_coverage"], "cue_onset_offset_diagnostics": run["alignment"]["offset_summary"]} for run in runs], "entries": entries, "limitations": ["ASR word timings are estimates, not forced alignment or human listening.", "Publisher cues have interval-proxy timing. No constant publisher-to-media offset is assumed.", "Only transcript text/time context is linked; no geometry track is identified or associated with the audio speaker.", "The thresholds are authored quality gates, not learned behavioral labels.", "No raw audio or full transcript text is included in this report."]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    for record in manifest["clips"]:
        if record["id"] not in updates:
            continue
        context = record["context"]
        context.update(alignment="asr_word_matches" if updates[record["id"]] else "unverified_nearby_context", alignment_evidence_id=record["id"], alignment_evidence_path=destination.relative_to(root).as_posix(), alignment_evidence_sha256=digest)
        context["alignment_note"] = "Exact ASR and publisher artifacts were hash-checked and their token matches recomputed against this acquired video clock. The linked report gives within-window word/context coverage. Timings are ASR estimates, not forced alignment or a constant cue offset; image-to-speaker association and human listening remain unreviewed."
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report
