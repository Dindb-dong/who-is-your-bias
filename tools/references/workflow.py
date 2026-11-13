"""Bind candidate voice references to explicit text, timing, and review grounds."""

from __future__ import annotations

import array
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
import unicodedata
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from tools.media.common import MediaError, artifact_path, finite_number, load_manifest, local_file, sha256, timestamp, verify_artifact, write_json
from tools.media.prepare import prepare_clip

CHECKS = ("publisher_attribution", "transcript_text", "clip_timing", "single_speaker", "audio_quality")
LABEL_BASES = {"explicit_publisher_label", "same_paragraph_label_continuation"}


def read_json(path: str | Path) -> dict:
    file = local_file(path, max_bytes=16 * 1024 * 1024)
    def pairs(items):
        data = {}
        for key, value in items:
            if key in data:
                raise MediaError("Duplicate JSON key")
            data[key] = value
        return data
    def nonfinite(value):
        raise MediaError("Nonfinite JSON value")
    try:
        data = json.loads(file.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise MediaError("Expected a UTF-8 JSON document") from exc
    if not isinstance(data, dict):
        raise MediaError("Expected a JSON object")
    return data


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text, re.UNICODE)


def publisher_words(publisher: dict) -> list[dict]:
    if publisher.get("schema_version") != "publisher_transcript.v1":
        raise MediaError("Use a canonical publisher_transcript.v1 artifact")
    segments = publisher.get("segments")
    if not isinstance(segments, list) or not 1 <= len(segments) <= 30_000:
        raise MediaError("Publisher caption segments are missing or too numerous")
    result = []
    identifiers = set()
    for cue in segments:
        if not isinstance(cue, dict) or cue.get("id") in identifiers:
            raise MediaError("Publisher cue IDs must be unique")
        identifiers.add(cue.get("id"))
        if publisher.get("timing_kind") == "untimed_text":
            if cue.get("start") is not None or publisher.get("source_timeline") is not None:
                raise MediaError("Untimed publisher text cannot declare numeric cue times or a media clock")
            onset = None
        else:
            onset = finite_number(cue.get("start"), "publisher cue onset", 0, 86400)
        spans = cue.get("speaker_spans")
        if not isinstance(spans, list):
            raise MediaError("Publisher text must retain explicit speaker spans")
        index = 0
        for span in spans:
            if span.get("kind") != "caption_text":
                continue
            text = span.get("text")
            if not isinstance(text, str):
                raise MediaError("Publisher span text is missing")
            for token in tokens(text):
                result.append({"token": token, "cue_id": cue["id"], "cue_word_index": index,
                    "publisher_start": onset, "speaker": cue.get("speaker"),
                    "speaker_basis": cue.get("speaker_basis"), "span_speaker": span.get("speaker")})
                index += 1
    if not result:
        raise MediaError("No publisher text tokens found")
    return result


def asr_words(asr: dict) -> list[dict]:
    if asr.get("schema_version") != "transcript.v1":
        raise MediaError("Use a transcript.v1 ASR artifact")
    offset = finite_number(asr.get("source_start_seconds"), "ASR source offset")
    result = []
    last_start = -1.0
    for segment in asr.get("segments", []):
        for word in segment.get("words", []):
            start = finite_number(word.get("start"), "ASR word start", 0, 120.1)
            end = finite_number(word.get("end"), "ASR word end", 0, 120.1)
            if end < start or start < last_start:
                raise MediaError("ASR word times must be ordered and nonnegative")
            last_start = start
            text = word.get("word")
            if not isinstance(text, str) or not text.strip():
                continue
            probability = word.get("confidence", word.get("probability"))
            if probability is not None:
                probability = finite_number(probability, "ASR word confidence", 0, 1)
            result.append({"word": text, "tokens": tokens(text), "start": start, "end": end,
                "source_start": offset + start, "source_end": offset + end, "confidence": probability})
    if not result:
        raise MediaError("ASR word timestamps are required for reference selection")
    if len(result) > 10_000:
        raise MediaError("ASR context contains too many words")
    return result


def align_transcripts(publisher: dict, asr: dict) -> dict:
    if publisher.get("source_id") != asr.get("source_id"):
        raise MediaError("Publisher and ASR source IDs differ")
    pub = publisher_words(publisher)
    words = asr_words(asr)
    observed = [{"token": token, "word_index": i} for i, word in enumerate(words) for token in word["tokens"]]
    matcher = SequenceMatcher(None, [item["token"] for item in observed], [item["token"] for item in pub], autojunk=False)
    matches = []
    anchors = []
    for block in matcher.get_matching_blocks():
        for delta in range(block.size):
            ai, pi = block.a + delta, block.b + delta
            token = observed[ai]
            word = words[token["word_index"]]
            source = pub[pi]
            match = {"asr_token_index": ai, "publisher_token_index": pi, "asr_word_index": token["word_index"],
                "cue_id": source["cue_id"], "speaker": source["speaker"],
                "speaker_basis": source["speaker_basis"], "source_start_seconds": word["source_start"],
                "source_end_seconds": word["source_end"], "publisher_cue_start_seconds": source["publisher_start"]}
            matches.append(match)
            if source["cue_word_index"] == 0 and source["publisher_start"] is not None:
                anchors.append({"cue_id": source["cue_id"], "native_word_start_seconds": word["source_start"],
                    "publisher_cue_start_seconds": source["publisher_start"],
                    "native_minus_publisher_seconds": round(word["source_start"] - source["publisher_start"], 4)})
    offsets = [anchor["native_minus_publisher_seconds"] for anchor in anchors]
    return {
        "schema_version": "publisher_asr_alignment.v1", "source_id": publisher["source_id"],
        "publisher_timeline": publisher.get("source_timeline"), "media_timeline": asr.get("source_timeline"),
        "asr_tokens": len(observed), "matched_tokens": len(matches),
        "token_coverage": round(len(matches) / len(observed), 6), "matches": matches, "anchors": anchors,
        "offset_summary": {"median_seconds": statistics.median(offsets), "minimum_seconds": min(offsets), "maximum_seconds": max(offsets)} if offsets else None,
        "scope": "Observed word matches in this context only. Caption onsets and ASR word onsets have different uncertainty; no global clock offset or forced alignment is established.",
        "speaker_method": "Publisher text labels and matched words/context, never voice or face resemblance.",
        "review_status": "candidate",
    }


def wave_metrics(path: Path) -> dict:
    with wave.open(str(path), "rb") as stream:
        channels, rate, width, count = stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getnframes()
        if channels != 1 or width != 2 or stream.getcomptype() != "NONE":
            raise MediaError("Reference WAV must be mono PCM16")
        raw = stream.readframes(count)
        if len(raw) != count * width:
            raise MediaError("Reference WAV is truncated")
    values = array.array("h", raw)
    if sys.byteorder != "little":
        values.byteswap()
    if not values:
        raise MediaError("Reference WAV is empty")
    rms = math.sqrt(sum((value / 32768) ** 2 for value in values) / len(values))
    peak = max(abs(value) for value in values) / 32768
    return {"duration_seconds": count / rate, "sample_rate": rate, "channels": channels,
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 3), "peak_dbfs": round(20 * math.log10(max(peak, 1e-12)), 3),
        "clipped_sample_fraction": sum(abs(value) >= 32767 for value in values) / len(values),
        "note": "Signal levels do not establish speaker identity, lack of music/noise, or subjective reference quality."}


def _artifact(path: Path, base: Path) -> dict:
    return {"path": path.relative_to(base).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}


def _fingerprint(manifest: dict) -> str:
    return digest_json({key: value for key, value in manifest.items() if key not in {"review", "review_ledger"}})


def _ledger_entry(previous: str | None, sequence: int, fingerprint: str, actor: str, decision: str, checks: dict, notes: str, listening: bool = False) -> dict:
    entry = {"sequence": sequence, "created_at": timestamp(), "previous_sha256": previous,
        "bundle_fingerprint": fingerprint, "actor": actor, "decision": decision, "checks": checks,
        "listening_review_completed": listening, "notes": notes}
    return dict(entry, sha256=digest_json(entry))


def prepare_reference(
    media_manifest: str | Path, publisher_path: str | Path, asr_path: str | Path,
    source_media: str | Path, output_directory: str | Path, *, speaker: str,
    cue_ids: list[str], start_seconds: float, duration_seconds: float,
) -> dict:
    """Prepare a candidate; this operation never declares human/audio review."""
    start = finite_number(start_seconds, "reference source start")
    duration = finite_number(duration_seconds, "reference duration", 8, 15)
    end = start + duration
    media_path, media = load_manifest(media_manifest)
    verify_artifact(media_path, media, "audio")
    original = local_file(source_media)
    if sha256(original) != media.get("input", {}).get("sha256"):
        raise MediaError("Source media does not match the ASR context preparation input hash")
    pub_file, asr_file = local_file(publisher_path), local_file(asr_path)
    publisher, asr = read_json(pub_file), read_json(asr_file)
    if asr.get("source_id") != media["source_id"] or asr.get("source_timeline") != media["source_timeline"]:
        raise MediaError("ASR source identity or clock does not match the context media")
    if asr.get("audio_sha256") != media["artifacts"]["audio"]["sha256"]:
        raise MediaError("ASR was not computed from the verified context audio")
    if abs(finite_number(asr.get("source_start_seconds"), "ASR offset") - media["source_start_seconds"]) > 0.001:
        raise MediaError("ASR and prepared context offsets differ")
    if start < media["source_start_seconds"] or end > media["source_start_seconds"] + media["clip"]["duration_seconds"]:
        raise MediaError("Reference selection is outside the ASR context")
    if not speaker or not isinstance(speaker, str) or len(speaker) > 200:
        raise MediaError("Supply the explicit publisher speaker label")
    if not cue_ids or len(cue_ids) != len(set(cue_ids)):
        raise MediaError("Choose distinct publisher cue IDs")
    selected = [cue for cue in publisher.get("segments", []) if cue.get("id") in cue_ids]
    if len(selected) != len(cue_ids) or [cue["id"] for cue in selected] != cue_ids:
        raise MediaError("Selected cue IDs must exist in publisher source order")
    if any(cue.get("speaker") != speaker or cue.get("speaker_basis") not in LABEL_BASES for cue in selected):
        raise MediaError("Every selected cue must have the same explicit or paragraph-supported publisher speaker; unknown/mixed cues cannot establish attribution")
    all_ids = [cue["id"] for cue in publisher["segments"]]
    first, last = all_ids.index(cue_ids[0]), all_ids.index(cue_ids[-1])
    if all_ids[first:last + 1] != cue_ids:
        raise MediaError("Selected publisher cues must be consecutive")
    words = asr_words(asr)
    # Adding a duration can move an exact adjacent-word boundary by a few
    # floating-point ULPs (601.22 + 9.94 -> 611.1600000000001). Such rounding
    # must not pull the following complete word into an otherwise valid clip.
    chosen = [word for word in words if word["source_start"] < end - 1e-7 and word["source_end"] > start + 1e-7]
    if not chosen or any(word["source_start"] < start - 0.001 or word["source_end"] > end + 0.001 for word in chosen):
        raise MediaError("Reference boundaries cut an ASR word; choose boundaries outside observed words")
    text = " ".join(word["word"].strip() for word in chosen)
    if not text or len(text) > 4000:
        raise MediaError("Reference transcript must have 1–4000 characters")
    expected_tokens = [item["token"] for item in publisher_words(publisher) if item["cue_id"] in cue_ids]
    observed_tokens = tokens(text)
    alignment = align_transcripts(publisher, asr)
    if len(observed_tokens) < 10:
        raise MediaError("Reference needs at least 10 observed words for contextual matching")
    target_matches = [item for item in alignment["matches"] if item["source_start_seconds"] >= start - 1e-7 and item["source_end_seconds"] <= end + 1e-7]
    attribution_agrees = bool(target_matches) and all(item["cue_id"] in cue_ids and item["speaker"] == speaker for item in target_matches)
    exact_agreement = observed_tokens == expected_tokens and len(target_matches) == len(observed_tokens)
    if not attribution_agrees:
        raise MediaError("ASR word alignment does not support the selected publisher attribution")
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise MediaError("Reference output directory already exists")
    input_offset = finite_number(media["input"].get("source_offset_seconds", 0), "original input source offset")
    prepared = prepare_clip(original, output, source_id=media["source_id"], source_url=media["source_url"],
        source_timeline=media["source_timeline"], start=start - input_offset, duration=duration,
        input_source_offset=input_offset, sample_rate=24000, include_video=False)
    try:
        (output / "reference.txt").write_text(text + "\n", encoding="utf-8")
        context = {"schema_version": "reference_publisher_context.v1", "source_id": publisher["source_id"],
            "source_url": publisher.get("source_url"), "source_timeline": publisher.get("source_timeline"),
            "full_transcript_sha256": sha256(pub_file), "publisher_caption_sha256": publisher.get("publisher_caption_sha256"),
            "selected_cue_ids": cue_ids, "segments": publisher["segments"][max(0, first - 3):last + 4]}
        write_json(output / "publisher-context.json", context)
        write_json(output / "asr-context.json", asr)
        write_json(output / "alignment.json", alignment)
        artifacts = {"audio": _artifact(output / "audio.wav", output), "reference_text": _artifact(output / "reference.txt", output),
            "publisher_context": _artifact(output / "publisher-context.json", output), "asr_context": _artifact(output / "asr-context.json", output),
            "alignment": _artifact(output / "alignment.json", output), "media_manifest": _artifact(output / "manifest.json", output)}
        manifest = {
            "schema_version": "voice_reference.v1", "created_at": timestamp(),
            "source_id": media["source_id"], "source_url": media["source_url"], "source_timeline": media["source_timeline"],
            "source_start_seconds": prepared["source_start_seconds"], "source_end_seconds": prepared["source_end_seconds"],
            "source_media_sha256": sha256(original), "clip": prepared["clip"], "artifacts": artifacts,
            "publisher_attribution": {"speaker_label": speaker, "cue_ids": cue_ids, "transcript_sha256": sha256(pub_file),
                "method": "Explicit publisher text labels plus actual ASR word/context matching", "biometric_identity_inference": False},
            "automatic_checks": {"exact_normalized_text_agreement": exact_agreement, "reference_token_count": len(observed_tokens),
                "matched_reference_tokens": len(target_matches), "word_boundaries_not_cut": True,
                "minimum_asr_word_confidence": min((word["confidence"] for word in chosen if word["confidence"] is not None), default=None)},
            "audio_metrics": wave_metrics(output / "audio.wav"),
            "limitations": ["Automatic matching is not a listening review.", "No voice or face resemblance was used for attribution.",
                "ASR word boundaries are estimates; publisher subtitle clocks are not interchangeable with native media clocks.",
                "A reference review is separate from permission to distribute a voice or likeness asset."],
        }
        checks = {name: {"status": "unknown", "grounds": "Awaiting explicit review."} for name in CHECKS}
        entry = _ledger_entry(None, 1, _fingerprint(manifest), "preparation_tool", "candidate", checks, "Prepared from verified file hashes and publisher/ASR context; no automatic approval.")
        (output / "review-ledger.jsonl").write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["review"] = {"status": "candidate", "checks": checks, "listening_review_completed": False}
        manifest["review_ledger"] = {"path": "review-ledger.jsonl", "entries": 1, "head_sha256": entry["sha256"]}
        write_json(output / "reference.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(output)
        raise


def verify_reference(manifest_path: str | Path) -> dict:
    path = local_file(manifest_path, max_bytes=1024 * 1024)
    manifest = read_json(path)
    if manifest.get("schema_version") != "voice_reference.v1":
        raise MediaError("Expected a voice_reference.v1 manifest")
    for name, item in manifest.get("artifacts", {}).items():
        file = artifact_path(path, item.get("path", ""))
        if sha256(file) != item.get("sha256") or file.stat().st_size != item.get("bytes"):
            raise MediaError(f"Reference artifact changed: {name}")
    if not {"audio", "reference_text", "publisher_context", "asr_context", "alignment", "media_manifest"}.issubset(manifest.get("artifacts", {})):
        raise MediaError("Reference manifest is missing required artifacts")
    ledger = manifest.get("review_ledger", {})
    ledger_file = artifact_path(path, ledger.get("path", ""))
    if ledger_file.stat().st_size > 2 * 1024 * 1024:
        raise MediaError("Review ledger is too large")
    entries = []
    previous = None
    fingerprint = _fingerprint(manifest)
    for line in ledger_file.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        digest = entry.get("sha256")
        content = {key: value for key, value in entry.items() if key != "sha256"}
        if digest != digest_json(content) or entry.get("previous_sha256") != previous or entry.get("sequence") != len(entries) + 1 or entry.get("bundle_fingerprint") != fingerprint:
            raise MediaError("Review ledger hash chain or bundle binding is invalid")
        entries.append(entry)
        previous = digest
    if not entries or len(entries) != ledger.get("entries") or previous != ledger.get("head_sha256"):
        raise MediaError("Review ledger does not match its manifest")
    latest = entries[-1]
    review = manifest.get("review", {})
    if (review.get("status") != latest.get("decision") or review.get("checks") != latest.get("checks")
        or review.get("listening_review_completed") != latest.get("listening_review_completed")
        or len(entries) > 1 and review.get("reviewer") != latest.get("actor")):
        raise MediaError("Review status differs from the bound ledger")
    if review.get("status") == "reviewed" and (
        not review.get("listening_review_completed") or any(latest["checks"].get(name, {}).get("status") != "verified" for name in CHECKS)
    ):
        raise MediaError("Reviewed reference lacks completed explicit checks")
    return manifest


def review_reference(manifest_path: str | Path, assessment: dict) -> dict:
    path = local_file(manifest_path)
    manifest = verify_reference(path)
    actor, notes = assessment.get("reviewer"), assessment.get("notes", "")
    if not isinstance(actor, str) or not actor.strip() or len(actor) > 200:
        raise MediaError("Name the reviewer; a tool must not imply a human reviewed it")
    if not isinstance(notes, str) or len(notes) > 4000:
        raise MediaError("Review notes must be at most 4000 characters")
    checks = assessment.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECKS):
        raise MediaError("Supply each attribution, text, timing, speaker, and audio-quality check")
    for name, check in checks.items():
        if not isinstance(check, dict) or check.get("status") not in {"verified", "unknown", "rejected"}:
            raise MediaError(f"Invalid review check: {name}")
        if not isinstance(check.get("grounds"), str) or not check["grounds"].strip() or len(check["grounds"]) > 4000:
            raise MediaError(f"Record concrete review grounds for {name}")
    listening = assessment.get("listening_review_completed")
    if not isinstance(listening, bool):
        raise MediaError("Explicitly state whether listening review was completed")
    statuses = [check["status"] for check in checks.values()]
    decision = "rejected" if "rejected" in statuses else "reviewed" if all(status == "verified" for status in statuses) and listening else "candidate"
    entry = _ledger_entry(manifest["review_ledger"]["head_sha256"], manifest["review_ledger"]["entries"] + 1,
        _fingerprint(manifest), actor.strip(), decision, checks, notes, listening)
    ledger_file = artifact_path(path, manifest["review_ledger"]["path"])
    with ledger_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    manifest["review"] = {"status": decision, "checks": checks, "listening_review_completed": listening, "reviewer": actor.strip()}
    manifest["review_ledger"].update(entries=entry["sequence"], head_sha256=entry["sha256"])
    write_json(path, manifest, overwrite=True)
    verify_reference(path)
    return manifest
