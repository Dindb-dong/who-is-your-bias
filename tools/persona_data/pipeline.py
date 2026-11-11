"""Build reproducible local corpora without publishing complete source wording."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import unicodedata
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bias.corpus import Corpus
from bias.models import Evidence
from tools.research.collect import collect_source

from .parsers import PARSER_VERSION, digest, parse, sha_text


PROJECTION_VERSION = "persona-projections.v1"
TOPICS = {
    "artificial intelligence": ("artificial intelligence", " ai ", "neural network", "grok"),
    "robots": ("robot", "optimus"), "space": ("space", "rocket", "starship", "orbit"),
    "mars": ("mars", "multiplanet", "multi-planet"), "energy": ("energy", "solar", "battery", "electricity"),
    "electric cars": ("tesla", "electric car", "vehicle", "model 3", "model s"),
    "manufacturing": ("manufactur", "factory", "production"), "engineering": ("engineering", "design", "physics"),
    "learning": ("learn", "school", "university", "education", "reading", "read books"),
    "work": ("working", "work ethic", "work hours", "career", "hard work"),
    "risk": ("risk", "failure", "fail", "fear"), "motivation": ("motivat", "inspir", "purpose"),
    "neuralink": ("neuralink", "brain", "consciousness"), "tunnels": ("tunnel", "boring company", "traffic"),
    "social media": ("twitter", "tweet", "social media"), "government": ("government", "regulat", "politic", "doge"),
    "economics": ("cost", "price", "profit", "econom", "money"), "humanity": ("humanity", "civilization", "future of"),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def load_registry(directory: Path) -> tuple[dict, dict]:
    registry = read_json(directory / "source-families.json")
    registration = read_json(directory / "split-registration.json")
    if registry.get("schema_version") != "persona-source-families.v1" or registration.get("schema_version") != "persona-split-registration.v1":
        raise ValueError("Unknown source/split registration version")
    sources = registry["sources"]
    source_ids = [row["id"] for row in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate source ID")
    assignments = {}
    family_splits = {}
    for event in registration["events"]:
        family = event["event_family_id"]
        if event["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        if family in family_splits and family_splits[family] != event["split"]:
            raise ValueError("Recording family crosses data splits")
        family_splits[family] = event["split"]
        for identifier in event["source_ids"]:
            if identifier in assignments:
                raise ValueError("Duplicate source assignment")
            assignments[identifier] = (event["source_event_id"], family, event["split"])
    for source in sources:
        if assignments.get(source["id"]) != (source["source_event_id"], source["event_family_id"], source["split"]):
            raise ValueError("Source does not match frozen family/split registration")
        if source.get("prior_exposure", {}).get("original_qa_split") in {"validation", "test"} and source["split"] != "validation":
            raise ValueError("Exposed pilot evaluation events must remain development-only")
    return registry, registration


def acquire_sources(registry_dir: Path, output: Path, identifiers: list[str]) -> dict:
    registry, registration = load_registry(registry_dir)
    by_id = {row["id"]: row for row in registry["sources"]}
    if not identifiers or len(identifiers) != len(set(identifiers)) or any(key not in by_id for key in identifiers):
        raise ValueError("Choose unique registered source IDs")
    if any(by_id[key]["split"] == "test" for key in identifiers):
        raise ValueError("Reserved test events cannot enter acquisition/training through this command")
    if output.exists():
        raise ValueError("Acquisition output exists; preserve the earlier snapshot")
    output.mkdir(parents=True)
    attempts = []
    for identifier in identifiers:
        try:
            record = collect_source(by_id[identifier], output / "raw")
            attempts.append({"source_id": identifier, "status": "acquired", "record": record})
        except Exception as exc:
            # A per-source failure is part of the report; no substitute source.
            attempts.append({"source_id": identifier, "status": "failed", "error": str(exc)})
        write_json(output / "acquisition-attempts.json", attempts)
    return {"registered_split_sha256": digest(registration), "attempts": attempts}


def _normal(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _topics(question: str, answer: str) -> list[str]:
    value = " " + (question + " " + answer).lower() + " "
    return [topic for topic, markers in TOPICS.items() if any(marker in value for marker in markers)] or ["conversation"]


def runtime_source(source: dict, transcript: dict, *, acquired_at: str) -> dict:
    """Project inventory metadata into the truthful local-originals contract."""
    result = dict(source)
    result["summary_inventory_curation"] = source.get("curation")
    result["curation"] = "original_publisher_qa_structural_agent_review"
    result["raw_material_included"] = True
    result["original_text_included"] = True
    result["raw_recordings_included"] = False
    result["distribution"] = "local_selected_original_text_only"
    result["reviewed_at"] = acquired_at[:10]
    result["ingested_at"] = acquired_at
    result["transcript_status"] = "structurally_checked_original_publisher_dialogue"
    result["timeline"] = transcript["source_timeline"]
    result["source_artifact_sha256"] = transcript["input_sha256"]
    result["transcript_sha256"] = transcript["transcript_sha256"]
    if transcript.get("source_scope"):
        result["summary_inventory_attribution"] = source.get("attribution")
        result["attribution"] = transcript["source_scope"]
        result["source_type"] = "congressional_hearing_transcript"
        result["transcript_adapter_version"] = transcript["adapter_version"]
        result["transcript_adapter_sha256"] = transcript["adapter_sha256"]
    if source["id"] == "dwarkesh-2026":
        result["recorded_at"] = None
        result["date_provenance"] = (
            "Inspected publisher HTML has datePublished 2026-02-05T16:45:08+00:00 and a February 5 byline. "
            "No recording date is stated; video upload/create timestamps are not treated as recording dates. "
            "Later post updated_at metadata is not a new interview or current-position endorsement."
        )
    return result


def extract_exchanges(source: dict, transcript: dict, *, acquired_at: str) -> tuple[list[dict], list[dict]]:
    turns = transcript["turns"]
    accepted, rejected = [], []
    for index in range(len(turns) - 1):
        question, answer = turns[index:index+2]
        if question["speaker"] in {None, "Elon Musk"} or answer["speaker"] != "Elon Musk":
            continue
        identifier = f"qa26-{source['id']}-{question['id']}"
        qtext, atext = question["text"], answer["text"]
        if question.get("question_eligible") is False:
            rejected.append({"id": identifier, "reason": "not_direct_committee_question",
                             "question_sha256": sha_text(qtext), "answer_sha256": sha_text(atext)})
            continue
        reason = None
        if not qtext or not atext:
            reason = "empty_text"
        elif len(qtext) > 6000 or len(atext) > 12000:
            reason = "whole_exchange_exceeds_runtime_contract"
        elif len(question["cue_ids"]) > 300 or len(answer["cue_ids"]) > 500:
            reason = "whole_exchange_exceeds_cue_contract"
        if reason:
            rejected.append({"id": identifier, "reason": reason, "question_sha256": sha_text(qtext), "answer_sha256": sha_text(atext)})
            continue
        following = turns[index+2] if index+2 < len(turns) else None
        uncertain_end = (following is not None and following["speaker"] is None
                         and following["speaker_basis"] not in {"chapter_heading"})
        interrupted = answer.get("response_complete") is False
        completeness = "excerpt" if uncertain_end or interrupted else "complete_exchange"
        context_turn = turns[index-1] if index else None
        context = context_turn["text"] if context_turn and context_turn["speaker"] and len(context_turn["text"]) <= 3000 else ""
        timeline = transcript["source_timeline"]
        qstart = question.get("start") if timeline else None
        astart = answer.get("start") if timeline else None
        if qstart is not None and astart is not None and astart < qstart:
            rejected.append({"id": identifier, "reason": "nonmonotonic_publisher_time"})
            continue
        has_time = timeline is not None and (qstart is not None or astart is not None)
        review_basis = (
            f"Agent checked the publisher's speaker structure and adapter output, then applied the fixed source-wide extraction rule. "
            f"Question attribution: {question['speaker_basis']}; answer: {answer['speaker_basis']}. "
            "This is structural publisher-text review, not independent human, acoustic, factual-entailment or personality review. "
            "Consecutive attributed turns retain original wording; unknown blocks are never reassigned."
        )
        qa = {"question": qtext, "answer": atext, "context_before": context, "context_kind": "original_text" if context else "none",
              "original_language": "en", "questioner": question["speaker"], "respondent": "Elon Musk",
              "source_event_id": source["source_event_id"], "completeness": completeness,
              "excerpt_note": ("The printed response ends at an interruption; the attributed turn is preserved, but later qualifications are not assumed complete."
                               if interrupted else "An adjacent block has unknown attribution; this is only the contiguous attributed part of the published response, not a proven complete response."
                               if uncertain_end else None),
              "question_start_seconds": qstart, "answer_start_seconds": astart,
              "question_end_seconds": None, "answer_end_seconds": None,
              "question_end_proxy_seconds": None, "answer_end_proxy_seconds": None,
              "source_timeline": timeline if has_time else None,
              "timing_basis": "publisher_transcript_timestamps_only" if has_time else "unavailable",
              "transcript_locator": {"question_cue_ids": question["cue_ids"], "answer_cue_ids": answer["cue_ids"],
                                     "context_cue_ids": context_turn["cue_ids"] if context else [], "transcript_sha256": transcript["transcript_sha256"]},
              "review": {"reviewer_type": "agent", "reviewer": "OpenAI coding agent; publisher-structure review",
                         "reviewed_at": acquired_at[:10], "basis": review_basis,
                         "speaker_status": "publisher_labels_checked", "timing_status": "not_audio_verified"},
              "annotation_basis": "Generic conversation label only; topical tags are deterministic lexical hints, not reviewed situation or emotional inferences.",
              "taxonomy_version": "situations.v1"}
        evidence = Evidence.model_validate({"id": identifier, "source_id": source["id"], "kind": "question_answer", "question_answer": qa,
                    "topics": _topics(qtext, atext), "situations": ["conversation"], "review_status": "reviewed", "confidence": "source_grounded",
                    "locator": {"label": f"Original publisher turns {question['id']} / {answer['id']}; {completeness}; unverified acoustic timing.",
                                "start_seconds": astart if has_time else None, "timeline": timeline if has_time else None},
                    "event_family_id": source["event_family_id"], "event_at": source.get("recorded_at"),
                    "published_at": source["published_at"], "ingested_at": acquired_at,
                    "source_artifact_sha256": transcript["input_sha256"],
                    "transformation_history": [PARSER_VERSION, *([transcript["adapter_version"]] if transcript.get("adapter_version") else []), "adjacent-attributed-turns.v1"]})
        accepted.append(evidence.model_dump(mode="json", exclude={"text"}))
    return accepted, rejected


def _media_eligibility(record: dict, entry: dict | None, modality: str, asset_root: Path) -> tuple[bool, str]:
    if entry is None:
        return False, f"no_reviewed_{modality}_alignment"
    qa = record["question_answer"]
    if entry.get("record_id") != record["id"] or entry.get("answer_sha256") != sha_text(qa["answer"]):
        return False, "media_text_identity_mismatch"
    if entry.get("source_id") != record["source_id"] or not entry.get("source_timeline"):
        return False, "media_source_clock_missing_or_mismatched"
    if (entry.get("transcript_sha256") != qa["transcript_locator"]["transcript_sha256"]
            or entry.get("source_artifact_sha256") != record["source_artifact_sha256"]
            or not entry.get("recording_id") or entry.get("source_clock_mapping_reviewed") is not True):
        return False, "media_source_mapping_unreviewed"
    if (entry.get("speaker") != "Elon Musk" or entry.get("review_status") != "accepted" or not entry.get("reviewer")
            or entry.get("reviewer_type") not in {"agent", "human"} or not entry.get("reviewed_at")):
        return False, "target_speaker_or_review_unaccepted"
    start, end = entry.get("start_seconds"), entry.get("end_seconds")
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float))
            or not isinstance(end, (int, float)) or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end):
        return False, "invalid_reviewed_interval"
    expected_basis = "acoustically_reviewed" if modality == "speech" else "visually_reviewed_continuous_track"
    if entry.get("alignment_basis") != expected_basis:
        return False, "unreviewed_alignment_basis"
    path = (asset_root / entry.get("file", "")).resolve()
    if not path.is_relative_to(asset_root.resolve()) or not path.is_file():
        return False, "missing_or_outside_media_asset"
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
        return False, "media_hash_mismatch"
    if entry.get("overlap") is not False or entry.get("cut") is not False:
        return False, "overlap_or_cut_not_cleared"
    if modality == "speech":
        try:
            with wave.open(str(path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
                if end > duration + 1 / audio.getframerate():
                    return False, "audio_interval_outside_asset"
        except (wave.Error, OSError, EOFError):
            return False, "invalid_pcm_wave"
        if entry.get("transcript_exact") is not True or entry.get("noise_accepted") is not True:
            return False, "speech_quality_unaccepted"
    else:
        if entry.get("face_visible") is not True or not isinstance(entry.get("visible_channels"), list) or not entry["visible_channels"]:
            return False, "visual_visibility_unreviewed"
        try:
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                                   capture_output=True, check=True, timeout=15)
            duration = float(json.loads(probe.stdout)["format"]["duration"])
            if not math.isfinite(duration) or end > duration:
                return False, "visual_interval_outside_asset"
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            return False, "invalid_video_asset"
    return True, "eligible"


def project_records(records: list[dict], sources: dict[str, dict], *, media_reviews: dict | None = None, asset_root: Path = Path(".")) -> tuple[dict[str, list], list[dict]]:
    projections = {key: [] for key in ("factual", "text_style", "aligned_speech", "visible_behavior")}
    decisions = []
    reviews = media_reviews or {}
    for record in records:
        source = sources[record["source_id"]]
        qa = record["question_answer"]
        training = source["split"] == "train"
        accepted = record["review_status"] == "reviewed" and qa["review"]["speaker_status"] != "unreviewed"
        complete = qa["completeness"] == "complete_exchange"
        factual = training and accepted and len(qa["answer"].split()) >= 8
        style = training and accepted and complete
        speech, speech_reason = _media_eligibility(record, reviews.get(record["id"], {}).get("speech"), "speech", asset_root)
        visual, visual_reason = _media_eligibility(record, reviews.get(record["id"], {}).get("visual"), "visual", asset_root)
        eligibility = {
            "factual": {"eligible": factual, "reason": "eligible" if factual else "held_out_or_unreviewed_or_short_response"},
            "text_style": {"eligible": style, "reason": "eligible" if style else "held_out_or_unreviewed_or_incomplete_exchange"},
            "aligned_speech": {"eligible": training and accepted and complete and speech, "reason": speech_reason if training and accepted and complete else "held_out_or_unreviewed_or_incomplete_exchange"},
            "visible_behavior": {"eligible": training and accepted and visual, "reason": visual_reason if training and accepted else "held_out_or_unreviewed"},
        }
        common = {"segment_id": record["id"], "source_id": record["source_id"], "event_family_id": source["event_family_id"],
                  "split": source["split"], "source_url": source["url"], "source_artifact_sha256": record["source_artifact_sha256"],
                  "transcript_sha256": qa["transcript_locator"]["transcript_sha256"], "answer_sha256": sha_text(qa["answer"]),
                  "question": qa["question"], "text": qa["answer"], "context": qa["context_before"], "language": "en",
                  "completeness": qa["completeness"], "published_at": source["published_at"], "ingested_at": record["ingested_at"]}
        for key, decision in eligibility.items():
            if decision["eligible"]:
                entry = dict(common)
                if key == "aligned_speech":
                    entry["media"] = reviews[record["id"]]["speech"]
                elif key == "visible_behavior":
                    entry["media"] = reviews[record["id"]]["visual"]
                projections[key].append(entry)
        decisions.append({"segment_id": record["id"], "eligibility": eligibility})
    return projections, decisions


def bind_projection_eligibility(records: list[dict], decisions: list[dict]) -> list[dict]:
    """Keep independently accepted roles attached to each runtime evidence row."""
    by_id = {row["segment_id"]: row["eligibility"] for row in decisions}
    if len(by_id) != len(decisions) or set(by_id) != {row["id"] for row in records}:
        raise ValueError("Projection decisions must cover each runtime record exactly once")
    keys = {"factual", "text_style", "aligned_speech", "visible_behavior"}
    projected = []
    for record in records:
        decision = by_id[record["id"]]
        if set(decision) != keys or any(type(value.get("eligible")) is not bool for value in decision.values()):
            raise ValueError("Each runtime projection requires an explicit eligibility boolean")
        projected.append({**record, "projection_version": PROJECTION_VERSION,
                          "projection_eligibility": {key: value["eligible"] for key, value in decision.items()}})
    return projected


def build(registry_dir: Path, acquisition_dir: Path, output: Path, *, report_dir: Path, recipe_path: Path | None = None,
          media_review_path: Path | None = None, asset_root: Path = Path(".")) -> dict:
    registry, registration = load_registry(registry_dir)
    if output.exists() or output.is_symlink():
        raise ValueError("Build output exists; choose a new path to preserve its snapshot")
    sources = {row["id"]: row for row in registry["sources"]}
    attempts = read_json(acquisition_dir / "acquisition-attempts.json")
    pinned = read_json(recipe_path) if recipe_path else None
    if pinned and pinned.get("split_registration_sha256") != digest(registration):
        raise ValueError("Pinned materialization belongs to another split registration")
    pinned_sources = {row["source_id"]: row for row in pinned.get("sources", [])} if pinned else {}
    records, rejected, source_reports, canonical = [], [], [], {}
    materialized_sources = {}
    normalized = {}
    for attempt in attempts:
        identifier = attempt["source_id"]
        if identifier not in sources:
            raise ValueError("Unregistered source in acquisition log")
        source = sources[identifier]
        if source["split"] == "test":
            raise ValueError("Reserved test data cannot enter a runtime/training build")
        if attempt["status"] != "acquired":
            source_reports.append({"source_id": identifier, "status": "acquisition_failed", "error": attempt.get("error")})
            continue
        record = attempt["record"]
        artifact = (acquisition_dir / "raw" / identifier / record["artifact"]).resolve()
        if not artifact.is_relative_to(acquisition_dir.resolve()) or hashlib.sha256(artifact.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Acquired source hash/path mismatch")
        try:
            transcript = parse(source, artifact)
        except (ValueError, OSError) as exc:
            source_reports.append({"source_id": identifier, "status": "not_original_qa", "reason": str(exc), "artifact_sha256": record["sha256"]})
            continue
        if pinned and pinned_sources.get(identifier, {}).get("transcript_sha256") != transcript["transcript_sha256"]:
            raise ValueError(f"Publisher transcript changed for {identifier}; make a separately reviewed recipe")
        extracted, failures = extract_exchanges(source, transcript, acquired_at=record["retrieved_at"])
        rejected.extend(failures)
        included = []
        for item in extracted:
            qa = item["question_answer"]
            key = _normal(qa["question"] + "\n" + qa["answer"])
            if key in normalized:
                rejected.append({"id": item["id"], "reason": "exact_normalized_exchange_duplicate", "duplicate_of": normalized[key]})
                continue
            normalized[key] = item["id"]
            included.append(item)
        records.extend(included)
        canonical[identifier] = transcript
        materialized_sources[identifier] = runtime_source(source, transcript, acquired_at=record["retrieved_at"])
        source_reports.append({"source_id": identifier, "event_family_id": source["event_family_id"], "status": "extracted",
                               "source_url": source["url"], "published_at": source["published_at"], "acquired_at": record["retrieved_at"],
                               "artifact_sha256": record["sha256"], "transcript_sha256": transcript["transcript_sha256"],
                               "attributed_turns": sum(t["speaker"] is not None for t in transcript["turns"]),
                               "candidate_exchanges": len(extracted) + len(failures), "accepted_exchanges": len(included),
                               "target_speaker_words": sum(len(t["text"].split()) for t in transcript["turns"] if t["speaker"] == "Elon Musk")})
        if transcript.get("adapter_version"):
            source_reports[-1].update(adapter_version=transcript["adapter_version"], adapter_sha256=transcript["adapter_sha256"],
                                      extraction_audit=transcript["extraction_audit"])
    if not records:
        raise ValueError("No original exchanges passed extraction")
    # Exact answer overlap across splits can leak held-out wording even if
    # questions differ. Trivial acknowledgments are retained with their context.
    long_answers = {}
    for record in records:
        answer = _normal(record["question_answer"]["answer"])
        split = sources[record["source_id"]]["split"]
        if len(answer.split()) >= 10 and answer in long_answers and long_answers[answer] != split:
            raise ValueError("Exact original answer appears across splits")
        long_answers[answer] = split
    reviews = read_json(media_review_path) if media_review_path else {}
    projections, decisions = project_records(records, sources, media_reviews=reviews, asset_root=asset_root)
    records = bind_projection_eligibility(records, decisions)
    recipe = {"schema_version": "persona-materialization.v1", "parser_version": PARSER_VERSION,
              "hash_definition": "Registry, split, recipe and non-TED transcript digests use sorted compact UTF-8 JSON with ensure_ascii=False; source artifacts and original text use exact bytes.",
              "split_registration_sha256": digest(registration), "source_registry_sha256": digest(registry),
              "sources": [row for row in source_reports if row["status"] == "extracted"],
              "selections": [{"id": row["id"], "source_id": row["source_id"], "question_sha256": sha_text(row["question_answer"]["question"]),
                              "answer_sha256": sha_text(row["question_answer"]["answer"]), "completeness": row["question_answer"]["completeness"],
                              "transcript_locator": row["question_answer"]["transcript_locator"]} for row in records],
              "original_wording_distributed": False}
    if pinned and pinned.get("selections") != recipe["selections"]:
        raise ValueError("Extracted original selections differ from pinned recipe")
    used_ids = sorted({row["source_id"] for row in records})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".persona-build-", dir=output.parent))
    try:
        runtime = temporary / "corpus"
        runtime.mkdir()
        write_json(runtime / "sources.json", [materialized_sources[key] for key in used_ids])
        write_json(runtime / "split-manifest.json", registration)
        profile = read_json(Path("data/qna/elon-musk-v1/profile.json"))
        profile["qa_corpus_id"] = "elon-musk-original-2026.v1"
        write_json(runtime / "profile.json", profile)
        write_jsonl(runtime / "evidence.jsonl", records)
        manifest = {"schema_version": "original-qa-corpus.v1", "id": profile["qa_corpus_id"], "materialization": "full_selected_exchanges_local",
                    "selection_recipe_sha256": digest(recipe), "split_manifest_sha256": digest(registration),
                    "record_count": len(records), "source_content_sha256": {key: canonical[key]["transcript_sha256"] for key in canonical},
                    "projection_version": PROJECTION_VERSION, "projection_counts": {key: len(rows) for key, rows in projections.items()},
                    "projection_decisions_sha256": digest(decisions),
                    "raw_transcripts_distributed": False, "review_note": "Agent-reviewed publisher structure and deterministic extraction; human and acoustic ratings remain unperformed."}
        write_json(runtime / "corpus-manifest.json", manifest)
        corpus = Corpus.load(runtime)
        for key, rows in projections.items():
            write_jsonl(temporary / "projections" / f"{key}.jsonl", rows)
        write_jsonl(temporary / "segments.jsonl", records)
        write_json(temporary / "projection-decisions.json", decisions)
        for key, transcript in canonical.items():
            write_json(temporary / "canonical" / f"{key}.json", transcript)
        report = {"schema_version": "persona-coverage.v1", "generated_at": datetime.now(timezone.utc).isoformat(),
                  "corpus": corpus.qa_report(), "source_events": len(used_ids), "registered_events": len(sources),
                  "fresh_test_events_reserved": sum(source["split"] == "test" for source in sources.values()),
                  "fresh_test_exchanges_acquired": 0, "projections": {key: len(rows) for key, rows in projections.items()},
                  "accepted_speech_hours": 0 if not projections["aligned_speech"] else sum(row["media"]["end_seconds"]-row["media"]["start_seconds"] for row in projections["aligned_speech"])/3600,
                  "raw_recording_hours": None, "target_speaker_audio_hours": None,
                  "source_reports": source_reports, "rejection_counts": dict(Counter(row["reason"] for row in rejected)),
                  "rejected": rejected, "temporal_coverage": dict(Counter(sources[key]["published_at"][:4] for key in used_ids)),
                  "event_dominance": {key: sum(row["source_id"] == key for row in records) for key in used_ids},
                  "recipe_sha256": digest(recipe), "source_registry_sha256": digest(registry),
                  "limitations": ["Structural agent review is not human review or factual-entailment evaluation.",
                                 "Repeated/quoted semantic duplicates beyond exact normalized wording remain to be reviewed.",
                                 "Source text does not establish clean speech duration, 2026 appearance, trained identity or efficacy.",
                                 "New holdouts remain reserved and unacquired; this is a development data expansion."]}
        write_json(temporary / "coverage.json", report)
        temporary.rename(output)
        # Shareable files deliberately contain no original questions or answers.
        write_json(report_dir / "coverage.json", report)
        write_json(report_dir / "materialization-recipes.json", recipe)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
