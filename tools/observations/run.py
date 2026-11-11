"""Build a reproducible, text-free observation report from local hashed media."""

from __future__ import annotations

import json
import platform
import resource
import time
from collections import Counter
from pathlib import Path

from bias.motion import validate_clip
from tools.media.common import MediaError, artifact_path, finite_number, load_manifest, local_file, sha256, verify_artifact, write_json
from tools.references.workflow import align_transcripts, digest_json, read_json

from . import VERSION
from .features import aligned_exchange, merge_speech_runs, visual_features


def within(root: Path, relative: str, *, max_bytes: int = 64 * 1024**2) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise MediaError("Observation artifact paths must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise MediaError("Observation artifact path escapes its supplied root")
    return local_file(path, max_bytes=max_bytes)


def _read_jsonl(path: Path, *, max_rows: int) -> list[dict]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(MediaError("Nonfinite JSON value")))
            if not isinstance(value, dict):
                raise MediaError("Expected JSON object rows")
            result.append(value)
            if len(result) > max_rows:
                raise MediaError("Too many observation rows")
    return result


def validate_splits(sources: list[dict], event_inventory: dict) -> dict[str, dict]:
    events: dict[str, dict] = {}
    source_ids: dict[str, str] = {}
    for event in event_inventory["events"]:
        event_id = event["source_event_id"]
        if event_id in events or event["split"] not in {"train", "validation", "test"}:
            raise MediaError("Source-event split inventory is ambiguous")
        events[event_id] = event
        for source in event["source_ids"]:
            if source in source_ids:
                raise MediaError("A source appears in multiple event assignments")
            source_ids[source] = event_id
    seen = set()
    observation_splits = {}
    for source in sources:
        if source["source_id"] in seen:
            raise MediaError("Duplicate observation source")
        seen.add(source["source_id"])
        event_id = source["source_event_id"]
        if source_ids.get(source["source_id"]) != event_id:
            raise MediaError("Observation source disagrees with the canonical event inventory")
        split = source["observation_split"]
        if split not in {"train", "validation", "test"}:
            raise MediaError("Unknown observation split")
        if event_id in observation_splits and observation_splits[event_id] != split:
            raise MediaError("Mirror publications cannot cross observation splits")
        if split == "train" and events[event_id]["split"] != "train":
            raise MediaError("A held-out text event cannot train observation features")
        observation_splits[event_id] = split
    if not any(value != "train" for value in observation_splits.values()):
        raise MediaError("Observation evaluation must retain an independent source event")
    return events


def _load_asr(directory: Path, publisher: dict, publisher_file: Path) -> dict:
    media_path, media = load_manifest(directory / "manifest.json")
    verify_artifact(media_path, media, "audio")
    asr_file = local_file(directory / "asr.json", max_bytes=16 * 1024**2)
    asr = read_json(asr_file)
    alignment_file = local_file(directory / "alignment.json", max_bytes=16 * 1024**2)
    alignment = read_json(alignment_file)
    if alignment.get("asr_artifact_sha256") != sha256(asr_file) or alignment.get("publisher_artifact_sha256") != sha256(publisher_file):
        raise MediaError("Alignment input artifact hash differs from the preserved run")
    recomputed = align_transcripts(publisher, asr)
    if recomputed["matches"] != alignment["matches"]:
        raise MediaError("Publisher/ASR matches could not be reproduced")
    if asr["source_id"] != media["source_id"] or asr["source_timeline"] != media["source_timeline"] or asr["audio_sha256"] != media["artifacts"]["audio"]["sha256"]:
        raise MediaError("ASR and prepared media provenance differ")
    if abs(asr["source_start_seconds"] - media["source_start_seconds"]) > 1e-6:
        raise MediaError("ASR offset differs from its prepared audio clock")
    return {"asr": asr, "alignment": alignment, "media": media, "provenance": {"run_id": directory.name, "asr_sha256": sha256(asr_file), "alignment_sha256": sha256(alignment_file), "media_manifest_sha256": sha256(media_path), "audio_sha256": asr["audio_sha256"], "input_sha256": media["input"]["sha256"], "engine": asr["engine"]}}


def _load_visual(window: dict, cache: Path, repository: Path, measurement_plan: dict) -> dict:
    directory = within(cache, f"motion-acquisition/offline-v021/{window['id']}/manifest.json").parent
    media_path, media = load_manifest(directory / "manifest.json")
    verify_artifact(media_path, media, "video")
    obs_manifest_file = local_file(directory / "observations-manifest.json", max_bytes=1024**2)
    observed = read_json(obs_manifest_file)
    obs_file = artifact_path(obs_manifest_file, observed["observations"]["path"])
    if sha256(obs_file) != observed["observations"]["sha256"] or sha256(media_path) != observed["preparation_manifest_sha256"]:
        raise MediaError("Visual observations or their prepared source changed")
    if observed["video_sha256"] != media["artifacts"]["video"]["sha256"] or observed["input_sha256"] != media["input"]["sha256"]:
        raise MediaError("Observed geometry input hashes disagree")
    if observed["source_id"] != window["source_id"] or media["source_id"] != window["source_id"]:
        raise MediaError("Observation window source mismatch")
    if observed["source_timeline"] != media["source_timeline"] or abs(observed["source_start_seconds"] - media["source_start_seconds"]) > 1e-6:
        raise MediaError("Observation window clock mismatch")
    acquisition = next((item for item in measurement_plan["acquisitions"] if item["source_id"] == window["source_id"]), None)
    if not acquisition or media["input"]["sha256"] != acquisition["input_sha256"]:
        raise MediaError("Observed input differs from the frozen acquisition")
    if abs(media["source_start_seconds"] - window["start_seconds"]) > 1e-6:
        raise MediaError("Observed input starts outside the planned source window")
    geometry_file = within(repository, f"apps/web/public/motion/{window['id']}.json", max_bytes=8 * 1024**2)
    geometry = read_json(geometry_file)
    validate_clip(geometry)
    if geometry["source"]["observations_sha256"] != sha256(obs_file) or geometry["source"]["video_sha256"] != observed["video_sha256"]:
        raise MediaError("Converted head geometry has different measurement provenance")
    if geometry["source"]["source_id"] != observed["source_id"] or geometry["source"]["source_timeline"] != observed["source_timeline"] or abs(geometry["source"]["source_start_seconds"] - observed["source_start_seconds"]) > 1e-6:
        raise MediaError("Converted head geometry uses a different source clock")
    rows = _read_jsonl(obs_file, max_rows=4000)
    for row in rows:
        if row.get("schema_version") != "observation.v1" or row["source_id"] != observed["source_id"] or row["source_timeline"] != observed["source_timeline"]:
            raise MediaError("Visual row provenance differs")
        if abs(finite_number(row["source_timestamp_seconds"], "source time") - media["source_start_seconds"] - finite_number(row["timestamp_seconds"], "clip time")) > 1e-6:
            raise MediaError("A frame's local and source clocks disagree")
    accepted = window["accepted_end_ms"] / 1000
    if abs(geometry["duration_ms"] / 1000 - accepted) > 0.001:
        raise MediaError("Camera exclusion differs from accepted geometry duration")
    exclusions = [(window["start_seconds"] + accepted, window["start_seconds"] + window["duration_seconds"])] if accepted < window["duration_seconds"] else []
    return {"rows": rows, "geometry": geometry, "exclusions": exclusions, "provenance": {"window_id": window["id"], "observations_sha256": sha256(obs_file), "observations_manifest_sha256": sha256(obs_manifest_file), "geometry_sha256": sha256(geometry_file), "prepared_video_sha256": observed["video_sha256"], "input_sha256": observed["input_sha256"], "runtime": {"name": "mediapipe", "version": observed["mediapipe_version"]}, "models": observed["models"]}}


def build_report(plan_path: str | Path, corpus_path: str | Path, cache_root: str | Path, repository_root: str | Path, output_path: str | Path, *, overwrite: bool = False) -> dict:
    wall = time.perf_counter()
    cpu = time.process_time()
    plan_file = local_file(plan_path, max_bytes=1024**2)
    plan = read_json(plan_file)
    if plan.get("schema_version") != "exchange-observation-plan.v1":
        raise MediaError("Expected exchange-observation-plan.v1")
    cache, repository = Path(cache_root).resolve(strict=True), Path(repository_root).resolve(strict=True)
    corpus = Path(corpus_path).resolve(strict=True)
    evidence_file = within(corpus, "evidence.jsonl", max_bytes=16 * 1024**2)
    split_file = within(repository, plan["split_manifest_path"])
    inventory = read_json(split_file)
    events = validate_splits(plan["sources"], inventory)
    measurement_file = within(repository, plan["motion_plan_path"])
    measurement = read_json(measurement_file)
    evidence = _read_jsonl(evidence_file, max_rows=3000)
    records = {row["id"]: row for row in evidence}
    if len(records) != len(evidence):
        raise MediaError("Duplicate Q&A evidence IDs")
    requested_ids = [record_id for source in plan["sources"] for record_id in source["exchange_ids"]]
    if len(set(requested_ids)) != len(requested_ids) or any(value not in records for value in requested_ids):
        raise MediaError("Observation plan exchange IDs are missing or duplicated")
    entries, provenance = [], []
    for source in plan["sources"]:
        publisher_file = within(cache, source["publisher_path"])
        publisher_envelope = read_json(publisher_file)
        publisher = publisher_envelope.get("transcript", publisher_envelope)
        if publisher["source_id"] != source["source_id"]:
            raise MediaError("Publisher transcript belongs to another source")
        if sha256(publisher_file) != source["publisher_sha256"]:
            raise MediaError("Publisher artifact differs from the frozen observation plan")
        runs = [_load_asr(within(cache, relative + "/manifest.json").parent, publisher, publisher_file) for relative in source["asr_directories"]]
        windows = [window for window in measurement["windows"] if window["source_id"] == source["source_id"]]
        visual = {window["id"]: _load_visual(window, cache, repository, measurement) for window in windows}
        provenance.append({"source_id": source["source_id"], "publisher_artifact_sha256": sha256(publisher_file), "publisher_transcript_digest": digest_json(publisher), "asr_runs": [run["provenance"] for run in runs], "visual_windows": [item["provenance"] for item in visual.values()]})
        for record_id in source["exchange_ids"]:
            record = records[record_id]
            qa = record.get("question_answer")
            if not isinstance(qa, dict) or record.get("kind") != "question_answer" or record.get("review_status") != "reviewed":
                raise MediaError("Observation linkage needs a reviewed original Q&A record")
            if record["source_id"] != source["source_id"] or qa["source_event_id"] != source["source_event_id"]:
                raise MediaError("Q&A source-event identity disagrees with the observation plan")
            locator = qa["transcript_locator"]
            # The TED acquisition adapter hashes canonical publisher caption
            # content. This is distinct from the file/envelope byte hash above.
            if locator["transcript_sha256"] != publisher.get("publisher_caption_sha256"):
                raise MediaError("Q&A locator references a different publisher transcript")
            speech_runs = []
            for run in runs:
                result = aligned_exchange(publisher, run["asr"], run["alignment"], question_cues=locator["question_cue_ids"], answer_cues=locator["answer_cue_ids"], questioner=qa["questioner"], respondent=qa["respondent"], language=qa["original_language"])
                if result["matched_answer_words"] or result["matched_question_words"]:
                    result["asr_run_id"] = run["provenance"]["run_id"]
                    speech_runs.append(result)
            linked_visual = []
            for window in windows:
                lo, hi = window["start_seconds"], window["start_seconds"] + window["duration_seconds"]
                # Clock linkage comes from matching words in the exact media;
                # overlap on subtitle times alone cannot link a face track.
                intervals = [run[field] for run in speech_runs for field in ("question_window_seconds", "answer_window_seconds") if run.get(field)]
                if not any(a < hi and lo < b for a, b in intervals):
                    continue
                measured = visual[window["id"]]
                feature = visual_features(measured["rows"], measured["geometry"], start=lo, end=hi, speech_runs=speech_runs, exclusions=measured["exclusions"], **plan["thresholds"])
                feature["window_id"] = window["id"]
                feature["status"] = "exchange_context_candidate_not_speaker_attributed"
                linked_visual.append(feature)
            entries.append({
                "exchange_id": record_id, "exchange_sha256": digest_json(record), "source_id": source["source_id"], "source_event_id": source["source_event_id"], "source_url": publisher["source_url"],
                "text_split": events[source["source_event_id"]]["split"], "observation_split": source["observation_split"],
                "situations": record["situations"], "topics": record["topics"], "original_language": qa["original_language"],
                "text_review": qa["review"], "exchange_completeness": qa["completeness"], "observation_scope": "original_cue_window_not_only_quoted_excerpt",
                "speech": merge_speech_runs(speech_runs), "visual_windows": linked_visual,
                "review_status": "candidate", "human_listening_completed": False, "runtime_eligible": False,
            })
    unique_windows = {window["window_id"]: window for row in entries for window in row["visual_windows"]}
    rate_windows = [window for row in entries for run in row["speech"]["runs"] for window in run["speech_rate_windows"]]
    report = {
        "schema_version": "exchange-observations.v1", "id": plan["id"], "implementation_version": VERSION,
        "inputs": {"plan_sha256": sha256(plan_file), "qa_corpus_sha256": sha256(evidence_file), "source_event_split_sha256": sha256(split_file), "motion_plan_sha256": sha256(measurement_file), "features_implementation_sha256": sha256(Path(__file__).with_name("features.py")), "runner_implementation_sha256": sha256(Path(__file__))},
        "thresholds": plan["thresholds"], "threshold_origin": plan["threshold_origin"],
        "event_inventory": inventory["events"], "observation_split_note": "Observation validation retains whole source events. A text-training event may be held out separately for motion; it is not thereby an unseen text event.",
        "provenance": provenance, "entries": entries,
        "summary": {"exchanges": len(entries), "independent_source_events": len({row["source_event_id"] for row in entries}), "exchanges_by_observation_split": dict(Counter(row["observation_split"] for row in entries)), "speech_runs": sum(len(row["speech"]["runs"]) for row in entries), "speech_rate_windows": len(rate_windows), "visual_windows_unique": len(unique_windows), "sampled_visual_frames_unique": sum(window["sampled_frames"] for window in unique_windows.values()), "excluded_visual_frames_unique": sum(window["excluded_frames"] for window in unique_windows.values()), "smile_candidate_intervals": sum(len(window["visible_smile"]["intervals"]) for window in unique_windows.values()), "reviewed_pre_answer_pauses": 0, "reviewed_laughter_intervals": 0, "human_voice_ratings": 0, "runtime_eligible_exchanges": 0, "unmeasured_source_events": [event["source_event_id"] for event in inventory["events"] if event["source_event_id"] not in {row["source_event_id"] for row in entries}]},
        "execution": {"python": platform.python_version(), "platform": platform.system(), "machine": platform.machine(), "wall_seconds": round(time.perf_counter() - wall, 6), "process_cpu_seconds": round(time.process_time() - cpu, 6), "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024), "scope": "Recomputation from existing hashed ASR and visual model outputs; excludes original media acquisition/model inference. Shared-host timing is not an isolated throughput benchmark."},
        "limitations": ["The pilot has only two observed events; correlated frames are not independent samples.", "No human listening, acoustic-boundary review, image-to-speaker association, or semantic gesture accuracy is claimed.", "Smiles are visual geometry candidates, audible laughter requires listening, and emotional state is never inferred.", "Known camera-cut tails are excluded; camera-motion review remains required before accepting nod/tilt or hand gestures.", "No raw interview text, original audio or video is copied into this report.", "The language-specific speech units are not interchangeable; cross-language rate comparisons require a shared acoustic annotation method.", "Candidate observations never automatically enter runtime voice or motion selection."],
    }
    write_json(Path(output_path), report, overwrite=overwrite)
    return report
