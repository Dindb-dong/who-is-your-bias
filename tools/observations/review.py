"""Explicit source-observation review without manufacturing human judgments."""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

from tools.media.common import MediaError, finite_number, sha256, write_json
from tools.references.workflow import digest_json, read_json

CHECKS = ("watched_source", "listened_to_source", "source_clock_verified", "image_speaker_association_verified", "track_continuity_verified", "camera_motion_and_cuts_reviewed", "recording_continuity_verified", "acoustic_turn_boundaries_verified")
KINDS = {"visible_smile", "audible_laughter", "head_nod", "head_tilt", "hand_wave", "hand_emphasis", "hand_unobservable"}


def make_review_bundle(report_path: str | Path, output_directory: str | Path) -> dict:
    file = Path(report_path).resolve(strict=True)
    report = read_json(file)
    if report.get("schema_version") != "exchange-observations.v1":
        raise MediaError("Expected an exchange observation report")
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    table = []
    for row in report["entries"]:
        entries.append({
            "exchange_id": row["exchange_id"], "exchange_sha256": row["exchange_sha256"], "source_event_id": row["source_event_id"],
            "decision": "not_reviewed", "checks": {check: False for check in CHECKS},
            "evidence_note": "", "acoustic_boundaries": {"question_end_seconds": None, "answer_start_seconds": None, "answer_end_seconds": None},
            "annotations": [],
        })
        table.append("<tr><td>" + html.escape(row["exchange_id"]) + "</td><td>" + html.escape(", ".join(row["situations"])) + "</td><td>" + html.escape(row["observation_split"]) + "</td><td>" + str(len(row["speech"]["runs"])) + "</td><td>" + str(len(row["visual_windows"])) + '</td><td><a target="_blank" rel="noreferrer" href="' + html.escape(row["source_url"], quote=True) + '">Original recording</a></td></tr>')
    template = {"schema_version": "exchange-observation-review.v1", "report_sha256": sha256(file), "reviewer": None, "reviewer_type": None, "reviewed_at": None, "entries": entries}
    write_json(out / "review-template.json", template)
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Interview observation review</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#132536;max-width:1100px;margin:48px auto;padding:0 24px;background:#fafbfc}h1{font-size:32px}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #dfe5ea}th{background:#eef3f6}a{color:#005fb0}code{background:#eef3f6;padding:2px 5px}li{margin:8px 0}.notice{border-left:4px solid #ca8500;background:#fff5db;padding:16px}</style><h1>Interview observation review</h1><p class="notice">All observations start as candidates. No human listening or gesture judgments have been recorded.</p><p>Review the original recordings and the exact local measurement clips. Times in the report refer to the acquired recording clock; publisher subtitle onsets are a separate clock. The original recording link alone does not verify the mapping.</p><table><thead><tr><th>Exchange</th><th>Situation</th><th>Observation split</th><th>Audio windows</th><th>Video windows</th><th>Source</th></tr></thead><tbody>""" + "".join(table) + """</tbody></table><h2>Review steps</h2><ol><li>Identify the source event, track, and respondent using the source attribution and conversation, then verify their association with the selected audio/video.</li><li>Check continuity and camera motion. A visible smile is distinct from audible laughter. A hand outside the frame is unobservable.</li><li>For a pre-answer pause, listen to the actual question ending and first answer sound. Leave the value unknown if editing or clock alignment makes it unknowable.</li><li>Fill <a href="review-template.json">the review template</a> with the reviewer type, evidence note, checks you actually completed, and source-clock annotations.</li><li>Import the annotations with <code>python -m tools.observations review</code>. Only accepted human observations from training events can enter the exported runtime index.</li></ol><p>Allowed annotation kinds: visible_smile, audible_laughter, head_nod, head_tilt, hand_wave, hand_emphasis, hand_unobservable. Each annotation needs start_seconds, end_seconds, phase (listening/speaking/unknown), and observable (true/false). Hand annotations also need side (left/right/both). Set acoustic boundaries only after the corresponding listening and continuity checks.</p><p>Voice naturalness and similarity are evaluated separately using the existing generated-speech listening bundle. This page does not infer voice quality from ASR or source geometry.</p></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    return {"report_sha256": sha256(file), "entries": len(entries), "reviewed_entries": 0, "output_directory": str(out)}


def import_review(report_path: str | Path, annotations_path: str | Path, output_path: str | Path, *, overwrite: bool = False) -> dict:
    file = Path(report_path).resolve(strict=True)
    report, review = read_json(file), read_json(annotations_path)
    if report.get("schema_version") != "exchange-observations.v1" or review.get("schema_version") != "exchange-observation-review.v1":
        raise MediaError("Expected observation report and matching review schema")
    if review.get("report_sha256") != sha256(file):
        raise MediaError("Review belongs to another immutable observation report")
    rows = {row["exchange_id"]: row for row in report["entries"]}
    submitted = review.get("entries")
    if not isinstance(submitted, list) or len(submitted) > len(rows):
        raise MediaError("Review entries are missing or excessive")
    reviewed = [item for item in submitted if item.get("decision") != "not_reviewed"]
    reviewer_type = review.get("reviewer_type")
    if reviewed:
        if reviewer_type not in {"human", "agent"} or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip() or not isinstance(review.get("reviewed_at"), str) or not review["reviewed_at"].strip():
            raise MediaError("Completed reviews require an explicit actor, actor type, and date")
    seen, accepted, rejected, unreviewed = set(), [], [], []
    for item in submitted:
        record_id = item.get("exchange_id")
        if record_id not in rows or record_id in seen:
            raise MediaError("Unknown or duplicate reviewed exchange")
        seen.add(record_id)
        row = rows[record_id]
        if item.get("exchange_sha256") != row["exchange_sha256"] or item.get("source_event_id") != row["source_event_id"]:
            raise MediaError("Review exchange content or event identity changed")
        decision = item.get("decision")
        if decision not in {"accept", "reject", "not_reviewed"}:
            raise MediaError("Unknown review decision")
        if decision == "not_reviewed":
            if item.get("annotations") or any(item.get("acoustic_boundaries", {}).values()):
                raise MediaError("An unreviewed exchange cannot contain accepted annotations")
            unreviewed.append(record_id)
            continue
        if not isinstance(item.get("evidence_note"), str) or len(item["evidence_note"].strip()) < 12:
            raise MediaError("Document the actual observation evidence")
        checks = item.get("checks", {})
        if set(checks) != set(CHECKS) or any(type(value) is not bool for value in checks.values()):
            raise MediaError("Every completed review needs explicit boolean checks")
        if decision == "reject":
            rejected.append({"exchange_id": record_id, "evidence_note": item["evidence_note"]})
            continue
        if not checks["source_clock_verified"]:
            raise MediaError("Accepted annotations need a verified source clock")
        intervals = []
        audio_intervals = []
        for run in row["speech"]["runs"]:
            spans = [run[field] for field in ("question_window_seconds", "answer_window_seconds") if run.get(field)]
            if spans:
                audio_intervals.append([min(span[0] for span in spans), max(span[1] for span in spans)])
        visual_intervals = [[window["window_start_seconds"], window["window_end_seconds"]] for window in row["visual_windows"]]
        intervals = audio_intervals + visual_intervals
        if not intervals:
            raise MediaError("No observed source interval is available for review")
        lower, upper = min(interval[0] for interval in intervals), max(interval[1] for interval in intervals)
        annotations = item.get("annotations")
        if not isinstance(annotations, list) or len(annotations) > 100:
            raise MediaError("Annotations must be a bounded list")
        validated = []
        for annotation in annotations:
            kind = annotation.get("kind")
            if kind not in KINDS or annotation.get("phase") not in {"listening", "speaking", "unknown"} or type(annotation.get("observable")) is not bool:
                raise MediaError("Observation kind, phase, or observability is invalid")
            start = finite_number(annotation.get("start_seconds"), "annotation start", lower, upper)
            end = finite_number(annotation.get("end_seconds"), "annotation end", start + 0.001, upper)
            modality_intervals = audio_intervals if kind == "audible_laughter" else visual_intervals
            if not any(lo <= start and end <= hi for lo, hi in modality_intervals):
                raise MediaError("Annotation crosses an unobserved source gap")
            if not annotation["observable"] and kind != "hand_unobservable":
                raise MediaError("Unobserved data cannot affirm a behavior")
            if kind == "hand_unobservable" and annotation["observable"]:
                raise MediaError("Unobservable hands must not be marked observable")
            if kind.startswith("hand_") and annotation.get("side") not in {"left", "right", "both"}:
                raise MediaError("A hand annotation needs its visible/occluded side")
            if kind == "audible_laughter":
                if not checks["listened_to_source"]:
                    raise MediaError("Audible laughter requires actual listening")
            elif not checks["watched_source"] or not checks["track_continuity_verified"]:
                raise MediaError("Visual behavior requires a watched continuous image track")
            if not checks["image_speaker_association_verified"]:
                raise MediaError("A named-speaker observation requires verified speaker association")
            if kind in {"head_nod", "head_tilt", "hand_wave", "hand_emphasis"} and not checks["camera_motion_and_cuts_reviewed"]:
                raise MediaError("Head/hand gestures require camera-motion and cut review")
            for window in row["visual_windows"]:
                if kind != "audible_laughter" and any(start < exclusion["end_seconds"] and exclusion["start_seconds"] < end for exclusion in window["exclusions"]):
                    raise MediaError("A visual annotation overlaps an excluded camera interval")
            validated.append(dict(annotation, start_seconds=start, end_seconds=end))
        boundaries = item.get("acoustic_boundaries", {})
        if set(boundaries) != {"question_end_seconds", "answer_start_seconds", "answer_end_seconds"}:
            raise MediaError("Acoustic boundaries must explicitly retain unknown values")
        pause = None
        if any(value is not None for value in boundaries.values()):
            if not all(checks[key] for key in ("listened_to_source", "recording_continuity_verified", "acoustic_turn_boundaries_verified")):
                raise MediaError("Pause measurement requires listening, continuity, and acoustic boundary review")
            if any(value is None for value in boundaries.values()):
                raise MediaError("Reviewed pause needs all three acoustic boundaries")
            q_end = finite_number(boundaries["question_end_seconds"], "question ending", lower, upper)
            a_start = finite_number(boundaries["answer_start_seconds"], "answer onset", lower, upper)
            a_end = finite_number(boundaries["answer_end_seconds"], "answer ending", a_start + 0.001, upper)
            if a_start < q_end:
                raise MediaError("Overlapping turns are not a positive pre-answer pause")
            if not any(run.get("question_end_seconds_candidate") is not None and run.get("answer_start_seconds_candidate") is not None and run.get("answer_end_seconds_candidate") is not None for run in row["speech"]["runs"]):
                raise MediaError("The observed audio does not cover complete turn boundaries")
            pause = round(a_start - q_end, 6)
        if not validated and pause is None:
            raise MediaError("An accepted review must provide at least one actual observation")
        accepted.append({"exchange_id": record_id, "source_event_id": row["source_event_id"], "text_split": row["text_split"], "observation_split": row["observation_split"], "situations": row["situations"], "reviewer_type": reviewer_type, "annotations": validated, "pre_answer_pause_seconds": pause, "checks": checks, "evidence_note": item["evidence_note"], "runtime_eligible": reviewer_type == "human" and row["text_split"] == "train" and row["observation_split"] == "train"})
    unreviewed += sorted(set(rows) - seen)
    output = {"schema_version": "reviewed-exchange-observations.v1", "report_sha256": sha256(file), "annotations_sha256": sha256(Path(annotations_path)), "reviewer": review.get("reviewer"), "reviewer_type": reviewer_type, "reviewed_at": review.get("reviewed_at"), "accepted": accepted, "rejected": rejected, "unreviewed": unreviewed, "runtime_index": [row for row in accepted if row["runtime_eligible"]], "summary": {"accepted": len(accepted), "rejected": len(rejected), "unreviewed": len(unreviewed), "human_reviewed": len(accepted) + len(rejected) if reviewer_type == "human" else 0, "agent_reviewed": len(accepted) + len(rejected) if reviewer_type == "agent" else 0, "runtime_eligible": sum(row["runtime_eligible"] for row in accepted), "accepted_by_event": dict(Counter(row["source_event_id"] for row in accepted))}, "limitations": ["Review records explicit supplied judgments; it cannot prove a reviewer listened merely from a checkbox.", "Agent judgments remain distinct from human judgments and cannot populate the runtime index.", "Validation/test events remain excluded even after a positive review.", "Observed source behavior is not a prediction of the person's reaction to a new question."]}
    output["review_fingerprint"] = digest_json(output)
    write_json(Path(output_path), output, overwrite=overwrite)
    return output
