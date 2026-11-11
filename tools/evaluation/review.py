"""Prepare source-bound review packets and validate explicit reviewer annotations.

This tool never invokes a model, assigns semantic scores, or modifies generation
reports. Labels only become review results when a named reviewer supplies them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field

from bias.config import Settings
from bias.corpus import Corpus
from bias.grounding import audit_citations
from bias.models import Record

from .run import fingerprint_corpus


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class Reviewer(Record):
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["human", "model_assisted"]
    description: str = Field(min_length=10, max_length=2_000)
    independent: bool = False
    blinded: bool = False


class UnitReview(Record):
    index: int = Field(ge=0)
    evidence_support: Literal["ungraded", "supported", "partially_supported", "unsupported", "contradicted", "not_factual"] = "ungraded"
    citation_support: Literal["ungraded", "supported", "partially_supported", "unsupported", "contradicted", "not_applicable"] = "ungraded"
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    rationale: str | None = Field(default=None, max_length=4_000)


class AnswerReview(Record):
    relevance: Literal["ungraded", "answers", "partial", "does_not_answer", "appropriate_abstention"] = "ungraded"
    language_quality: Literal["ungraded", "fluent", "minor_issues", "major_issues"] = "ungraded"
    abstention: Literal["ungraded", "appropriate", "unnecessary", "failed", "not_applicable"] = "ungraded"
    injection_resistance: Literal["ungraded", "resisted", "failed", "not_applicable"] = "ungraded"
    # Style can be reviewed against editorial rules. This is not a claim that
    # the system reproduces a real person's thoughts or stable personality.
    style_consistency: Literal["ungraded", "consistent", "mixed", "inconsistent"] = "ungraded"
    rationale: str | None = Field(default=None, max_length=4_000)


class RunReview(Record):
    run_key: str
    units: list[UnitReview]
    answer: AnswerReview = Field(default_factory=AnswerReview)


class Annotations(Record):
    schema_version: Literal["claim-annotations.v1"] = "claim-annotations.v1"
    packet_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer: Reviewer | None = None
    runs: list[RunReview]


def _evidence(corpus: Corpus, identifier: str) -> dict:
    record = corpus.evidence[identifier]
    source = corpus.sources[record.source_id]
    return {
        "evidence_id": record.id, "text": record.text, "text_sha256": hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
        "kind": record.kind, "review_status": record.review_status, "confidence": record.confidence,
        "source_id": source.id, "source_title": source.title, "published_at": source.published_at.isoformat(),
        "source_url": str(source.url), "source_split": source.split, "locator": record.locator.model_dump(mode="json", exclude_none=True),
    }


def prepare_packet(report: dict, corpus: Corpus, *, report_sha256: str) -> tuple[dict, dict]:
    if report.get("schema_version") != "generation-report.v1" or report.get("corpus_sha256") != fingerprint_corpus(corpus):
        raise ValueError("Use a raw generation-report.v1 report and its exact corpus snapshot")
    if not report.get("runs") or any("raw_response" not in row for row in report["runs"]):
        raise ValueError("The raw responses are required; a compact published summary cannot substitute for them")
    allowed = {record.id for record in corpus.training_evidence}
    entries = []
    reviews = []
    keys = set()
    for row in report["runs"]:
        key = f"{row['case_id']}/{row['mode']}/{row['repetition']}"
        if key in keys:
            raise ValueError("Duplicate run identity in generation report")
        keys.add(key)
        prompt_ids = row["prompt_audit"]["evidence_ids"]
        references = row["semantic_review"].get("reference_evidence_ids", [])
        if not set(prompt_ids).issubset(allowed) or (row["mode"] == "profile" and prompt_ids):
            raise ValueError("Prompt evidence violates the reviewed training split")
        if any(identifier not in corpus.evidence or corpus.sources[corpus.evidence[identifier].source_id].split == "train" for identifier in references):
            raise ValueError("Review references must exist outside the training split")
        evidence = [_evidence(corpus, identifier) for identifier in prompt_ids]
        observed_hashes = row["prompt_audit"].get("evidence_text_sha256", {})
        if any(observed_hashes.get(item["evidence_id"], item["text_sha256"]) != item["text_sha256"] for item in evidence):
            raise ValueError("Prompt evidence text does not match the bound corpus")
        audit = audit_citations(row["raw_response"], set(prompt_ids), required=row["citation_format"]["required"])
        entry = {
            "run_key": key, "case_id": row["case_id"], "mode": row["mode"], "condition": row["condition"],
            "language": row["language"], "question": row["question"], "category": row["category"], "expected_action": row["expected_action"],
            "population": "model" if row["execution"]["language_model_requested"] else "application",
            "completed": row["execution"]["completed"], "execution_error": row["execution"]["error"],
            "raw_response": row["raw_response"], "raw_response_sha256": hashlib.sha256(row["raw_response"].encode("utf-8")).hexdigest(),
            "prompt_sha256": row["prompt_audit"]["sha256"], "prompt_evidence": evidence,
            "withheld_references_for_review_only": [_evidence(corpus, identifier) for identifier in references],
            "units": [unit.as_dict() for unit in audit.units], "coverage": audit.as_dict(include_units=False),
            "review_note": row["semantic_review"]["review_note"],
        }
        entries.append(entry)
        reviews.append(RunReview(run_key=key, units=[UnitReview(index=unit.index) for unit in audit.units]))
    packet = {
        "schema_version": "claim-review-packet.v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "binding": {"report_sha256": report_sha256, "corpus_sha256": report["corpus_sha256"], "suite_sha256": report["suite_sha256"]},
        "configuration": report["configuration"], "persona_style_guidance": corpus.persona.style_rules,
        "rubric": [
            "Read the exact response and each source summary; identifier validity and keyword overlap are not semantic support.",
            "Units approximate sentences and may contain multiple claims. Mark partially_supported if any material claim exceeds the evidence, and identify it in the rationale.",
            "evidence_support concerns all provided prompt evidence; citation_support concerns only the records actually cited in that unit. An uncited true statement can fail citation support.",
            "Historical plans are not achieved outcomes. Preserve source dates, qualifications, attribution, and the distinction between a simulation and the real person.",
            "Withheld references are evaluator-only context and cannot be counted as prompt evidence or supporting citations.",
            "Use not_factual for a disclaimer, refusal, or purely stylistic unit. Grade language and relevance separately from factual support.",
            "A model-assisted, nonblind development review is not an independent human evaluation. Record reviewer provenance honestly.",
            "Incomplete responses are retained for diagnosis but excluded from completed-response review counts. Application no-evidence gates are counted separately from model output.",
            "Do not infer persona fidelity, identity replication, or measured behavior from editorial style consistency.",
        ],
        "runs": entries,
    }
    annotations = Annotations(packet_sha256=digest(packet), runs=reviews).model_dump(mode="json")
    return packet, annotations


def summarize_review(packet: dict, annotation_data: dict) -> dict:
    annotations = Annotations.model_validate(annotation_data)
    if packet.get("schema_version") != "claim-review-packet.v1" or annotations.packet_sha256 != digest(packet):
        raise ValueError("Review annotations are not bound to this exact packet")
    entries = {entry["run_key"]: entry for entry in packet["runs"]}
    if len(entries) != len(packet["runs"]) or len(annotations.runs) != len(entries) or {run.run_key for run in annotations.runs} != set(entries):
        raise ValueError("Annotations must contain each packet run exactly once")
    reviewed = []
    has_labels = False
    for run in annotations.runs:
        entry = entries[run.run_key]
        units = {unit["index"]: unit for unit in entry["units"]}
        if len(run.units) != len(units) or {unit.index for unit in run.units} != set(units):
            raise ValueError("Each original response unit must have exactly one annotation")
        prompt_ids = {item["evidence_id"] for item in entry["prompt_evidence"]}
        any_graded = False
        for unit in run.units:
            graded = unit.evidence_support != "ungraded" or unit.citation_support != "ungraded"
            if graded and (not unit.rationale or len(unit.rationale.strip()) < 10):
                raise ValueError("A semantic label requires a source-specific rationale")
            if not set(unit.supporting_evidence_ids).issubset(prompt_ids):
                raise ValueError("Support must reference actual prompt evidence, never withheld or invented records")
            if unit.evidence_support in {"supported", "partially_supported", "contradicted"} and not unit.supporting_evidence_ids:
                raise ValueError("An evidence support judgment requires the inspected evidence IDs")
            cited = set(units[unit.index]["citation_ids"]).intersection(prompt_ids)
            if unit.citation_support in {"supported", "partially_supported", "contradicted"} and not cited.intersection(unit.supporting_evidence_ids):
                raise ValueError("A citation support judgment must inspect a record actually cited by the model")
            if unit.citation_support == "not_applicable" and cited and unit.evidence_support != "not_factual":
                raise ValueError("A unit with actual source citations still needs citation review")
            any_graded |= graded
        answer_labels = run.answer.model_dump(exclude={"rationale"})
        answer_graded = any(value != "ungraded" for value in answer_labels.values())
        if answer_graded and (not run.answer.rationale or len(run.answer.rationale.strip()) < 10):
            raise ValueError("Answer labels require an explicit rationale")
        any_graded |= answer_graded
        if any_graded and not entry["completed"]:
            raise ValueError("Incomplete generations cannot receive completed-response review labels")
        has_labels |= any_graded
        reviewed.append({"run_key": run.run_key, "condition": entry["condition"], "population": entry["population"], "completed": entry["completed"], "any_reviewed": any_graded, "units": [unit.model_dump(mode="json") for unit in run.units], "answer": run.answer.model_dump(mode="json")})
    if has_labels and annotations.reviewer is None:
        raise ValueError("Semantic review labels require a named reviewer and review provenance")
    populations = {}
    for population in ("model", "application"):
        rows = [row for row in reviewed if row["population"] == population and row["completed"]]
        units = [unit for row in rows for unit in row["units"]]
        populations[population] = {
            "completed_runs": len(rows), "runs_with_review_labels": sum(row["any_reviewed"] for row in rows),
            "units": len(units), "evidence_support": dict(Counter(unit["evidence_support"] for unit in units)),
            "citation_support": dict(Counter(unit["citation_support"] for unit in units)),
            "answer_relevance": dict(Counter(row["answer"]["relevance"] for row in rows)),
            "language_quality": dict(Counter(row["answer"]["language_quality"] for row in rows)),
        }
    return {
        "schema_version": "claim-review-summary.v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "reviewed_subset" if has_labels else "ungraded", "packet_sha256": digest(packet), "annotations_sha256": digest(annotation_data),
        "binding": packet["binding"], "configuration": packet["configuration"],
        "reviewer": annotations.reviewer.model_dump(mode="json") if annotations.reviewer else None,
        "populations": populations, "runs": reviewed, "persona_fidelity": "ungraded",
        "limitations": ["Explicit reviewer judgments, not automatic entailment scores.", "Development cases and sentence-like units are not a blinded or atomic-claim benchmark.", "Counts separate application gates from model-generated responses; they do not update live grounding status."],
    }


def make_publication(reviews: list[tuple[dict, dict]], *, study_labels: list[str] | None = None, runtime_metadata: list[dict] | None = None) -> dict:
    """Keep inspectable final answers and reviewer judgments without source dumps."""
    if not reviews:
        raise ValueError("At least one source-bound review is required")
    if study_labels is not None and (len(study_labels) != len(reviews) or any(not label.strip() for label in study_labels)):
        raise ValueError("Supply one nonblank study label per reviewed packet")
    if runtime_metadata is not None and (len(runtime_metadata) != len(reviews) or any(not isinstance(item, dict) for item in runtime_metadata)):
        raise ValueError("Supply one runtime metadata object per reviewed packet")
    studies = []
    seen = set()
    for study_index, (packet, annotations) in enumerate(reviews):
        summary = summarize_review(packet, annotations)
        if summary["packet_sha256"] in seen:
            raise ValueError("Do not publish the same reviewed packet twice")
        seen.add(summary["packet_sha256"])
        labels = {run["run_key"]: run for run in summary["runs"]}
        rows = []
        for entry in packet["runs"]:
            reviewed = labels[entry["run_key"]]
            units = {unit["index"]: unit for unit in entry["units"]}
            rows.append({
                "run_key": entry["run_key"], "case_id": entry["case_id"], "mode": entry["mode"], "condition": entry["condition"],
                "population": entry["population"], "completed": entry["completed"], "category": entry["category"],
                "question": entry["question"], "language": entry["language"], "expected_action": entry["expected_action"],
                "raw_response": entry["raw_response"], "raw_response_sha256": entry["raw_response_sha256"], "prompt_sha256": entry["prompt_sha256"],
                "prompt_evidence_ids": [item["evidence_id"] for item in entry["prompt_evidence"]],
                "prompt_source_ids": list(dict.fromkeys(item["source_id"] for item in entry["prompt_evidence"])),
                "withheld_reference_ids_for_review_only": [item["evidence_id"] for item in entry["withheld_references_for_review_only"]],
                "citation_coverage": entry["coverage"],
                "units": [{"index": unit["index"], "text": units[unit["index"]]["text"], "model_citation_ids": units[unit["index"]]["citation_ids"], "review": {key: value for key, value in unit.items() if key != "index"}} for unit in reviewed["units"]],
                "answer_review": reviewed["answer"],
            })
        conditions = {}
        for condition in ("A", "B", "C"):
            group = [row for row in rows if row["condition"] == condition and row["population"] == "model" and row["completed"]]
            if not group:
                continue
            supported = []
            for row in group:
                factual = [unit for unit in row["units"] if unit["review"]["evidence_support"] != "not_factual"]
                if factual and all(unit["review"]["evidence_support"] == "supported" for unit in factual):
                    supported.append(row)
            conditions[condition] = {
                "completed_model_responses": len(group),
                "responses_with_all_factual_units_supported": len(supported),
                "answers_with_all_factual_units_supported": sum(row["answer_review"]["relevance"] == "answers" for row in supported),
                "citation_coverage": dict(Counter(row["citation_coverage"]["status"] for row in group)),
                "evidence_support_units": dict(Counter(unit["review"]["evidence_support"] for row in group for unit in row["units"])),
                "language_quality": dict(Counter(row["answer_review"]["language_quality"] for row in group)),
                "injection_probe_labels": dict(Counter(row["answer_review"]["injection_resistance"] for row in group if row["category"] == "injection")),
            }
        studies.append({
            "label": study_labels[study_index] if study_labels else f"Study {study_index + 1}: {packet['configuration'].get('model')} / {packet['configuration'].get('prompt_version', 'unspecified')}",
            "configuration": packet["configuration"], "binding": packet["binding"],
            "runtime_metadata": runtime_metadata[study_index] if runtime_metadata else None,
            "runtime_metadata_sha256": digest(runtime_metadata[study_index]) if runtime_metadata else None,
            "packet_sha256": summary["packet_sha256"], "annotations_sha256": summary["annotations_sha256"],
            "reviewer": summary["reviewer"], "review_status": summary["status"],
            "populations": summary["populations"], "condition_reviews": conditions, "runs": rows,
        })
    return {
        "schema_version": "semantic-review-publication.v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "development_only": True, "studies": studies, "persona_fidelity": "ungraded",
        "definitions": {
            "evidence_support": "Reviewer judgment against provided prompt evidence, not external-world truth.",
            "all_factual_units_supported": "At least one factual unit, every factual unit labeled supported; nonfactual wrappers/refusals are excluded. Relevance is counted separately.",
            "citation_coverage": "Mechanical trailing-marker coverage, never a semantic score.",
            "unit": "A sentence-like formatting unit; it may contain several claims and is not an atomic-fact annotation.",
        },
        "limitations": [
            "Inspect reviewer provenance: implementation-agent reviews are nonblind model-assisted development judgments, not independent human evaluations.",
            "The same small declared cases informed prompt revisions. These exploratory counts are not a held-out benchmark or a persona-fidelity score.",
            "Model sizes, runtimes, quantization, and prompt versions can differ across studies; use the configuration and runtime reports before making causal comparisons.",
            "Application gates and incomplete generations are excluded from completed-model response counts.",
            "No original transcripts or reasoning traces are included. Exact final model answers and their negative examples are retained for reviewability.",
        ],
    }


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create an immutable packet and blank annotation template")
    prepare.add_argument("--report", type=Path, required=True)
    prepare.add_argument("--corpus", type=Path, default=Settings.from_env().corpus_dir)
    prepare.add_argument("--output", type=Path, required=True)
    summarize = commands.add_parser("summarize", help="Validate explicit annotations and summarize their labels")
    summarize.add_argument("--packet", type=Path, required=True)
    summarize.add_argument("--annotations", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    publish = commands.add_parser("publish", help="Retain exact final answers and judgments in a compact review artifact")
    publish.add_argument("--review", type=Path, nargs=2, action="append", required=True, metavar=("PACKET", "ANNOTATIONS"))
    publish.add_argument("--label", action="append", help="Explicit study/model label; repeat in the same order as --review")
    publish.add_argument("--runtime-metadata", type=Path, action="append", help="Measured runtime metadata JSON; repeat in the same order as --review")
    publish.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix != ".json":
        parser.error("Choose a new .json output path")
    try:
        if args.command == "prepare":
            annotation_path = args.output.with_suffix(".annotations.json")
            if annotation_path.exists():
                parser.error("The companion annotation template already exists")
            raw = args.report.read_bytes()
            packet, annotations = prepare_packet(json.loads(raw), Corpus.load(args.corpus), report_sha256=hashlib.sha256(raw).hexdigest())
            _write_new(args.output, packet)
            _write_new(annotation_path, annotations)
            print(json.dumps({"packet": str(args.output), "annotations": str(annotation_path), "semantic_review": "ungraded"}))
        elif args.command == "summarize":
            summary = summarize_review(json.loads(args.packet.read_text()), json.loads(args.annotations.read_text()))
            _write_new(args.output, summary)
            print(json.dumps({"summary": str(args.output), "status": summary["status"], "populations": summary["populations"]}))
        else:
            publication = make_publication(
                [(json.loads(packet.read_text()), json.loads(annotations.read_text())) for packet, annotations in args.review],
                study_labels=args.label,
                runtime_metadata=[json.loads(path.read_text()) for path in args.runtime_metadata] if args.runtime_metadata else None,
            )
            _write_new(args.output, publication)
            print(json.dumps({"publication": str(args.output), "studies": len(publication["studies"]), "persona_fidelity": "ungraded"}))
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
