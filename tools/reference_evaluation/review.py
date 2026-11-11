"""Export condition-blinded comparison material and ingest actual judgments."""

from __future__ import annotations

import argparse
import html
import json
import secrets
import statistics
from pathlib import Path

from .protocol import digest, load_inputs, qa_data, strip_citations

DIMENSIONS = ("content_viewpoint", "wording_structure", "situation_response")
RUBRIC = {
    "content_viewpoint": "1 contradicts the reference position; 2 mostly misses it; 3 partial alignment; 4 close with minor differences; 5 same substantive position.",
    "wording_structure": "1 unlike the reference wording and structure; 2 weak resemblance; 3 some similar patterns; 4 close; 5 strongly similar phrasing and sentence structure. Do not reward verbatim copying by itself.",
    "situation_response": "1 mismatches the question's conversational situation; 2 weak fit; 3 plausible but generic; 4 close response pattern; 5 same kind of reaction as the reference. Leave null when context is insufficient.",
    "factual_support": "Reviewed separately with supplied training evidence: supported, partly_supported, unsupported, not_assessable. Similarity to an actual answer does not prove support from retrieved sources.",
}


def export_bundle(report: dict, corpus, output: Path) -> dict:
    if output.exists():
        raise ValueError("Choose a new review directory")
    if report.get("schema_version") != "actual-qa-report.v1":
        raise ValueError("Unsupported reference report")
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Report runs must be an array")
    identities = [(row["case_id"], row["repetition"], row["condition"]) for row in runs]
    if len(set(identities)) != len(identities) or any(condition not in {"A", "B", "C"} for _, _, condition in identities):
        raise ValueError("Each case/repetition/condition must occur at most once")
    private, reviewer = output / "private", output / "reviewer"
    private.mkdir(parents=True)
    reviewer.mkdir()
    groups = {}
    for row in runs:
        groups.setdefault((row["case_id"], row["repetition"]), []).append(row)
    samples, answer_key, support = [], {}, {}
    for group_key in sorted(groups):
        rows = list(groups[group_key])
        secrets.SystemRandom().shuffle(rows)
        for row in rows:
            qa = qa_data(corpus.evidence[row["reference_id"]])
            sample_id = "sample-" + secrets.token_hex(6)
            samples.append({"sample_id": sample_id, "question": qa["question"], "reference_answer": qa["answer"], "reference_completeness": qa.get("completeness", "unknown"), "reference_excerpt_note": qa.get("excerpt_note"), "context_before": qa.get("context_before", ""), "context_kind": qa.get("context_kind", "none"), "candidate": strip_citations(row["generation"]["text"]), "generation_complete": row["generation"]["completed"], "language": row["language"]})
            answer_key[sample_id] = {"case_id": row["case_id"], "condition": row["condition"], "repetition": row["repetition"], "source_event_id": row["source_event_id"], "completed": row["generation"]["completed"]}
            support[sample_id] = {"raw_candidate": row["generation"]["text"], "training_examples": [qa_data(corpus.evidence[item["id"]]) for item in row["prompt_audit"]["selected"]], "note": "Open only after the blinded comparison; the evidence may reveal the condition."}
    # Commit to the private condition mapping without exposing it to reviewers.
    # The private salt prevents brute-forcing a small number of condition labels.
    key_payload = {"report_sha256": digest(report), "key": answer_key, "salt": secrets.token_hex(32)}
    public = {"schema_version": "reference-review-bundle.v1", "report_sha256": digest(report), "answer_key_commitment": digest(key_payload), "rubric": RUBRIC, "samples": samples, "blinding": "Condition labels, record IDs, scores and citation markers are withheld. Wording may still reveal a condition. Keep private/ separate from reviewers."}
    bundle_sha = digest(public)
    template = {"schema_version": "reference-review.v1", "bundle_sha256": bundle_sha, "reviewer_id": "", "reviewer_kind": None, "method": "", "judgments": [{"sample_id": sample["sample_id"], **{key: None for key in DIMENSIONS}, "factual_support": None, "note": ""} for sample in samples]}
    (reviewer / "samples.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n")
    (reviewer / "review-template.json").write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n")
    (private / "answer-key.json").write_text(json.dumps({"bundle_sha256": bundle_sha, **key_payload}, indent=2) + "\n")
    (private / "support-review.json").write_text(json.dumps(support, ensure_ascii=False, indent=2) + "\n")
    articles = []
    for sample in samples:
        articles.append("<article><h2>" + html.escape(sample["sample_id"]) + "</h2><h3>Adjacent context (" + html.escape(sample["context_kind"]) + ")</h3><p>" + html.escape(sample["context_before"] or "Not supplied") + "</p><h3>Question</h3><p>" + html.escape(sample["question"]) + "</p><h3>Actual reference (" + html.escape(sample["reference_completeness"]) + ")</h3><p>" + html.escape(sample["reference_answer"]) + "</p><p>Reference scope: " + html.escape(sample["reference_excerpt_note"] or "Complete supplied exchange") + "</p><h3>Candidate</h3><p>" + html.escape(sample["candidate"]) + "</p><p>Generation " + ("complete" if sample["generation_complete"] else "incomplete — record missing quality judgments explicitly") + "</p></article>")
    (reviewer / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Blinded interview comparison</title><style>body{max-width:900px;margin:3rem auto;padding:0 1rem;font:16px/1.6 system-ui;background:#f6f5f0;color:#202425}article{padding:1.5rem;margin:1.5rem 0;background:white;border:1px solid #d9ddda;border-radius:12px}h3{font-size:14px;color:#466057}p{white-space:pre-wrap}</style><h1>Interview response comparison</h1><p>Use review-template.json. Leave uncertain judgments null. Record reviewer kind and method. Keep private/ away from reviewers.</p>' + ''.join('<p><b>' + html.escape(key) + '</b>: ' + html.escape(value) + '</p>' for key, value in RUBRIC.items()) + ''.join(articles) + '</html>')
    return {"samples": len(samples), "bundle_sha256": bundle_sha, "human_ratings": 0, "path": str(output)}


def ingest_review(review: dict, bundle: dict, key: dict) -> dict:
    bundle_sha = digest(bundle)
    if review.get("schema_version") != "reference-review.v1" or review.get("bundle_sha256") != bundle_sha or key.get("bundle_sha256") != bundle_sha:
        raise ValueError("Review, blinded samples and answer key do not match")
    key_payload = {name: key.get(name) for name in ("report_sha256", "key", "salt")}
    if digest(key_payload) != bundle.get("answer_key_commitment") or key.get("report_sha256") != bundle.get("report_sha256"):
        raise ValueError("Private answer key differs from its frozen report commitment")
    if review.get("reviewer_kind") not in {"human", "model_assisted"} or not isinstance(review.get("reviewer_id"), str) or not review["reviewer_id"].strip() or not isinstance(review.get("method"), str) or not review["method"].strip():
        raise ValueError("Declare the actual reviewer and method; never label model review as human")
    expected = {sample["sample_id"] for sample in bundle["samples"]}
    if set(key.get("key", {})) != expected:
        raise ValueError("Answer key sample set mismatch")
    if any(type(key["key"][sample["sample_id"]].get("completed")) is not bool or key["key"][sample["sample_id"]]["completed"] != sample["generation_complete"] or key["key"][sample["sample_id"]].get("condition") not in {"A", "B", "C"} for sample in bundle["samples"]):
        raise ValueError("Public completion state and private experiment mapping disagree")
    judgments = review.get("judgments")
    if not isinstance(judgments, list):
        raise ValueError("Judgments must be an array")
    seen, rows = set(), []
    for judgment in judgments:
        sample = judgment.get("sample_id")
        if sample not in expected or sample in seen:
            raise ValueError("Unknown or duplicate review sample")
        seen.add(sample)
        for dimension in DIMENSIONS:
            score = judgment.get(dimension)
            if score is not None and (type(score) is not int or not 1 <= score <= 5):
                raise ValueError("Ordinal ratings must be integers 1–5 or null")
        if judgment.get("factual_support") not in {None, "supported", "partly_supported", "unsupported", "not_assessable"}:
            raise ValueError("Invalid separate source-support judgment")
        if not key["key"][sample]["completed"] and any(judgment.get(d) is not None for d in DIMENSIONS):
            raise ValueError("Incomplete model responses remain ungraded for fidelity; keep failure notes")
        if any(judgment.get(d) is not None for d in DIMENSIONS) and not str(judgment.get("note", "")).strip():
            raise ValueError("Ratings require an evidence-based reviewer note")
        rows.append({**judgment, **key["key"][sample]})
    comparisons = {}
    for dimension in DIMENSIONS:
        by_case = {}
        for row in rows:
            if row.get(dimension) is not None:
                by_case.setdefault((row["case_id"], row["repetition"], row["source_event_id"]), {})[row["condition"]] = row[dimension]
        differences = [(case[2], values["C"] - values["B"]) for case, values in by_case.items() if {"B", "C"}.issubset(values)]
        by_event = {}
        for event, difference in differences:
            by_event.setdefault(event, []).append(difference)
        event_means = {event: statistics.mean(values) for event, values in by_event.items()}
        mean = statistics.mean(event_means.values()) if event_means else None
        comparisons[dimension] = {"paired_cases": len(differences), "independent_events": len(event_means), "event_mean_C_minus_B": event_means, "event_balanced_C_minus_B": mean, "descriptive_result": "ungraded" if mean is None else "C_higher" if mean > 0 else "C_lower" if mean < 0 else "equal", "inference": "descriptive_pilot_no_significance_claim"}
    return {"schema_version": "reference-review-summary.v1", "bundle_sha256": bundle_sha, "report_sha256": key["report_sha256"], "reviewer_kind": review["reviewer_kind"], "reviewer_id": review["reviewer_id"], "method": review["method"], "submitted_samples": len(rows), "missing_samples": sorted(expected - seen), "graded_samples": sum(any(row.get(d) is not None for d in DIMENSIONS) for row in rows), "dimensions": comparisons, "source_support": dict(__import__("collections").Counter(row["factual_support"] for row in rows if row.get("factual_support") is not None)), "human_review_complete": review["reviewer_kind"] == "human" and len(rows) == len(expected) and all(all(row.get(d) is not None for d in DIMENSIONS) for row in rows), "judgments": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--corpus", required=True, type=Path)
    export.add_argument("--plan", required=True, type=Path)
    export.add_argument("--report", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--bundle", required=True, type=Path)
    ingest.add_argument("--review", required=True, type=Path)
    ingest.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "export":
        corpus, plan, _ = load_inputs(args.corpus, args.plan)
        report = json.loads(args.report.read_text())
        if report.get("plan_sha256") != digest(plan.model_dump(mode="json")):
            raise ValueError("Report does not match the frozen reference plan")
        print(json.dumps(export_bundle(report, corpus, args.output)))
    else:
        result = ingest_review(json.loads(args.review.read_text()), json.loads((args.bundle / "reviewer/samples.json").read_text()), json.loads((args.bundle / "private/answer-key.json").read_text()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({key: value for key, value in result.items() if key != "judgments"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
