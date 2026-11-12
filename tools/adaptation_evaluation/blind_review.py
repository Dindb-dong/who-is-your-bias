"""Package collected M6 outputs for source-grounded, condition-blind review.

This offline helper reads finished collector artifacts. It never runs inference,
changes frozen inputs, or substitutes for the collector's receipt validation.
Only the resulting reviewer/ directory is intended for blinded reviewers.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import secrets
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .protocol import CONDITIONS, _context, digest, validate_plan
from .run import RATINGS


SCHEMA = "text-agent-blind-review.v1"
SHA256 = re.compile(r"[a-f0-9]{64}")
COLLECTION_FILES = (
    "comparison.json", "review-summary.json", "blind-review/reviewer/samples.json",
    "blind-review/private/answer-key.json", "blind-review/private/failed-attempts.json",
    "blind-review/private/comparison.json",
)
DEFAULT_RUBRIC = Path(__file__).resolve().parents[2] / "reports/jensen-evaluation-v1/agent-review-rubric.md"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path):
    raw = path.read_bytes()
    return json.loads(raw), _sha(raw)


def _write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _count(value) -> bool:
    return type(value) is int and value >= 0


def _load_collected(plan_path: Path, directory: Path) -> dict:
    plan, plan_file_sha = _read(plan_path)
    validate_plan(plan)
    loaded, hashes = {}, {}
    for name in COLLECTION_FILES:
        loaded[name], hashes[name] = _read(directory / name)
    summary = loaded["comparison.json"]
    original = loaded["blind-review/private/comparison.json"]
    core = {key: value for key, value in summary.items() if key != "local_execution"}
    execution = summary.get("local_execution", {})
    supervisors = execution.get("supervisors", [])
    if (core != original or summary.get("schema_version") != "text-adaptation-comparison-summary.v1"
            or summary.get("plan_sha256") != plan["plan_sha256"]
            or summary.get("exposure") != plan["exposure"]
            or set(summary.get("conditions", {})) != set(CONDITIONS)
            or summary.get("human_review") != "ungraded" or summary.get("efficacy_claim") is not False
            or len(supervisors) != len(CONDITIONS)
            or {row.get("condition") for row in supervisors} != set(CONDITIONS)
            or any(not SHA256.fullmatch(execution.get(key, ""))
                   for key in ("contract_sha256", "durability_receipts_file_sha256"))
            or any(not SHA256.fullmatch(row.get("supervisor_sha256", "")) for row in supervisors)):
        raise ValueError("A matching finished three-condition collector package is required")

    legacy = loaded["blind-review/reviewer/samples.json"]
    key = loaded["blind-review/private/answer-key.json"]
    failures = loaded["blind-review/private/failed-attempts.json"]
    review = loaded["review-summary.json"]
    if (legacy.get("plan_sha256") != plan["plan_sha256"] or not isinstance(key, dict)
            or not isinstance(failures, list) or not isinstance(legacy.get("samples"), list)
            or review.get("human_review") != "ungraded"):
        raise ValueError("Collected review metadata differs from its frozen plan")
    cases = {case["id"]: case for case in plan["cases"]}
    samples = legacy["samples"]
    identifiers = [sample.get("id") for sample in samples]
    if (len(set(identifiers)) != len(identifiers) or set(identifiers) != set(key)
            or review.get("samples") != len(samples) or review.get("failed_attempts") != len(failures)):
        raise ValueError("Collected review samples, key or failure counts are incomplete")

    seen, complete_counts, failure_counts = set(), Counter(), Counter()
    for sample in samples:
        entry = key[sample["id"]]
        case = cases.get(entry.get("case_id"))
        condition = entry.get("condition")
        pair = (condition, entry.get("case_id"))
        if case is None or condition not in CONDITIONS or pair in seen:
            raise ValueError("Duplicate or foreign completed case in collected review")
        seen.add(pair)
        complete_counts[condition] += 1
        if (sample.get("case_id") != case["id"] or sample.get("question") != case["question"]
                or sample.get("reference_answer") != case.get("reference_answer")
                or sample.get("factual_evidence") != _context(case["messages"]["facts"])["factual_evidence"]
                or not isinstance(sample.get("answer"), str)
                or _sha(sample["answer"].encode()) != entry.get("output_sha256")
                or summary["conditions"][condition].get("output_sha256", {}).get(case["id"]) != entry.get("output_sha256")
                or sample.get("ratings") != {name: None for name in RATINGS}
                or sample.get("reviewer") is not None or sample.get("reviewer_type") is not None):
            raise ValueError("Collected answer, evidence, reference, hash or ungraded fields changed")
    for failure in failures:
        condition, case_id = failure.get("condition"), failure.get("case_id")
        pair = (condition, case_id)
        if condition not in CONDITIONS or case_id not in cases or pair in seen:
            raise ValueError("Duplicate or foreign failed attempt in collected review")
        seen.add(pair)
        failure_counts[condition] += 1

    counts, unattempted = {}, []
    for condition in CONDITIONS:
        result = summary["conditions"][condition]
        attempted, completed = result.get("attempted"), result.get("completed")
        outputs = result.get("output_sha256", {})
        attempted_ids = {case_id for c, case_id in seen if c == condition}
        failure_errors = Counter(row.get("error") for row in failures
                                 if row.get("condition") == condition and row.get("error"))
        if (not _count(attempted) or not _count(completed) or not isinstance(outputs, dict)
                or completed != complete_counts[condition]
                or attempted != completed + failure_counts[condition]
                or set(outputs) != attempted_ids or not attempted_ids <= set(cases)
                or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in outputs.values())
                or result.get("errors") != dict(failure_errors)
                or result.get("run_status") not in {"completed", "failed", "deadline_exhausted"}):
            raise ValueError("Collected complete/failed/unattempted denominators are inconsistent")
        skipped = [case_id for case_id in cases if case_id not in attempted_ids]
        counts[condition] = {"planned": len(cases), "attempted": attempted, "completed": completed,
                             "failed": failure_counts[condition], "unattempted": len(skipped)}
        unattempted.extend({"condition": condition, "case_id": case_id} for case_id in skipped)
    return {"plan": plan, "samples": samples, "key": key, "failures": failures,
            "summary": summary, "counts": counts, "unattempted": unattempted,
            "input_manifest": {"plan_path": str(plan_path.resolve()), "plan_file_sha256": plan_file_sha,
                "plan_sha256": plan["plan_sha256"], "collection_path": str(directory.resolve()),
                "collection_files_sha256": hashes}}


def _pool(case: dict, pool_id: str, exposure: str) -> dict:
    facts = _context(case["messages"]["facts"])["factual_evidence"]
    styles = _context(case["messages"]["style"])["style_examples"]
    if ([source.get("example_id") for source in styles] != case["style_ids"]
            or len(set(case["style_ids"])) != len(styles)):
        raise ValueError("Candidate style source IDs differ from the frozen plan")
    pointers = []
    for role, sources in (("factual_evidence", facts), ("style_examples", styles)):
        for index, source in enumerate(sources):
            pointers.append({"role": role, "source_id": source.get("id", source.get("example_id")),
                "json_pointer": f"/pools/{pool_id}/{role}/{index}", "source_sha256": digest(source)})
    return {"case_id": case["id"], "language": case["language"], "exposure": exposure,
            "question": case["question"], "system_instruction": case["messages"]["facts"][0]["content"],
            "fixed_history": copy.deepcopy(case["messages"]["facts"][2:-1]),
            "factual_evidence": copy.deepcopy(facts), "style_examples": copy.deepcopy(styles),
            "evidence_pointers": pointers,
            "support_policy": "Only factual_evidence supports factual claims; style_examples are common candidate wording sources, not evidence that an output received them."}


REVIEWER_README = """# Blinded response review

Read `rubric.md` before reviewing. `samples.json` contains complete, unchanged
answers and null human fields. Look up each sample's `source_pool_id` in
`source-pools.json`; every output for that case shares the exact same factual
and candidate style pool. Evidence pointers address that JSON file. The saved
system instruction and fixed history describe the applicable constraints.

Assess support and requested coverage against the factual passages first.
`reference-answers.json` is a separate second-stage comparison aid: those
answers were not supplied to generation and cannot rescue unsupported claims.
Candidate style sources can expose copying; they cannot support factual claims
or reveal whether a particular answer actually received them.

Leave human ratings null unless an identified human supplies them. Record agent
findings separately. This package performs no semantic grading or inference.
Failed and unattempted cases remain in the coordinator's private denominator
record and are not silently promoted to completed samples.

Share only this `reviewer/` directory. Do not give reviewers the sibling
`private/` directory, original collection, condition key, or input manifest.
Metadata does not identify conditions; model-generated content is retained
verbatim and may itself reveal stylistic or identity clues. Record prior
condition exposure rather than claiming perfect blinding in that situation.
"""


def package_reviews(inputs: list[tuple[Path, Path]], output: Path, *, rubric: Path = DEFAULT_RUBRIC) -> dict:
    """Validate completed collections, then create a new offline review package."""
    if not inputs or output.exists():
        raise ValueError("Provide at least one plan/collection pair and a new output directory")
    bundles = [_load_collected(plan, directory) for plan, directory in inputs]
    if len({bundle["plan"]["plan_sha256"] for bundle in bundles}) != len(bundles):
        raise ValueError("A frozen plan may appear only once in a review package")
    rubric_bytes = rubric.read_bytes()
    pools, references, candidates, diagnostics = {}, {}, [], []
    for bundle in bundles:
        plan = bundle["plan"]
        pool_ids = {}
        for case in plan["cases"]:
            pool_id = f"pool-{len(pools) + 1:04d}"
            pool_ids[case["id"]] = pool_id
            pools[pool_id] = _pool(case, pool_id, plan["exposure"])
            references[pool_id] = {"case_id": case["id"], "reference_answer": case.get("reference_answer"),
                "reference_sha256": case["reference_sha256"],
                "role": "comparison_only_not_supplied_factual_evidence"}
        for sample in bundle["samples"]:
            candidates.append((sample, bundle["key"][sample["id"]], plan["plan_sha256"], pool_ids[sample["case_id"]]))
        diagnostics.append({"plan_sha256": plan["plan_sha256"], "exposure": plan["exposure"],
            "denominators": bundle["counts"], "failed_attempts": bundle["failures"],
            "unattempted_cases": bundle["unattempted"], "collected_comparison": bundle["summary"]})
    # A fresh OS-random permutation prevents reconstructing the original
    # exporter's public seed/order. Its realized mapping survives in the key.
    secrets.SystemRandom().shuffle(candidates)
    samples, key = [], {}
    for index, (legacy, entry, plan_sha, pool_id) in enumerate(candidates, 1):
        identifier = f"sample-{index:04d}"
        samples.append({"id": identifier, "case_id": legacy["case_id"], "source_pool_id": pool_id,
            "question": legacy["question"], "answer": legacy["answer"],
            "ratings": {name: None for name in RATINGS}, "reviewer": None, "reviewer_type": None})
        key[identifier] = {**entry, "plan_sha256": plan_sha, "source_pool_id": pool_id,
                           "original_sample_id": legacy["id"]}
    envelope = {"schema_version": SCHEMA, "human_review": "ungraded", "efficacy_claim": False}
    documents = {
        "reviewer/samples.json": {**envelope, "samples": samples},
        "reviewer/source-pools.json": {**envelope, "pools": pools},
        "reviewer/reference-answers.json": {**envelope, "references": references},
        "private/answer-key.json": {**envelope, "key": key},
        "private/execution-diagnostics.json": {**envelope, "inputs": diagnostics},
        "private/input-manifest.json": {**envelope, "inputs": [bundle["input_manifest"] for bundle in bundles]},
    }
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "reviewer").mkdir(mode=0o700)
    (output / "private").mkdir(mode=0o700)
    for name, value in documents.items():
        _write(output / name, value)
    (output / "reviewer/README.md").write_text(REVIEWER_README, encoding="utf-8")
    (output / "reviewer/rubric.md").write_bytes(rubric_bytes)
    totals = Counter()
    for bundle in bundles:
        for counts in bundle["counts"].values():
            totals.update(counts)
    manifest = {**envelope, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "packaged_ungraded", "plans": len(bundles), "source_pools": len(pools),
        "denominators": dict(totals), "review_samples": len(samples), "share_only": "reviewer/",
        "packager_sha256": _sha(Path(__file__).read_bytes()), "rubric_sha256": _sha(rubric_bytes),
        "validation_scope": "Checks collector package consistency and complete answer hashes; does not contact remote storage or re-execute identity, generation or semantic checks.",
        "files_sha256": {str(path.relative_to(output)): _sha(path.read_bytes())
                         for path in sorted(output.rglob("*")) if path.is_file()}}
    # The manifest is written last; its absence means packaging did not finish.
    _write(output / "manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", nargs=2, type=Path, required=True,
                        metavar=("PLAN", "COLLECTION"), help="Repeat for development and heldout collections")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    args = parser.parse_args(argv)
    try:
        result = package_reviews(args.input, args.output, rubric=args.rubric)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"Blind review packaging failed: {exc}\n")
    print(json.dumps({key: result[key] for key in ("status", "plans", "review_samples", "denominators", "human_review", "share_only")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
