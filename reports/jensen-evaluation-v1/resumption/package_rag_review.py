"""Package only genuinely restored RAG results for supplied-evidence review.

This implements the preserved m6_rag_reviewer_plan.json, without changing the
registered diagnostic or treating it as a three-condition comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from tools.adaptation_evaluation.protocol import _context, digest  # noqa: E402
from tools.adaptation_evaluation.run import RATINGS  # noqa: E402


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def review_documents(plan, report, persistence):
    """Transform already-validated actual outputs; no retrieval or grading."""
    pools, references = {}, {}
    for index, case in enumerate(plan["cases"], 1):
        pool_id = f"pool-{index:04d}"
        messages = case["messages"]["facts"]
        context = _context(messages)
        facts = context["factual_evidence"]
        require(context["style_examples"] == [], "RAG pools cannot introduce style examples")
        pools[pool_id] = {
            "case_id": case["id"], "question": case["question"], "language": case["language"],
            "system_instruction": messages[0]["content"], "fixed_history": messages[2:-1],
            "factual_evidence": facts, "style_examples": [],
            "factual_sha256": case["factual_sha256"], "prompt_sha256": case["prompt_sha256"]["facts"],
            "evidence_pointers": [{"source_id": source["id"], "source_sha256": digest(source),
                "json_pointer": f"/pools/{pool_id}/factual_evidence/{position}"}
                for position, source in enumerate(facts)],
            "support_policy": "Only these actually supplied factual passages support factual claims. Empty evidence stays empty.",
        }
        references[pool_id] = {"case_id": case["id"], "reference_answer": case.get("reference_answer"),
            "reference_sha256": case["reference_sha256"], "role": "second_stage_only_not_supplied_evidence"}
    pool_by_case = {pool["case_id"]: key for key, pool in pools.items()}
    completed = [row for row in report["runs"] if row["generation"]["completed"] is True]
    secrets.SystemRandom().shuffle(completed)
    samples = []
    for index, row in enumerate(completed, 1):
        generation, raw = row["generation"], row["provider_generation"]
        require(raw.get("completed") is True and raw.get("error") is None and generation.get("error") is None
                and not row.get("validation_errors") and row["identity_check"] == {"before": "verified", "after": "verified"}
                and generation["text"] == raw["text"], "A completed sample lacks original output/identity integrity")
        pool_id = pool_by_case[row["case_id"]]
        samples.append({"id": f"sample-{index:04d}", "case_id": row["case_id"], "source_pool_id": pool_id,
            "question": pools[pool_id]["question"], "answer": generation["text"],
            "output_sha256": sha(generation["text"].encode()), "ratings": {key: None for key in RATINGS},
            "reviewer": None, "reviewer_type": None})
    counts = {"registered": len(plan["rag_diagnostic"]["registered_case_ids"]), "planned": len(plan["cases"]),
        "attempted": len(report["runs"]), "completed": len(completed),
        "failed": len(report["runs"]) - len(completed), "unattempted": len(plan["cases"]) - len(report["runs"])}
    require(all(counts[key] == persistence["observation"][key] for key in counts), "Verified RAG denominators differ")
    envelope = {"schema_version": "jensen-rag-agent-review.v1", "scope": "separate_single_base_offline_rag",
                "human_review": "ungraded", "efficacy_claim": False}
    documents = {
        "reviewer/samples.json": encode({**envelope, "samples": samples}),
        "reviewer/source-pools.json": encode({**envelope, "pools": pools}),
        "private/reference-answers.json": encode({**envelope, "references": references}),
        "private/execution-diagnostics.json": encode({**envelope, "denominators": counts,
            "failed_attempts": [row for row in report["runs"] if row["generation"]["completed"] is not True],
            "unattempted_case_ids": report.get("skipped_case_ids", [])}),
    }
    return documents, counts, len(pools)


README = """# Separate heldout RAG support review

Read rubric.md first. This packet contains one base-model diagnostic, not a
matched adapter comparison. Use samples.json and source-pools.json. Only the
exact captured factual passages can support a claim. Empty pools are deliberate;
do not borrow manually selected evidence, dropped retrieval hits, corpus-level
answerability explanations, private reference answers, or your own knowledge.
Preserve the actual system instruction when judging evidence-gap handling.

Record categorical agent support, citation applicability, requested coverage,
evidence limits, and language/format findings with evidence pointers and exact
output hashes. Human fields stay null. Failed/unattempted cases remain in the
coordinator's private denominator. Save support findings before requesting the
separate private reference answers for any second-stage comparison. Share only
this reviewer/ directory; the other files are coordinator material.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--artifact-receipt", type=Path, required=True)
    parser.add_argument("--inference-python", type=Path, default=ROOT / ".venv-inference/bin/python")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "Preserve existing reviewer packets")
    # Reuse the full final archive/member/process/health validation immediately
    # before reading actual private outputs. This is not a receipt-only check.
    subprocess.run([sys.executable, str(Path(__file__).with_name("deliver_remaining.py")), "rag-verify",
        "--mode", "final", "--artifact-root", str(args.artifact_root.absolute()),
        "--artifact-receipt", str(args.artifact_receipt.absolute()),
        "--inference-python", str(args.inference_python.absolute())], check=True)
    result_path = args.artifact_root / "m6-heldout-rag-bf16-v2-result.json"
    result_raw = result_path.read_bytes()
    persistence = json.loads(result_raw)
    require(persistence["mode"] == "final" and persistence["status"] == "saved_and_fresh_readback_verified"
            and persistence["observation"]["validation"] == "closed_invocation_revalidated",
            "Only final freshly verified RAG evidence can be reviewed")
    restored = Path(persistence["private_restored_root"])
    plan_raw = (restored / "inputs/plan.json").read_bytes()
    report_raw = (restored / "condition-output/condition/run.json").read_bytes()
    for name, raw in (("inputs/plan.json", plan_raw), ("condition-output/condition/run.json", report_raw)):
        require(persistence["archive_members"][name] == {"bytes": len(raw), "sha256": sha(raw)},
                "A freshly validated input changed before reviewer packaging")
    plan, report = json.loads(plan_raw), json.loads(report_raw)
    documents, counts, pool_count = review_documents(plan, report, persistence)
    documents["reviewer/README.md"] = README.encode()
    documents["reviewer/rubric.md"] = (ROOT / "reports/jensen-evaluation-v1/agent-review-rubric.md").read_bytes()
    documents["private/input-manifest.json"] = encode({
        "plan_file_sha256": sha(plan_raw), "report_file_sha256": sha(report_raw),
        "persistence_result_file_sha256": sha(result_raw), "persistence_result": persistence,
        "packet_tool_sha256": sha(Path(__file__).read_bytes()),
    })
    manifest = {"schema_version": "jensen-rag-agent-review-manifest.v1", "plan_sha256": plan["plan_sha256"],
        "denominators": counts, "source_pools": pool_count, "review_samples": counts["completed"],
        "share_only": "reviewer/", "private_references_included": True, "human_review": "ungraded", "efficacy_claim": False,
        "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(documents.items())}}
    args.output.mkdir(parents=True, mode=0o700)
    for name, raw in documents.items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("xb") as stream:
            stream.write(raw)
    with (args.output / "manifest.json").open("xb") as stream:
        stream.write(encode(manifest))
    print(json.dumps({"output": str(args.output), "denominators": counts, "review_samples": counts["completed"],
        "source_pools": pool_count, "human_review": "ungraded", "remote_persistence_performed": False}))


if __name__ == "__main__":
    main()
