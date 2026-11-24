"""Reproducible retrieval regression, not a generated-answer likeness score."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bias.corpus import Corpus
from bias.jensen_public_retrieval import JensenPublicRetriever
from bias.retrieval import Retriever
from bias.snapshots import PersonaSnapshot, select_temporal

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data/evaluation/jensen-public-rag-2026-11-24.json"


def evaluate(output: Path) -> dict:
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    old = Corpus.load(ROOT / "data/personas/jensen-huang")
    values = Corpus.load(ROOT / "data/personas/jensen-huang-values-v2")
    expanded = Corpus.load(ROOT / "data/personas/jensen-huang-public-v3")
    snapshot = PersonaSnapshot.load(ROOT / "data/snapshots/jensen-public-2026-11-24.json", expanded)
    configurations = {"baseline_32_lexical": (old, Retriever(old)),
                      "baseline_60_lexical": (values, Retriever(values)),
                      "expanded_legacy_lexical": (expanded, Retriever(expanded)),
                      "expanded_public_bm25": (expanded, JensenPublicRetriever(expanded))}
    rows = []
    for name, (corpus, retriever) in configurations.items():
        for case in cases:
            started = time.perf_counter()
            result = retriever.search(case["question"], limit=4, behavior=True)
            ids = [hit.evidence.id for hit in result.hits]
            sources = [hit.evidence.source_id for hit in result.hits]
            expected = case.get("expected_sources", [])
            topical_hit = bool(set(expected) & set(sources)) if expected else not sources
            evidence_ids = case.get("expected_ids", [])
            # Source-family recall is comparable across old/new corpus IDs.
            # A new-record-only target must not make the old baseline fail by
            # construction. Specific new-turn recovery is reported separately.
            hit = topical_hit
            rows.append({"configuration": name, "case_id": case["id"], "language": case["language"],
                         "question": case["question"], "kind": case["kind"], "expected_sources": expected,
                         "expected_ids": evidence_ids, "retrieved_ids": ids, "retrieved_sources": sources,
                         "passed": hit, "specific_new_turn_hit": bool(set(evidence_ids) & set(ids)) if evidence_ids else None,
                         "reason": result.reason,
                         "retrieval_ms": round((time.perf_counter()-started)*1000, 3)})
    summary = {}
    for name in configurations:
        matching = [row for row in rows if row["configuration"] == name]
        positive = [row for row in matching if row["kind"] == "source_question"]
        controls = [row for row in matching if row["kind"] == "abstention_control"]
        summary[name] = {"positive_hits_at_4": sum(row["passed"] for row in positive),
                         "positive_cases": len(positive),
                         "controls_without_hits": sum(row["passed"] for row in controls),
                         "control_cases": len(controls),
                         "failures": [row["case_id"] for row in matching if not row["passed"]]}
    general = select_temporal(expanded, snapshot, "public interview experiences")
    report = {"schema_version": "jensen-public-retrieval-report.v1",
              "scope": "Authored development/regression questions. Tests retrieval of relevant records, not answer truth, human likeness, fluent speech, or unseen-interview generalization. Raw retriever comparison is distinct from production routing and temporal filtering.",
              "corpus_sha256": snapshot.corpus_sha256, "summary": summary,
              "snapshot_general_eligible_records": len(general.eligible_ids),
              "snapshot_general_exclusions": general.exclusion_reasons,
              "results": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/jensen-public-rag-2026-11-24/retrieval.json")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output)["summary"], ensure_ascii=False, indent=2))
