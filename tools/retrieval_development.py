"""Compare frozen development cases without distributing original source text."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from bias.corpus import Corpus
from bias.jensen_retrieval import JensenConceptRetriever, VERSION
from bias.retrieval import Retriever, RETRIEVER_VERSION


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(corpus_path: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("exposure") != "development":
        raise ValueError("This candidate comparison is development only")
    for name, expected in protocol["files_sha256"].items():
        if Path(name).name != name or sha(corpus_path / name) != expected:
            raise ValueError("Frozen corpus bytes changed")
    corpus = Corpus.load(corpus_path)
    eligible = {e.id for e in corpus.training_evidence}
    if any(not set(case["relevant_ids"]) <= eligible for case in protocol["cases"]):
        raise ValueError("Relevance labels include an unavailable/non-training record")
    conditions = {}
    for name, cls, version in (("archived-lexical", Retriever, RETRIEVER_VERSION),
                               ("jensen-concept-lexical", JensenConceptRetriever, VERSION)):
        started = time.perf_counter()
        retriever = cls(corpus)
        build_ms = (time.perf_counter() - started) * 1000
        runs = []
        for case in protocol["cases"]:
            timings, expected_ids = [], None
            for _ in range(protocol["repetitions"]):
                started = time.perf_counter()
                result = retriever.search(case["question"], limit=protocol["limit"])
                timings.append((time.perf_counter() - started) * 1000)
                ids = [hit.evidence.id for hit in result.hits]
                if expected_ids is not None and ids != expected_ids:
                    raise ValueError("A repeated deterministic search changed its result")
                expected_ids = ids
            relevant = set(case["relevant_ids"])
            runs.append({"id": case["id"], "kind": case["kind"], "language": case["language"],
                "retrieved_ids": ids, "gold_hit_at_1": bool(ids and ids[0] in relevant),
                "gold_hit_at_k": bool(set(ids) & relevant), "abstained": not bool(ids),
                "confidence": result.confidence, "reason": result.reason,
                "median_search_ms": round(statistics.median(timings), 4),
                "matched_terms": [list(hit.matched_terms) for hit in result.hits]})
        supported = [row for row in runs if row["kind"] == "supported"]
        controls = [row for row in runs if row["kind"] == "unsupported"]
        conditions[name] = {"version": version, "index_build_ms": round(build_ms, 4),
            "indexed_records": len(retriever.records), "runs": runs,
            "supported_recall_at_1": sum(row["gold_hit_at_1"] for row in supported) / len(supported),
            "supported_recall_at_k": sum(row["gold_hit_at_k"] for row in supported) / len(supported),
            "unsupported_false_hit_count": sum(not row["abstained"] for row in controls),
            "median_query_ms": round(statistics.median(row["median_search_ms"] for row in runs), 4)}
    baseline = {r["id"]: r for r in conditions["archived-lexical"]["runs"]}
    candidate = conditions["jensen-concept-lexical"]
    regressions = [r["id"] for r in candidate["runs"]
                   if baseline[r["id"]]["gold_hit_at_k"] and not r["gold_hit_at_k"]]
    gates = protocol["acceptance"]
    passed = (candidate["supported_recall_at_k"] >= gates["supported_recall_at_4_min"]
              and candidate["unsupported_false_hit_count"] <= gates["unsupported_false_hit_count_max"]
              and len(regressions) <= gates["regressions_from_baseline_max"])
    root = Path(__file__).resolve().parents[1]
    return {"schema_version": "jensen-retrieval-comparison.v1", "exposure": "development",
        "protocol_sha256": sha(protocol_path), "files_sha256": protocol["files_sha256"],
        "implementation_sha256": {name: sha(root / name) for name in (
            "src/bias/retrieval.py", "src/bias/jensen_retrieval.py", "src/bias/query_policy.py",
            "tools/retrieval_development.py")},
        "conditions": conditions, "regressions_from_baseline": regressions,
        "development_gates_passed": passed, "promotion": "retain_default_lexical_keep_candidate_opt_in",
        "citation_entailment_evaluated": False, "human_fidelity_evaluated": False,
        "limitations": protocol["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new results path")
    result = compare(args.corpus, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"conditions": {k: {m: v[m] for m in (
        "supported_recall_at_1", "supported_recall_at_k", "unsupported_false_hit_count", "median_query_ms")}
        for k, v in result["conditions"].items()}, "development_gates_passed": result["development_gates_passed"]}))


if __name__ == "__main__":
    main()
