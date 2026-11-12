"""Execute the predeclared multilingual retrieval comparison on private corpus bytes."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from bias.corpus import Corpus
from bias.hybrid_retrieval import HybridPolicy, HybridRetriever, SentenceTransformerEmbedder
from tools.retrieval_development import sha


def run(corpus_path: Path, settings_path: Path, output: Path):
    import torch
    settings = json.loads(settings_path.read_text())
    if settings.get("exposure") != "development" or output.exists():
        raise ValueError("Use frozen development settings and a new output directory")
    root = Path(__file__).resolve().parents[1]
    cases_path = root / settings["cases_path"]
    if sha(cases_path) != settings["parent_protocol_sha256"]:
        raise ValueError("Development questions changed after hybrid protocol freeze")
    protocol = json.loads(cases_path.read_text())
    for name, expected in protocol["files_sha256"].items():
        if Path(name).name != name or sha(corpus_path / name) != expected:
            raise ValueError("Frozen corpus changed")
    corpus = Corpus.load(corpus_path)
    torch.set_num_threads(settings["threads"])
    embedder = SentenceTransformerEmbedder(settings["model"], revision=settings["revision"],
        query_prefix=settings["query_prefix"], passage_prefix=settings["passage_prefix"],
        device=settings["device"], max_tokens=settings["max_tokens"], batch_size=settings["batch_size"])
    retriever = HybridRetriever(corpus, embedder,
        policy=HybridPolicy(dense_min_similarity=settings["dense_min_similarity"], max_style_examples=0))
    output.mkdir(parents=True)
    started = time.perf_counter()
    retriever.warm_up()
    build_seconds = time.perf_counter() - started
    print(json.dumps({"stage": "index_ready", "seconds": round(build_seconds, 3),
                      "records": len(retriever.records)}), flush=True)
    rows = []
    for case in protocol["cases"]:
        timings, first_ids = [], None
        for _ in range(protocol["repetitions"]):
            started = time.perf_counter()
            result = retriever.search(case["question"], limit=protocol["limit"], include_style=False)
            timings.append((time.perf_counter() - started) * 1000)
            ids = [hit.evidence.id for hit in result.hits]
            if first_ids is not None and ids != first_ids:
                raise ValueError("Repeated retrieval result changed")
            first_ids = ids
        relevant = set(case["relevant_ids"])
        rows.append({"id": case["id"], "kind": case["kind"], "language": case["language"],
            "retrieved_ids": ids, "gold_hit_at_1": bool(ids and ids[0] in relevant),
            "gold_hit_at_k": bool(set(ids) & relevant), "abstained": not bool(ids),
            "reason": result.reason, "cold_query_ms": round(timings[0], 4),
            "warm_query_median_ms": round(statistics.median(timings[1:]), 4),
            "indexed_units": result.trace.get("indexed_units"),
            "abstention_reason": result.trace.get("abstention_reason")})
        (output / (case["id"] + ".trace.json")).write_text(json.dumps(result.trace, ensure_ascii=False, indent=2) + "\n")
        (output / "progress.json").write_text(json.dumps({"completed": len(rows), "runs": rows}, indent=2) + "\n")
        print(json.dumps({"case": case["id"], "gold_hit_at_k": rows[-1]["gold_hit_at_k"], "abstained": not bool(ids)}), flush=True)
    supported = [r for r in rows if r["kind"] == "supported"]
    controls = [r for r in rows if r["kind"] == "unsupported"]
    baseline = json.loads((root / "reports/jensen-retrieval-v2/comparison-v1.json").read_text())["conditions"]["archived-lexical"]["runs"]
    baseline = {r["id"]: r for r in baseline}
    regressions = [r["id"] for r in supported if baseline[r["id"]]["gold_hit_at_k"] and not r["gold_hit_at_k"]]
    recall = sum(r["gold_hit_at_k"] for r in supported) / len(supported)
    false_hits = sum(not r["abstained"] for r in controls)
    gates = protocol["acceptance"]
    result = {"schema_version": "jensen-hybrid-comparison.v1", "exposure": "development",
        "settings_sha256": sha(settings_path), "protocol_sha256": sha(cases_path),
        "corpus_files_sha256": protocol["files_sha256"], "embedding": embedder.identity,
        "policy": asdict(retriever.policy), "index_build_seconds": round(build_seconds, 3),
        "indexed_records": len(retriever.records), "indexed_units": max(r.get("indexed_units") or 0 for r in rows),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "sentence-transformers")},
        "supported_recall_at_1": sum(r["gold_hit_at_1"] for r in supported) / len(supported),
        "supported_recall_at_k": recall, "unsupported_false_hit_count": false_hits,
        "regressions_from_baseline": regressions,
        "development_gates_passed": recall >= gates["supported_recall_at_4_min"] and false_hits <= gates["unsupported_false_hit_count_max"] and len(regressions) <= gates["regressions_from_baseline_max"],
        "promotion": "retain_default_lexical_keep_hybrid_opt_in", "runs": rows,
        "median_cold_query_ms": round(statistics.median(r["cold_query_ms"] for r in rows), 4),
        "median_warm_query_ms": round(statistics.median(r["warm_query_median_ms"] for r in rows), 4),
        "citation_entailment_evaluated": False, "human_fidelity_evaluated": False,
        "implementation_sha256": {n: sha(root / n) for n in ("src/bias/hybrid_retrieval.py", "src/bias/retrieval.py", "src/bias/query_policy.py", "tools/retrieval_hybrid_development.py")},
        "limitations": settings["limitations"] + protocol["limitations"]}
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("supported_recall_at_k", "unsupported_false_hit_count", "development_gates_passed")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.corpus, args.settings, args.output)


if __name__ == "__main__":
    main()
