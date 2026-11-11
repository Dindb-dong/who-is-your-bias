"""Run actual dense inference; retain misses, false support and model identity."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from bias.corpus import Corpus, source_event_id
from bias.hybrid_retrieval import CrossEncoderReranker, HYBRID_VERSION, HybridPolicy, HybridRetriever, SentenceTransformerEmbedder
from bias.retrieval import RETRIEVER_VERSION, Retriever


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def capture_inputs(paths: dict[str, Path]) -> dict[str, str | None]:
    """Snapshot optional/required inputs without silently ignoring deletions."""
    return {name: digest(path) if path.is_file() else None for name, path in paths.items()}


def input_integrity(before: dict, after: dict) -> dict:
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    return {"stable": not changed, "changed_inputs": changed, "before": before, "after": after,
            "scope": "A run with changed inputs is diagnostic only and must be rerun before comparison claims."}


def validate_cases(payload: dict, corpus: Corpus) -> list[dict]:
    if payload.get("schema_version") != "retrieval-development-cases.v1" or payload.get("split") != "development":
        raise ValueError("This runner accepts explicitly registered development cases only")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty list")
    identifiers = set()
    training = {record.id for record in corpus.training_evidence}
    for item in cases:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] in identifiers:
            raise ValueError("Case IDs must be unique strings")
        identifiers.add(item["id"])
        if not isinstance(item.get("query"), str) or not item["query"].strip() or item.get("language") not in {"en", "ko"}:
            raise ValueError("Cases need a nonblank query and declared language")
        if item.get("answerability") not in {"answerable", "absent"}:
            raise ValueError("Cases need a reviewed answerability label")
        expected = item.get("expected_evidence_ids")
        if not isinstance(expected, list) or len(set(expected)) != len(expected) or any(identifier not in training for identifier in expected):
            raise ValueError("Supporting evidence must belong to the reviewed training corpus")
        if bool(expected) != (item["answerability"] == "answerable"):
            raise ValueError("Answerable cases need supporting IDs; absent cases cannot have them")
        if item.get("expected_source_ids") and set(item["expected_source_ids"]) != {corpus.evidence[identifier].source_id for identifier in expected}:
            raise ValueError("Expected source IDs do not match the supporting evidence")
        if item.get("support_text_sha256"):
            if len(expected) != 1 or hashlib.sha256(corpus.evidence[expected[0]].text.encode()).hexdigest() != item["support_text_sha256"]:
                raise ValueError("Supporting text changed after answerability review")
        if not item.get("review_basis"):
            raise ValueError("Each answerability label must record its review basis")
    return cases


def summarize(rows: list[dict], condition: str) -> dict:
    answered = [row for row in rows if row["answerability"] == "answerable" and not row["conditions"][condition].get("error")]
    absent = [row for row in rows if row["answerability"] == "absent" and not row["conditions"][condition].get("error")]
    matched = sum(bool(set(row["expected_evidence_ids"]).intersection(row["conditions"][condition]["ids"])) for row in answered)
    reciprocal = 0.0
    recall = 0.0
    for row in answered:
        ids = row["conditions"][condition]["ids"]
        expected = set(row["expected_evidence_ids"])
        recall += len(expected.intersection(ids)) / len(expected)
        reciprocal += next((1 / index for index, identifier in enumerate(ids, 1) if identifier in expected), 0)
    false_support = sum(bool(row["conditions"][condition]["ids"]) for row in absent)
    errors = sum(bool(row["conditions"][condition].get("error")) for row in rows)
    return {"answerable_cases": len(answered), "retrieval_hits": matched,
            "hit_rate_at_k": matched / len(answered) if answered else None,
            "mean_recall_at_k": recall / len(answered) if answered else None,
            "mrr_at_k": reciprocal / len(answered) if answered else None,
            "absent_cases": len(absent), "absent_cases_with_candidates": false_support,
            "false_support_candidate_rate": false_support / len(absent) if absent else None,
            "failures": errors,
            "scope": "Candidate retrieval against agent-reviewed development labels; not answer entailment or human persona evaluation."}


def model_files(model: str, revision: str) -> list[dict]:
    from huggingface_hub import snapshot_download
    directory = Path(snapshot_download(model, revision=revision, local_files_only=True))
    return [{"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(directory.rglob("*")) if path.is_file()]


def run(args) -> dict:
    import torch
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[2]
    implementation_paths = ["src/bias/hybrid_retrieval.py", "src/bias/retrieval.py", "src/bias/situations.py",
                            "src/bias/corpus.py", "src/bias/models.py", "tools/retrieval_evaluation/__main__.py"]
    corpus_paths = ["evidence.jsonl", "sources.json", "profile.json", "corpus-manifest.json", "split-manifest.json"]
    inputs = {f"corpus/{name}": args.corpus / name for name in corpus_paths}
    inputs.update({name: root / name for name in implementation_paths})
    inputs["cases"] = args.cases
    initial_inputs = capture_inputs(inputs)
    corpus = Corpus.load(args.corpus)
    case_payload = json.loads(args.cases.read_text())
    cases = validate_cases(case_payload, corpus)
    if capture_inputs(inputs) != initial_inputs:
        raise ValueError("Inputs changed while loading the registered corpus or case manifest")
    policy = HybridPolicy(dense_min_similarity=args.dense_threshold, max_chunk_tokens=args.chunk_tokens,
                          reranker_min_score=args.reranker_min_score)
    embedder = SentenceTransformerEmbedder(args.model, revision=args.revision, device=args.device,
                                          query_prefix=args.query_prefix, passage_prefix=args.passage_prefix)
    reranker = CrossEncoderReranker(args.reranker_model, revision=args.reranker_revision, device=args.device) if args.reranker_model else None
    hybrid = HybridRetriever(corpus, embedder, policy=policy, reranker=reranker)
    lexical = Retriever(corpus)
    started = time.perf_counter()
    rows = []
    for index, case in enumerate(cases):
        row = dict(case, conditions={})
        for name, retriever, behavior in [("lexical", lexical, False), ("hybrid_B", hybrid, False), ("hybrid_C", hybrid, True)]:
            tick = time.perf_counter()
            try:
                result = retriever.search(case["query"], limit=args.limit, behavior=behavior)
                row["conditions"][name] = {"ids": [hit.evidence.id for hit in result.hits],
                    "scores": [hit.score for hit in result.hits], "reason": result.reason,
                    "elapsed_ms": round((time.perf_counter() - tick) * 1000, 3),
                    "trace": getattr(result, "trace", None)}
            except Exception as exc:
                row["conditions"][name] = {"ids": [], "error": type(exc).__name__, "message": str(exc),
                    "elapsed_ms": round((time.perf_counter() - tick) * 1000, 3)}
        rows.append(row)
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{len(cases)} development cases", flush=True)
    versions = {}
    for package in ["sentence-transformers", "transformers", "torch", "numpy", "tokenizers", "huggingface-hub"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    artifacts = []
    artifact_error = None
    reranker_artifacts = []
    try:
        artifacts = model_files(args.model, args.revision)
        if args.reranker_model:
            reranker_artifacts = model_files(args.reranker_model, args.reranker_revision)
    except Exception as exc:
        artifact_error = {"error": type(exc).__name__, "message": str(exc)}
    return {"schema_version": "hybrid-development-report.v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Development-only retrieval comparison. Existing exposed corpus; agent-written questions and labels. No fresh held-out-event efficacy claim.",
            "case_manifest_sha256": initial_inputs["cases"], "case_manifest": case_payload.get("id"),
            "corpus_sha256": hybrid.corpus_sha256, "corpus_id": (corpus.manifest or {}).get("id"),
            "training_records": len(corpus.training_evidence),
            "style_records": len(corpus.style_evidence),
            "training_events": len({source_event_id(corpus.sources[record.source_id]) for record in corpus.training_evidence}),
            "input_sha256": {name: initial_inputs[f"corpus/{name}"] for name in corpus_paths},
            "implementation_sha256": {path: initial_inputs[path] for path in implementation_paths},
            "input_integrity": input_integrity(initial_inputs, capture_inputs(inputs)),
            "retriever_versions": {"lexical": RETRIEVER_VERSION, "hybrid": HYBRID_VERSION},
            "model": embedder.identity, "model_files": artifacts, "model_file_hash_error": artifact_error,
            "reranker": reranker.identity if reranker else None, "reranker_files": reranker_artifacts,
            "runtime": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "python": platform.python_version(),
                        "omp_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
                        "torch_threads": torch.get_num_threads(), "packages": versions},
            "limit": args.limit, "elapsed_seconds": round(time.perf_counter() - started, 3),
            "metrics": {name: summarize(rows, name) for name in ["lexical", "hybrid_B", "hybrid_C"]},
            "review": {"labels": "agent-reviewed", "human_review": "ungraded", "tuning_split": "development",
                       "factual_entailment": "not_evaluated", "persona_style": "not_evaluated"},
            "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--revision", required=True, help="Pinned model revision (commit SHA), never a moving main branch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dense-threshold", type=float, default=0.84)
    parser.add_argument("--chunk-tokens", type=int, default=240)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-revision")
    parser.add_argument("--reranker-min-score", type=float)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new filename to preserve previous runs")
    if not __import__("re").fullmatch(r"[a-f0-9]{40}", args.revision):
        parser.error("--revision must be a 40-character pinned commit SHA")
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be between 1 and 32")
    if args.reranker_model and (not args.reranker_revision or not __import__("re").fullmatch(r"[a-f0-9]{40}", args.reranker_revision)):
        parser.error("--reranker-model requires a pinned 40-character --reranker-revision")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    args.output.write_bytes(gzip.compress(serialized, mtime=0) if args.output.suffix == ".gz" else serialized)
    print(json.dumps(report["metrics"], indent=2))
    if not report["input_integrity"]["stable"]:
        raise SystemExit("Inputs changed during execution; preserved diagnostic report is not a valid completed comparison")


if __name__ == "__main__":
    main()
