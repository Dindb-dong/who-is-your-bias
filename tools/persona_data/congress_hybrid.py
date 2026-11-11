"""Run frozen congressional paraphrases through actual pinned hybrid API retrieval."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def snapshot_files(path: Path) -> list[dict]:
    return [{"path": str(p.relative_to(path)), "bytes": p.stat().st_size, "sha256": file_hash(p)}
            for p in sorted(path.rglob("*")) if p.is_file()]


def model_snapshot(model: str, revision: str) -> tuple[Path, list[dict]]:
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(model, revision=revision)
    if info.sha != revision:
        raise ValueError("Resolved retrieval model revision changed")
    names = {item.rfilename for item in info.siblings}
    weight = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
    if weight not in names:
        raise ValueError("Pinned retrieval model lacks the expected weight artifact")
    path = Path(snapshot_download(model, revision=revision, allow_patterns=["*.json", "*.model", weight],
                                  ignore_patterns=["onnx/*", "openvino/*"], max_workers=2))
    return path, snapshot_files(path)


def run(corpus_path: Path, snapshot_path: Path, cases_path: Path, output: Path) -> dict:
    from bias.api import create_app
    from bias.config import Settings, repository_root
    from bias.corpus import Corpus
    from fastapi.testclient import TestClient
    import torch

    if output.exists():
        raise ValueError("Use a new hybrid diagnostic output path")
    cases_bytes = cases_path.read_bytes()
    registration = json.loads(cases_bytes)
    if registration.get("schema_version") != "congress-hybrid-api-cases.v1" or registration.get("exposure") != "development":
        raise ValueError("Use the separately registered development paraphrases")
    policy = registration["policy"]
    if policy["device"] != "cpu" or policy["threads"] != 2:
        raise ValueError("This diagnostic uses the frozen two-thread CPU policy")
    torch.set_num_threads(2)
    corpus = Corpus.load(corpus_path)
    for case in registration["cases"]:
        record = corpus.evidence[case["expected_evidence_id"]]
        if record.source_id != case["expected_source_id"] or hashlib.sha256(record.text.encode()).hexdigest() != case["support_text_sha256"]:
            raise ValueError("Expected source passage changed after case registration")
    root = repository_root()
    inputs = {"snapshot": snapshot_path, "cases": cases_path,
              **{"corpus/" + path.name: path for path in corpus_path.iterdir() if path.is_file()},
              **{str(path.relative_to(root)): path for path in (root / "src/bias").glob("*.py")},
              "tools/persona_data/congress_hybrid.py": Path(__file__).resolve()}
    initial = {name: file_hash(path) for name, path in inputs.items()}
    report = {"schema_version": "congress-hybrid-api-diagnostic.v1", "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat(), "case_manifest_sha256": hashlib.sha256(cases_bytes).hexdigest(),
              "transport": "in_process_asgi_testclient", "exposure": "development", "policy": policy,
              "llm_inference": False, "tts_inference": False, "human_review": "ungraded", "efficacy_claim": False,
              "runtime": {"python": platform.python_version(), "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "sentence-transformers", "numpy", "huggingface-hub", "tokenizers")}},
              "input_hashes_before": initial, "model_files": {}, "cases": [],
              "limitations": ["Six developer-written paraphrases with prior lexical exposure; no fresh held-out efficacy claim.",
                  "Candidate and citation matches are not human relevance, factual entailment or persona ratings.",
                  "No threshold changes or query rewrites after the hybrid run starts; failures remain in its denominator.",
                  "CPU ASGI timings include this corpus and model state; they do not measure network, generation, speech or target-device latency."]}
    stage = "model_acquisition"
    started = time.perf_counter()
    try:
        model_paths = {}
        for kind in ("embedding", "reranker"):
            model, revision = policy[kind + "_model"], policy[kind + "_revision"]
            model_paths[kind], report["model_files"][kind] = model_snapshot(model, revision)
        # Downloads are excluded from the API-startup and per-query metrics.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        settings = Settings(corpus_dir=corpus_path, persona_snapshot=snapshot_path, retrieval_backend="hybrid",
            retrieval_prewarm=True, embedding_model=policy["embedding_model"], embedding_revision=policy["embedding_revision"],
            reranker_model=policy["reranker_model"], reranker_revision=policy["reranker_revision"],
            dense_min_similarity=policy["dense_min_similarity"], reranker_min_score=policy["reranker_min_score"],
            retrieval_limit=policy["retrieval_limit"], style_examples=policy["style_examples"])
        stage = "api_startup_prewarm"
        startup = time.perf_counter()
        with TestClient(create_app(settings)) as client:
            report["api_startup_ms"] = round((time.perf_counter() - startup) * 1000, 3)
            health = client.get("/api/health").json()
            report.update(snapshot=health["persona_snapshot"], corpus=health["corpus"], retrieval=health["retrieval"])
            # Loading may additionally cache the model card. Freeze the complete
            # actual inventory after startup, before any registered query.
            report["acquired_model_files"] = report["model_files"]
            report["model_files"] = {kind: snapshot_files(path) for kind, path in model_paths.items()}
            for kind, before in report["acquired_model_files"].items():
                loaded = {item["path"]: item for item in report["model_files"][kind]}
                if any(loaded.get(item["path"]) != item for item in before):
                    raise ValueError("Pinned retrieval artifact changed during model loading")
            retriever = client.app.state.engine.retriever
            modules = {"embedding": retriever.embedder._model, "reranker": retriever.reranker._model.model}
            report["runtime"]["loaded_models"] = {}
            for kind, module in modules.items():
                parameter = next(module.parameters())
                report["runtime"]["loaded_models"][kind] = {"device": str(parameter.device), "dtype": str(parameter.dtype),
                    "parameters": sum(p.numel() for p in module.parameters()), "training": module.training}
            stage = "case_execution"
            for case in registration["cases"]:
                tick = time.perf_counter()
                row = {**case, "status": "running"}
                try:
                    response = client.post("/api/chat", json={"message": case["query"], "language": case["language"]})
                    response.raise_for_status()
                    turn = response.json()
                    ids = turn["grounding"]["retrieved_ids"]
                    cited = [citation["evidence_id"] for citation in turn["citations"]]
                    prefix = "qa26-" + case["expected_source_id"] + "-"
                    row.update(status="completed", provider=turn["provider"], temporal_mode=turn["context"]["temporal"]["mode"],
                        grounding_status=turn["grounding"]["status"], retrieved_ids=ids, citation_ids=cited,
                        expected_passage_retrieved=case["expected_evidence_id"] in ids,
                        expected_passage_cited=case["expected_evidence_id"] in cited,
                        expected_source_retrieved=any(identifier.startswith(prefix) for identifier in ids),
                        retrieval_trace=turn["context"].get("retrieval"),
                        response_text_sha256=hashlib.sha256(turn["text"].encode()).hexdigest())
                except Exception as exc:
                    row.update(status="failed", failure_type=type(exc).__name__)
                row["elapsed_ms"] = round((time.perf_counter() - tick) * 1000, 3)
                report["cases"].append(row)
                print(json.dumps({"case": row["id"], "status": row["status"], "expected_passage_retrieved": row.get("expected_passage_retrieved")}), flush=True)
        stage = "post_run_integrity"
        for kind, path in model_paths.items():
            files = snapshot_files(path)
            if files != report["model_files"][kind]:
                raise ValueError("Retrieval model files changed during execution")
        report["model_files_stable"] = True
        report["status"] = "completed" if all(row["status"] == "completed" for row in report["cases"]) else "completed_with_case_failures"
    except Exception as exc:
        chain, cause = [], exc
        while cause is not None and len(chain) < 6:
            chain.append(type(cause).__name__)
            cause = cause.__cause__ or cause.__context__
        report.update(status="failed", failure={"stage": stage, "exception_types": chain})
    report["input_hashes_after"] = {name: file_hash(path) for name, path in inputs.items()}
    report["input_integrity_stable"] = report["input_hashes_after"] == initial
    if not report["input_integrity_stable"]:
        report["status"] = "invalid_input_drift"
    report["summary"] = {"registered_cases": len(registration["cases"]), "attempted": len(report["cases"]),
        "completed": sum(row["status"] == "completed" for row in report["cases"]),
        "expected_passages_retrieved": sum(row.get("expected_passage_retrieved", False) for row in report["cases"]),
        "expected_passages_cited": sum(row.get("expected_passage_cited", False) for row in report["cases"]),
        "expected_sources_retrieved": sum(row.get("expected_source_retrieved", False) for row in report["cases"])}
    report.update(completed_at=datetime.now(timezone.utc).isoformat(), total_seconds=round(time.perf_counter() - started, 3))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as out:
        out.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "snapshot", "cases", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.corpus, args.snapshot, args.cases, args.output)
    print(json.dumps({"status": result["status"], "summary": result["summary"], "failure": result.get("failure")}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
