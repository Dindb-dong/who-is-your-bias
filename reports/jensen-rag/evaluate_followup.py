"""Measure Jensen calibration and warmed chat against a running local Ollama."""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from bias.api import create_app
from bias.config import ProviderSettings, Settings
from bias.snapshots import select_temporal

HERE = Path(__file__).resolve().parent
OLLAMA = "http://127.0.0.1:11434/v1"
EMBEDDING = "batiai/qwen3-embedding:0.6b"
BASELINE = json.loads((HERE / "korean-retrieval-comparison.json").read_text())
ORIGINAL = [run for run in BASELINE["runs"] if run["backend"] == "lexical"]
CASES = json.loads((HERE / "korean-calibration.json").read_text())


def settings(backend, threshold=0.84, model=None, llm_base_url=OLLAMA):
    return replace(
        Settings.from_env(), retrieval_backend=backend, retrieval_limit=4, retrieval_prewarm=True,
        embedding_backend="http", embedding_base_url=OLLAMA, embedding_model=EMBEDDING,
        embedding_revision=None, embedding_query_prefix="query: ", embedding_passage_prefix="passage: ",
        dense_min_similarity=threshold, reranker_model=None, reranker_revision=None,
        reranker_min_score=None, style_examples=False, llm_max_tokens=160,
        llm=ProviderSettings(llm_base_url, model) if model else ProviderSettings(None, None),
    )


def dump(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def calibration():
    app = create_app(settings("hybrid"))

    def run(engine, cases, threshold):
        retriever = engine.retriever
        retriever.policy = replace(retriever.policy, dense_min_similarity=threshold)
        results = []
        for query, expected in cases:
            temporal = select_temporal(engine.corpus, engine.snapshot, query)
            hits = retriever.search(
                query, behavior=True, original_query=query, allow_live=True,
                eligible_ids=temporal.eligible_ids, exclusion_reasons=temporal.exclusion_reasons,
                temporal_scores=temporal.temporal_scores,
            ).hits if temporal.eligible_ids else ()
            results.append({"query": query, "expected": expected, "hits": [h.evidence.id for h in hits]})
        return results

    def counts(rows):
        return {
            "targets": sum(row["expected"] in row["hits"] for row in rows if row["expected"]),
            "top1_targets": sum(row["hits"][:1] == [row["expected"]] for row in rows if row["expected"]),
            "control_hits": sum(bool(row["hits"]) for row in rows if row["expected"] is None),
        }

    with TestClient(app):
        engine = app.state.engine
        sweep = []
        for threshold in (0.84, 0.65, 0.60, 0.57, 0.54, 0.51, 0.48, 0.45):
            rows = run(engine, CASES["calibration"], threshold)
            row = {"threshold": threshold, **counts(rows)}
            sweep.append(row)
            print(row, flush=True)
        baseline = sweep[0]
        eligible = [row for row in sweep if row["control_hits"] <= baseline["control_hits"]]
        selected = max(eligible, key=lambda row: (row["targets"], row["top1_targets"], row["threshold"]))
        original = [(r["query"], next(iter(r["expected_ids"]), None)) for r in ORIGINAL]
        evaluation = {}
        for name, cases in (("held_out", CASES["held_out"]), ("original", original)):
            evaluation[name] = {}
            for threshold in dict.fromkeys((0.84, selected["threshold"])):
                rows = run(engine, cases, threshold)
                evaluation[name][str(threshold)] = {**counts(rows), "runs": rows}
        dump("korean-threshold-results.json", {
            "embedding": EMBEDDING, "selected": selected, "sweep": sweep, "evaluation": evaluation,
        })


def chat_run(client, query):
    start = time.perf_counter()
    response = client.post("/api/chat/stream", json={"message": query, "language": "ko"})
    wall_ms = round((time.perf_counter() - start) * 1000, 3)
    response.raise_for_status()
    events = [block.split("\ndata: ", 1) for block in response.text.strip().split("\n\n")]
    result = next((json.loads(data)["turn"] for event, data in events if event == "event: result"), None)
    errors = [json.loads(data) for event, data in events if event == "event: error"]
    if result is None:
        return {"wall_ms": wall_ms, "errors": errors or ["missing result"]}
    return {
        "wall_ms": wall_ms, "metrics": result["metrics"], "answer": result["text"],
        "grounding": result["grounding"]["status"], "retrieved_ids": result["grounding"]["retrieved_ids"],
        "cited_ids": [citation["evidence_id"] for citation in result["citations"]],
    }


def benchmark(models, threshold, llm_base_url, guard_checks):
    for model in models:
        runs, controls = [], []
        for backend in ("lexical", "hybrid"):
            with TestClient(create_app(settings(backend, threshold, model, llm_base_url))) as client:
                warm = chat_run(client, "AI 공장이 무엇을 생산해?")
                assert "errors" not in warm, warm
                for original in ORIGINAL:
                    row = chat_run(client, original["query"])
                    runs.append({"backend": backend, "case": original["case"], **row})
                    print(model, backend, original["case"], row["wall_ms"], row.get("cited_ids"), flush=True)
                if guard_checks and backend == "hybrid":
                    for query, expected in CASES["held_out"]:
                        if expected is None:
                            row = chat_run(client, query)
                            controls.append({"query": query, **row})
                            print(model, "guard control", row["wall_ms"], flush=True)
        dump(f"{model.replace(':', '-')}-korean-results.json", {
            "model": model, "llm_base_url": llm_base_url, "threshold": threshold,
            "max_tokens": 160, "runs": runs, "guard_controls": controls,
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*")
    parser.add_argument("--threshold", type=float, default=0.84)
    parser.add_argument("--llm-base-url", default=OLLAMA)
    parser.add_argument("--guard-checks", action="store_true")
    args = parser.parse_args()
    if args.models:
        benchmark(args.models, args.threshold, args.llm_base_url, args.guard_checks)
    else:
        calibration()
