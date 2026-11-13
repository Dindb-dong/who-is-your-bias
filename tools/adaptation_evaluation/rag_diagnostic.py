"""Freeze default lexical retrieval-to-prompt behavior, then run one base condition.

Preparation captures Engine's actual messages without producing any model
response. Source-bearing plans and traces belong in private artifact storage.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from bias.config import ProviderSettings, Settings
from bias.corpus import Corpus, source_event_id
from bias.engine import Engine
from bias.models import ChatRequest
from bias.retrieval import RETRIEVER_VERSION, Retriever
from bias.snapshots import corpus_digest
from tools.persona_data.heldout import load_heldout
from .local_condition import make_execution_contract, run_local_condition, validate_execution_contract
from .protocol import CONDITIONS, SCHEMA, _context, digest, implementation_hashes, validate_plan

DIAGNOSTIC_SCHEMA = "heldout-default-lexical-generation.v1"
EXTRA_IMPLEMENTATION = ("tools/adaptation_evaluation/rag_diagnostic.py", "src/bias/engine.py",
    "src/bias/dialogue.py", "src/bias/query_policy.py", "src/bias/situations.py", "src/bias/providers.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rag_implementation_hashes() -> dict:
    root = Path(__file__).resolve().parents[2]
    return {name: _sha(root / name) for name in EXTRA_IMPLEMENTATION}


class _PromptCaptured(Exception):
    pass


class _CaptureOnly:
    messages = None

    async def stream(self, messages):
        self.messages = copy.deepcopy(messages)
        raise _PromptCaptured
        yield ""  # Async-generator interface; no token is ever produced.


class _RecordingRetriever:
    def __init__(self, delegate, corpus, searches):
        self.delegate, self.corpus, self.searches = delegate, corpus, searches

    def with_corpus(self, corpus):
        return type(self)(self.delegate.with_corpus(corpus), corpus, self.searches)

    def search(self, query, **kwargs):
        result = self.delegate.search(query, **kwargs)
        self.searches.append({"query": query, "arguments": kwargs,
            "eligible_count": len(self.delegate.records),
            "eligible_ids_sha256": digest(sorted(e.id for e in self.delegate.records)),
            "confidence": result.confidence, "reason": result.reason,
            "hits": [{"rank": index + 1, "id": hit.evidence.id, "source_id": hit.evidence.source_id,
                "event_family_id": source_event_id(self.corpus.sources[hit.evidence.source_id]),
                "score": hit.score, "matched_terms": list(hit.matched_terms)} for index, hit in enumerate(result.hits)]})
        return result


async def capture_runtime_prompt(engine: Engine, case: dict) -> dict:
    """Forward unchanged retrieval, interrupt at the actual model boundary."""
    if (engine.settings.retrieval_backend != "lexical" or type(engine.retriever) is not Retriever
            or engine.settings.style_examples or engine.settings.retrieval_limit != 4
            or not engine.settings.llm.configured or engine.snapshot is None):
        raise ValueError("Use default lexical top-four retrieval, no style examples and an explicit snapshot")
    if case.get("context_before") or case.get("context"):
        raise ValueError("This diagnostic deliberately omits earlier source context")
    capture, searches = _CaptureOnly(), []
    original_llm, original_retriever = engine.llm, engine.retriever
    engine.llm = capture
    engine.retriever = _RecordingRetriever(original_retriever, engine.corpus, searches)
    outcome = {"case_id": case["id"], "status": "preparing", "model_generations": 0, "searches": searches}
    try:
        turn = await engine.chat(ChatRequest(message=case["question"], mode="rag",
            language=case.get("language", case.get("original_language", "en"))))
        outcome.update(status="runtime_did_not_generate", runtime_result=turn.model_dump(mode="json"))
    except _PromptCaptured:
        context = _context(capture.messages)
        factual = context["factual_evidence"]
        if context["style_examples"]:
            raise ValueError("Default lexical capture unexpectedly supplied wording examples")
        ids = [item["id"] for item in factual]
        eligible = {item.id for item in engine.corpus.training_evidence}
        if not set(ids) <= eligible:
            raise ValueError("A non-training record reached the captured factual prompt")
        for item in factual:
            evidence = engine.corpus.evidence[item["id"]]
            source = engine.corpus.sources[evidence.source_id]
            if source_event_id(source) == case.get("event_family_id"):
                raise ValueError("A heldout reference family reached the prompt")
            if item["text"] != evidence.text:
                raise ValueError("The runtime shortened or changed a factual passage")
        retrieved = [hit["id"] for search in searches for hit in search["hits"]]
        outcome.update(status="prompt_captured_no_generation", messages=capture.messages,
            factual_ids=ids, factual_sha256=digest(factual),
            no_retrieved_evidence=not retrieved,
            budget={"counter": engine.counter.identity, "exact_model_tokens": engine.counter.exact,
                "input_units": engine.counter.count(capture.messages),
                "context_window": engine.settings.llm_context_window,
                "reserved_output_tokens": engine.settings.llm_max_tokens,
                "dropped_evidence_ids": [key for key in retrieved if key not in ids],
                "supplied_evidence_ids": ids, "style_example_ids": [], "history_turns": 0})
    except Exception as exc:
        # Retain registration and a failed pre-generation case. Never invent a
        # replacement prompt or silently change retrieval to make it answerable.
        outcome.update(status="pre_generation_failed", failure_type=type(exc).__name__, failure=str(exc))
    finally:
        engine.llm, engine.retriever = original_llm, original_retriever
    return outcome


async def freeze_rag_plan(engine: Engine, cases: list[dict], prior_plan: dict,
                          answerability: list[dict], *, identifier: str) -> tuple[dict, list[dict]]:
    validate_plan(prior_plan)
    if (prior_plan["exposure"] != "heldout" or prior_plan["corpus_sha256"] != corpus_digest(engine.corpus)
            or engine.snapshot is None or engine.snapshot.summary() != prior_plan["snapshot"]):
        raise ValueError("Reuse the heldout questions against the exact previously checked training corpus")
    prior = {row["id"]: row for row in prior_plan["cases"]}
    if [row["id"] for row in cases] != list(prior) or [row["id"] for row in answerability] != list(prior):
        raise ValueError("Preserve the original heldout question order and answerability rows")
    expected_review = prior_plan["input_identity"]["generation_subset"]["answerability_review_sha256"]
    if digest(answerability) != expected_review:
        raise ValueError("The pre-generation answerability labels or rationales changed")
    if engine.settings.llm_max_tokens != 160 or engine.settings.llm_context_window != 4096:
        raise ValueError("Keep the registered 160-token/4096-context budget")
    engine.corpus.validate_split_isolation()
    frozen, traces = [], []
    for original in cases:
        previous = prior[original["id"]]
        if (digest(original) != previous["input_case_sha256"]
                or original["question"] != previous["question"]
                or digest(original.get("reference_answer")) != previous["reference_sha256"]):
            raise ValueError("Heldout input, question or original private reference changed")
        trace = await capture_runtime_prompt(engine, original)
        traces.append(trace)
        if trace["status"] != "prompt_captured_no_generation":
            continue
        messages = trace["messages"]
        if (messages[-1]["content"] != original["question"]
                or any(row.get("reference_answer") and row["reference_answer"] in json.dumps(messages, ensure_ascii=False)
                       for row in cases)):
            raise ValueError("A question changed or a private heldout reference reached the prompt")
        frozen.append({"id": original["id"], "question": original["question"],
            "language": original.get("language", original.get("original_language", "en")),
            "event_family_id": original["event_family_id"], "reference_answer": original.get("reference_answer"),
            "reference_sha256": previous["reference_sha256"], "question_sha256": previous["question_sha256"],
            "input_case_sha256": previous["input_case_sha256"], "reference_review_status": original.get("reference_review_status", "ungraded"),
            "factual_ids": trace["factual_ids"], "style_ids": [], "factual_sha256": trace["factual_sha256"],
            "selection_policy": "actual_default_engine_lexical_top4_then_whole_passage_budget",
            "budget": trace["budget"], "has_factual_support_candidates": bool(trace["factual_ids"]),
            "style_contrast_present": False,
            "messages": {condition: copy.deepcopy(messages) for condition in CONDITIONS},
            "prompt_sha256": {condition: digest(messages) for condition in CONDITIONS}})
    diagnostic = {"schema_version": DIAGNOSTIC_SCHEMA, "execution_conditions": ["facts"],
        "source_fixed_evidence_plan_sha256": prior_plan["plan_sha256"],
        "registered_case_ids": list(prior), "generation_case_ids": [row["id"] for row in frozen],
        "pre_generation_results": [{key: trace[key] for key in ("case_id", "status", "failure_type", "failure") if key in trace}
                                   for trace in traces],
        "retriever": {"backend": "lexical", "version": RETRIEVER_VERSION, "limit": 4, "style_examples": False},
        "answerability_sha256": digest(answerability), "answerability": answerability,
        "retrieval_trace_sha256": digest(traces), "implementation_sha256": rag_implementation_hashes(),
        "scope": "Offline default retrieval-to-generation diagnostic; one base condition, no matched adapter comparison or live API latency claim",
        "model_generations_at_freeze": 0, "tuning_on_heldout_outputs": False}
    plan = {"schema_version": SCHEMA, "id": identifier, "created_at": datetime.now(timezone.utc).isoformat(),
        "exposure": "heldout", "input_identity": prior_plan["input_identity"], "base_identity": prior_plan["base_identity"],
        "corpus_sha256": corpus_digest(engine.corpus), "snapshot": engine.snapshot.summary(),
        "implementation_sha256": implementation_hashes(), "policy": copy.deepcopy(prior_plan["policy"]),
        "cases": frozen, "rag_diagnostic": diagnostic, "human_review": "ungraded", "efficacy_claim": False}
    plan["plan_sha256"] = digest(plan)
    if frozen:
        validate_rag_plan(plan)
    return plan, traces


def validate_rag_plan(plan: dict) -> None:
    validate_plan(plan)
    diagnostic = plan.get("rag_diagnostic", {})
    if (diagnostic.get("schema_version") != DIAGNOSTIC_SCHEMA
            or diagnostic.get("execution_conditions") != ["facts"]
            or diagnostic.get("implementation_sha256") != rag_implementation_hashes()
            or any(row["style_ids"] or len({digest(value) for value in row["messages"].values()}) != 1 for row in plan["cases"])):
        raise ValueError("The single-base RAG diagnostic implementation or prompts changed")


def public_diagnostic(plan: dict, traces: list[dict]) -> dict:
    diagnostic = plan["rag_diagnostic"]
    return {"schema_version": DIAGNOSTIC_SCHEMA, "status": "frozen_no_model_generations",
        "plan_sha256": plan["plan_sha256"], "source_fixed_evidence_plan_sha256": diagnostic["source_fixed_evidence_plan_sha256"],
        "corpus_sha256": plan["corpus_sha256"], "execution_conditions": ["facts"], "scope": diagnostic["scope"],
        "registered_cases": len(diagnostic["registered_case_ids"]), "planned_generations": len(plan["cases"]),
        "attempted_generations": 0, "pre_generation_results": [
            {key: row[key] for key in ("case_id", "status", "failure_type") if key in row}
            for row in diagnostic["pre_generation_results"]],
        "retriever": diagnostic["retriever"], "answerability_sha256": diagnostic["answerability_sha256"],
        "answerability_counts": dict(Counter(row["answerability"] for row in diagnostic["answerability"])),
        "answerability": [{key: row[key] for key in ("id", "answerability", "question_sha256", "reference_sha256")}
                          for row in diagnostic["answerability"]],
        "retrieval_trace_sha256": diagnostic["retrieval_trace_sha256"],
        "implementation_sha256": diagnostic["implementation_sha256"],
        "cases": [{"id": trace["case_id"], "status": trace["status"],
            "retrieved_ids": [hit["id"] for search in trace["searches"] for hit in search["hits"]],
            "supplied_ids": trace.get("factual_ids", []), "budget": trace.get("budget"),
            "searches": [{"query_sha256": digest(search["query"]), "eligible_count": search["eligible_count"],
                "eligible_ids_sha256": search["eligible_ids_sha256"], "confidence": search["confidence"],
                "reason": search["reason"], "hits": [{key: hit[key] for key in ("rank", "id", "source_id", "event_family_id", "score")}
                                                       for hit in search["hits"]]} for search in trace["searches"]]}
                  for trace in traces],
        "source_wording_distributed": False, "human_review": "ungraded", "efficacy_claim": False,
        "limitations": ["Answerability labels concern the whole eligible training corpus, not guaranteed retrieval success.",
            "No evidence or insufficient evidence remains a diagnostic outcome; no handpicked passage replaces it.",
            "Original earlier-speaker source context is omitted, and private reference answers are never supplied.",
            "The three-condition plan envelope is reused for runner compatibility; only its facts condition is registered here."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--corpus", type=Path, required=True)
    freeze.add_argument("--heldout", type=Path, required=True)
    freeze.add_argument("--prior-plan", type=Path, required=True)
    freeze.add_argument("--answerability", type=Path, required=True)
    freeze.add_argument("--snapshot", type=Path, required=True)
    freeze.add_argument("--tokenizer", type=Path, required=True)
    freeze.add_argument("--base-contract", type=Path, required=True)
    freeze.add_argument("--id", required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--timeout", type=float, default=180)
    for command in (freeze, run):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        plan, contract = (json.loads(path.read_text()) for path in (args.plan, args.contract))
        validate_rag_plan(plan)
        validate_execution_contract(contract, plan)
        runtime = contract["runtime"]
        result = run_local_condition(args.plan, args.contract, "facts", args.output,
            dtype=runtime["dtype"], threads=runtime["threads"], timeout=args.timeout)
        print(json.dumps({key: result.get(key) for key in ("condition", "status", "attempted", "completed", "server_stopped")}))
        return
    if args.output.exists():
        raise ValueError("Preserve frozen artifacts; choose a new output directory")
    _, originals = load_heldout(args.heldout)
    prior = json.loads(args.prior_plan.read_text())
    base_contract = json.loads(args.base_contract.read_text())
    validate_execution_contract(base_contract, prior)
    if _sha(args.heldout / "manifest.json") != prior["input_identity"]["generation_subset"]["manifest_file_sha256"]:
        raise ValueError("The actual heldout dataset differs from the original frozen references")
    source_cases = {row["id"]: row for row in originals}
    cases = []
    for row in prior["cases"]:
        source = source_cases[row["id"]]
        if digest(source) != prior["input_identity"]["generation_subset"]["original_case_sha256"][row["id"]]:
            raise ValueError("An original heldout source case changed")
        derived = copy.deepcopy(source)
        derived.update(context_before="", context=[], context_sha256=digest([]))
        cases.append(derived)
    labels = json.loads(args.answerability.read_text())
    engine = Engine(Settings(corpus_dir=args.corpus, persona_snapshot=args.snapshot,
        llm=ProviderSettings("http://127.0.0.1:9/v1", "capture-only-no-network"),
        llm_max_tokens=160, llm_context_window=4096, llm_tokenizer_path=args.tokenizer,
        retrieval_backend="lexical", retrieval_prewarm=False, retrieval_limit=4, style_examples=False), Corpus.load(args.corpus))
    plan, traces = asyncio.run(freeze_rag_plan(engine, cases, prior, labels, identifier=args.id))
    contract = make_execution_contract([plan["plan_sha256"]], base_contract["base_identity"],
        dtype=base_contract["runtime"]["dtype"], threads=base_contract["runtime"]["threads"])
    for name, value in (("plan.json", plan), ("retrieval-traces.json", traces), ("answerability.json", labels),
                        ("execution-contract.json", contract), ("summary.json", public_diagnostic(plan, traces))):
        _write(args.output / name, value)
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "registered_cases": len(cases),
        "planned_generations": len(plan["cases"]), "actual_generations": 0}))


if __name__ == "__main__":
    main()
