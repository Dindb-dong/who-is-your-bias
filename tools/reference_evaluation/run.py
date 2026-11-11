"""Run the same actual model on every frozen A/B/C reference question."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

from bias.config import ProviderSettings, Settings
from tools.evaluation.observe import StreamObservation
from tools.evaluation.run import citation_report

from .protocol import CONDITIONS, PROMPT_VERSION, PROTOCOL_VERSION, Plan, build_condition, digest, freeze_plan, lexical_diagnostics, load_inputs, qa_data, validate_plan


async def generate(provider: ProviderSettings, messages: list[dict], *, max_tokens: int, timeout: float, seed: int, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    if not provider.configured:
        raise ValueError("An actual language-model endpoint is required")
    if not 1 <= max_tokens <= 4096 or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError("Invalid generation bounds")
    payload = {"model": provider.model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens, "temperature": 0, "top_p": 1, "seed": seed}
    observation = StreamObservation()
    fragments, error, complete = [], None, False
    first_token_ms = None
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(transport=transport, timeout=timeout, follow_redirects=False) as client:
                async with client.stream("POST", f"{provider.base_url}/chat/completions", headers=provider.headers, json=payload) as response:
                    observation.http_statuses.append(response.status_code)
                    if response.status_code != 200:
                        raise ValueError(f"provider_http_{response.status_code}")
                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        raise ValueError("provider_not_sse")
                    size = 0
                    async for line in response.aiter_lines():
                        size += len(line)
                        if len(line) > 65536 or size > 2_000_000:
                            raise ValueError("provider_stream_limit")
                        observation.feed((line + "\n").encode())
                        if not line.startswith("data:"):
                            continue
                        value = line[5:].strip()
                        if value == "[DONE]":
                            break
                        if not value:
                            continue
                        event = json.loads(value)
                        if not isinstance(event, dict) or "error" in event:
                            raise ValueError("provider_stream_error")
                        choices = event.get("choices", [])
                        if not choices:
                            continue
                        choice = choices[0]
                        part = choice.get("delta", {}).get("content")
                        if part is not None:
                            if not isinstance(part, str):
                                raise ValueError("provider_nontext_delta")
                            if part:
                                if first_token_ms is None:
                                    first_token_ms = (time.perf_counter() - started) * 1000
                                fragments.append(part)
                                if sum(map(len, fragments)) > 20000:
                                    raise ValueError("provider_output_limit")
                        finish = choice.get("finish_reason")
                        if finish:
                            complete = finish == "stop"
                            if not complete:
                                error = "provider_finish_" + str(finish)
                    if not fragments:
                        error, complete = "provider_empty", False
                    elif not complete and error is None:
                        error = "provider_incomplete"
    except (TimeoutError, httpx.TimeoutException):
        error, complete = "provider_timeout", False
    except httpx.HTTPError:
        error, complete = "provider_network", False
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
        # Do not persist provider response bodies, URLs containing credentials,
        # or unknown exception text in a shareable research report.
        error, complete = str(exc) if isinstance(exc, ValueError) and str(exc).startswith("provider_") else "provider_protocol", False
    return {"text": "".join(fragments), "completed": complete, "error": error, "first_text_ms": round(first_token_ms, 3) if first_token_ms is not None else None, "wall_ms": round((time.perf_counter() - started) * 1000, 3), "finish_reason": observation.finish_reason, "response_model": observation.response_model, "usage": observation.usage, "usage_status": observation.usage_status, "http_statuses": observation.http_statuses, "request_sha256": digest(payload)}


def summarize(rows: list[dict]) -> dict:
    conditions = {}
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        complete = [row for row in group if row["generation"]["completed"]]
        conditions[condition] = {"attempted": len(group), "completed": len(complete), "errors": dict(Counter(row["generation"]["error"] for row in group if row["generation"]["error"])), "no_examples": sum(not row["prompt_audit"]["selected"] for row in group), "wall_ms_median": statistics.median(row["generation"]["wall_ms"] for row in complete) if complete else None, "token_f1_mean_diagnostic": statistics.mean(row["lexical_diagnostic"]["token_unigram_f1"] for row in complete) if complete else None}
    by_case = {}
    for row in rows:
        by_case.setdefault((row["case_id"], row["repetition"]), {})[row["condition"]] = row
    pairs = [group for group in by_case.values() if "B" in group and "C" in group]
    changes = sum([item["id"] for item in group["B"]["prompt_audit"]["selected"]] != [item["id"] for item in group["C"]["prompt_audit"]["selected"]] for group in pairs)
    return {"conditions": conditions, "paired_bc_retrieval_cases": len(pairs), "bc_selection_or_order_changes": changes, "human_review": "ungraded", "content_and_persona_conclusion": "requires_separate_judgments", "note": "A retrieval difference or lexical overlap increase is not evidence of better persona fidelity."}


async def run(plan: Plan, corpus, manifest: dict, provider: ProviderSettings, *, model_identity: dict, max_tokens: int = 256, context_chars: int = 9000, retrieval_limit: int = 4, timeout: float = 180, total_seconds: float = 3600, repetitions: int = 1, seed: int = 42, transport_factory=None, checkpoint=None) -> dict:
    validate_plan(plan, corpus, manifest)
    if not provider.configured or not model_identity.get("revision") or not model_identity.get("runtime"):
        raise ValueError("Declare the model revision and runtime for real inference")
    if not 1 <= repetitions <= 5 or not math.isfinite(total_seconds) or not 0 < total_seconds <= 14400:
        raise ValueError("Invalid run budget")
    schedule = []
    conditions = tuple(CONDITIONS)
    for repetition in range(repetitions):
        for index, case in enumerate(plan.cases):
            offset = (index + repetition) % 3
            for condition in conditions[offset:] + conditions[:offset]:
                schedule.append((case, condition, repetition + 1))
    rows = []
    started = time.perf_counter()
    for case, condition, repetition in schedule:
        remaining = total_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            break
        messages, audit = build_condition(corpus, case, condition, retrieval_limit=retrieval_limit, context_chars=context_chars)
        result = await generate(provider, messages, max_tokens=max_tokens, timeout=min(timeout, remaining), seed=seed + repetition - 1, transport=transport_factory() if transport_factory else None)
        reference = qa_data(corpus.evidence[case.reference_id])
        allowed = {item["id"] for item in audit["selected"]}
        row = {"case_id": case.id, "source_event_id": case.source_event_id, "reference_id": case.reference_id, "reference_completeness": reference.get("completeness", "unknown"), "language": case.language, "condition": condition, "mode": CONDITIONS[condition], "repetition": repetition, "prompt_audit": audit, "generation": result, "citation_format": citation_report(result["text"], allowed, required=bool(allowed)), "lexical_diagnostic": lexical_diagnostics(result["text"], reference["answer"]), "judgments": {"human": None, "model_assisted": None}}
        rows.append(row)
        if checkpoint:
            checkpoint(row)
    skipped = [{"case_id": case.id, "condition": condition, "repetition": repetition, "reason": "total_deadline"} for case, condition, repetition in schedule[len(rows):]]
    return {"schema_version": "actual-qa-report.v1", "created_at": datetime.now(timezone.utc).isoformat(), "plan": plan.model_dump(mode="json"), "plan_sha256": digest(plan.model_dump(mode="json")), "model_identity": model_identity, "configuration": {"model": provider.model, "max_tokens": max_tokens, "context_chars": context_chars, "retrieval_limit": retrieval_limit, "temperature": 0, "top_p": 1, "seed": seed, "timeout_seconds": timeout, "session_policy": "fresh_request_no_history", "condition_order": "rotating_sequential", "prompt_version": PROMPT_VERSION, "same_instructions_all_conditions": True, "all_conditions_call_model_even_without_examples": True, "transport": "test_injected" if transport_factory else "actual_http"}, "environment": {"platform": platform.platform(), "python": platform.python_version(), "logical_cpu_count": os.cpu_count(), "timing_scope": "execution_host_not_intended_demo_hardware"}, "execution": {"planned": len(schedule), "attempted": len(rows), "skipped": skipped, "wall_ms": round((time.perf_counter() - started) * 1000, 3)}, "summary": summarize(rows), "runs": rows}


def main() -> int:
    from bias.corpus import Corpus
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--corpus", required=True, type=Path)
    freeze.add_argument("--id", required=True)
    freeze.add_argument("--split", choices=["validation", "test"], default="test")
    freeze.add_argument("--output", required=True, type=Path)
    execute = sub.add_parser("run")
    execute.add_argument("--corpus", required=True, type=Path)
    execute.add_argument("--plan", required=True, type=Path)
    execute.add_argument("--output", required=True, type=Path)
    execute.add_argument("--model-identity", required=True, type=Path, help="JSON with model revision, runtime, quantization and file hash")
    execute.add_argument("--max-tokens", type=int, default=256)
    execute.add_argument("--context-chars", type=int, default=9000)
    execute.add_argument("--retrieval-limit", type=int, default=4)
    execute.add_argument("--repetitions", type=int, choices=range(1, 6), default=1)
    execute.add_argument("--total-seconds", type=float, default=3600)
    execute.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.output.suffix != ".json" or args.output.exists():
        parser.error("Choose a new .json output; historical runs are never overwritten")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "freeze":
        corpus = Corpus.load(args.corpus)
        manifest = json.loads((args.corpus / "split-manifest.json").read_text())
        plan = freeze_plan(corpus, manifest, plan_id=args.id, split=args.split)
        args.output.write_text(plan.model_dump_json(indent=2) + "\n")
        print(json.dumps({"plan": str(args.output), "cases": len(plan.cases), "events": len({case.source_event_id for case in plan.cases}), "inference_performed": False}))
        return 0
    corpus, plan, manifest = load_inputs(args.corpus, args.plan)
    identity = json.loads(args.model_identity.read_text())
    checkpoint_path = args.output.with_suffix(".jsonl")
    with checkpoint_path.open("x") as stream:
        def checkpoint(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"case": row["case_id"], "condition": row["condition"], "completed": row["generation"]["completed"], "wall_ms": row["generation"]["wall_ms"], "error": row["generation"]["error"]}), flush=True)
        report = asyncio.run(run(plan, corpus, manifest, Settings.from_env().llm, model_identity=identity, max_tokens=args.max_tokens, context_chars=args.context_chars, retrieval_limit=args.retrieval_limit, timeout=args.timeout, total_seconds=args.total_seconds, repetitions=args.repetitions, checkpoint=checkpoint))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return int(bool(report["execution"]["skipped"]) or any(not row["generation"]["completed"] for row in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
