"""Execute the real Engine/provider in A/B/C conditions without grading identity."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from contextlib import aclosing
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from bias.config import Settings
from bias.corpus import Corpus, source_event_id
from bias.engine import Engine
from bias.grounding import audit_citations
from bias.models import ChatRequest, ChatTurn
from bias.prompts import PROMPT_BUILDERS
from bias.providers import ProviderError

from .cases import GenerationCase, GenerationSuite, read_suite, suite_fingerprint, validate_suite
from .observe import AuditedLLM, ObservedTransport, StreamObservation

CONDITIONS = {"profile": "A", "rag": "B", "behavior": "C"}
LIMITATIONS = [
    "These are declared development cases, not a blinded generalization or persona-fidelity benchmark.",
    "Citation format, identifiers, and sentence-like trailing-marker coverage are checked mechanically; factual entailment, relevance, persona resemblance, Korean quality, and injection resistance remain ungraded.",
    "Citation units are a formatting heuristic, not extracted or verified factual claims. Disclaimers and unusual abbreviations can produce uncited units.",
    "No repair request or automatic citation insertion occurs. The raw model response is retained even when incomplete or unsupported.",
    "No-evidence RAG responses bypass the language model; their abstention does not measure model abstention skill.",
    "Withheld references never enter application prompts. Exposure during base-model pretraining cannot be ruled out by this evaluation.",
    "Provider-reported token usage is kept separate from decoded text chunks; missing token counts remain null.",
    "Runs are sequential with fresh sessions, and condition order rotates across cases. First-load and hardware effects can still influence latency.",
]


def fingerprint_corpus(corpus: Corpus) -> str:
    payload = {"profile": corpus.persona.model_dump(mode="json"), "sources": [corpus.sources[key].model_dump(mode="json") for key in sorted(corpus.sources)], "evidence": [corpus.evidence[key].model_dump(mode="json") for key in sorted(corpus.evidence)]}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def citation_report(raw: str, allowed: set[str], required: bool) -> dict:
    audit = audit_citations(raw, allowed, required=required)
    status = "invalid" if audit.invalid_ids or audit.malformed_markers else "valid" if audit.valid_ids else "missing" if required else "not_required"
    return {"required": required, "status": status, "valid_ids": list(audit.valid_ids), "invalid_ids": list(audit.invalid_ids), "malformed_markers": list(audit.malformed_markers), "coverage": audit.as_dict(), "semantic_entailment": "ungraded"}


def retrieval_check(case: GenerationCase, mode: str, retrieved: list[str], corpus: Corpus) -> dict:
    if mode == "profile":
        return {"status": "not_applicable", "reason": "Profile-only condition has no retrieval."}
    if case.expected_action == "abstain":
        return {"status": "pass" if not retrieved else "fail", "check": "no_evidence_gate", "model_abstention": "not_measured_by_this_check"}
    if not (case.expected_evidence_ids or case.expected_source_ids):
        return {"status": "ungraded"}
    sources = {corpus.evidence[key].source_id for key in retrieved}
    hit = bool(set(case.expected_evidence_ids).intersection(retrieved) or set(case.expected_source_ids).intersection(sources))
    return {"status": "pass" if hit else "fail", "check": "declared_top_k_target", "semantic_answer_quality": "ungraded"}


async def run_case(settings: Settings, corpus: Corpus, case: GenerationCase, mode: str, repetition: int, transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None, total_budget_seconds: float | None = None, *, prompt_version: str = "compact") -> dict:
    observation = StreamObservation()
    provider = AuditedLLM(settings.llm, timeout=settings.request_timeout, max_chars=settings.max_response_chars, max_tokens=settings.llm_max_tokens, corpus=corpus, transport=ObservedTransport(observation, transport_factory))
    engine = Engine(settings, corpus, llm=provider, prompt_builder=PROMPT_BUILDERS[prompt_version])
    request = ChatRequest(message=case.question, mode=mode, language=case.language)
    started = time.perf_counter()
    first_text_ms = None
    fragments = []
    events = []
    turn: ChatTurn | None = None
    error = None
    try:
        async with asyncio.timeout(total_budget_seconds if total_budget_seconds is not None else settings.request_timeout + 3):
            async with aclosing(engine.events(request)) as stream:
                async for event, payload in stream:
                    events.append(event)
                    if event == "token":
                        if first_text_ms is None:
                            first_text_ms = (time.perf_counter() - started) * 1000
                        fragments.append(payload["text"])
                    elif event == "result":
                        turn = ChatTurn.model_validate(payload["turn"])
    except ProviderError as exc:
        error = {"code": exc.code, "status": exc.status, "message": exc.public_message}
    except TimeoutError:
        matrix_deadline = total_budget_seconds is not None and total_budget_seconds < settings.request_timeout + 2
        error = {"code": "evaluation_deadline" if matrix_deadline else "engine_timeout", "status": 504, "message": "The evaluation's total deadline interrupted this run." if matrix_deadline else "The application engine exceeded its deadline."}
    except Exception as exc:
        error = {"code": "evaluation_or_engine_error", "type": type(exc).__name__, "message": "The run could not complete; inspect configuration and test declarations."}
    wall_ms = (time.perf_counter() - started) * 1000
    raw = "".join(fragments)
    retrieved = turn.grounding.retrieved_ids if turn else provider.context_ids
    prompt_ids = provider.context_ids
    forbidden_events = {source_event_id(corpus.sources[key]) for key in case.forbidden_source_ids}
    leaks = [key for key in prompt_ids if corpus.sources[corpus.evidence[key].source_id].split != "train" or source_event_id(corpus.sources[corpus.evidence[key].source_id]) in forbidden_events]
    inference = provider.calls > 0
    required = mode != "profile" and turn is not None and turn.provider == "openai" and bool(retrieved)
    citations = citation_report(raw, set(retrieved) if mode != "profile" else set(), required)
    citations["response_complete"] = turn is not None
    if error:
        citations["partial_marker_status"] = citations["status"] if raw else None
        citations["status"] = "not_evaluated"
        citations["reason"] = "An incomplete response is excluded from completed-answer citation checks."
    row = {
        "case_id": case.id, "condition": CONDITIONS[mode], "mode": mode, "repetition": repetition,
        "question": case.question, "language": case.language, "category": case.category, "expected_action": case.expected_action,
        "execution": {"completed": turn is not None, "language_model_requested": inference, "provider": turn.provider if turn else "openai" if inference else None, "http_requests": observation.requests, "http_statuses": observation.http_statuses, "error": error, "events": events},
        "latency": {"wall_ms": round(wall_ms, 3), "first_visible_text_ms": round(first_text_ms, 3) if first_text_ms is not None else None, "engine_metrics": turn.metrics.model_dump() if turn else None},
        "tokens": {"usage": observation.usage, "usage_status": observation.usage_status, "visible_characters": len(raw), "visible_chunks": len(fragments), "finish_reason": observation.finish_reason, "requested_max_tokens": settings.llm_max_tokens, "response_model": observation.response_model},
        "prompt_audit": {"version": prompt_version, "sha256": provider.prompt_sha256, "characters": provider.prompt_chars, "evidence_ids": prompt_ids, "evidence_text_sha256": provider.evidence_text_sha256, "held_out_or_forbidden_ids": leaks, "reference_metadata_forwarded": False, "fresh_session": True},
        "retrieved_ids": retrieved,
        "retrieval_check": retrieval_check(case, mode, retrieved, corpus),
        "citation_format": citations,
        "raw_response": raw,
        "turn": turn.model_dump(mode="json") if turn else None,
        "semantic_review": {"status": "ungraded", "reviewer": None, "review_note": case.review_note, "reference_evidence_ids": case.reference_evidence_ids, "factual_entailment": None, "answer_relevance": None, "abstention_appropriateness": None, "persona_fidelity": None, "injection_resistance": None},
    }
    # A fresh Engine is discarded after each condition; no previous condition's
    # answer, reference answer, or conversation memory can enter the next prompt.
    return row


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for mode, condition in CONDITIONS.items():
        group = [row for row in rows if row["mode"] == mode]
        if not group:
            continue
        latency = [row["latency"]["wall_ms"] for row in group]
        usage_rows = [row for row in group if row["tokens"]["usage"] is not None]
        required = [row for row in group if row["citation_format"]["required"]]
        summary[condition] = {
            "mode": mode, "runs": len(group), "completed": sum(row["execution"]["completed"] for row in group),
            "model_requests": sum(row["execution"]["language_model_requested"] for row in group),
            "errors": dict(Counter(row["execution"]["error"]["code"] for row in group if row["execution"]["error"])),
            "required_citation_format_checks": len(required), "valid_citation_format": sum(row["citation_format"]["status"] == "valid" for row in required),
            "complete_citation_coverage": sum(row["citation_format"]["coverage"]["status"] == "complete" for row in required),
            "citation_coverage_status_counts": dict(Counter(row["citation_format"]["coverage"]["status"] for row in required)),
            "citation_status_counts": dict(Counter(row["citation_format"]["status"] for row in group)),
            "prompt_split_leaks": sum(bool(row["prompt_audit"]["held_out_or_forbidden_ids"]) for row in group),
            "wall_ms_median": round(statistics.median(latency), 3), "wall_ms_min": min(latency), "wall_ms_max": max(latency),
            "usage_reported_runs": len(usage_rows), "completion_tokens_reported": sum(row["tokens"]["usage"]["completion_tokens"] for row in usage_rows) if usage_rows else None,
            "semantic_reviews": 0, "persona_score": None,
        }
    return summary


async def run_suite(settings: Settings, corpus: Corpus, suite: GenerationSuite, *, modes: tuple[str, ...] = ("profile", "rag", "behavior"), repetitions: int = 1, allow_preview: bool = False, max_total_seconds: float = 600, transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None, progress: Callable[[dict], None] | None = None, checkpoint: Callable[[dict], None] | None = None, prompt_version: str = "compact") -> dict:
    validate_suite(suite, corpus)
    if prompt_version not in PROMPT_BUILDERS:
        raise ValueError("Choose a declared prompt version: " + ", ".join(PROMPT_BUILDERS))
    if not modes or len(set(modes)) != len(modes) or any(mode not in CONDITIONS for mode in modes):
        raise ValueError("Choose distinct profile, rag, or behavior conditions")
    if not 1 <= repetitions <= 5:
        raise ValueError("Use between one and five repetitions")
    if not math.isfinite(max_total_seconds) or not 0.01 <= max_total_seconds <= 3600:
        raise ValueError("Total evaluation time must be between 0.01 and 3600 seconds")
    if not settings.llm.configured and not allow_preview:
        raise ValueError("Configure BIAS_LLM_BASE_URL and BIAS_LLM_MODEL, or explicitly select --allow-preview")
    plan = []
    for repetition in range(1, repetitions + 1):
        for index, case in enumerate(suite.cases):
            offset = (index + repetition - 1) % len(modes)
            order = modes[offset:] + modes[:offset]
            for mode in order:
                plan.append((case, mode, repetition))
    rows = []
    started = time.perf_counter()
    deadline = started + max_total_seconds
    for case, mode, repetition in plan:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        row = await run_case(settings, corpus, case, mode, repetition, transport_factory, remaining, prompt_version=prompt_version)
        rows.append(row)
        if checkpoint:
            checkpoint(row)
        if progress:
            progress({"case_id": case.id, "condition": row["condition"], "completed": row["execution"]["completed"], "citation_format": row["citation_format"]["status"], "error": row["execution"]["error"], "wall_ms": row["latency"]["wall_ms"]})
    return {
        "schema_version": "generation-report.v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_id": suite.id, "suite_sha256": hashlib.sha256(suite_fingerprint(suite)).hexdigest(), "corpus_sha256": fingerprint_corpus(corpus),
        "declaration": suite.declaration, "development_only": True,
        "execution_scope": {"planned_runs": len(plan), "attempted_runs": len(rows), "max_total_seconds": max_total_seconds, "wall_ms": round((time.perf_counter() - started) * 1000, 3), "skipped_runs": [{"case_id": case.id, "condition": CONDITIONS[mode], "mode": mode, "repetition": repetition, "reason": "total_deadline"} for case, mode, repetition in plan[len(rows):]]},
        "configuration": {"model": settings.llm.model, "prompt_version": prompt_version, "llm_configured": settings.llm.configured, "max_tokens": settings.llm_max_tokens, "timeout_seconds": settings.request_timeout, "retrieval_limit": settings.retrieval_limit, "modes": list(modes), "repetitions": repetitions, "transport": "test_injected" if transport_factory else "actual_http", "condition_order": "rotating_sequential", "session_policy": "new_engine_and_session_per_condition"},
        "corpus": corpus.stats(), "quality": corpus.quality_report(), "summary": summarize(rows), "runs": rows, "limitations": LIMITATIONS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--case", action="append", dest="case_ids", help="Select a declared case; may be repeated")
    parser.add_argument("--modes", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--repetitions", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--prompt-version", choices=list(PROMPT_BUILDERS), default="compact", help="Select a frozen baseline or an explicitly versioned grounding candidate")
    parser.add_argument("--max-total-seconds", type=float, default=600)
    parser.add_argument("--allow-preview", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/local/generation-evaluation.json"))
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("Choose each mode at most once.")
    settings = Settings.from_env()
    corpus = Corpus.load(settings.corpus_dir)
    suite = read_suite(args.cases)
    if args.case_ids:
        missing = set(args.case_ids) - {case.id for case in suite.cases}
        if missing:
            parser.error("Unknown case IDs: " + ", ".join(sorted(missing)))
        suite = suite.model_copy(update={"cases": [case for case in suite.cases if case.id in args.case_ids]})
    validate_suite(suite, corpus)
    if args.validate_only:
        print(json.dumps({"valid": True, "cases": len(suite.cases), "corpus": corpus.stats(), "inference_performed": False}))
        return 0
    if args.output.exists():
        parser.error("The output already exists; choose a new report path.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_suffix(".jsonl")
    if args.output.suffix != ".json" or checkpoint_path.exists():
        parser.error("Use a new .json output path with no existing companion .jsonl checkpoint.")
    if not settings.llm.configured and not args.allow_preview:
        parser.error("Configure BIAS_LLM_BASE_URL and BIAS_LLM_MODEL, or explicitly select --allow-preview")
    with checkpoint_path.open("x", encoding="utf-8") as checkpoint_stream:
        def save_checkpoint(row: dict) -> None:
            checkpoint_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            checkpoint_stream.flush()

        try:
            report = asyncio.run(run_suite(settings, corpus, suite, modes=tuple(args.modes), repetitions=args.repetitions, allow_preview=args.allow_preview, max_total_seconds=args.max_total_seconds, transport_factory=None, progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True), checkpoint=save_checkpoint, prompt_version=args.prompt_version))
        except KeyboardInterrupt:
            print(json.dumps({"status": "cancelled", "completed_runs_checkpoint": str(checkpoint_path), "semantic_review": "ungraded"}), flush=True)
            return 130
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "summary": report["summary"], "semantic_review": "ungraded"}, ensure_ascii=False))
    return 1 if report["execution_scope"]["skipped_runs"] or any(row["execution"]["error"] or row["prompt_audit"]["held_out_or_forbidden_ids"] for row in report["runs"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
