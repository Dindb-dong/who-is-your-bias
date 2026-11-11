"""Run frozen comparison conditions against the verified local language service."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import platform
import random
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from bias.config import ProviderSettings
from services.language.adapters import repository_id
from tools.evaluation.run import citation_report
from tools.reference_evaluation.protocol import lexical_diagnostics
from tools.reference_evaluation.run import generate

from .protocol import CONDITIONS, _context, digest, validate_plan

IDENTITY_FIELDS = ("base_model", "base_revision", "base_checkpoint_manifest_sha256",
                   "tokenizer_contract_sha256", "chat_template_options")
RATINGS = ("factual_support", "temporal_fidelity", "style_similarity", "answer_quality",
           "unsupported_first_person", "repetitive_mannerisms", "quotation_memorization")


def _runtime_identity(runtime: dict) -> dict:
    required = ("device", "dtype", "backend", "context_window", "threads", "attention", "base_parameters")
    if (not isinstance(runtime, dict)
            or any(not isinstance(runtime.get(key), str) or not runtime[key].strip()
                   for key in ("device", "dtype", "backend", "attention"))
            or any(type(runtime.get(key)) is not int or runtime[key] <= 0
                   for key in ("context_window", "threads", "base_parameters"))):
        raise ValueError("Actual language runtime precision/device and measured size identity is required")
    versions = runtime.get("versions")
    if not isinstance(versions, dict) or any(not isinstance(versions.get(key), str) or not versions[key].strip()
                                           for key in ("torch", "transformers")):
        raise ValueError("Actual Torch and Transformers versions are required")
    defaults = runtime.get("generation_defaults")
    if (not isinstance(defaults, dict) or type(defaults.get("seed")) is not int or not 0 <= defaults["seed"] < 2**32
            or type(defaults.get("top_k")) is not int or not 1 <= defaults["top_k"] <= 100
            or any(isinstance(defaults.get(key), bool) or not isinstance(defaults.get(key), (int, float))
                   or not math.isfinite(defaults[key]) for key in ("temperature", "top_p"))
            or not 0 <= defaults["temperature"] <= 2 or not 0 < defaults["top_p"] <= 1):
        raise ValueError("Actual language generation defaults are required")
    return {**{key: runtime[key] for key in required},
            "versions": {key: versions[key] for key in ("torch", "transformers")},
            "generation_defaults": {key: defaults[key] for key in ("seed", "temperature", "top_p", "top_k")}}


def _adapter_identity(adapter: dict, identity: dict, condition: str, expected_sha: str | None) -> None:
    if not isinstance(adapter, dict):
        raise ValueError("The runtime must explicitly identify its adapter selection")
    if condition != "adapter":
        if any(adapter.get(key) is not False for key in ("configured", "loaded", "verified")):
            raise ValueError("Base and contextual-style conditions cannot have an adapter loaded")
        return
    hashes = ("run_sha256", "artifact_manifest_sha256", "dataset_sha256")
    families = adapter.get("training_event_families")
    if (not isinstance(expected_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha)
            or not all(adapter.get(key) is True for key in ("configured", "loaded", "verified"))
            or adapter.get("run_sha256") != expected_sha
            or any(not isinstance(adapter.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", adapter[key]) for key in hashes)
            or any(adapter.get(key) != identity.get(key) for key in ("base_model", "base_revision",
                                  "base_checkpoint_manifest_sha256", "tokenizer_contract_sha256"))
            or not isinstance(families, list) or not families
            or any(not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", item) for item in families)
            or len(set(families)) != len(families)
            or type(adapter.get("optimizer_steps")) is not int or adapter["optimizer_steps"] < 1):
        raise ValueError("The adapter condition requires its exact verified completed training run")


def _expected_request(plan: dict, condition: str, served_model: str, case: dict) -> dict:
    return {"model": served_model, "messages": case["messages"]["style" if condition == "larger" else condition],
            "stream": True, "stream_options": {"include_usage": True}, "max_tokens": plan["policy"]["max_tokens"],
            "temperature": 0, "top_p": 1, "seed": plan["policy"]["seed"]}


def _health_url(provider: ProviderSettings) -> str:
    parsed = urlsplit(provider.base_url or "")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        raise ValueError("Comparison endpoints must use the local language service /v1 contract")
    return urlunsplit((parsed.scheme, parsed.netloc, path[:-3] + "/health", "", ""))


async def read_identity(provider: ProviderSettings, *, condition: str, adapter_sha256: str | None = None,
                        transport=None, deadline: float | None = None) -> dict:
    remaining = 15 if deadline is None else min(15., deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("Comparison absolute deadline exhausted")
    async with asyncio.timeout(remaining):
        async with httpx.AsyncClient(timeout=remaining, transport=transport, follow_redirects=False) as client:
            response = await client.get(_health_url(provider), headers=provider.headers)
            response.raise_for_status()
            health = response.json()
    if health.get("status") != "ready" or health.get("model_loaded") is not True or health.get("model") != provider.model:
        raise ValueError("Load the declared language model before measuring a comparison")
    identity, adapter = health.get("identity", {}), health.get("adapter", {})
    if (identity.get("schema_version") != "language-model-identity.v1"
            or not re.fullmatch(r"[a-f0-9]{40}", identity.get("base_revision") or "")
            or not repository_id(identity.get("base_model"))
            or any(not re.fullmatch(r"[a-f0-9]{64}", identity.get(k) or "")
                   for k in ("base_checkpoint_manifest_sha256", "tokenizer_contract_sha256"))
            or identity.get("chat_template_options") != {"enable_thinking": False}):
        raise ValueError("Actual immutable base/tokenizer identity is required for a controlled comparison")
    _adapter_identity(adapter, identity, condition, adapter_sha256)
    comparable = _runtime_identity(health.get("runtime", {}))
    return {"identity": {key: identity[key] for key in IDENTITY_FIELDS}, "adapter": adapter, "runtime": comparable}


def _validate_base(plan: dict, observed: dict, condition: str) -> None:
    if condition == "larger":
        return
    expected = plan.get("base_identity") or {}
    if any(observed["identity"].get(key) != value for key, value in expected.items()):
        raise ValueError("Serving base differs from the preregistered comparison base")
    if plan["exposure"] == "heldout" and condition == "adapter":
        families = observed["adapter"].get("training_event_families")
        if not isinstance(families, list) or set(families) & {c["event_family_id"] for c in plan["cases"]}:
            raise ValueError("Adapter training-family isolation from held-out references is unverified")


def _validate_report_identity(plan: dict, observed: dict, condition: str, expected_adapter_sha: str | None = None) -> None:
    identity, adapter = observed.get("identity", {}), observed.get("adapter", {})
    if (set(identity) != set(IDENTITY_FIELDS)
            or not repository_id(identity.get("base_model"))
            or not re.fullmatch(r"[a-f0-9]{40}", identity.get("base_revision") or "")
            or identity.get("chat_template_options") != {"enable_thinking": False}
            or any(not re.fullmatch(r"[a-f0-9]{64}", identity.get(k) or "")
                   for k in ("base_checkpoint_manifest_sha256", "tokenizer_contract_sha256"))):
        raise ValueError("Saved model identity is incomplete")
    _adapter_identity(adapter, identity, condition, expected_adapter_sha)
    _runtime_identity(observed.get("runtime", {}))
    _validate_base(plan, observed, condition)


async def run_condition(plan: dict, condition: str, provider: ProviderSettings, output: Path, *,
                        adapter_sha256: str | None = None, timeout: float = 180,
                        total_seconds: float = 3600, transport=None) -> dict:
    validate_plan(plan)
    if condition not in (*CONDITIONS, "larger") or not provider.configured:
        raise ValueError("Choose a declared comparison condition and configured endpoint")
    if (not math.isfinite(timeout) or not 0 < timeout <= 600
            or not math.isfinite(total_seconds) or not 0 < total_seconds <= 14400):
        raise ValueError("Invalid bounded execution time")
    if output.exists():
        raise ValueError("Comparison run output already exists; preserve it and choose a new path")
    started, rows = time.monotonic(), []
    deadline = started + total_seconds
    identity = await read_identity(provider, condition=condition, adapter_sha256=adapter_sha256, transport=transport, deadline=deadline)
    _validate_base(plan, identity, condition)
    output.mkdir(parents=True)
    policy = plan["policy"]
    report = {"schema_version": "text-adaptation-condition-run.v1", "condition": condition,
        "plan_sha256": plan["plan_sha256"], "exposure": plan["exposure"],
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "model": identity, "served_model": provider.model, "requested_adapter_run_sha256": adapter_sha256,
        "policy": policy, "planned_case_ids": [c["id"] for c in plan["cases"]],
        "transport": "injected_test_transport" if transport is not None else "actual_http",
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "timing_scope": "warm_model_http_generation_no_speech_or_acoustic_measurement",
        "human_review": "ungraded", "efficacy_claim": False, "runs": rows}

    def checkpoint():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report.pop("run_sha256", None)
        report["run_sha256"] = digest(report)
        (output / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    checkpoint()
    try:
        for case in plan["cases"]:
            remaining = total_seconds - (time.monotonic() - started)
            if remaining <= 0:
                report["status"] = "deadline_exhausted"
                break
            before = await read_identity(provider, condition=condition, adapter_sha256=adapter_sha256, transport=transport, deadline=deadline)
            if before != identity:
                raise ValueError("Model identity changed during the frozen run")
            prompt_condition = "style" if condition == "larger" else condition
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Comparison deadline exhausted before generation")
            result = await generate(provider, case["messages"][prompt_condition],
                max_tokens=policy["max_tokens"], timeout=min(timeout, remaining), seed=policy["seed"], transport=transport)
            # Record what the provider returned before attempting another
            # network operation. A deadline, disconnect or changed health
            # identity must never turn an attempted generation into a skip.
            reported = copy.deepcopy(result)
            errors = []
            if reported.get("response_model") != provider.model:
                errors.append("provider_model_identity_changed")
            if reported.get("request_sha256") != digest(_expected_request(plan, condition, provider.model, case)):
                errors.append("generation_request_identity_changed")
            reference = case.get("reference_answer")
            row = {"case_id": case["id"], "condition": condition,
                "prompt_sha256": case["prompt_sha256"][prompt_condition],
                "factual_sha256": case["factual_sha256"], "factual_ids": case["factual_ids"],
                "style_ids": [] if condition == "facts" else case["style_ids"],
                "provider_generation": reported,
                "generation": {**copy.deepcopy(reported), "completed": False,
                               "error": errors[0] if errors else reported.get("error") or "post_generation_identity_pending"},
                "identity_check": {"before": "verified", "after": "pending"},
                "validation_errors": errors,
                "citation_format": citation_report(reported["text"], set(case["factual_ids"]), required=bool(case["factual_ids"])),
                "reference_surface_overlap": lexical_diagnostics(reported["text"], reference) if reference else None,
                "judgments": {key: None for key in RATINGS}}
            rows.append(row)
            with (output / "raw-generations.jsonl").open("a") as log:
                log.write(json.dumps({"case_id": case["id"], "condition": condition,
                                      "provider_generation": reported}, ensure_ascii=False) + "\n")
                log.flush()
            checkpoint()
            post_failure = None
            try:
                after = await read_identity(provider, condition=condition, adapter_sha256=adapter_sha256,
                                            transport=transport, deadline=deadline)
                if after != identity:
                    row["identity_check"].update(after="changed", observed_after=after)
                    raise ValueError("Model identity changed while generating a case")
                row["identity_check"]["after"] = "verified"
            except BaseException as exc:
                post_failure = exc
                if row["identity_check"]["after"] == "pending":
                    row["identity_check"]["after"] = "unverified"
                row["identity_check"]["failure_type"] = type(exc).__name__
                errors.append("post_generation_identity_changed" if row["identity_check"]["after"] == "changed"
                              else "post_generation_identity_unverified")
                if not isinstance(exc, (TimeoutError, httpx.HTTPError, ValueError)):
                    raise
            finally:
                row["generation"]["completed"] = reported.get("completed") is True and reported.get("error") is None and not errors
                row["generation"]["error"] = errors[0] if errors else reported.get("error")
                with (output / "attempts.jsonl").open("a") as log:
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log.flush()
                checkpoint()
            if post_failure is not None:
                report["status"] = "deadline_exhausted" if time.monotonic() >= deadline else "failed"
                report["failure_type"] = type(post_failure).__name__
                break
        else:
            report["status"] = "completed"
        report["skipped_case_ids"] = [c["id"] for c in plan["cases"][len(rows):]]
    except TimeoutError:
        report["status"] = "deadline_exhausted"
        report["skipped_case_ids"] = [c["id"] for c in plan["cases"][len(rows):]]
    except BaseException as exc:
        report["status"] = "failed"
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        checkpoint()
    return report


def compare_reports(plan: dict, reports: list[dict]) -> dict:
    validate_plan(plan)
    by_condition = {r["condition"]: r for r in reports}
    if len(by_condition) != len(reports) or not set(CONDITIONS) <= set(by_condition) or set(by_condition) - {*CONDITIONS, "larger"}:
        raise ValueError("Comparison requires exactly one facts/style/adapter report, with optional larger base")
    baseline = by_condition["facts"]["model"]
    expected_ids = [c["id"] for c in plan["cases"]]
    for condition, report in by_condition.items():
        sealed = dict(report)
        seal = sealed.pop("run_sha256", None)
        if digest(sealed) != seal:
            raise ValueError("Frozen condition report changed")
        if (report.get("schema_version") != "text-adaptation-condition-run.v1"
                or report.get("plan_sha256") != plan["plan_sha256"] or report.get("policy") != plan["policy"]
                or report.get("planned_case_ids") != expected_ids):
            raise ValueError("Cannot compare different plans, requests, or case selections")
        _validate_report_identity(plan, report["model"], condition, report.get("requested_adapter_run_sha256"))
        if condition != "larger" and (report["model"]["identity"] != baseline["identity"]
                                      or report["model"]["runtime"] != baseline["runtime"]):
            raise ValueError("Base/style/adapter must share exact base, tokenizer, precision and runtime")
        if condition == "larger" and report["model"]["runtime"]["base_parameters"] <= baseline["runtime"]["base_parameters"]:
            raise ValueError("The larger-base condition must have a measured larger parameter count")
        seen = set()
        for row in report["runs"]:
            if row["case_id"] not in expected_ids or row["case_id"] in seen or row["condition"] != condition:
                raise ValueError("Duplicate or foreign case in comparison results")
            seen.add(row["case_id"])
            case = next(c for c in plan["cases"] if c["id"] == row["case_id"])
            if (row["factual_sha256"] != case["factual_sha256"]
                    or row["prompt_sha256"] != case["prompt_sha256"]["style" if condition == "larger" else condition]):
                raise ValueError("Executed prompt differs from the registered paired input")
            payload = _expected_request(plan, condition, report["served_model"], case)
            generation, raw = row["generation"], row.get("provider_generation", {})
            if generation.get("request_sha256") != digest(payload) or raw.get("request_sha256") != digest(payload):
                raise ValueError("Actual generation request differs from frozen messages and decoding policy")
            if {k: v for k, v in generation.items() if k not in {"completed", "error"}} != {k: v for k, v in raw.items() if k not in {"completed", "error"}}:
                raise ValueError("Saved generation content differs from the original provider result")
            checks = row.get("identity_check", {})
            if generation.get("completed") and (raw.get("completed") is not True or raw.get("error") is not None
                    or generation.get("error") is not None or generation.get("response_model") != report["served_model"]
                    or checks.get("before") != "verified" or checks.get("after") != "verified" or row.get("validation_errors")):
                raise ValueError("Completed response lacks verified request and model identity")
    summaries = {}
    for condition, report in by_condition.items():
        complete = [r for r in report["runs"] if r["generation"]["completed"]]
        summaries[condition] = {"attempted": len(report["runs"]), "completed": len(complete),
            "run_status": report["status"], "run_sha256": report["run_sha256"], "model": report["model"],
            "errors": dict(Counter(r["generation"].get("error") for r in report["runs"] if r["generation"].get("error"))),
            "generation_ms_median": statistics.median(r["generation"]["wall_ms"] for r in complete) if complete else None,
            "output_sha256": {r["case_id"]: hashlib.sha256(r["generation"]["text"].encode()).hexdigest() for r in report["runs"]}}
    return {"schema_version": "text-adaptation-comparison-summary.v1", "plan_sha256": plan["plan_sha256"],
        "exposure": plan["exposure"], "conditions": summaries,
        "cases_with_style_contrast": sum(c["style_contrast_present"] for c in plan["cases"]),
        "cases_without_factual_candidates": sum(not c["factual_ids"] for c in plan["cases"]),
        "larger_model_compared": "larger" in by_condition, "human_review": "ungraded", "efficacy_claim": False,
        "limitations": ["Completion and citation syntax are not semantic entailment or persona similarity.",
            "Surface overlap can indicate memorization as well as matching wording; judge it separately.",
            "Development cases and a bounded training smoke do not establish held-out improvement.",
            "These are sequential model runs, not simultaneous audiovisual latency measurements."]}


def export_review(plan: dict, reports: list[dict], output: Path, *, seed: int = 42) -> dict:
    summary = compare_reports(plan, reports)
    if output.exists():
        raise ValueError("Blind review output exists")
    samples, key, failed = [], {}, []
    candidates = [(report, row) for report in reports for row in report["runs"]]
    random.Random(seed).shuffle(candidates)
    cases = {case["id"]: case for case in plan["cases"]}
    for report, row in candidates:
        if not row["generation"]["completed"]:
            failed.append({"case_id": row["case_id"], "condition": report["condition"], "error": row["generation"]["error"]})
            continue
        identifier = f"sample-{len(samples)+1:04d}"
        case = cases[row["case_id"]]
        samples.append({"id": identifier, "case_id": case["id"], "question": case["question"],
            "answer": row["generation"]["text"], "reference_answer": case.get("reference_answer"),
            "factual_evidence": _context(case["messages"]["facts"])["factual_evidence"],
            "ratings": {name: None for name in RATINGS}, "reviewer": None, "reviewer_type": None})
        key[identifier] = {"condition": report["condition"], "case_id": case["id"],
                           "output_sha256": hashlib.sha256(row["generation"]["text"].encode()).hexdigest()}
    (output / "reviewer").mkdir(parents=True)
    (output / "private").mkdir()
    for filename, content in [("reviewer/samples.json", {"plan_sha256": plan["plan_sha256"], "samples": samples}),
                              ("private/answer-key.json", key), ("private/failed-attempts.json", failed),
                              ("private/comparison.json", summary)]:
        (output / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    return {"samples": len(samples), "failed_attempts": len(failed), "human_review": "ungraded",
            "share_only": "reviewer/", "private_answer_key": "private/answer-key.json"}
