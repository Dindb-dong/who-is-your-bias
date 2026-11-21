"""Frozen, matched answer-quality probes for the optional Jensen persona graph.

No keyword score, canned answer, or evidence preview is a generated answer.
See docs/persona-graph-evaluation-protocol.md before interpreting a result.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time
from typing import Callable

import httpx

from bias.context_budget import ByteBudgetCounter, build_snapshot_messages
from bias.corpus import Corpus
from bias.models import ChatRequest
from bias.query_policy import public_evidence_limitation
from bias.retrieval import Retriever, SearchResult
from bias.snapshots import PersonaSnapshot, corpus_digest, select_temporal


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("baseline", "flat", "graph")
SCHEMA = "persona-graph-evaluation.v1"
MAX_RAW_CHARS = 100_000
PROVIDER_DIALECTS = ("recorded-gguf", "team-language")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class HTTPTokenCounter:
    """Count the exact template using the same server as local generation."""
    exact = True

    def __init__(self, url: str, *, model: str, revision: str):
        self.url = url
        self.identity = f"server-chat-template:{model}@{revision}"
        self.context_window: int | None = None
        self.cache: dict[str, int] = {}
        self.count([{"role": "user", "content": ""}])

    def count(self, messages: list[dict[str, str]]) -> int:
        key = digest(messages)
        if key not in self.cache:
            response = httpx.post(self.url, json={"messages": messages}, timeout=60)
            response.raise_for_status()
            data = response.json()
            count = data.get("prompt_tokens", data.get("count"))
            window = data.get("context_window")
            if type(count) is not int or count < 0 or type(window) is not int or window < 256:
                raise ValueError("Token-count service returned invalid token count/context window")
            if self.context_window is not None and window != self.context_window:
                raise ValueError("Token-count service changed context window during freeze")
            self.context_window = window
            self.cache[key] = count
        return self.cache[key]


def prepare_case(corpus, snapshot, graph, question: str, *, language: str = "en",
                 context_window: int = 16384, max_tokens: int = 256,
                 retrieval_limit: int = 4, retrieval_backend: str = "lexical",
                 counter=None) -> dict:
    """Use production retrieval, guards, temporal scope and prompt packing.

    This function deliberately accepts no expected values, source labels, ideal
    answers or judge rubric. All conditions have empty conversation history.
    """
    from bias.persona_graph import augment_result, select_advice_memory

    request = ChatRequest(message=question, language=language, mode="behavior")
    temporal = select_temporal(corpus, snapshot, question) if snapshot else None
    limitation = public_evidence_limitation(question)
    if retrieval_backend == "jensen-lexical":
        from bias.jensen_retrieval import JensenConceptRetriever
        retriever = JensenConceptRetriever(corpus)
    elif retrieval_backend == "lexical":
        retriever = Retriever(corpus)
    else:
        raise ValueError("This probe supports lexical or jensen-lexical retrieval; hybrid needs a separate frozen run.")
    query = question
    if limitation:
        result = SearchResult((), 0, limitation)
    elif temporal and not temporal.eligible_ids:
        result = SearchResult((), 0, temporal.reason)
    else:
        if temporal and len(temporal.eligible_ids) != len(corpus.training_evidence):
            filtered = Corpus(corpus.persona, corpus.sources,
                              {key: item for key, item in corpus.evidence.items() if key in temporal.eligible_ids})
            retriever = retriever.with_corpus(filtered)
        if temporal and temporal.intent.mode != "comparison":
            query = re.sub(r"(?<!\d)(?:19|20)\d{2}(?!\d)", "", query)
        result = retriever.search(query, limit=retrieval_limit, behavior=True, allow_live=bool(temporal))
    selection = None if limitation else select_advice_memory(
        graph, corpus, question, snapshot=snapshot, temporal=temporal)
    supplemented = augment_result(selection, corpus, result, limit=retrieval_limit) if selection else result
    conditions = {}
    for condition in CONDITIONS:
        selected = None if condition == "baseline" else selection
        retrieved = result if condition == "baseline" else supplemented
        speculative = not retrieved.hits and not limitation
        if limitation:
            # This is an existing deterministic production guard, not an LLM success.
            from bias.engine import evidence_preview
            conditions[condition] = {"messages": [], "supplied_ids": [], "budget": {},
                "speculative": False, "policy_guard": limitation,
                "deterministic_text": evidence_preview(request, retrieved)}
            continue
        package = build_snapshot_messages(corpus, request, retrieved, [], snapshot, temporal,
            context_window=context_window, output_tokens=max_tokens, counter=counter or ByteBudgetCounter(),
            speculative=speculative, live_concise=False, answer_policy=None,
            persona_selection=selected, persona_memory_format="graph" if condition == "graph" else "flat")
        conditions[condition] = {"messages": package.messages, "supplied_ids": list(package.supplied_ids),
            "budget": package.budget, "speculative": speculative, "policy_guard": None,
            "retrieval_reason": retrieved.reason}
    if conditions["flat"]["supplied_ids"] != conditions["graph"]["supplied_ids"]:
        raise ValueError("Flat and graph factual evidence differs; increase the common context budget.")
    included = [conditions[condition]["budget"].get("persona_memory", {}).get("included_value_ids", [])
                for condition in ("flat", "graph")]
    if included[0] != included[1]:
        raise ValueError("Flat and graph included value paths differ; increase the common context budget.")
    if selection is not None and selection.paths:
        for condition in ("flat", "graph"):
            if not conditions[condition]["budget"].get("persona_memory", {}).get("included"):
                raise ValueError("Selected persona memory did not fit in both conditions; freeze with a sufficient common budget.")
    return {"conditions": conditions, "selection": selection.trace() if selection else None,
            "temporal": {"mode": temporal.intent.mode, "eligible_ids": sorted(temporal.eligible_ids)} if temporal else None,
            "lexical_query": query}


def validate_registry(registry: dict, corpus: Corpus) -> None:
    if registry.get("schema_version") != "persona-graph-probes.v1" or not registry.get("cases"):
        raise ValueError("Unsupported or empty probe registry")
    ids = [case["id"] for case in registry["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs")
    for case in registry["cases"]:
        if case["category"] not in {"advice", "control"} or case["language"] != "en":
            raise ValueError("This frozen registry uses English advice/control probes")
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError("Empty question")
        if set(case["reference_evidence_ids"]) - set(corpus.evidence):
            raise ValueError("Unknown judge reference evidence")


def freeze_manifest(*, registry_path: Path, corpus_path: Path, graph_path: Path,
                    snapshot_path: Path, model: str, revision: str,
                    case_ids: list[str] | None = None, seed: int = 20261120,
                    repeats: int = 1, max_tokens: int = 256, context_window: int = 16384,
                    retrieval_backend: str = "lexical", provider_dialect: str = "recorded-gguf", counter=None) -> dict:
    from bias.persona_graph import PersonaGraph

    if repeats < 1 or max_tokens < 32 or not model or not revision:
        raise ValueError("Provide model/revision, repeats >= 1 and max_tokens >= 32")
    if provider_dialect not in PROVIDER_DIALECTS:
        raise ValueError("Unknown frozen provider dialect")
    corpus = Corpus.load(corpus_path)
    registry = read_json(registry_path)
    validate_registry(registry, corpus)
    cases = registry["cases"]
    if case_ids is not None:
        if len(case_ids) != len(set(case_ids)) or set(case_ids) - {case["id"] for case in cases}:
            raise ValueError("Case subset has duplicate or unknown IDs")
        cases = [case for case in cases if case["id"] in case_ids]
    if not cases:
        raise ValueError("Empty case subset")
    graph = PersonaGraph.load(graph_path, corpus)
    snapshot = PersonaSnapshot.load(snapshot_path, corpus)
    if getattr(counter, "context_window", context_window) < context_window:
        raise ValueError("Declared context window exceeds the token-count server's window")
    decode = {"temperature": 0, "top_p": 1, "max_tokens": max_tokens, "seed": seed}
    if provider_dialect == "recorded-gguf":
        decode.update(presence_penalty=0, chat_template_kwargs={"enable_thinking": False})
    configuration = {"model": model, "revision": revision, "decode": decode,
        "provider_dialect": provider_dialect,
        "server_policy": {"enable_thinking": False, "presence_penalty": 0,
            "verification": "request_fields" if provider_dialect == "recorded-gguf" else "team-language-health-and-versioned-service-contract"},
        "context_window": context_window, "retrieval_backend": retrieval_backend,
        "retrieval_limit": 4, "adaptive_responses": False, "live_concise": False,
        "history": "empty", "repeats": repeats, "seed": seed}
    prompts = {}
    selections = {}
    for case in cases:
        prepared = prepare_case(corpus, snapshot, graph, case["question"], language=case["language"],
            context_window=context_window, max_tokens=max_tokens,
            retrieval_backend=retrieval_backend, counter=counter)
        selections[case["id"]] = {key: value for key, value in prepared.items() if key != "conditions"}
        for condition, content in prepared["conditions"].items():
            key = f"{case['id']}:{condition}"
            prompts[key] = {**content, "prompt_sha256": digest(content["messages"])}
    rng = random.Random(seed)
    permutations = list(itertools.permutations(CONDITIONS))
    rng.shuffle(permutations)
    units = [(case["id"], repeat) for repeat in range(repeats) for case in cases]
    rng.shuffle(units)
    schedule = []
    for index, (case_id, repeat) in enumerate(units):
        for condition in permutations[index % len(permutations)]:
            prompt_key = f"{case_id}:{condition}"
            schedule.append({"attempt_id": f"{case_id}:r{repeat}:{condition}", "case_id": case_id,
                "repeat": repeat, "condition": condition, "prompt_key": prompt_key,
                "request_sha256": digest({"prompt": prompts[prompt_key], "configuration": configuration})})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    sources = {}
    for case in cases:
        judge_ids = set(case["reference_evidence_ids"])
        for condition in CONDITIONS:
            judge_ids.update(prompts[f"{case['id']}:{condition}"]["supplied_ids"])
        for evidence_id in sorted(judge_ids):
            evidence = corpus.evidence[evidence_id]
            source = corpus.sources[evidence.source_id]
            sources[evidence_id] = {"text": evidence.text, "source_id": source.id,
                "url": str(source.url), "published_at": source.published_at.isoformat(),
                "sha256": digest(evidence.model_dump(mode="json"))}
    manifest = {"schema_version": SCHEMA, "registry_id": registry["id"],
        "scope": "Synthetic English development probes; not unseen-event or human-persona validation.",
        "configuration": configuration, "conditions": list(CONDITIONS), "cases": cases,
        "prompts": prompts, "selections": selections, "schedule": schedule, "judge_sources": sources,
        "provenance": {"base_commit": commit, "corpus_sha256": corpus_digest(corpus),
            "registry_sha256": file_digest(registry_path), "graph_sha256": file_digest(graph_path),
            "snapshot_sha256": file_digest(snapshot_path), "protocol_sha256": file_digest(ROOT / "docs/persona-graph-evaluation-protocol.md"),
            "runner_sha256": file_digest(Path(__file__)),
            "production_prompt_builder_sha256": file_digest(ROOT / "src/bias/context_budget.py")}}
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def check_manifest(manifest: dict) -> None:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("schema_version") != SCHEMA or digest(unsigned) != manifest.get("manifest_sha256"):
        raise ValueError("Manifest contents changed after freeze")
    ids = [entry["attempt_id"] for entry in manifest["schedule"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate attempt IDs in manifest")


def completion_payload(messages: list[dict], configuration: dict) -> dict:
    """The dialect is frozen before inference; never adapt after HTTP failure."""
    dialect = configuration.get("provider_dialect")
    if dialect not in PROVIDER_DIALECTS:
        raise ValueError("Unknown frozen provider dialect")
    accepted = {"temperature", "top_p", "max_tokens", "seed"}
    if dialect == "recorded-gguf":
        accepted.update({"presence_penalty", "chat_template_kwargs"})
    if set(configuration["decode"]) - accepted:
        raise ValueError("Frozen decoding fields are unsupported by this provider dialect")
    return {"model": configuration["model"], "messages": messages,
            **configuration["decode"], "stream": False}


def verify_team_contract(health: dict, configuration: dict) -> dict:
    identity = health.get("identity", {})
    if health.get("model") != configuration["model"] or identity.get("chat_template_options", {}).get("enable_thinking") is not False:
        raise ValueError("Team language endpoint did not verify the frozen model and non-thinking template")
    revision = identity.get("base_revision")
    if revision is not None and revision != configuration["revision"]:
        raise ValueError("Team language endpoint revision differs from the frozen model revision")
    return {"model": health["model"], "base_revision": revision,
            "chat_template_options": identity["chat_template_options"],
            "tokenizer_contract_sha256": identity.get("tokenizer_contract_sha256"),
            "base_checkpoint_manifest_sha256": identity.get("base_checkpoint_manifest_sha256")}


class OpenAICompatibleProvider:
    """A real, non-streaming completion request. Never falls back to mock text."""
    def __init__(self, base_url: str, *, api_key: str | None = None, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("An http(s) OpenAI-compatible base URL is required")
        self.api_key = api_key
        self.timeout = timeout

    def __call__(self, messages: list[dict], configuration: dict) -> dict:
        payload = completion_payload(messages, configuration)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        server_contract = None
        with httpx.Client(timeout=self.timeout) as client:
            if configuration["provider_dialect"] == "team-language":
                health = client.get(self.base_url.removesuffix("/v1") + "/health", headers=headers)
                health.raise_for_status()
                server_contract = verify_team_contract(health.json(), configuration)
            response = client.post(self.base_url + "/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            # Retain generation metadata, not request/response headers or secrets.
            data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        return {"raw_text": message.get("content") or "", "reasoning_text": message.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"), "model": data.get("model"),
            "system_fingerprint": data.get("system_fingerprint"), "usage": data.get("usage", {}),
            "server_contract": server_contract}


def _check_results(manifest: dict, results: dict) -> None:
    if results.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Results belong to a different frozen manifest")
    planned = {entry["attempt_id"]: entry for entry in manifest["schedule"]}
    for attempt_id, result in results.get("attempts", {}).items():
        if attempt_id not in planned or result.get("request_sha256") != planned[attempt_id]["request_sha256"]:
            raise ValueError("Stored result request identity does not match this manifest")
        if result.get("result_sha256") != digest({key: value for key, value in result.items() if key != "result_sha256"}):
            raise ValueError("Stored result contents changed")


def run_manifest(manifest: dict, output_path: Path, provider: Callable, *, limit: int | None = None) -> dict:
    """Save one attempt at a time; interrupted runs resume without duplicate calls.

    ``provider(messages, configuration)`` must return the actual raw completion,
    finish_reason, model identity and usage. A callable permits local inference.
    There are no implicit retries: an error remains a measured failed attempt.
    """
    check_manifest(manifest)
    results = read_json(output_path) if output_path.exists() else {
        "schema_version": SCHEMA, "manifest_sha256": manifest["manifest_sha256"], "attempts": {}}
    _check_results(manifest, results)
    completed = 0
    for entry in manifest["schedule"]:
        if entry["attempt_id"] in results["attempts"]:
            continue
        if limit is not None and completed >= limit:
            break
        prompt = manifest["prompts"][entry["prompt_key"]]
        started = time.perf_counter()
        record = {**entry, "prompt_sha256": prompt["prompt_sha256"]}
        if prompt["policy_guard"]:
            record.update(status="policy_guard", raw_text=prompt["deterministic_text"],
                finish_reason="deterministic_guard", model=None, usage={})
        else:
            try:
                generated = provider(prompt["messages"], manifest["configuration"])
                raw = generated.get("raw_text", "")
                if not isinstance(raw, str):
                    raise ValueError("Provider returned non-text content")
                record.update({key: generated.get(key) for key in
                    ("finish_reason", "model", "system_fingerprint", "usage", "server_contract")})
                record["raw_text"] = raw[:MAX_RAW_CHARS]
                record["raw_text_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
                record["raw_text_truncated"] = len(raw) > MAX_RAW_CHARS
                reasoning = generated.get("reasoning_text") or ""
                record["reasoning_text"] = str(reasoning)[:MAX_RAW_CHARS]
                record["status"] = ("ok" if raw.strip() and generated.get("finish_reason") == "stop"
                    and not record["raw_text_truncated"] else "invalid_completion")
            except Exception as exc:
                # Error bodies and full exception messages may expose provider credentials.
                record.update(status="error", error_type=type(exc).__name__, raw_text="", finish_reason=None,
                    http_status=getattr(getattr(exc, "response", None), "status_code", None))
        record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        record["display_text"] = record["raw_text"]
        if prompt["speculative"] and record["raw_text"].strip():
            record["display_text"] = "Hypothetical simulation (no verified statement)\n\n" + record["raw_text"].lstrip()
        record["result_sha256"] = digest(record)
        results["attempts"][entry["attempt_id"]] = record
        atomic_json(output_path, results)
        completed += 1
    return results


def blind_packet(manifest: dict, results: dict, *, seed: int = 9127) -> tuple[dict, dict, dict]:
    check_manifest(manifest)
    _check_results(manifest, results)
    if len(results["attempts"]) != len(manifest["schedule"]):
        raise ValueError("Complete every planned attempt, including failed attempts, before blinding")
    cases = {case["id"]: case for case in manifest["cases"]}
    entries = list(results["attempts"].values())
    rng = random.Random(seed)
    rng.shuffle(entries)
    packet, key, template = [], {}, []
    for index, result in enumerate(entries):
        # Invalid/failed completions automatically score zero; partial text stays in raw results.
        if result["status"] not in {"ok", "policy_guard"}:
            continue
        rating_id = "answer-" + digest({"seed": seed, "index": index, "raw": result["result_sha256"]})[:12]
        case = cases[result["case_id"]]
        judge_ids = set(case["reference_evidence_ids"])
        for condition in CONDITIONS:
            judge_ids.update(manifest["prompts"][f"{case['id']}:{condition}"]["supplied_ids"])
        packet.append({"rating_id": rating_id, "question": case["question"], "answer": result["display_text"],
            "category": case["category"], "rubric": case["rubric"],
            "reference_sources": {identifier: manifest["judge_sources"][identifier] for identifier in sorted(judge_ids)},
            "answer_sha256": digest(result["display_text"])})
        key[rating_id] = {"attempt_id": result["attempt_id"], "result_sha256": result["result_sha256"],
                          "answer_sha256": digest(result["display_text"])}
        template.append({"rating_id": rating_id, "relevance": None, "fidelity": None, "application": None,
                         "unsupported_autobiography": None, "control_pass": None, "rationale": ""})
    shared = {"schema_version": SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
              "results_sha256": digest(results)}
    return ({**shared, "instructions": "Rate independently using the frozen protocol. Condition labels are hidden. Mentioning a value alone does not count as applying it. Do not consult the condition key.", "answers": packet},
            {**shared, "mapping": key},
            {**shared, "rater_id": "", "rater_kind": "", "ratings": template})


def _validate_ratings(ratings: dict, key: dict, cases_by_attempt: dict) -> dict:
    if any(ratings.get(field) != key[field] for field in ("manifest_sha256", "results_sha256")):
        raise ValueError("Ratings reference different answers")
    if ratings.get("rater_kind") not in {"human", "agent"} or not str(ratings.get("rater_id", "")).strip():
        raise ValueError("Each rating file needs an explicit human/agent rater identity")
    indexed = {}
    for rating in ratings.get("ratings", []):
        identifier = rating.get("rating_id")
        if identifier in indexed or identifier not in key["mapping"]:
            raise ValueError("Duplicate or unknown rating ID")
        category = cases_by_attempt[key["mapping"][identifier]["attempt_id"]]["category"]
        if not str(rating.get("rationale", "")).strip():
            raise ValueError("Each external rating must include a short rationale")
        if type(rating.get("unsupported_autobiography")) is not bool:
            raise ValueError("unsupported_autobiography must be an explicit boolean")
        if category == "advice":
            for field in ("relevance", "fidelity", "application"):
                if type(rating.get(field)) is not int or not 0 <= rating[field] <= 2:
                    raise ValueError(f"{field} must be an integer 0, 1, or 2")
            if rating.get("control_pass") is not None:
                raise ValueError("Advice probes do not have a control score")
        elif type(rating.get("control_pass")) is not bool:
            raise ValueError("Controls require an explicit control_pass boolean")
        indexed[identifier] = rating
    if set(indexed) != set(key["mapping"]):
        raise ValueError("Missing external ratings; do not silently reduce the denominator")
    return indexed


def paired_bootstrap(differences: list[float], *, seed: int = 377, samples: int = 5000) -> dict:
    if not differences:
        return {"n_cases": 0, "delta": None, "ci95": None}
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    return {"n_cases": len(differences), "delta": statistics.fmean(differences),
            "ci95": [means[int(.025 * (samples - 1))], means[int(.975 * (samples - 1))]],
            "method": "paired percentile bootstrap over cases; repeats/raters averaged within each case"}


def score_results(manifest: dict, results: dict, key: dict, rating_files: list[dict]) -> dict:
    check_manifest(manifest)
    _check_results(manifest, results)
    if set(results["attempts"]) != {entry["attempt_id"] for entry in manifest["schedule"]}:
        raise ValueError("Cannot score an incomplete run")
    if key.get("manifest_sha256") != manifest["manifest_sha256"] or key.get("results_sha256") != digest(results):
        raise ValueError("Blinding key does not match this run")
    valid = {identifier for identifier, item in results["attempts"].items() if item["status"] in {"ok", "policy_guard"}}
    mapped = [item["attempt_id"] for item in key["mapping"].values()]
    if len(mapped) != len(set(mapped)) or set(mapped) != valid:
        raise ValueError("Blinding key has missing, duplicate, or invalid attempts")
    for mapping in key["mapping"].values():
        result = results["attempts"][mapping["attempt_id"]]
        if mapping["result_sha256"] != result["result_sha256"] or mapping["answer_sha256"] != digest(result["display_text"]):
            raise ValueError("Blinding key answer hash mismatch")
    cases = {case["id"]: case for case in manifest["cases"]}
    cases_by_attempt = {identifier: cases[item["case_id"]] for identifier, item in results["attempts"].items()}
    if not rating_files or len({(item.get("rater_kind"), item.get("rater_id")) for item in rating_files}) != len(rating_files):
        raise ValueError("Provide at least one independently identified rater, without duplicates")
    indexed = [_validate_ratings(ratings, key, cases_by_attempt) for ratings in rating_files]
    rating_ids = {entry["attempt_id"]: identifier for identifier, entry in key["mapping"].items()}
    measured = []
    for attempt_id, result in results["attempts"].items():
        category = cases[result["case_id"]]["category"]
        row = {"attempt_id": attempt_id, "case_id": result["case_id"], "condition": result["condition"],
               "category": category, "status": result["status"]}
        scores = []
        unsupported = []
        dimensions = {field: [] for field in ("relevance", "fidelity", "application")}
        for rater in indexed:
            rating = rater[rating_ids[attempt_id]] if attempt_id in rating_ids else None
            success = False
            if rating is not None:
                success = (rating["relevance"] == 2 and rating["fidelity"] == 2 and rating["application"] >= 1
                           and not rating["unsupported_autobiography"]) if category == "advice" else (
                           rating["control_pass"] and not rating["unsupported_autobiography"])
            scores.append(float(success))
            unsupported.append(float(rating["unsupported_autobiography"]) if rating else 0.0)
            if category == "advice":
                for field in dimensions:
                    dimensions[field].append(float(rating[field]) if rating else 0.0)
        row.update(success=statistics.fmean(scores), unsupported_autobiography=statistics.fmean(unsupported))
        row.update({field: statistics.fmean(values) if values else None for field, values in dimensions.items()})
        measured.append(row)
    summaries = {}
    for condition in CONDITIONS:
        subset = [row for row in measured if row["condition"] == condition]
        advice = [row["success"] for row in subset if row["category"] == "advice"]
        controls = [row["success"] for row in subset if row["category"] == "control"]
        attempts = [item for item in results["attempts"].values() if item["condition"] == condition]
        budgets = [manifest["prompts"][f"{case['id']}:{condition}"]["budget"] for case in manifest["cases"]]
        input_units = [budget["input_units"] for budget in budgets if "input_units" in budget]
        actual_prompt_tokens = [(item.get("usage") or {}).get("prompt_tokens") for item in attempts]
        actual_prompt_tokens = [value for value in actual_prompt_tokens if type(value) is int]
        finish_reasons = {}
        for item in attempts:
            reason = str(item.get("finish_reason"))
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
        summaries[condition] = {"n_attempts": len(subset), "n_advice": len(advice), "n_controls": len(controls),
            "valid_model_completions": sum(row["status"] == "ok" for row in subset),
            "deterministic_policy_guards": sum(row["status"] == "policy_guard" for row in subset),
            "failed_or_truncated": sum(row["status"] not in {"ok", "policy_guard"} for row in subset),
            "advice_value_application_accuracy": statistics.fmean(advice) if advice else None,
            "advice_dimension_means_0_to_2": {field: statistics.fmean(row[field] for row in subset if row["category"] == "advice")
                                             if advice else None for field in ("relevance", "fidelity", "application")},
            "control_pass_rate": statistics.fmean(controls) if controls else None,
            "unsupported_autobiographical_assertion_rate_all_attempts": statistics.fmean(row["unsupported_autobiography"] for row in subset),
            "finish_reasons": finish_reasons,
            "provider_reported_models": sorted({str(item.get("model")) for item in attempts if item.get("model")}),
            "prompt_budget": {"counters": sorted({budget["counter"] for budget in budgets if "counter" in budget}),
                "all_exact_model_tokens": all(budget.get("exact_model_tokens") for budget in budgets if budget),
                "mean_input_units": statistics.fmean(input_units) if input_units else None,
                "max_input_units": max(input_units) if input_units else None,
                "mean_provider_prompt_tokens": statistics.fmean(actual_prompt_tokens) if actual_prompt_tokens else None}}
    deltas = {}
    for category in ("advice", "control"):
        for comparison in ("baseline", "flat"):
            differences = []
            for case in manifest["cases"]:
                if case["category"] != category:
                    continue
                by_condition = {condition: statistics.fmean(row["success"] for row in measured
                    if row["case_id"] == case["id"] and row["condition"] == condition) for condition in ("graph", comparison)}
                differences.append(by_condition["graph"] - by_condition[comparison])
            deltas[f"{category}:graph-minus-{comparison}"] = paired_bootstrap(differences)
    coverage = {}
    for category in ("advice", "control"):
        category_cases = [case for case in manifest["cases"] if case["category"] == category]
        reasons = {}
        for case in category_cases:
            selection = manifest["selections"][case["id"]]["selection"]
            reason = (selection.get("reason") or "selected") if selection else "production_public_evidence_guard"
            reasons[reason] = reasons.get(reason, 0) + 1
        coverage[category] = {"n_cases": len(category_cases), "memory_included": sum(
            manifest["prompts"][f"{case['id']}:graph"]["budget"].get("persona_memory", {}).get("included", False)
            for case in category_cases), "selection_reasons": reasons,
            "interpretation": "Routing/coverage diagnostics only, not answer quality."}
    return {"schema_version": SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
        "results_sha256": digest(results), "raters": [{"id": item["rater_id"], "kind": item["rater_kind"],
            "ratings_sha256": digest(item)} for item in rating_files], "conditions": summaries, "paired_deltas": deltas,
        "selection_coverage": coverage,
        "per_attempt": measured, "limitations": ["Small synthetic development set, not held-out event/personality validation.",
            "Agent ratings are AI review, not human ratings or the real person's ground truth.",
            "Graph-minus-baseline changes retrieval and interpreted values together; graph-minus-flat isolates representation, with unequal prompt lengths.",
            "No inference about live avatar latency, speech, Korean quality or production hybrid retrieval.",
            "Bootstrap intervals are descriptive and unstable for small case counts; do not tune on these answers."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--registry", type=Path, default=ROOT / "data/evaluation/jensen-persona-graph-v1.json")
    freeze.add_argument("--corpus", type=Path, default=ROOT / "data/personas/jensen-huang")
    freeze.add_argument("--graph", type=Path, required=True)
    freeze.add_argument("--snapshot", type=Path, default=ROOT / "data/snapshots/jensen-huang-2026-11-12.json")
    freeze.add_argument("--model", required=True)
    freeze.add_argument("--revision", required=True)
    freeze.add_argument("--case-ids", nargs="+")
    freeze.add_argument("--seed", type=int, default=20261120)
    freeze.add_argument("--repeats", type=int, default=1)
    freeze.add_argument("--max-tokens", type=int, default=256)
    freeze.add_argument("--context-window", type=int)
    freeze.add_argument("--token-count-url", help="Exact local template/tokenizer count endpoint; chooses its window unless overridden")
    freeze.add_argument("--retrieval-backend", choices=("lexical", "jensen-lexical"), default="lexical")
    freeze.add_argument("--provider-dialect", choices=PROVIDER_DIALECTS, default="recorded-gguf")
    freeze.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--base-url", default=os.getenv("PERSONA_EVAL_BASE_URL"))
    run.add_argument("--api-key-env", default="PERSONA_EVAL_API_KEY")
    run.add_argument("--timeout", type=float, default=300)
    run.add_argument("--limit", type=int)
    run.add_argument("--output", type=Path, required=True)
    blind = sub.add_parser("blind")
    blind.add_argument("--manifest", type=Path, required=True)
    blind.add_argument("--results", type=Path, required=True)
    blind.add_argument("--output-dir", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--key", type=Path, required=True)
    score.add_argument("--ratings", type=Path, nargs="+", required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.output.exists():
            raise SystemExit("Refusing to overwrite a frozen manifest; use a new run path.")
        counter = HTTPTokenCounter(args.token_count_url, model=args.model, revision=args.revision) if args.token_count_url else None
        context_window = args.context_window or (counter.context_window if counter else 16384)
        manifest = freeze_manifest(registry_path=args.registry, corpus_path=args.corpus, graph_path=args.graph,
            snapshot_path=args.snapshot, model=args.model, revision=args.revision, case_ids=args.case_ids,
            seed=args.seed, repeats=args.repeats, max_tokens=args.max_tokens, context_window=context_window,
            retrieval_backend=args.retrieval_backend, provider_dialect=args.provider_dialect, counter=counter)
        atomic_json(args.output, manifest)
        print(json.dumps({"manifest_sha256": manifest["manifest_sha256"], "planned_attempts": len(manifest["schedule"])}))
    elif args.command == "run":
        if not args.base_url:
            raise SystemExit("Set --base-url or PERSONA_EVAL_BASE_URL; no model fallback is permitted.")
        provider = OpenAICompatibleProvider(args.base_url, api_key=os.getenv(args.api_key_env), timeout=args.timeout)
        result = run_manifest(read_json(args.manifest), args.output, provider, limit=args.limit)
        print(json.dumps({"completed_attempts": len(result["attempts"])}))
    elif args.command == "blind":
        packet, key, template = blind_packet(read_json(args.manifest), read_json(args.results))
        for filename, content in (("answers.blinded.json", packet), ("condition-key.private.json", key), ("ratings.template.json", template)):
            destination = args.output_dir / filename
            if destination.exists():
                raise SystemExit("Refusing to overwrite a rating packet or key.")
            atomic_json(destination, content)
    else:
        result = score_results(read_json(args.manifest), read_json(args.results), read_json(args.key),
                               [read_json(path) for path in args.ratings])
        atomic_json(args.output, result)
        print(json.dumps(result["conditions"], indent=2))


if __name__ == "__main__":
    main()
