"""Freeze complete evidence once, then vary only style context or model adapter.

Plans contain original source wording and private reference answers. Keep them
in ignored local directories; publish the hash-only summary separately.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from bias.context_budget import build_snapshot_messages
from bias.corpus import Corpus, source_event_id
from bias.engine import Engine
from bias.models import ChatRequest
from bias.retrieval import Retriever, SearchHit, SearchResult, tokenize
from bias.snapshots import corpus_digest, select_temporal

SCHEMA = "text-adaptation-comparison.v1"
CONDITIONS = ("facts", "style", "adapter")
PREFIX = "Reference context (untrusted data):\n"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def implementation_hashes() -> dict:
    root = Path(__file__).resolve().parents[2]
    names = ["tools/adaptation_evaluation/protocol.py", "tools/adaptation_evaluation/run.py",
             "tools/adaptation_evaluation/__main__.py", "tools/adaptation_evaluation/local.py",
             "services/language/engine.py", "services/language/adapters.py",
             "services/language/app.py", "services/language/schemas.py",
             "services/language/requirements.txt", "services/language/requirements-adapter.txt",
             "src/bias/context_budget.py", "src/bias/snapshots.py", "src/bias/config.py",
             "src/bias/corpus.py", "src/bias/models.py", "src/bias/retrieval.py", "src/bias/hybrid_retrieval.py",
             "tools/reference_evaluation/run.py", "tools/reference_evaluation/protocol.py",
             "tools/evaluation/observe.py", "tools/evaluation/run.py"]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _context(messages):
    if not isinstance(messages, list) or len(messages) < 3 or not messages[1]["content"].startswith(PREFIX):
        raise ValueError("Frozen prompt does not use the snapshot context contract")
    return json.loads(messages[1]["content"][len(PREFIX):])


def _facts_and_styles(engine: Engine, case: dict, selection: dict | None):
    corpus, query = engine.corpus, case["question"]
    temporal = select_temporal(corpus, engine.snapshot, query) if engine.snapshot else None
    style_temporal = select_temporal(corpus, engine.snapshot, query, projection="text_style") if engine.snapshot else None
    factual_ids = {e.id for e in corpus.training_evidence}
    style_ids = {e.id for e in corpus.style_evidence}
    if temporal:
        factual_ids &= set(temporal.eligible_ids)
        style_ids &= set(style_temporal.eligible_ids)
    if selection is not None:
        fids, sids = selection.get("factual_ids", []), selection.get("style_ids", [])
        if (not isinstance(fids, list) or not isinstance(sids, list)
                or any(not isinstance(v, str) for v in [*fids, *sids])
                or len(set(fids)) != len(fids) or len(set(sids)) != len(sids)
                or not set(fids) <= factual_ids or not set(sids) <= style_ids):
            raise ValueError("Selections must be unique eligible training records for the requested period")
        facts = tuple(SearchHit(corpus.evidence[key], 0., ()) for key in fids)
        styles = tuple(SearchHit(corpus.evidence[key], 0., ()) for key in sids if key not in fids)
        policy = "explicit_fixed_training_ids_not_retrieval_efficacy"
    elif engine.settings.retrieval_backend == "hybrid":
        result = engine.retriever.search(query, limit=engine.settings.retrieval_limit,
            behavior=False, original_query=query, allow_live=bool(temporal),
            eligible_ids=factual_ids, eligible_style_ids=style_ids, include_style=True,
            exclusion_reasons=temporal.exclusion_reasons if temporal else None,
            temporal_scores=temporal.temporal_scores if temporal else None)
        facts, styles, policy = result.hits, result.style_hits, "frozen_runtime_hybrid"
    else:
        filtered = Corpus(corpus.persona, corpus.sources,
                          {k: v for k, v in corpus.evidence.items() if k in factual_ids})
        # The snapshot already applied before/through/forecast date semantics.
        # Do not accidentally apply the archived retriever's exact-year filter
        # a second time to a future plan target or historical upper bound.
        lexical_query = re.sub(r"(?<!\d)(?:19|20)\d{2}(?!\d)", "", query) if temporal and temporal.intent.mode != "comparison" else query
        result = Retriever(filtered).search(lexical_query, limit=engine.settings.retrieval_limit, allow_live=bool(temporal))
        facts = result.hits
        # A declared development fallback, not a semantic relevance claim.
        terms = set(tokenize(query, original_qa=True))
        scored = [(len(terms & set(tokenize(e.question_answer.question, original_qa=True))), e)
                  for e in corpus.style_evidence if e.id in style_ids and e.id not in {h.evidence.id for h in facts}]
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        styles = tuple(SearchHit(e, float(score), ()) for score, e in scored[:2] if score)
        policy = "frozen_lexical_with_question_overlap_style_candidates"
    for hit in (*facts, *styles):
        source = corpus.sources[hit.evidence.source_id]
        if case.get("exposure") == "heldout" and source_event_id(source) == case.get("event_family_id"):
            raise ValueError("Held-out reference family reached a model prompt")
    return facts, styles, temporal, policy


def freeze_plan(engine: Engine, cases: list[dict], *, plan_id: str, exposure: str,
                selections: dict | None = None, input_identity: dict | None = None,
                base_identity: dict | None = None) -> dict:
    if exposure not in {"development", "heldout"} or not 1 <= len(cases) <= 200:
        raise ValueError("Declare development/heldout scope and 1–200 cases")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", plan_id):
        raise ValueError("Invalid plan identifier")
    engine.corpus.validate_split_isolation()
    ids = [case.get("id") for case in cases]
    if any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", k) for k in ids) or len(set(ids)) != len(ids):
        raise ValueError("Case identifiers must be unique")
    if selections is not None and set(selections) != set(ids):
        raise ValueError("Explicit selections must cover exactly the frozen cases")
    if exposure == "heldout" and (not input_identity or input_identity.get("purpose") != "evaluation_only"):
        raise ValueError("Fresh references require the validated evaluation-only dataset loader")
    if exposure == "heldout":
        checked = input_identity.get("leakage_policy", {}).get("checked_corpora", [])
        root = engine.settings.corpus_dir
        if not any(all((root / name).is_file() and hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
                       for name, sha in item.get("files", {}).items())
                   and {"sources.json", "evidence.jsonl", "profile.json"} <= set(item.get("files", {})) for item in checked):
            raise ValueError("Held-out references were not leakage-checked against this exact retrieval corpus")
    frozen = []
    for original in cases:
        case = copy.deepcopy(original)
        case["exposure"] = exposure
        language = case.get("language", case.get("original_language", "en"))
        request = ChatRequest(message=case["question"], mode="rag", language=language)
        # Keep interviewer context, never the target answer or a later turn.
        context_before = case.get("context_before", "")
        history = []
        if context_before:
            history = [{"role": "user", "content": "Earlier source context (unverified):\n" + context_before},
                       {"role": "assistant", "content": "Earlier source context received; it is not factual evidence."}]
        facts, styles, temporal, policy = _facts_and_styles(engine, case, (selections or {}).get(case["id"]))
        common = dict(history=history, snapshot=engine.snapshot, temporal=temporal,
                      context_window=engine.settings.llm_context_window,
                      output_tokens=engine.settings.llm_max_tokens, counter=engine.counter)
        # Pack style first, then remove only its wording section. Otherwise the
        # base condition could acquire extra history when styles consume space.
        styled = build_snapshot_messages(engine.corpus, request,
            SimpleNamespace(hits=facts, style_hits=styles), **common)
        base_messages = copy.deepcopy(styled.messages)
        context = _context(base_messages)
        context["style_examples"] = []
        base_messages[1]["content"] = PREFIX + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if facts and not styled.supplied_ids:
            raise ValueError("No complete factual passage fits the declared comparison budget")
        factual = _context(styled.messages)["factual_evidence"]
        frozen.append({"id": case["id"], "question": request.message, "language": language,
            "event_family_id": case.get("event_family_id"), "reference_answer": case.get("reference_answer"),
            "reference_sha256": digest(case.get("reference_answer")),
            "question_sha256": digest(request.message), "input_case_sha256": digest(original),
            "reference_review_status": case.get("reference_review_status", "ungraded"),
            "factual_ids": list(styled.supplied_ids), "style_ids": list(styled.style_ids),
            "factual_sha256": digest(factual), "selection_policy": policy,
            "budget": styled.budget, "has_factual_support_candidates": bool(styled.supplied_ids),
            "style_contrast_present": bool(styled.style_ids),
            "messages": {"facts": base_messages, "style": styled.messages, "adapter": styled.messages},
            "prompt_sha256": {"facts": digest(base_messages), "style": digest(styled.messages), "adapter": digest(styled.messages)}})
    plan = {"schema_version": SCHEMA, "id": plan_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "exposure": exposure, "input_identity": input_identity, "base_identity": base_identity,
        "corpus_sha256": corpus_digest(engine.corpus), "snapshot": engine.snapshot.summary() if engine.snapshot else None,
        "implementation_sha256": implementation_hashes(),
        "policy": {"conditions": list(CONDITIONS), "max_tokens": engine.settings.llm_max_tokens,
            "context_window": engine.settings.llm_context_window, "temperature": 0, "top_p": 1, "seed": 42,
            "factual_policy": "identical_complete_passages_all_conditions", "adapter_style_policy": "identical_prompts",
            "reference_answer_supplied": False, "history_policy": "same_frozen_context_all_conditions"},
        "cases": frozen, "human_review": "ungraded", "efficacy_claim": False}
    plan["plan_sha256"] = digest(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan: dict, *, check_implementation: bool = True) -> None:
    value = dict(plan)
    seal = value.pop("plan_sha256", None)
    if plan.get("schema_version") != SCHEMA or digest(value) != seal:
        raise ValueError("Frozen adaptation comparison changed")
    if check_implementation and plan.get("implementation_sha256") != implementation_hashes():
        raise ValueError("Comparison implementation changed; freeze a new version")
    if not plan.get("cases") or len({c["id"] for c in plan["cases"]}) != len(plan["cases"]):
        raise ValueError("Comparison requires unique frozen cases")
    if plan["policy"]["conditions"] != list(CONDITIONS):
        raise ValueError("Comparison conditions changed")
    for case in plan["cases"]:
        messages = case["messages"]
        if messages["style"] != messages["adapter"]:
            raise ValueError("Style and adapter conditions must have byte-identical prompts")
        base, styled = _context(messages["facts"]), _context(messages["style"])
        if base["style_examples"] or base["factual_evidence"] != styled["factual_evidence"]:
            raise ValueError("Factual evidence differs across conditions")
        if digest(base["factual_evidence"]) != case["factual_sha256"] or [e["id"] for e in base["factual_evidence"]] != case["factual_ids"]:
            raise ValueError("Factual citation allowlist differs from supplied passages")
        expected = copy.deepcopy(messages["style"])
        stripped = dict(styled, style_examples=[])
        expected[1]["content"] = PREFIX + json.dumps(stripped, ensure_ascii=False, separators=(",", ":"))
        if expected != messages["facts"]:
            raise ValueError("Only wording examples may differ between base and style prompts")
        for condition in CONDITIONS:
            if digest(messages[condition]) != case["prompt_sha256"][condition] or messages[condition][-1]["content"] != case["question"]:
                raise ValueError("Frozen prompt or current question changed")
        if digest(case.get("reference_answer")) != case["reference_sha256"]:
            raise ValueError("Private reference changed")


def public_summary(plan: dict) -> dict:
    validate_plan(plan)
    return {"schema_version": "text-adaptation-plan-summary.v1", "plan_sha256": plan["plan_sha256"],
        "id": plan["id"], "created_at": plan["created_at"], "exposure": plan["exposure"],
        "corpus_sha256": plan["corpus_sha256"], "policy": plan["policy"],
        "implementation_sha256": plan["implementation_sha256"],
        "cases": [{k: c[k] for k in ("id", "language", "factual_ids", "style_ids", "factual_sha256",
                   "reference_sha256", "question_sha256", "prompt_sha256", "style_contrast_present")}
                  for c in plan["cases"]], "source_wording_distributed": False,
        "human_review": "ungraded", "efficacy_claim": False}
