"""Frozen matched-evidence ablation of compact values and public delivery cues.

Authored transfer probes, not a held-out interview benchmark or acoustic test.
Uses production selectors, prompt packing, guards and the existing real runner.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import re
import statistics

from bias.context_budget import build_snapshot_messages
from bias.corpus import Corpus
from bias.models import ChatRequest
from bias.persona_delivery import PersonaDelivery
from bias.persona_graph import PersonaGraph, augment_result, select_advice_memory
from bias.query_policy import public_evidence_limitation
from bias.retrieval import Retriever, SearchResult
from bias.snapshots import PersonaSnapshot, corpus_digest, select_temporal
from tools import persona_graph_eval as shared

ROOT = shared.ROOT
REPORT = ROOT / "reports/persona-delivery-2026-11-21"
CORPUS = ROOT / "data/personas/jensen-huang-values-v2"
CONDITIONS = ("matched_evidence", "compact_values", "compact_delivery")
FIELDS = ("relevance", "fidelity", "application", "reaction", "style")


def prepare(corpus, snapshot, graph, delivery, question, *, counter):
    # No evaluator rubric or reference label is an input to these selectors.
    request = ChatRequest(message=question, language="en", mode="behavior")
    temporal = select_temporal(corpus, snapshot, question)
    limitation = public_evidence_limitation(question)
    selection = None if limitation else select_advice_memory(
        graph, corpus, question, snapshot=snapshot, temporal=temporal, limit=1)
    communication = None if limitation else delivery.select(question, snapshot=snapshot, temporal=temporal)
    eligible = {k: v for k, v in corpus.evidence.items() if k in temporal.eligible_ids}
    result = SearchResult((), 0, limitation or temporal.reason)
    if not limitation and eligible:
        scoped = Corpus(corpus.persona, corpus.sources, eligible)
        query = re.sub(r"(?<!\d)(?:19|20)\d{2}(?!\d)", "", question) if temporal.intent.mode != "comparison" else question
        result = Retriever(scoped).search(query, limit=4, behavior=True, allow_live=True)
    if selection:
        result = augment_result(selection, corpus, result, limit=4)
    prompts = {}
    for condition in CONDITIONS:
        if limitation:
            from bias.engine import evidence_preview
            prompts[condition] = {"messages": [], "supplied_ids": [], "budget": {},
                "speculative": False, "policy_guard": limitation,
                "deterministic_text": evidence_preview(request, result)}
            continue
        package = build_snapshot_messages(corpus, request, result, [], snapshot, temporal,
            context_window=8192, output_tokens=256, counter=counter,
            speculative=not result.hits, live_concise=True,
            persona_selection=selection if condition != "matched_evidence" else None,
            persona_memory_format="compact",
            delivery_selection=communication if condition == "compact_delivery" else None)
        prompts[condition] = {"messages": package.messages, "supplied_ids": list(package.supplied_ids),
            "budget": package.budget, "speculative": not result.hits, "policy_guard": None}
    if len({tuple(p["supplied_ids"]) for p in prompts.values()}) != 1:
        raise ValueError("Factual evidence must be identical in all conditions")
    if selection and selection.paths:
        for condition in CONDITIONS[1:]:
            if not prompts[condition]["budget"].get("persona_memory", {}).get("included"):
                raise ValueError("Selected memory was not included; increase the common budget before freeze")
    if communication and communication.observations:
        expected = {o.id for o in communication.observations}
        actual = set(prompts["compact_delivery"]["budget"]["persona_delivery"]["included_ids"])
        if actual != expected:
            raise ValueError("Selected delivery did not fit before freeze")
    return prompts, {"values": selection.trace() if selection else None,
                     "delivery": communication.trace() if communication else None}


def freeze(counter):
    if (REPORT / "manifest.json").exists() or (REPORT / "results.json").exists():
        raise ValueError("Frozen artifacts already exist; resume run or use a new --report-dir with a registry")
    corpus = Corpus.load(CORPUS)
    snapshot_path = ROOT / "data/snapshots/jensen-values-2026-11-21.json"
    snapshot = PersonaSnapshot.load(snapshot_path, corpus)
    graph = PersonaGraph.load(CORPUS / "persona_graph.v2.json", corpus)
    delivery = PersonaDelivery.load(CORPUS / "persona_delivery.v1.json", corpus)
    registry = shared.read_json(REPORT / "registry.json")
    shared.validate_registry({**registry, "schema_version": "persona-graph-probes.v1"}, corpus)
    if counter.context_window != 8192:
        raise ValueError("Expected pinned 8192-token runtime")
    configuration = {"model": "Qwen3-4B-Q4_K_M", "revision": "bc640142c66e1fdd12af0bd68f40445458f3869b",
        "provider_dialect": "recorded-gguf", "decode": {"temperature": 0, "top_p": 1,
            "max_tokens": 256, "seed": 20261122, "presence_penalty": 0,
            "chat_template_kwargs": {"enable_thinking": False}},
        "context_window": 8192, "history": "empty", "repeats": 1, "retrieval_limit": 4,
        "retrieval_backend": "lexical", "live_concise": True, "persona_path_limit": 1}
    prompts, selections, sources = {}, {}, {}
    for case in registry["cases"]:
        packed, selections[case["id"]] = prepare(corpus, snapshot, graph, delivery, case["question"], counter=counter)
        for condition, prompt in packed.items():
            prompts[f"{case['id']}:{condition}"] = {**prompt, "prompt_sha256": shared.digest(prompt["messages"])}
        ids = set(case["reference_evidence_ids"]) | set(packed[CONDITIONS[0]]["supplied_ids"])
        for key in ids:
            evidence = corpus.evidence[key]; source = corpus.sources[evidence.source_id]
            sources[key] = {"text": evidence.text, "source_id": source.id, "url": str(source.url)}
    rng = random.Random(20261122)
    cases = list(registry["cases"]); rng.shuffle(cases)
    order = list(CONDITIONS); rng.shuffle(order)
    schedule = []
    for index, case in enumerate(cases):
        offset = index % len(order)
        for condition in order[offset:] + order[:offset]:
            key = f"{case['id']}:{condition}"
            schedule.append({"attempt_id": f"{case['id']}:r0:{condition}", "case_id": case["id"],
                "repeat": 0, "condition": condition, "prompt_key": key,
                "request_sha256": shared.digest({"prompt": prompts[key], "configuration": configuration})})
    code = ["src/bias/persona_delivery.py", "src/bias/persona_rules.py", "src/bias/context_budget.py",
            "tools/persona_delivery_eval.py"]
    manifest = {"schema_version": shared.SCHEMA, "design_version": "matched-delivery.v1",
        "scope": "Fresh authored development transfer probes; AI ratings only; no acoustic or unseen-event validation.",
        "registry_id": registry["id"], "cases": registry["cases"], "conditions": list(CONDITIONS),
        "configuration": configuration, "prompts": prompts, "selections": selections,
        "schedule": schedule, "judge_sources": sources, "rubric": registry["rubric"],
        "contrast_interpretation": {"compact_values-minus-matched_evidence": "Conditional interpretation and compact prompt guidance with matched facts; not graph retrieval benefit.",
            "compact_delivery-minus-compact_values": "Additional source-bound reaction/style guidance; not acoustic identity."},
        "provenance": {"corpus_sha256": corpus_digest(corpus), "graph_sha256": graph.sha256,
            "delivery_sha256": delivery.sha256, "snapshot_sha256": shared.file_digest(snapshot_path),
            "registry_sha256": shared.file_digest(REPORT / "registry.json"),
            "code_sha256": {p: shared.file_digest(ROOT / p) for p in code}}}
    manifest["manifest_sha256"] = shared.digest(manifest)
    shared.atomic_json(REPORT / "manifest.json", manifest)
    return manifest


def _blind_materials(manifest, results):
    shared.check_manifest(manifest); shared._check_results(manifest, results)
    if manifest.get("design_version") != "matched-delivery.v1" or manifest.get("conditions") != list(CONDITIONS):
        raise ValueError("Expected the matched-evidence delivery design")
    if set(results["attempts"]) != {s["attempt_id"] for s in manifest["schedule"]}:
        raise ValueError("Complete all planned attempts before blinding")
    for entry in manifest["schedule"]:
        row = results["attempts"][entry["attempt_id"]]
        if any(row.get(key) != entry[key] for key in
               ("attempt_id", "case_id", "condition", "repeat", "prompt_key", "request_sha256")):
            raise ValueError("Stored attempt labels differ from frozen schedule")
        if row.get("prompt_sha256") != manifest["prompts"][entry["prompt_key"]]["prompt_sha256"]:
            raise ValueError("Stored prompt identity differs from frozen prompt")
    cases = {c["id"]: c for c in manifest["cases"]}
    items = [results["attempts"][entry["attempt_id"]] for entry in manifest["schedule"]]
    random.Random(931572).shuffle(items)
    packet, mapping = [], {}
    for index, result in enumerate(items):
        identifier = "answer-" + shared.digest([index, result["result_sha256"]])[:12]
        mapping[identifier] = result["attempt_id"]
        if result["status"] not in {"ok", "policy_guard"}:
            continue
        case = cases[result["case_id"]]
        ids = set(case["reference_evidence_ids"]) | set(manifest["prompts"][f"{case['id']}:{CONDITIONS[0]}"]["supplied_ids"])
        packet.append({"rating_id": identifier, "question": case["question"], "answer": result["display_text"],
            "category": case["category"], "case_rubric": case["rubric"],
            "reference_sources": {k: manifest["judge_sources"][k] for k in sorted(ids)}})
    return {"rubric": manifest["rubric"], "answers": packet}, mapping


def blind(manifest, results):
    packet, mapping = _blind_materials(manifest, results)
    shared.atomic_json(REPORT / "blind-packet.json", packet)
    shared.atomic_json(REPORT / "condition-key.private.json", mapping)


def summarize(manifest, results, rating_paths):
    expected_packet, expected_mapping = _blind_materials(manifest, results)
    mapping = shared.read_json(REPORT / "condition-key.private.json")
    packet = shared.read_json(REPORT / "blind-packet.json")
    if mapping != expected_mapping or packet != expected_packet:
        raise ValueError("Blinded packet or condition mapping differs from frozen actual results")
    if len(rating_paths) != 2 or len({p.resolve() for p in rating_paths}) != 2:
        raise ValueError("Exactly two distinct rating files are required")
    if len({shared.file_digest(p) for p in rating_paths}) != 2:
        raise ValueError("A duplicate rating file is not a second independent rater")
    valid_ids = {item["rating_id"] for item in packet["answers"]}
    judges, rater_ids = [], set()
    for path in rating_paths:
        data = shared.read_json(path)
        rater_id = data.get("rater_id")
        if not isinstance(rater_id, str) or not rater_id.strip() or rater_id.strip() in rater_ids:
            raise ValueError("Each rater needs a distinct nonblank rater_id")
        rater_ids.add(rater_id.strip())
        rows = {row["rating_id"]: row for row in data["ratings"]}
        if set(rows) != valid_ids or len(rows) != len(data["ratings"]):
            raise ValueError("Every blinded answer needs exactly one rating per judge")
        for row in rows.values():
            if any(type(row[f]) is not int or not 0 <= row[f] <= 2 for f in FIELDS):
                raise ValueError("Rating dimensions must be integer 0, 1 or 2")
            if type(row["safe"]) is not bool or not row.get("reason"):
                raise ValueError("Missing safety verdict or reason")
        judges.append({mapping[key]: value for key, value in rows.items()})
    cases = {c["id"]: c for c in manifest["cases"]}
    out = {"manifest_sha256": manifest["manifest_sha256"], "results_sha256": shared.digest(results),
        "judge_files": {str(p.name): shared.file_digest(p) for p in rating_paths},
        "rating_type": "independent blinded AI judgments, not human validation", "conditions": {}}
    for condition in CONDITIONS:
        rows = [r for r in results["attempts"].values() if r["condition"] == condition]
        advice = [r for r in rows if cases[r["case_id"]]["category"] == "advice"]
        controls = [r for r in rows if cases[r["case_id"]]["probe_type"] == "factual"]
        style_probes = [r for r in rows if cases[r["case_id"]]["probe_type"] == "style"]
        def score(r, judge, field):
            return judge.get(r["attempt_id"], {}).get(field, 0)
        def passes(r, judge):
            return all(score(r, judge, f) == 2 for f in ("relevance", "fidelity", "application"))
        generated = [r for r in rows if r["status"] == "ok"]
        out["conditions"][condition] = {
            "attempts": len(rows), "valid_model_completions": len(generated),
            "deterministic_guards": sum(r["status"] == "policy_guard" for r in rows),
            "failed_or_truncated": sum(r["status"] not in {"ok", "policy_guard"} for r in rows),
            "n_advice": len(advice), "conditional_advice_pass_rate": statistics.mean(passes(r,j) for r in advice for j in judges),
            "both_judges_advice_passes": sum(all(passes(r,j) for j in judges) for r in advice),
            "advice_dimension_means_0_to_2": {f: statistics.mean(score(r,j,f) for r in advice for j in judges) for f in FIELDS},
            "control_safe_rate": statistics.mean(bool(score(r,j,"safe")) for r in controls for j in judges),
            "style_probe_mean_0_to_2": statistics.mean(score(r,j,"style") for r in style_probes for j in judges),
            "style_probe_safe_rate": statistics.mean(bool(score(r,j,"safe")) for r in style_probes for j in judges),
            "median_full_completion_seconds_cpu": statistics.median(r["elapsed_seconds"] for r in generated),
            "median_output_words": statistics.median(len(re.findall(r"\b[\w'-]+\b", r["raw_text"])) for r in generated)}
    shared.atomic_json(REPORT / "summary.json", out)
    return out


def main():
    global REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "blind", "summarize"))
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    args = parser.parse_args()
    REPORT = args.report_dir.resolve()
    base = os.getenv("PERSONA_EVAL_BASE_URL", "http://127.0.0.1:8129/v1")
    if args.action == "freeze":
        counter = shared.HTTPTokenCounter(base.removesuffix("/v1") + "/count_tokens",
            model="Qwen3-4B-Q4_K_M", revision="bc640142c66e1fdd12af0bd68f40445458f3869b")
        result = freeze(counter)
        print("Frozen", len(result["schedule"]), "attempts", result["manifest_sha256"])
        return
    manifest = shared.read_json(REPORT / "manifest.json")
    path = REPORT / "results.json"
    if args.action == "run":
        result = shared.run_manifest(manifest, path, shared.OpenAICompatibleProvider(base), limit=args.limit)
        print("Recorded", len(result["attempts"]), "of", len(manifest["schedule"]), "attempts", flush=True)
    elif args.action == "blind":
        blind(manifest, shared.read_json(path))
    else:
        print(summarize(manifest, shared.read_json(path), [REPORT / "ratings-a.json", REPORT / "ratings-b.json"]))


if __name__ == "__main__":
    main()
