"""Four-condition, source-grounded conditional-persona evaluation.

This is a prospective authored transfer probe, not held-out original interview
answer accuracy. No keyword match is treated as generated-answer quality.
The existing runner handles atomic attempts, errors, hashes and provider calls;
this module adds a frozen four-condition design and explicit context ratings.
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
import random
import statistics

from bias.corpus import Corpus
from bias.persona_graph import PersonaGraph
from bias.snapshots import PersonaSnapshot, corpus_digest
from tools import persona_graph_eval as previous


ROOT = previous.ROOT
CONDITIONS = ("baseline", "expanded_rag", "flat", "graph")
DESIGN = "conditional-values.v2"
FIELDS = ("relevance", "fidelity", "application", "context_sensitivity")


def validate_comparison_corpora(baseline: Corpus, corpus: Corpus) -> None:
    # Coverage dates describe the expanded evidence range. They are not the
    # persona name/style guidance supplied to the generator; allow ONLY these
    # metadata differences while retaining strict equality for all other fields.
    metadata_dates = {"corpus_start", "corpus_cutoff"}
    if corpus.persona.model_dump(exclude=metadata_dates) != baseline.persona.model_dump(exclude=metadata_dates):
        raise ValueError("Keep the same persona card across all conditions")
    if not set(baseline.evidence) <= set(corpus.evidence):
        raise ValueError("Expanded corpus must retain the original evidence")
    for key, evidence in baseline.evidence.items():
        if corpus.evidence[key] != evidence or corpus.sources[evidence.source_id] != baseline.sources[evidence.source_id]:
            raise ValueError("Expanded corpus changes original evidence or source metadata")


def validate_registry(registry: dict, corpus: Corpus) -> None:
    if registry.get("schema_version") != "persona-rules-probes.v2":
        raise ValueError("Unsupported conditional-persona registry")
    adapted = {**registry, "schema_version": "persona-graph-probes.v1"}
    previous.validate_registry(adapted, corpus)
    pairs: dict[str, list] = {}
    for case in registry["cases"]:
        if not isinstance(case.get("rubric"), dict):
            raise ValueError("Every case needs a judge-only rubric")
        if case["category"] == "advice":
            if not case.get("pair_id") or not case["rubric"].get("context_sensitivity"):
                raise ValueError("Advice needs a paired context contrast and context rubric")
            pairs.setdefault(case["pair_id"], []).append(case)
    if any(len(cases) != 2 for cases in pairs.values()):
        raise ValueError("Each context contrast must have exactly two variants")


def prepare_conditions(baseline_corpus, baseline_snapshot, baseline_graph,
                       corpus, snapshot, graph, question, **kwargs) -> dict:
    """Neither retrieval nor generation receives evaluator references/rubrics."""
    base = previous.prepare_case(baseline_corpus, baseline_snapshot, baseline_graph, question, **kwargs)
    expanded = previous.prepare_case(corpus, snapshot, graph, question, **kwargs)
    return {"conditions": {
        "baseline": base["conditions"]["baseline"],
        "expanded_rag": expanded["conditions"]["baseline"],
        "flat": expanded["conditions"]["flat"],
        "graph": expanded["conditions"]["graph"],
    }, "selection": expanded["selection"], "temporal": expanded["temporal"],
       "lexical_query": expanded["lexical_query"]}


def freeze_manifest(*, registry_path: Path, corpus_path: Path, graph_path: Path,
                    snapshot_path: Path, baseline_corpus_path: Path,
                    baseline_snapshot_path: Path, baseline_graph_path: Path,
                    model: str, revision: str, seed: int = 20261121,
                    repeats: int = 1, max_tokens: int = 256, context_window: int = 16384,
                    retrieval_backend: str = "lexical", provider_dialect: str = "recorded-gguf",
                    case_ids: list[str] | None = None, counter=None) -> dict:
    if repeats < 1 or max_tokens < 32 or not model or not revision:
        raise ValueError("Provide pinned model/revision, repeats >= 1 and max_tokens >= 32")
    if provider_dialect not in previous.PROVIDER_DIALECTS:
        raise ValueError("Unsupported provider dialect")
    corpus = Corpus.load(corpus_path)
    baseline = Corpus.load(baseline_corpus_path)
    validate_comparison_corpora(baseline, corpus)
    registry = previous.read_json(registry_path)
    validate_registry(registry, corpus)
    cases = registry["cases"]
    if case_ids is not None:
        if len(case_ids) != len(set(case_ids)) or set(case_ids) - {case["id"] for case in cases}:
            raise ValueError("Case subset has duplicate or unknown IDs")
        cases = [case for case in cases if case["id"] in case_ids]
        validate_registry({**registry, "cases": cases}, corpus)
    graph = PersonaGraph.load(graph_path, corpus)
    old_graph = PersonaGraph.load(baseline_graph_path, baseline)
    snapshot = PersonaSnapshot.load(snapshot_path, corpus)
    old_snapshot = PersonaSnapshot.load(baseline_snapshot_path, baseline)
    if getattr(counter, "context_window", context_window) < context_window:
        raise ValueError("Declared context exceeds exact token server window")
    decode = {"temperature": 0, "top_p": 1, "max_tokens": max_tokens, "seed": seed}
    if provider_dialect == "recorded-gguf":
        decode.update(presence_penalty=0, chat_template_kwargs={"enable_thinking": False})
    configuration = {"model": model, "revision": revision, "decode": decode,
        "provider_dialect": provider_dialect, "context_window": context_window,
        "retrieval_backend": retrieval_backend, "retrieval_limit": 4,
        "adaptive_responses": False, "live_concise": False, "history": "empty",
        "repeats": repeats, "seed": seed}
    prompts, selections = {}, {}
    for case in cases:
        prepared = prepare_conditions(baseline, old_snapshot, old_graph, corpus, snapshot, graph,
            case["question"], language=case["language"], context_window=context_window,
            max_tokens=max_tokens, retrieval_backend=retrieval_backend, counter=counter)
        selections[case["id"]] = {key: value for key, value in prepared.items() if key != "conditions"}
        for condition, prompt in prepared["conditions"].items():
            prompts[f"{case['id']}:{condition}"] = {**prompt, "prompt_sha256": previous.digest(prompt["messages"])}
    rng = random.Random(seed)
    # Latin-square rotations balance position every four case/repeat blocks.
    order = list(CONDITIONS)
    rng.shuffle(order)
    orders = [order[index:] + order[:index] for index in range(len(order))]
    units = list(itertools.product([case["id"] for case in cases], range(repeats)))
    rng.shuffle(units)
    schedule = []
    for index, (case_id, repeat) in enumerate(units):
        for condition in orders[index % len(orders)]:
            key = f"{case_id}:{condition}"
            schedule.append({"attempt_id": f"{case_id}:r{repeat}:{condition}", "case_id": case_id,
                "repeat": repeat, "condition": condition, "prompt_key": key,
                "request_sha256": previous.digest({"prompt": prompts[key], "configuration": configuration})})
    sources = {}
    for case in cases:
        ids = set(case["reference_evidence_ids"])
        for condition in CONDITIONS:
            ids.update(prompts[f"{case['id']}:{condition}"]["supplied_ids"])
        for identifier in sorted(ids):
            evidence = corpus.evidence[identifier]
            source = corpus.sources[evidence.source_id]
            sources[identifier] = {"text": evidence.text, "source_id": source.id,
                "url": str(source.url), "published_at": source.published_at.isoformat(),
                "sha256": previous.digest(evidence.model_dump(mode="json"))}
    manifest = {"schema_version": previous.SCHEMA, "design_version": DESIGN,
        "registry_id": registry["id"], "scope": registry["scope"],
        "configuration": configuration, "conditions": list(CONDITIONS), "cases": cases,
        "prompts": prompts, "selections": selections, "schedule": schedule, "judge_sources": sources,
        "contrast_interpretation": {
            "expanded_rag-minus-baseline": "Expansion of the retrieval corpus only.",
            "flat-minus-expanded_rag": "Conditional interpretations PLUS graph-based evidence selection; not a pure representation effect.",
            "graph-minus-flat": "Matched evidence, claims, conditions and exceptions; relation serialization differs and prompt lengths can differ.",
        },
        "provenance": {"baseline_corpus_sha256": corpus_digest(baseline),
            "expanded_corpus_sha256": corpus_digest(corpus), "registry_sha256": previous.file_digest(registry_path),
            "persona_card_allowed_metadata_differences": ["corpus_start", "corpus_cutoff"],
            "baseline_graph_sha256": previous.file_digest(baseline_graph_path),
            "graph_sha256": previous.file_digest(graph_path), "snapshot_sha256": previous.file_digest(snapshot_path),
            "baseline_snapshot_sha256": previous.file_digest(baseline_snapshot_path),
            "runner_sha256": previous.file_digest(Path(__file__)),
            "shared_runner_sha256": previous.file_digest(Path(previous.__file__)),
            "production_prompt_builder_sha256": previous.file_digest(ROOT / "src/bias/context_budget.py")}}
    manifest["manifest_sha256"] = previous.digest(manifest)
    return manifest


def check_manifest(manifest: dict) -> None:
    previous.check_manifest(manifest)
    if manifest.get("design_version") != DESIGN or manifest.get("conditions") != list(CONDITIONS):
        raise ValueError("Expected the frozen four-condition design")
    for case in manifest["cases"]:
        for condition in CONDITIONS:
            key = f"{case['id']}:{condition}"
            if key not in manifest["prompts"]:
                raise ValueError("Missing condition prompt")


def run_manifest(manifest: dict, output_path: Path, provider, *, limit: int | None = None) -> dict:
    check_manifest(manifest)
    return previous.run_manifest(manifest, output_path, provider, limit=limit)


def blind_packet(manifest: dict, results: dict, *, seed: int = 29021) -> tuple[dict, dict, dict]:
    check_manifest(manifest)
    previous._check_results(manifest, results)
    if set(results["attempts"]) != {entry["attempt_id"] for entry in manifest["schedule"]}:
        raise ValueError("Complete every planned attempt before blinding")
    cases = {case["id"]: case for case in manifest["cases"]}
    entries = list(results["attempts"].values())
    random.Random(seed).shuffle(entries)
    answers, mapping, ratings = [], {}, []
    for index, result in enumerate(entries):
        if result["status"] not in {"ok", "policy_guard"}:
            continue
        identifier = "answer-" + previous.digest({"seed": seed, "index": index, "result": result["result_sha256"]})[:12]
        case = cases[result["case_id"]]
        ids = set(case["reference_evidence_ids"])
        for condition in CONDITIONS:
            ids.update(manifest["prompts"][f"{case['id']}:{condition}"]["supplied_ids"])
        answers.append({"rating_id": identifier, "question": case["question"],
            "answer": result["display_text"], "category": case["category"], "rubric": case["rubric"],
            "reference_sources": {key: manifest["judge_sources"][key] for key in sorted(ids)},
            "answer_sha256": previous.digest(result["display_text"])})
        mapping[identifier] = {"attempt_id": result["attempt_id"], "result_sha256": result["result_sha256"],
            "answer_sha256": previous.digest(result["display_text"])}
        ratings.append({"rating_id": identifier, **{field: None for field in FIELDS},
            "unsupported_autobiography": None, "control_pass": None, "rationale": ""})
    shared = {"schema_version": previous.SCHEMA, "design_version": DESIGN,
        "manifest_sha256": manifest["manifest_sha256"], "results_sha256": previous.digest(results)}
    return ({**shared, "instructions":
        "Rate independently without consulting a condition key. Each dimension is 0=absent/contradictory, "
        "1=partial, 2=clear and source-consistent. Relevance: answers the question. Fidelity: no contradiction "
        "or unsupported attribution to the person. Application: concrete action rather than value name-dropping. "
        "Context_sensitivity: uses the stated constraint and respects the relevant exception. "
        "Historical paraphrases are not verbatim quotes or observed private feelings. "
        "For controls use only control_pass and unsupported_autobiography, leaving numeric dimensions null. "
        "Judge novel applications as plausible source-consistent advice, never as a known real Jensen answer.",
        "answers": answers}, {**shared, "mapping": mapping},
        {**shared, "rater_id": "", "rater_kind": "", "ratings": ratings})


def score_results(manifest: dict, results: dict, key: dict, rating_files: list[dict]) -> dict:
    check_manifest(manifest)
    previous._check_results(manifest, results)
    expected = {entry["attempt_id"] for entry in manifest["schedule"]}
    if set(results["attempts"]) != expected:
        raise ValueError("Cannot score an incomplete run")
    if key.get("manifest_sha256") != manifest["manifest_sha256"] or key.get("results_sha256") != previous.digest(results):
        raise ValueError("Blinding key does not match this run")
    valid = {identifier for identifier, result in results["attempts"].items() if result["status"] in {"ok", "policy_guard"}}
    mapped = [entry["attempt_id"] for entry in key["mapping"].values()]
    if len(mapped) != len(set(mapped)) or set(mapped) != valid:
        raise ValueError("Blinding key omits, duplicates or adds valid answers")
    for entry in key["mapping"].values():
        result = results["attempts"][entry["attempt_id"]]
        if entry["result_sha256"] != result["result_sha256"] or entry["answer_sha256"] != previous.digest(result["display_text"]):
            raise ValueError("Blinding key answer hash mismatch")
    cases = {case["id"]: case for case in manifest["cases"]}
    by_attempt = {identifier: cases[result["case_id"]] for identifier, result in results["attempts"].items()}
    if not rating_files or len({(r.get("rater_kind"), r.get("rater_id")) for r in rating_files}) != len(rating_files):
        raise ValueError("Provide distinct identified human/agent raters")
    indexed = []
    for ratings in rating_files:
        entries = previous._validate_ratings(ratings, key, by_attempt)
        for identifier, rating in entries.items():
            category = by_attempt[key["mapping"][identifier]["attempt_id"]]["category"]
            value = rating.get("context_sensitivity")
            if category == "advice" and (type(value) is not int or not 0 <= value <= 2):
                raise ValueError("Advice context_sensitivity must be 0, 1 or 2")
            if category == "control" and value is not None:
                raise ValueError("Controls do not have a context_sensitivity score")
        indexed.append(entries)
    rating_ids = {entry["attempt_id"]: identifier for identifier, entry in key["mapping"].items()}
    measured = []
    for identifier, result in results["attempts"].items():
        case = cases[result["case_id"]]
        scores, dimensions, unsupported = [], {field: [] for field in FIELDS}, []
        for entries in indexed:
            rating = entries.get(rating_ids.get(identifier))
            success = False
            if rating:
                success = (rating["relevance"] == 2 and rating["fidelity"] == 2 and rating["application"] >= 1
                    and rating["context_sensitivity"] == 2 and not rating["unsupported_autobiography"]) if case["category"] == "advice" else (
                    rating["control_pass"] and not rating["unsupported_autobiography"])
            scores.append(float(success))
            unsupported.append(float(rating["unsupported_autobiography"]) if rating else 0.0)
            if case["category"] == "advice":
                for field in FIELDS:
                    dimensions[field].append(float(rating[field]) if rating else 0.0)
        measured.append({"attempt_id": identifier, "case_id": case["id"], "pair_id": case.get("pair_id"),
            "condition": result["condition"], "category": case["category"], "status": result["status"],
            "success": statistics.fmean(scores), "rater_success": scores,
            "unsupported_autobiography": statistics.fmean(unsupported),
            **{field: statistics.fmean(values) if values else None for field, values in dimensions.items()}})
    summaries = {}
    for condition in CONDITIONS:
        rows = [row for row in measured if row["condition"] == condition]
        advice = [row for row in rows if row["category"] == "advice"]
        controls = [row for row in rows if row["category"] == "control"]
        attempts = [result for result in results["attempts"].values() if result["condition"] == condition]
        generated = [result for result in attempts if result["status"] != "policy_guard"]
        pair_success = []
        for pair_id in sorted({row["pair_id"] for row in advice}):
            pair = [row for row in advice if row["pair_id"] == pair_id]
            for rater in range(len(indexed)):
                # Pair is passed only if both variants pass for this rater; repeats averaged per variant.
                case_scores = [statistics.fmean(row["rater_success"][rater] for row in pair if row["case_id"] == case_id)
                               for case_id in sorted({row["case_id"] for row in pair})]
                pair_success.append(min(case_scores))
        summaries[condition] = {"n_attempts": len(attempts), "n_advice": len(advice), "n_controls": len(controls),
            "valid_model_completions": sum(result["status"] == "ok" for result in attempts),
            "deterministic_policy_guards": sum(result["status"] == "policy_guard" for result in attempts),
            "failed_or_truncated": sum(result["status"] not in {"ok", "policy_guard"} for result in attempts),
            "conditional_advice_pass_rate": statistics.fmean(row["success"] for row in advice) if advice else None,
            "paired_context_joint_pass_rate": statistics.fmean(pair_success) if pair_success else None,
            "advice_dimension_means_0_to_2": {field: statistics.fmean(row[field] for row in advice) if advice else None for field in FIELDS},
            "control_pass_rate": statistics.fmean(row["success"] for row in controls) if controls else None,
            "unsupported_autobiography_rate": statistics.fmean(row["unsupported_autobiography"] for row in rows),
            "model_request_median_seconds": statistics.median(result["elapsed_seconds"] for result in generated) if generated else None,
            "mean_provider_prompt_tokens": statistics.fmean((result.get("usage") or {})["prompt_tokens"] for result in generated
                if type((result.get("usage") or {}).get("prompt_tokens")) is int)
                if any(type((result.get("usage") or {}).get("prompt_tokens")) is int for result in generated) else None}
    deltas = {}
    for newer, older in (("expanded_rag", "baseline"), ("flat", "expanded_rag"), ("graph", "flat"), ("graph", "baseline")):
        # Cluster paired variants together; do not count them as independent cases.
        differences = []
        for pair_id in sorted({case["pair_id"] for case in cases.values() if case["category"] == "advice"}):
            means = {condition: statistics.fmean(row["success"] for row in measured if row["pair_id"] == pair_id
                and row["condition"] == condition) for condition in (newer, older)}
            differences.append(means[newer] - means[older])
        estimate = previous.paired_bootstrap(differences)
        estimate["method"] = "Paired percentile bootstrap over contrast pairs; variants, repeats and raters averaged within each pair. Descriptive for small samples."
        deltas[f"{newer}-minus-{older}"] = estimate
    return {"design_version": DESIGN, "manifest_sha256": manifest["manifest_sha256"],
        "results_sha256": previous.digest(results), "conditions": summaries, "paired_deltas": deltas,
        "raters": [{"id": r["rater_id"], "kind": r["rater_kind"], "sha256": previous.digest(r)} for r in rating_files],
        "per_attempt": measured, "limitations": [
            "Authored novel-situation pilot; not held-out real Jensen answers, a personality diagnosis, or a population accuracy estimate.",
            "AI ratings are not human judgments. Expected principles/rubrics never enter generation prompts.",
            "Expanded RAG changes data; flat adds interpreted rules plus evidence selection; only graph-minus-flat isolates serialization.",
            "Paired context joint pass requires both variants; it is not an independent psychological measure.",
            "CPU model request timings exclude TTS, browser and real avatar end-to-end latency.",
            "Failed/truncated attempts remain in denominators. Rerunning after tuning requires a separately labeled development run."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--registry", type=Path, default=ROOT / "data/evaluation/jensen-persona-rules-v2.json")
    freeze.add_argument("--corpus", type=Path, required=True)
    freeze.add_argument("--graph", type=Path, required=True)
    freeze.add_argument("--snapshot", type=Path, required=True)
    freeze.add_argument("--baseline-corpus", type=Path, default=ROOT / "data/personas/jensen-huang")
    freeze.add_argument("--baseline-graph", type=Path, default=ROOT / "data/personas/jensen-huang/persona_graph.v1.json")
    freeze.add_argument("--baseline-snapshot", type=Path, default=ROOT / "data/snapshots/jensen-huang-2026-11-12.json")
    freeze.add_argument("--model", required=True)
    freeze.add_argument("--revision", required=True)
    freeze.add_argument("--seed", type=int, default=20261121)
    freeze.add_argument("--repeats", type=int, default=1)
    freeze.add_argument("--max-tokens", type=int, default=256)
    freeze.add_argument("--context-window", type=int)
    freeze.add_argument("--case-ids", nargs="+")
    freeze.add_argument("--token-count-url")
    freeze.add_argument("--retrieval-backend", choices=("lexical", "jensen-lexical"), default="lexical")
    freeze.add_argument("--provider-dialect", choices=previous.PROVIDER_DIALECTS, default="recorded-gguf")
    freeze.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--base-url", default=os.getenv("PERSONA_EVAL_BASE_URL"))
    run.add_argument("--api-key-env", default="PERSONA_EVAL_API_KEY")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--limit", type=int)
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
        counter = previous.HTTPTokenCounter(args.token_count_url, model=args.model, revision=args.revision) if args.token_count_url else None
        manifest = freeze_manifest(registry_path=args.registry, corpus_path=args.corpus, graph_path=args.graph,
            snapshot_path=args.snapshot, baseline_corpus_path=args.baseline_corpus,
            baseline_snapshot_path=args.baseline_snapshot, baseline_graph_path=args.baseline_graph,
            model=args.model, revision=args.revision, seed=args.seed, repeats=args.repeats, max_tokens=args.max_tokens,
            context_window=args.context_window or (counter.context_window if counter else 16384),
            retrieval_backend=args.retrieval_backend, provider_dialect=args.provider_dialect,
            case_ids=args.case_ids, counter=counter)
        previous.atomic_json(args.output, manifest)
    elif args.command == "run":
        if not args.base_url:
            parser.error("Provide --base-url or PERSONA_EVAL_BASE_URL")
        provider = previous.OpenAICompatibleProvider(args.base_url, api_key=os.getenv(args.api_key_env))
        run_manifest(previous.read_json(args.manifest), args.output, provider, limit=args.limit)
    elif args.command == "blind":
        packet, key, template = blind_packet(previous.read_json(args.manifest), previous.read_json(args.results))
        for name, content in (("answers.blinded.json", packet), ("condition-key.private.json", key), ("ratings.template.json", template)):
            previous.atomic_json(args.output_dir / name, content)
    else:
        result = score_results(previous.read_json(args.manifest), previous.read_json(args.results),
            previous.read_json(args.key), [previous.read_json(path) for path in args.ratings])
        previous.atomic_json(args.output, result)


if __name__ == "__main__":
    main()
