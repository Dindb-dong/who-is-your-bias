"""Development-only response-guidance follow-up to a preserved persona pilot.

Known questions and reviewed outputs make this a diagnostic iteration, never
a held-out accuracy claim. The original four-condition artifacts are immutable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import statistics

from bias.context_budget import PERSONA_RULES_RESPONSE_VERSION, persona_rules_response_instruction
from bias.corpus import Corpus
from bias.persona_graph import PersonaGraph
from bias.snapshots import PersonaSnapshot, corpus_digest
from tools import persona_graph_eval as shared
from tools import persona_rules_eval as original_runner


ROOT = shared.ROOT
DESIGN = "conditional-values-development-guidance.v1"
CONDITION = "guided_graph"
FIELDS = ("relevance", "fidelity", "application", "context_sensitivity")
META_MARKERS = ("persona_memory", "persona-memory", "schema_version", "rule_ids")


def selected_cases(original: dict) -> list[dict]:
    original_runner.check_manifest(original)
    cases = [case for case in original["cases"] if case["category"] == "advice"
             and original["prompts"][f"{case['id']}:graph"]["budget"].get("persona_memory", {}).get("included")]
    if len(cases) != 8 or original["configuration"]["repeats"] != 1:
        raise ValueError("This diagnostic requires all eight original memory-included advice cases, once each")
    return cases


def assert_guidance_only(old: dict, new: dict, *, language: str) -> None:
    """Reject hidden changes to evidence, task, interpretation, or truncation."""
    before, after = old["messages"], new["messages"]
    if len(before) != len(after) or not before or before[0]["role"] != "system" or after[0]["role"] != "system":
        raise ValueError("Expected identical message layout with one initial system instruction")
    expected = persona_rules_response_instruction(language, default_length=True)
    if after[0]["content"] != before[0]["content"] + expected:
        raise ValueError("Only the explicit response-guidance system suffix may change")
    if after[1:] != before[1:]:
        raise ValueError("Reference context, persona memory, history, and user question must be byte-identical")
    for field in ("supplied_ids", "speculative", "policy_guard", "retrieval_reason"):
        if old.get(field) != new.get(field):
            raise ValueError(f"Guidance changed {field}")
    old_memory = old["budget"].get("persona_memory", {})
    new_memory = new["budget"].get("persona_memory", {})
    if not new_memory.get("included") or old_memory.get("included_value_ids") != new_memory.get("included_value_ids"):
        raise ValueError("Guidance must not drop or change a selected complete memory")
    if not new["budget"].get("exact_model_tokens"):
        raise ValueError("The diagnostic requires exact counts from the generation model tokenizer")


def freeze_manifest(*, original_manifest_path: Path, original_results_path: Path,
                    corpus_path: Path, graph_path: Path, snapshot_path: Path, counter) -> dict:
    original = shared.read_json(original_manifest_path)
    cases = selected_cases(original)
    if not getattr(counter, "exact", False):
        raise ValueError("Use the original model's exact tokenizer server")
    configuration = deepcopy(original["configuration"])
    if getattr(counter, "context_window", 0) != configuration["context_window"]:
        raise ValueError("The tokenizer server window differs from the original frozen window")
    corpus = Corpus.load(corpus_path)
    if (corpus_digest(corpus) != original["provenance"]["expanded_corpus_sha256"]
            or shared.file_digest(graph_path) != original["provenance"]["graph_sha256"]
            or shared.file_digest(snapshot_path) != original["provenance"]["snapshot_sha256"]):
        raise ValueError("Data, graph, or snapshot changed; this is not a guidance-only diagnostic")
    graph = PersonaGraph.load(graph_path, corpus)
    snapshot = PersonaSnapshot.load(snapshot_path, corpus)
    prior_results = shared.read_json(original_results_path)
    shared._check_results(original, prior_results)
    if len(prior_results["attempts"]) != len(original["schedule"]):
        raise ValueError("Preserve the completed original run before a development follow-up")
    prompts, selections = {}, {}
    for case in cases:
        prepared = shared.prepare_case(corpus, snapshot, graph, case["question"], language=case["language"],
            context_window=configuration["context_window"], max_tokens=configuration["decode"]["max_tokens"],
            retrieval_limit=configuration["retrieval_limit"], retrieval_backend=configuration["retrieval_backend"], counter=counter)
        new = prepared["conditions"]["graph"]
        old = original["prompts"][f"{case['id']}:graph"]
        assert_guidance_only(old, new, language=case["language"])
        if prepared["selection"] != original["selections"][case["id"]]["selection"]:
            raise ValueError("The selected rules, evidence or selection trace changed")
        key = f"{case['id']}:{CONDITION}"
        prompts[key] = {**new, "prompt_sha256": shared.digest(new["messages"]),
                       "original_prompt_sha256": old["prompt_sha256"]}
        selections[case["id"]] = deepcopy(original["selections"][case["id"]])
    selected = {case["id"] for case in cases}
    schedule = []
    for entry in original["schedule"]:
        if entry["condition"] != "graph" or entry["case_id"] not in selected:
            continue
        key = f"{entry['case_id']}:{CONDITION}"
        schedule.append({"attempt_id": f"{entry['case_id']}:r0:{CONDITION}", "case_id": entry["case_id"],
            "repeat": 0, "condition": CONDITION, "prompt_key": key,
            "original_attempt_id": entry["attempt_id"],
            "request_sha256": shared.digest({"prompt": prompts[key], "configuration": configuration})})
    graph_json = shared.read_json(graph_path)
    markers = sorted(set(META_MARKERS) | {item["id"] for item in graph_json["claims"] + graph_json["rules"]})
    manifest = {"schema_version": shared.SCHEMA, "design_version": DESIGN,
        "scope": "Development diagnostic on all eight previously evaluated memory-included advice cases; not held-out or independent validation.",
        "conditions": [CONDITION], "configuration": configuration, "cases": deepcopy(cases),
        "prompts": prompts, "selections": selections, "schedule": schedule,
        "judge_sources": deepcopy(original["judge_sources"]),
        "observational_markers": {"match": "case_sensitive_substring", "markers": markers,
            "meaning": "User-facing internal-identifier leakage diagnostic only; not a persona-quality score."},
        "provenance": {"original_manifest_sha256": original["manifest_sha256"],
            "original_manifest_file_sha256": shared.file_digest(original_manifest_path),
            "original_results_file_sha256": shared.file_digest(original_results_path),
            "original_results_sha256": shared.digest(prior_results),
            "graph_sha256": shared.file_digest(graph_path), "snapshot_sha256": shared.file_digest(snapshot_path),
            "corpus_sha256": corpus_digest(corpus), "guidance_version": PERSONA_RULES_RESPONSE_VERSION,
            "context_budget_sha256": shared.file_digest(ROOT / "src/bias/context_budget.py"),
            "runner_sha256": shared.file_digest(Path(__file__)),
            "shared_runner_sha256": shared.file_digest(Path(shared.__file__)),
            "guidance_only_assertions_passed": True}}
    manifest["manifest_sha256"] = shared.digest(manifest)
    check_manifest(manifest)
    return manifest


def check_manifest(manifest: dict) -> None:
    shared.check_manifest(manifest)
    if manifest.get("design_version") != DESIGN or manifest.get("conditions") != [CONDITION]:
        raise ValueError("Expected the separately labeled graph-only development design")
    cases = {case["id"] for case in manifest["cases"]}
    schedule = manifest["schedule"]
    if len(cases) != 8 or len(schedule) != 8 or {entry["case_id"] for entry in schedule} != cases:
        raise ValueError("Do not drop or duplicate any of the eight development cases")
    if any(entry["condition"] != CONDITION or entry["repeat"] != 0 for entry in schedule):
        raise ValueError("Unexpected condition or repeat in the diagnostic")


def run_manifest(manifest: dict, output: Path, provider, *, limit=None) -> dict:
    check_manifest(manifest)
    return shared.run_manifest(manifest, output, provider, limit=limit)


def diagnostics(manifest: dict, results: dict, original: dict, prior_results: dict) -> dict:
    check_manifest(manifest)
    shared._check_results(manifest, results)
    original_runner.check_manifest(original)
    shared._check_results(original, prior_results)
    if (original["manifest_sha256"] != manifest["provenance"]["original_manifest_sha256"]
            or shared.digest(prior_results) != manifest["provenance"]["original_results_sha256"]):
        raise ValueError("Original run differs from the diagnostic's preserved reference")
    if len(results["attempts"]) != 8:
        raise ValueError("Finish all eight attempts, including errors, before summarizing")
    markers = manifest["observational_markers"]["markers"]
    paired = []
    for entry in manifest["schedule"]:
        before = prior_results["attempts"][entry["original_attempt_id"]]
        after = results["attempts"][entry["attempt_id"]]
        row = {"case_id": entry["case_id"]}
        for label, result in (("original_graph", before), (CONDITION, after)):
            raw = result.get("raw_text", "")
            row[label] = {"status": result["status"], "finish_reason": result.get("finish_reason"),
                "word_count_whitespace": len(raw.split()), "internal_markers": [marker for marker in markers if marker in raw],
                "request_seconds": result["elapsed_seconds"], "completion_tokens": (result.get("usage") or {}).get("completion_tokens")}
        paired.append(row)
    summaries = {}
    for condition in ("original_graph", CONDITION):
        rows = [row[condition] for row in paired]
        summaries[condition] = {"n_attempts": len(rows), "normal_completions": sum(row["status"] == "ok" for row in rows),
            "length_limited": sum(row["finish_reason"] == "length" for row in rows),
            "errors_or_invalid": sum(row["status"] not in {"ok", "policy_guard"} for row in rows),
            "answers_with_internal_markers": sum(bool(row["internal_markers"]) for row in rows),
            "median_word_count_whitespace": statistics.median(row["word_count_whitespace"] for row in rows),
            "median_request_seconds": statistics.median(row["request_seconds"] for row in rows)}
    return {"design_version": DESIGN, "manifest_sha256": manifest["manifest_sha256"],
        "results_sha256": shared.digest(results), "conditions": summaries, "paired_cases": paired,
        "limitations": ["Same known questions after inspecting original outputs: development diagnostics, not held-out gains.",
            "Marker/length metrics are execution diagnostics, not factuality or value-faithfulness scores.",
            "All eight original selected cases included, including successes and failures; original results unchanged.",
            "CPU request times exclude TTS and avatar rendering. Different execution times can include host-load differences."]}


def blind_packet(manifest: dict, results: dict, *, seed=112129):
    check_manifest(manifest)
    shared._check_results(manifest, results)
    if len(results["attempts"]) != 8:
        raise ValueError("Complete all eight attempts before blinding")
    cases = {case["id"]: case for case in manifest["cases"]}
    entries = list(results["attempts"].values())
    random.Random(seed).shuffle(entries)
    answers, mapping, ratings = [], {}, []
    for index, result in enumerate(entries):
        if result["status"] != "ok":
            continue
        identifier = "answer-" + shared.digest({"seed": seed, "index": index, "result": result["result_sha256"]})[:12]
        case = cases[result["case_id"]]
        source_ids = set(case["reference_evidence_ids"]) | set(manifest["prompts"][result["prompt_key"]]["supplied_ids"])
        answers.append({"rating_id": identifier, "question": case["question"], "answer": result["display_text"],
            "category": "advice", "rubric": case["rubric"],
            "reference_sources": {key: manifest["judge_sources"][key] for key in sorted(source_ids)},
            "answer_sha256": shared.digest(result["display_text"])})
        mapping[identifier] = {"attempt_id": result["attempt_id"], "result_sha256": result["result_sha256"],
            "answer_sha256": shared.digest(result["display_text"])}
        ratings.append({"rating_id": identifier, **{field: None for field in FIELDS},
            "unsupported_autobiography": None, "control_pass": None, "rationale": ""})
    common = {"schema_version": shared.SCHEMA, "design_version": DESIGN,
        "manifest_sha256": manifest["manifest_sha256"], "results_sha256": shared.digest(results)}
    return ({**common, "instructions": "Use the original independently written rating guide and the same four dimensions. These are development answers on known questions; assess them without consulting prior ratings, the condition key or unblinded results. Invalid outputs remain failures outside this packet.",
             "answers": answers}, {**common, "mapping": mapping},
            {**common, "rater_id": "", "rater_kind": "", "ratings": ratings})


def write_blinding_files(manifest: dict, results: dict, output_dir: Path, private_key_path: Path) -> None:
    # Do not rely solely on .gitignore: a public key defeats the blind packet.
    if private_key_path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Save the condition key outside the Git checkout, then archive it privately")
    packet, key, template = blind_packet(manifest, results)
    shared.atomic_json(output_dir / "answers.blinded.json", packet)
    shared.atomic_json(output_dir / "ratings.template.json", template)
    shared.atomic_json(private_key_path, key)


def score_results(manifest: dict, results: dict, key: dict, rating_files: list[dict]) -> dict:
    check_manifest(manifest)
    shared._check_results(manifest, results)
    if len(results["attempts"]) != 8:
        raise ValueError("Cannot score an incomplete development run")
    if key.get("manifest_sha256") != manifest["manifest_sha256"] or key.get("results_sha256") != shared.digest(results):
        raise ValueError("Condition key does not match the development outputs")
    valid = {identifier for identifier, row in results["attempts"].items() if row["status"] == "ok"}
    mapped = [entry["attempt_id"] for entry in key["mapping"].values()]
    if len(mapped) != len(set(mapped)) or set(mapped) != valid:
        raise ValueError("Condition key has missing or duplicated valid answers")
    for entry in key["mapping"].values():
        row = results["attempts"][entry["attempt_id"]]
        if entry["result_sha256"] != row["result_sha256"] or entry["answer_sha256"] != shared.digest(row["display_text"]):
            raise ValueError("Condition key answer hash mismatch")
    cases = {case["id"]: case for case in manifest["cases"]}
    by_attempt = {identifier: cases[row["case_id"]] for identifier, row in results["attempts"].items()}
    if not rating_files or len({(r.get("rater_kind"), r.get("rater_id")) for r in rating_files}) != len(rating_files):
        raise ValueError("Provide distinct identified raters")
    indexed = []
    for file in rating_files:
        ratings = shared._validate_ratings(file, key, by_attempt)
        for rating in ratings.values():
            if type(rating.get("context_sensitivity")) is not int or not 0 <= rating["context_sensitivity"] <= 2:
                raise ValueError("context_sensitivity must be 0, 1 or 2")
        indexed.append(ratings)
    ids = {entry["attempt_id"]: identifier for identifier, entry in key["mapping"].items()}
    rows = []
    for identifier, result in results["attempts"].items():
        scores = []
        flags = []
        for ratings in indexed:
            rating = ratings.get(ids.get(identifier))
            scores.append(float(bool(rating) and rating["relevance"] == 2 and rating["fidelity"] == 2
                and rating["application"] >= 1 and rating["context_sensitivity"] == 2
                and not rating["unsupported_autobiography"]))
            if rating:
                flags.append(float(rating["unsupported_autobiography"]))
        rows.append({"attempt_id": identifier, "case_id": result["case_id"], "status": result["status"],
            "success": statistics.fmean(scores), "unsupported_autobiography": statistics.fmean(flags) if flags else None})
    flags = [row["unsupported_autobiography"] for row in rows if row["unsupported_autobiography"] is not None]
    return {"design_version": DESIGN, "manifest_sha256": manifest["manifest_sha256"],
        "results_sha256": shared.digest(results), "n_scheduled": 8, "n_rated_valid": len(valid),
        "unassessed_invalid": 8-len(valid), "conditional_advice_pass_rate": statistics.fmean(row["success"] for row in rows),
        "unsupported_autobiography_rate_among_rated_valid": statistics.fmean(flags) if flags else None,
        "raters": [{"id": r["rater_id"], "kind": r["rater_kind"], "sha256": shared.digest(r)} for r in rating_files],
        "per_attempt": rows, "limitation": "Known-case development assessment after prompt revision, not an independent generalization or held-out accuracy gain."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--original-manifest", type=Path, required=True)
    freeze.add_argument("--original-results", type=Path, required=True)
    freeze.add_argument("--corpus", type=Path, required=True)
    freeze.add_argument("--graph", type=Path, required=True)
    freeze.add_argument("--snapshot", type=Path, required=True)
    freeze.add_argument("--token-count-url", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--base-url", default=os.getenv("PERSONA_EVAL_BASE_URL"))
    run.add_argument("--api-key-env", default="PERSONA_EVAL_API_KEY")
    run.add_argument("--limit", type=int)
    run.add_argument("--output", type=Path, required=True)
    diag = sub.add_parser("diagnostics")
    for name in ("manifest", "results", "original-manifest", "original-results", "output"):
        diag.add_argument("--"+name, type=Path, required=True)
    blind = sub.add_parser("blind")
    for name in ("manifest", "results", "output-dir", "private-key"):
        blind.add_argument("--"+name, type=Path, required=True)
    score = sub.add_parser("score")
    for name in ("manifest", "results", "key", "output"):
        score.add_argument("--"+name, type=Path, required=True)
    score.add_argument("--ratings", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        config = shared.read_json(args.original_manifest)["configuration"]
        counter = shared.HTTPTokenCounter(args.token_count_url, model=config["model"], revision=config["revision"])
        manifest = freeze_manifest(original_manifest_path=args.original_manifest, original_results_path=args.original_results,
            corpus_path=args.corpus, graph_path=args.graph, snapshot_path=args.snapshot, counter=counter)
        shared.atomic_json(args.output, manifest)
    elif args.command == "run":
        if not args.base_url:
            parser.error("Provide --base-url or PERSONA_EVAL_BASE_URL")
        run_manifest(shared.read_json(args.manifest), args.output,
            shared.OpenAICompatibleProvider(args.base_url, api_key=os.getenv(args.api_key_env)), limit=args.limit)
    elif args.command == "diagnostics":
        shared.atomic_json(args.output, diagnostics(shared.read_json(args.manifest), shared.read_json(args.results),
            shared.read_json(args.original_manifest), shared.read_json(args.original_results)))
    elif args.command == "blind":
        write_blinding_files(shared.read_json(args.manifest), shared.read_json(args.results), args.output_dir, args.private_key)
    else:
        shared.atomic_json(args.output, score_results(shared.read_json(args.manifest), shared.read_json(args.results),
            shared.read_json(args.key), [shared.read_json(path) for path in args.ratings]))


if __name__ == "__main__":
    main()
