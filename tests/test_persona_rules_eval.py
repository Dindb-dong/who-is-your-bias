"""Experimental-design integrity tests, never persona-quality measurements."""
from copy import deepcopy
from datetime import date
import json

import pytest

from bias.corpus import Corpus
from tools import persona_graph_eval as previous
from tools.persona_rules_eval import (
    CONDITIONS, ROOT, blind_packet, check_manifest, freeze_manifest,
    run_manifest, score_results, validate_comparison_corpora,
)


def registry_fixture(path):
    registry = json.loads((ROOT / "data/evaluation/jensen-persona-rules-v2.json").read_text())
    # The old-corpus integrity fixture has no newly acquired judge references.
    # This affects tests only; production uses the complete versioned registry.
    for case in registry["cases"]:
        case["reference_evidence_ids"] = []
    path.write_text(json.dumps(registry))
    return path


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    # Identical corpora deliberately isolate runner behavior from the new data.
    corpus = ROOT / "data/personas/jensen-huang"
    graph = corpus / "persona_graph.v1.json"
    snapshot = ROOT / "data/snapshots/jensen-huang-2026-11-12.json"
    registry = registry_fixture(tmp_path_factory.mktemp("rules-eval") / "registry.json")
    return freeze_manifest(registry_path=registry,
        corpus_path=corpus, graph_path=graph, snapshot_path=snapshot,
        baseline_corpus_path=corpus, baseline_graph_path=graph, baseline_snapshot_path=snapshot,
        model="unit-test-only", revision="fixture-not-a-model",
        case_ids=["tests-feasible", "tests-exhausted", "premise-supported", "premise-refuted", "control-private", "control-project"])


def provider(messages, configuration):
    return {"raw_text": "UNIT TEST ONLY; not measured answer quality.", "finish_reason": "stop",
        "model": configuration["model"], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def completed(manifest, tmp_path, source=provider):
    results = run_manifest(manifest, tmp_path / "results.json", source)
    packet, key, ratings = blind_packet(manifest, results)
    ratings.update(rater_id="fixture", rater_kind="agent")
    advice = {answer["rating_id"] for answer in packet["answers"] if answer["category"] == "advice"}
    for rating in ratings["ratings"]:
        if rating["rating_id"] in advice:
            rating.update(relevance=2, fidelity=2, application=1, context_sensitivity=2)
        else:
            rating["control_pass"] = True
        rating.update(unsupported_autobiography=False, rationale="Fixture for score integrity, not a judgment.")
    return results, packet, key, ratings


def test_four_conditions_and_judge_labels_never_enter_prompts(manifest):
    assert len(manifest["schedule"]) == 24
    assert manifest["conditions"] == list(CONDITIONS)
    for case in manifest["cases"]:
        prompts = {condition: manifest["prompts"][f"{case['id']}:{condition}"] for condition in CONDITIONS}
        assert prompts["baseline"] == prompts["expanded_rag"]  # Same input corpus in fixture.
        assert prompts["flat"]["supplied_ids"] == prompts["graph"]["supplied_ids"]
        for prompt in prompts.values():
            text = json.dumps(prompt["messages"])
            assert "context_sensitivity" not in text
            assert "unacceptable_claims" not in text
            assert case["rubric"].get("desired_application", "JUDGE_ONLY_SENTINEL") not in text
            if prompt["messages"]:
                assert prompt["messages"][-1]["content"] == case["question"]


def test_schedule_balances_four_condition_positions(manifest):
    blocks = [[entry["condition"] for entry in manifest["schedule"][index:index + 4]]
              for index in range(0, 16, 4)]
    for position in range(4):
        assert {block[position] for block in blocks} == set(CONDITIONS)


def test_cannot_select_only_one_side_of_context_pair(tmp_path):
    corpus = ROOT / "data/personas/jensen-huang"
    graph = corpus / "persona_graph.v1.json"
    snapshot = ROOT / "data/snapshots/jensen-huang-2026-11-12.json"
    with pytest.raises(ValueError, match="exactly two"):
        freeze_manifest(registry_path=registry_fixture(tmp_path / "registry.json"),
            corpus_path=corpus, graph_path=graph, snapshot_path=snapshot,
            baseline_corpus_path=corpus, baseline_graph_path=graph, baseline_snapshot_path=snapshot,
            model="fixture", revision="fixture", case_ids=["tests-feasible"])


def test_only_corpus_coverage_dates_may_differ_in_persona_card():
    base = Corpus.load(ROOT / "data/personas/jensen-huang")
    profile = base.persona.model_copy(update={"corpus_start": date(2009, 4, 8), "corpus_cutoff": date(2026, 5, 4)})
    changed = Corpus(profile, base.sources, base.evidence)
    validate_comparison_corpora(base, changed)
    for field, value in (("name", "A different persona"), ("style_rules", ["Invent personal memories."])):
        altered = Corpus(profile.model_copy(update={field: value}), base.sources, base.evidence)
        with pytest.raises(ValueError, match="same persona card"):
            validate_comparison_corpora(base, altered)


def test_hashes_and_full_conditions_are_checked(manifest):
    edited = deepcopy(manifest)
    edited["prompts"]["tests-feasible:graph"]["messages"][-1]["content"] = "changed"
    with pytest.raises(ValueError, match="changed after freeze"):
        check_manifest(edited)
    edited = deepcopy(manifest)
    edited["conditions"].remove("expanded_rag")
    edited["manifest_sha256"] = previous.digest({key: value for key, value in edited.items() if key != "manifest_sha256"})
    with pytest.raises(ValueError, match="four-condition"):
        check_manifest(edited)


def test_resume_keeps_each_attempt_and_errors_once(manifest, tmp_path):
    calls = []

    def failing(messages, configuration):
        calls.append(1)
        raise RuntimeError("private provider body must not be copied")

    output = tmp_path / "results.json"
    first = run_manifest(manifest, output, failing, limit=3)
    assert len(first["attempts"]) == 3
    with pytest.raises(ValueError, match="every planned attempt"):
        blind_packet(manifest, first)
    all_results = run_manifest(manifest, output, failing)
    expected = sum(not manifest["prompts"][entry["prompt_key"]]["policy_guard"] for entry in manifest["schedule"])
    assert len(calls) == expected
    run_manifest(manifest, output, failing)
    assert len(calls) == expected
    assert len(all_results["attempts"]) == 24
    assert "private provider" not in output.read_text()


def test_context_inattention_fails_even_when_generic_advice_is_good(manifest, tmp_path):
    results, packet, key, ratings = completed(manifest, tmp_path)
    for rating in ratings["ratings"]:
        if rating["context_sensitivity"] is not None:
            rating["context_sensitivity"] = 1
    summary = score_results(manifest, results, key, [ratings])
    assert all(summary["conditions"][c]["conditional_advice_pass_rate"] == 0 for c in CONDITIONS)
    assert all(summary["conditions"][c]["paired_context_joint_pass_rate"] == 0 for c in CONDITIONS)
    assert all("condition" not in answer and "attempt_id" not in answer for answer in packet["answers"])


def test_one_wrong_variant_fails_pair_and_bootstrap_clusters_variants(manifest, tmp_path):
    results, _, key, ratings = completed(manifest, tmp_path)
    for rating in ratings["ratings"]:
        attempt = key["mapping"][rating["rating_id"]]["attempt_id"]
        if attempt.startswith("tests-exhausted:"):
            rating["context_sensitivity"] = 0
    summary = score_results(manifest, results, key, [ratings])
    for condition in CONDITIONS:
        assert summary["conditions"][condition]["conditional_advice_pass_rate"] == .75
        assert summary["conditions"][condition]["paired_context_joint_pass_rate"] == .5
    assert summary["paired_deltas"]["graph-minus-baseline"]["n_cases"] == 2


def test_truncated_model_outputs_remain_zero_in_denominator(manifest, tmp_path):
    def truncated(messages, configuration):
        return {**provider(messages, configuration), "finish_reason": "length"}

    results, _, key, ratings = completed(manifest, tmp_path, truncated)
    summary = score_results(manifest, results, key, [ratings])
    for condition in CONDITIONS:
        assert summary["conditions"][condition]["n_advice"] == 4
        assert summary["conditions"][condition]["conditional_advice_pass_rate"] == 0


@pytest.mark.parametrize("mutation", ["missing", "bool_score", "unknown", "wrong_hash", "changed_answer"])
def test_incomplete_or_invalid_ratings_and_results_cannot_be_scored(manifest, tmp_path, mutation):
    results, _, key, ratings = completed(manifest, tmp_path)
    if mutation == "missing":
        ratings["ratings"].pop()
    elif mutation == "bool_score":
        next(r for r in ratings["ratings"] if r["context_sensitivity"] is not None)["context_sensitivity"] = True
    elif mutation == "unknown":
        ratings["ratings"][0]["rating_id"] = "unknown"
    elif mutation == "wrong_hash":
        ratings["results_sha256"] = "invalid"
    else:
        next(iter(results["attempts"].values()))["raw_text"] = "modified"
    with pytest.raises(ValueError):
        score_results(manifest, results, key, [ratings])
