"""Guidance-only experiment integrity. Fake counters/providers are fixtures."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bias.context_budget import persona_rules_response_instruction
from tools import persona_graph_eval as shared
from tools import persona_rules_diagnostic as diagnostic


ROOT = diagnostic.ROOT
REPORT = ROOT / "reports/persona-rules-2026-11-21"


class FixtureCounter:
    exact = True  # Mock the tokenizer contract; never a measured token count.
    context_window = 8192
    identity = "unit-test-only-not-a-tokenizer"

    def count(self, messages):
        return 100 + len(json.dumps(messages)) // 4


@pytest.fixture(scope="module")
def manifest():
    return diagnostic.freeze_manifest(original_manifest_path=REPORT / "manifest.json",
        original_results_path=REPORT / "results.json", corpus_path=ROOT / "data/personas/jensen-huang-values-v2",
        graph_path=ROOT / "data/personas/jensen-huang-values-v2/persona_graph.v2.json",
        snapshot_path=ROOT / "data/snapshots/jensen-values-2026-11-21.json", counter=FixtureCounter())


def fixture_provider(messages, configuration):
    return {"raw_text": "UNIT TEST FIXTURE, never a measured model answer.", "finish_reason": "stop",
            "model": configuration["model"], "usage": {"prompt_tokens": 123, "completion_tokens": 12}}


def complete(manifest, tmp_path, provider=fixture_provider):
    results = diagnostic.run_manifest(manifest, tmp_path / "results.json", provider)
    packet, key, ratings = diagnostic.blind_packet(manifest, results)
    ratings.update(rater_id="unit-test-fixture", rater_kind="agent")
    for row in ratings["ratings"]:
        row.update(relevance=2, fidelity=2, application=1, context_sensitivity=2,
            unsupported_autobiography=False, rationale="Fixture only; no actual quality assessment.")
    return results, packet, key, ratings


def test_exactly_all_original_selected_cases_and_configuration_retained(manifest):
    original = shared.read_json(REPORT / "manifest.json")
    expected = {case["id"] for case in diagnostic.selected_cases(original)}
    assert {case["id"] for case in manifest["cases"]} == expected
    assert len(manifest["schedule"]) == 8
    assert manifest["conditions"] == ["guided_graph"]
    assert manifest["configuration"] == original["configuration"]
    for case in manifest["cases"]:
        assert case == next(item for item in original["cases"] if item["id"] == case["id"])
        before = original["prompts"][case["id"]+":graph"]
        after = manifest["prompts"][case["id"]+":guided_graph"]
        assert after["messages"][1:] == before["messages"][1:]
        assert after["messages"][0]["content"] == before["messages"][0]["content"] + persona_rules_response_instruction("en", default_length=True)
        assert after["supplied_ids"] == before["supplied_ids"]
        assert case["rubric"]["desired_application"] not in json.dumps(after["messages"])


@pytest.mark.parametrize("mutation", ["context", "question", "system", "sources", "drop_memory"])
def test_narrow_system_suffix_contract_rejects_hidden_changes(manifest, mutation):
    original = shared.read_json(REPORT / "manifest.json")
    case_id = manifest["cases"][0]["id"]
    before = original["prompts"][case_id+":graph"]
    after = deepcopy(manifest["prompts"][case_id+":guided_graph"])
    if mutation == "context":
        after["messages"][1]["content"] += "extra source"
    elif mutation == "question":
        after["messages"][-1]["content"] += "answer in exactly two words"
    elif mutation == "system":
        after["messages"][0]["content"] += "extra optimization"
    elif mutation == "sources":
        after["supplied_ids"].pop()
    else:
        after["budget"]["persona_memory"]["included"] = False
    with pytest.raises(ValueError):
        diagnostic.assert_guidance_only(before, after, language="en")


def test_modified_graph_cannot_be_called_guidance_only(tmp_path):
    graph = tmp_path / "graph.json"
    graph.write_text((ROOT / "data/personas/jensen-huang-values-v2/persona_graph.v2.json").read_text()+"\n")
    with pytest.raises(ValueError, match="Data, graph, or snapshot changed"):
        diagnostic.freeze_manifest(original_manifest_path=REPORT / "manifest.json", original_results_path=REPORT / "results.json",
            corpus_path=ROOT / "data/personas/jensen-huang-values-v2", graph_path=graph,
            snapshot_path=ROOT / "data/snapshots/jensen-values-2026-11-21.json", counter=FixtureCounter())


def test_approximate_counter_is_rejected():
    counter = FixtureCounter()
    counter.exact = False
    with pytest.raises(ValueError, match="exact tokenizer"):
        diagnostic.freeze_manifest(original_manifest_path=REPORT / "manifest.json", original_results_path=REPORT / "results.json",
            corpus_path=ROOT / "data/personas/jensen-huang-values-v2",
            graph_path=ROOT / "data/personas/jensen-huang-values-v2/persona_graph.v2.json",
            snapshot_path=ROOT / "data/snapshots/jensen-values-2026-11-21.json", counter=counter)


def test_partial_resume_and_failures_are_preserved_once(manifest, tmp_path):
    calls = []

    def failure(messages, configuration):
        calls.append(1)
        raise RuntimeError("a secret error body must not be saved")

    path = tmp_path / "results.json"
    partial = diagnostic.run_manifest(manifest, path, failure, limit=2)
    with pytest.raises(ValueError, match="all eight"):
        diagnostic.blind_packet(manifest, partial)
    final = diagnostic.run_manifest(manifest, path, failure)
    assert len(calls) == 8
    diagnostic.run_manifest(manifest, path, failure)
    assert len(calls) == 8
    assert len(final["attempts"]) == 8
    assert "secret error body" not in path.read_text()


def test_private_key_must_be_outside_checkout(manifest, tmp_path):
    results, _, _, _ = complete(manifest, tmp_path)
    with pytest.raises(ValueError, match="outside the Git checkout"):
        diagnostic.write_blinding_files(manifest, results, tmp_path / "public", ROOT / "reports/condition-key.private.json")
    diagnostic.write_blinding_files(manifest, results, tmp_path / "public", tmp_path / "private/key.json")
    assert not (tmp_path / "public/condition-key.private.json").exists()
    assert (tmp_path / "private/key.json").exists()
    packet = shared.read_json(tmp_path / "public/answers.blinded.json")
    assert all("attempt_id" not in answer and "condition" not in answer for answer in packet["answers"])


def test_invalid_outputs_fail_primary_but_do_not_imply_no_autobiography(manifest, tmp_path):
    def truncated(messages, configuration):
        return {**fixture_provider(messages, configuration), "finish_reason": "length"}

    results, _, key, ratings = complete(manifest, tmp_path, truncated)
    summary = diagnostic.score_results(manifest, results, key, [ratings])
    assert summary["conditional_advice_pass_rate"] == 0
    assert summary["n_rated_valid"] == 0
    assert summary["unassessed_invalid"] == 8
    assert summary["unsupported_autobiography_rate_among_rated_valid"] is None


@pytest.mark.parametrize("mutation", ["missing", "context_bool", "answer_hash", "result_edit"])
def test_invalid_ratings_and_tampered_outputs_rejected(manifest, tmp_path, mutation):
    results, _, key, ratings = complete(manifest, tmp_path)
    if mutation == "missing":
        ratings["ratings"].pop()
    elif mutation == "context_bool":
        ratings["ratings"][0]["context_sensitivity"] = True
    elif mutation == "answer_hash":
        next(iter(key["mapping"].values()))["answer_sha256"] = "wrong"
    else:
        next(iter(results["attempts"].values()))["raw_text"] = "changed"
    with pytest.raises(ValueError):
        diagnostic.score_results(manifest, results, key, [ratings])


def test_exact_case_sensitive_marker_definition_and_all_attempt_denominator(manifest, tmp_path):
    def marked(messages, configuration):
        return {**fixture_provider(messages, configuration), "raw_text": "persona_memory PERSONA_MEMORY", "finish_reason": "length"}

    results, _, _, _ = complete(manifest, tmp_path, marked)
    report = diagnostic.diagnostics(manifest, results, shared.read_json(REPORT / "manifest.json"), shared.read_json(REPORT / "results.json"))
    assert report["conditions"]["guided_graph"]["answers_with_internal_markers"] == 8
    assert report["conditions"]["guided_graph"]["length_limited"] == 8
    assert all(row["guided_graph"]["internal_markers"] == ["persona_memory"] for row in report["paired_cases"])
