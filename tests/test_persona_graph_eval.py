"""Experiment integrity checks; fake provider fixtures are not quality results."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.persona_graph_eval import (
    CONDITIONS, HTTPTokenCounter, ROOT, atomic_json, blind_packet, check_manifest, digest,
    completion_payload, freeze_manifest, paired_bootstrap, prepare_case, run_manifest, score_results,
    verify_team_contract,
)


@pytest.fixture(scope="module")
def manifest():
    return freeze_manifest(
        registry_path=ROOT / "data/evaluation/jensen-persona-graph-v1.json",
        corpus_path=ROOT / "data/personas/jensen-huang",
        graph_path=ROOT / "data/personas/jensen-huang/persona_graph.v1.json",
        snapshot_path=ROOT / "data/snapshots/jensen-huang-2026-11-12.json",
        model="unit-test-only-not-a-model", revision="fixture-not-a-quality-result",
        case_ids=["advice-01", "advice-03", "advice-05", "advice-07", "control-02", "control-03"],
    )


def test_generation_prompts_exclude_judge_targets_and_keep_flat_graph_semantics(manifest):
    assert len(manifest["schedule"]) == 18
    for case in manifest["cases"]:
        prompts = {condition: manifest["prompts"][f"{case['id']}:{condition}"] for condition in CONDITIONS}
        assert prompts["flat"]["supplied_ids"] == prompts["graph"]["supplied_ids"]
        for prompt in prompts.values():
            rendered = json.dumps(prompt["messages"])
            assert "expected_value_ids" not in rendered
            assert "desired_application" not in rendered
            assert case["rubric"].get("desired_application", "JUDGE_ONLY_SENTINEL") not in rendered
            if prompt["messages"]:
                assert prompt["messages"][-1]["content"] == case["question"]
        if prompts["graph"]["budget"].get("persona_memory", {}).get("included"):
            flat = json.loads(prompts["flat"]["messages"][1]["content"].split("\n", 1)[1])["persona_memory"]
            graph = json.loads(prompts["graph"]["messages"][1]["content"].split("\n", 1)[1])["persona_memory"]
            assert flat["nodes"] == graph["nodes"]
            assert flat["relation_statements"] == [
                f"{edge['source']} {edge['relation'].replace('_', ' ')} {edge['target']} (editorial interpretation)."
                for edge in graph["relations"]]
            assert prompts["flat"]["messages"][0] == prompts["graph"]["messages"][0]


def test_six_case_block_counterbalances_condition_order(manifest):
    orders = [tuple(item["condition"] for item in manifest["schedule"][index:index + 3])
              for index in range(0, len(manifest["schedule"]), 3)]
    assert len(set(orders)) == 6
    for position in range(3):
        assert {condition: sum(order[position] == condition for order in orders) for condition in CONDITIONS} == {
            condition: 2 for condition in CONDITIONS}


def test_manifest_detects_postfreeze_prompt_edits(manifest):
    modified = deepcopy(manifest)
    modified["prompts"]["advice-01:graph"]["messages"][-1]["content"] = "changed"
    with pytest.raises(ValueError, match="changed after freeze"):
        check_manifest(modified)


def provider_fixture(messages, configuration):
    assert configuration["model"] == "unit-test-only-not-a-model"
    return {"raw_text": "UNIT TEST FIXTURE. Never a measured persona answer.", "finish_reason": "stop",
            "model": configuration["model"], "usage": {"prompt_tokens": 42, "completion_tokens": 12}}


def completed_with_ratings(manifest, tmp_path, *, provider=provider_fixture):
    results = run_manifest(manifest, tmp_path / "results.json", provider)
    packet, key, ratings = blind_packet(manifest, results)
    ratings.update(rater_id="unit-test-fixture-not-human-review", rater_kind="agent")
    categories = {entry["rating_id"]: entry["category"] for entry in packet["answers"]}
    for rating in ratings["ratings"]:
        if categories[rating["rating_id"]] == "advice":
            rating.update(relevance=2, fidelity=2, application=1)
        else:
            rating["control_pass"] = True
        rating.update(unsupported_autobiography=False, rationale="UNIT TEST FIXTURE; no answer quality assessed.")
    return results, packet, key, ratings


def test_atomic_resume_calls_each_attempt_once_and_never_retries_errors(manifest, tmp_path):
    calls = []

    def raising_provider(messages, config):
        calls.append(digest(messages))
        raise RuntimeError("sensitive provider text must not be recorded")

    path = tmp_path / "results.json"
    partial = run_manifest(manifest, path, raising_provider, limit=2)
    assert len(partial["attempts"]) == 2
    with pytest.raises(ValueError, match="every planned attempt"):
        blind_packet(manifest, partial)
    results = run_manifest(manifest, path, raising_provider)
    expected_calls = sum(not manifest["prompts"][entry["prompt_key"]]["policy_guard"] for entry in manifest["schedule"])
    assert len(calls) == expected_calls
    assert len(results["attempts"]) == len(manifest["schedule"])
    run_manifest(manifest, path, raising_provider)
    assert len(calls) == expected_calls
    assert "sensitive provider text" not in path.read_text()


def test_existing_policy_guards_do_not_call_model_or_count_as_generations(manifest, tmp_path):
    results, packet, key, ratings = completed_with_ratings(manifest, tmp_path)
    summary = score_results(manifest, results, key, [ratings])
    for condition in CONDITIONS:
        assert summary["conditions"][condition]["deterministic_policy_guards"] == 2
        assert summary["conditions"][condition]["valid_model_completions"] == 4
    # Raters cannot infer the condition from an explicit label or attempt ID.
    assert all("condition" not in answer and "attempt_id" not in answer for answer in packet["answers"])


def test_truncation_is_zero_and_not_removed_from_advice_denominator(manifest, tmp_path):
    def truncated_provider(messages, config):
        return {**provider_fixture(messages, config), "finish_reason": "length"}

    results, _, key, ratings = completed_with_ratings(manifest, tmp_path, provider=truncated_provider)
    summary = score_results(manifest, results, key, [ratings])
    for condition in CONDITIONS:
        assert summary["conditions"][condition]["n_advice"] == 4
        assert summary["conditions"][condition]["failed_or_truncated"] == 4
        assert summary["conditions"][condition]["advice_value_application_accuracy"] == 0
        assert summary["conditions"][condition]["control_pass_rate"] == 1


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "bool_score", "wrong_hash", "no_rationale"])
def test_external_rating_schema_rejects_invalid_or_incomplete_scores(manifest, tmp_path, mutation):
    results, packet, key, ratings = completed_with_ratings(manifest, tmp_path)
    advice_ids = {item["rating_id"] for item in packet["answers"] if item["category"] == "advice"}
    advice = next(item for item in ratings["ratings"] if item["rating_id"] in advice_ids)
    if mutation == "missing":
        ratings["ratings"].pop()
    elif mutation == "duplicate":
        ratings["ratings"].append(deepcopy(ratings["ratings"][0]))
    elif mutation == "unknown":
        ratings["ratings"][0]["rating_id"] = "not-a-generated-answer"
    elif mutation == "bool_score":
        advice["fidelity"] = True
    elif mutation == "wrong_hash":
        ratings["results_sha256"] = "altered"
    else:
        advice["rationale"] = ""
    with pytest.raises(ValueError):
        score_results(manifest, results, key, [ratings])


def test_mention_only_and_invented_autobiography_never_pass_primary_metric(manifest, tmp_path):
    results, packet, key, ratings = completed_with_ratings(manifest, tmp_path)
    advice_ids = {item["rating_id"] for item in packet["answers"] if item["category"] == "advice"}
    for index, rating in enumerate(item for item in ratings["ratings"] if item["rating_id"] in advice_ids):
        if index % 2:
            rating["application"] = 0
        else:
            rating["unsupported_autobiography"] = True
    summary = score_results(manifest, results, key, [ratings])
    assert all(summary["conditions"][condition]["advice_value_application_accuracy"] == 0 for condition in CONDITIONS)


def test_answer_mutation_rejected_on_resume_and_rating(manifest, tmp_path):
    results, _, key, ratings = completed_with_ratings(manifest, tmp_path)
    result = next(iter(results["attempts"].values()))
    result["raw_text"] = "tampered"
    atomic_json(tmp_path / "results.json", results)
    with pytest.raises(ValueError, match="contents changed"):
        run_manifest(manifest, tmp_path / "results.json", provider_fixture)
    with pytest.raises(ValueError, match="contents changed"):
        score_results(manifest, results, key, [ratings])


def test_paired_bootstrap_is_case_based_reproducible_and_handles_no_cases():
    first = paired_bootstrap([1, 0, -1, 1])
    assert first == paired_bootstrap([1, 0, -1, 1])
    assert first["n_cases"] == 4
    assert first["delta"] == 0.25
    assert first["ci95"][0] <= first["delta"] <= first["ci95"][1]
    assert paired_bootstrap([])["delta"] is None


def test_exact_token_counter_uses_server_window_and_caches_identical_prompts(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"prompt_tokens": 37, "context_window": 8192}

    def post(url, *, json, timeout):
        calls.append((url, json))
        return Response()

    monkeypatch.setattr("tools.persona_graph_eval.httpx.post", post)
    counter = HTTPTokenCounter("http://127.0.0.1/count_tokens", model="test-model", revision="test-revision")
    messages = [{"role": "user", "content": "test exact token count"}]
    assert counter.count(messages) == counter.count(messages) == 37
    assert counter.context_window == 8192
    assert counter.exact
    assert len(calls) == 2  # Initial window check plus one unique prompt.
    assert "test-revision" in counter.identity


def test_team_dialect_payload_matches_the_actual_team_service_schema(manifest):
    from services.language.schemas import CompletionRequest

    configuration = deepcopy(manifest["configuration"])
    configuration["provider_dialect"] = "team-language"
    configuration["decode"] = {"temperature": 0, "top_p": 1, "max_tokens": 256, "seed": 42}
    payload = completion_payload([{"role": "user", "content": "test"}], configuration)
    parsed = CompletionRequest.model_validate(payload)
    assert parsed.temperature == 0 and parsed.seed == 42
    assert "chat_template_kwargs" not in payload and "presence_penalty" not in payload
    configuration["decode"]["chat_template_kwargs"] = {"enable_thinking": False}
    with pytest.raises(ValueError, match="unsupported"):
        completion_payload([], configuration)
    configuration["provider_dialect"] = "unknown"
    with pytest.raises(ValueError, match="Unknown"):
        completion_payload([], configuration)


def test_team_dialect_requires_verified_non_thinking_and_matching_model(manifest):
    configuration = manifest["configuration"]
    health = {"model": configuration["model"], "identity": {
        "base_revision": configuration["revision"], "chat_template_options": {"enable_thinking": False}}}
    assert verify_team_contract(health, configuration)["chat_template_options"] == {"enable_thinking": False}
    health["identity"]["chat_template_options"]["enable_thinking"] = True
    with pytest.raises(ValueError, match="non-thinking"):
        verify_team_contract(health, configuration)


def test_different_complete_path_fallbacks_are_rejected_as_unmatched():
    from bias.corpus import Corpus
    from bias.persona_graph import PersonaGraph
    from bias.snapshots import PersonaSnapshot

    class UnequalRepresentationCounter:
        identity = "synthetic-budget-test-only"
        exact = False

        def count(self, messages):
            context = json.loads(messages[1]["content"].split("\n", 1)[1])
            memory = context.get("persona_memory", {})
            values = sum(node["kind"] == "value" for node in memory.get("nodes", []))
            # Makes two flat paths fit, while only one complete typed path fits.
            return 500 + values * (2000 if "relations" in memory else 700)

    corpus = Corpus.load(ROOT / "data/personas/jensen-huang")
    graph = PersonaGraph.load(ROOT / "data/personas/jensen-huang/persona_graph.v1.json", corpus)
    snapshot = PersonaSnapshot.load(ROOT / "data/snapshots/jensen-huang-2026-11-12.json", corpus)
    with pytest.raises(ValueError, match="included value paths differ"):
        prepare_case(corpus, snapshot, graph,
            "How should I delegate to a junior engineer and test a prototype?",
            context_window=4000, counter=UnequalRepresentationCounter())
