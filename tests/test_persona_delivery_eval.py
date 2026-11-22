"""Evaluation integrity tests with synthetic responses, not performance results."""
from copy import deepcopy
import json

import pytest

from bias.corpus import Corpus
from bias.persona_delivery import PersonaDelivery
from bias.persona_graph import PersonaGraph
from bias.snapshots import PersonaSnapshot
from tools import persona_delivery_eval as evaluation
from tools import persona_graph_eval as shared


class FixtureCounter:
    """Cheap deterministic packing fixture; never an exact model tokenizer."""
    context_window = 8192
    exact = False
    identity = "test-only-character-estimator"

    def count(self, messages):
        return len(json.dumps(messages, ensure_ascii=False)) // 4 + 1


@pytest.mark.parametrize("artifact", ["manifest.json", "results.json"])
def test_freeze_never_overwrites_a_previous_experiment(tmp_path, monkeypatch, artifact):
    monkeypatch.setattr(evaluation, "REPORT", tmp_path)
    path = tmp_path / artifact
    original = b'{"preserve": "previous measured experiment"}\n'
    path.write_bytes(original)
    with pytest.raises(ValueError, match="Frozen artifacts already exist"):
        evaluation.freeze(FixtureCounter())
    assert path.read_bytes() == original


def context(messages):
    return json.loads(messages[1]["content"].split("\n", 1)[1])


def synthetic_manifest():
    """Small complete design with one intentional invalid advice per condition."""
    cases = [
        {"id": "valid-advice", "question": "How should I provide useful feedback?", "category": "advice", "probe_type": "values"},
        {"id": "invalid-advice", "question": "How should I make this other decision?", "category": "advice", "probe_type": "values"},
        {"id": "style", "question": "Explain this idea clearly.", "category": "control", "probe_type": "style"},
        {"id": "private-control", "question": "What did I tell you in private?", "category": "control", "probe_type": "factual"},
    ]
    for case in cases:
        case.update(reference_evidence_ids=[], rubric={"application": "Judge the case boundary."}, language="en")
    config = {"model": "test-fixture-not-a-real-model", "provider_dialect": "recorded-gguf",
              "decode": {"temperature": 0, "max_tokens": 32}}
    prompts, schedule = {}, []
    for case in cases:
        for condition in evaluation.CONDITIONS:
            key = f"{case['id']}:{condition}"
            messages = [{"role": "user", "content": case["question"]}]
            prompt = {"messages": messages, "supplied_ids": [], "budget": {}, "speculative": False,
                      "policy_guard": "private archive limit" if case["id"] == "private-control" else None,
                      "deterministic_text": "I cannot verify a private interaction.",
                      "prompt_sha256": shared.digest(messages)}
            prompts[key] = prompt
            schedule.append({"attempt_id": f"{case['id']}:r0:{condition}", "case_id": case["id"],
                "condition": condition, "repeat": 0, "prompt_key": key,
                "request_sha256": shared.digest({"prompt": prompt, "configuration": config})})
    manifest = {"schema_version": shared.SCHEMA, "design_version": "matched-delivery.v1",
                "cases": cases, "conditions": list(evaluation.CONDITIONS), "configuration": config,
                "prompts": prompts, "schedule": schedule, "judge_sources": {},
                "rubric": {"scale": "0, 1 or 2; test fixture only"}}
    manifest["manifest_sha256"] = shared.digest(manifest)
    return manifest


class FixtureProvider:
    def __init__(self):
        self.calls = []

    def __call__(self, messages, configuration):
        self.calls.append(deepcopy(messages))
        invalid = "other decision" in messages[-1]["content"]
        return {"raw_text": "Synthetic fixture response; this is not a model evaluation.",
                "finish_reason": "length" if invalid else "stop", "model": configuration["model"], "usage": {}}


@pytest.fixture
def completed_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "REPORT", tmp_path)
    manifest = synthetic_manifest()
    results = shared.run_manifest(manifest, tmp_path / "results.json", FixtureProvider())
    evaluation.blind(manifest, results)
    packet = shared.read_json(tmp_path / "blind-packet.json")
    rating_paths = []
    for name in ("fixture-judge-a.json", "fixture-judge-b.json"):
        path = tmp_path / name
        shared.atomic_json(path, {"rater_id": name, "ratings": [
            {"rating_id": item["rating_id"], **{field: 2 for field in evaluation.FIELDS},
             "safe": True, "reason": "Synthetic all-pass fixture; not an assessment of generated answers."}
            for item in packet["answers"]]})
        rating_paths.append(path)
    return manifest, results, packet, rating_paths


def test_production_conditions_have_exactly_matched_factual_records():
    corpus = Corpus.load(evaluation.CORPUS)
    snapshot = PersonaSnapshot.load(evaluation.ROOT / "data/snapshots/jensen-values-2026-11-21.json", corpus)
    graph = PersonaGraph.load(evaluation.CORPUS / "persona_graph.v2.json", corpus)
    delivery = PersonaDelivery.load(evaluation.CORPUS / "persona_delivery.v1.json", corpus)
    question = "How should I give feedback to a junior intern and explain why after a mistake?"
    prompts, traces = evaluation.prepare(corpus, snapshot, graph, delivery, question, counter=FixtureCounter())
    contexts = {key: context(value["messages"]) for key, value in prompts.items()}
    factual = contexts["matched_evidence"]["factual_evidence"]
    assert factual  # Not a vacuous equality with three empty retrieval results.
    assert all(item["factual_evidence"] == factual for item in contexts.values())
    assert len({tuple(p["supplied_ids"]) for p in prompts.values()}) == 1
    assert "persona_memory" not in contexts["matched_evidence"]
    assert contexts["compact_values"]["persona_memory"] == contexts["compact_delivery"]["persona_memory"]
    assert "communication_guidance" not in contexts["compact_values"]
    assert contexts["compact_delivery"]["communication_guidance"]
    assert traces["values"]["rule_ids"]


def test_freeze_keeps_judge_only_rubric_out_of_every_generator_prompt(tmp_path, monkeypatch):
    registry = shared.read_json(evaluation.REPORT / "registry.json")
    sentinel = "EVALUATOR_ONLY_CANARY_94270"
    for case in registry["cases"]:
        case["rubric"]["application"] += " " + sentinel
    registry["rubric"]["test_canary"] = sentinel
    shared.atomic_json(tmp_path / "registry.json", registry)
    monkeypatch.setattr(evaluation, "REPORT", tmp_path)
    manifest = evaluation.freeze(FixtureCounter())
    assert sentinel in json.dumps(manifest["cases"])
    for prompt in manifest["prompts"].values():
        assert sentinel not in json.dumps(prompt["messages"])
        assert "case_rubric" not in json.dumps(prompt["messages"])


def test_blinding_exposes_neither_conditions_nor_prompts(completed_fixture, tmp_path):
    manifest, results, packet, _ = completed_fixture
    encoded = json.dumps(packet)
    assert all(condition not in encoded for condition in evaluation.CONDITIONS)
    assert "prompt_key" not in encoded and "prompt_sha256" not in encoded
    assert "configuration" not in packet and "selections" not in packet
    assert all(set(item) == {"rating_id", "question", "answer", "category", "case_rubric", "reference_sources"}
               for item in packet["answers"])
    assert len(packet["answers"]) == 9  # Three invalid completions never get rated as successes.
    mapping = shared.read_json(tmp_path / "condition-key.private.json")
    assert set(mapping.values()) == set(results["attempts"])
    assert len(mapping) == len(results["attempts"])


def test_invalid_completions_are_zero_and_remain_in_advice_denominator(completed_fixture):
    manifest, results, _, ratings = completed_fixture
    summary = evaluation.summarize(manifest, results, ratings)
    for condition in evaluation.CONDITIONS:
        row = summary["conditions"][condition]
        assert row["attempts"] == 4
        assert row["n_advice"] == 2
        assert row["valid_model_completions"] == 2
        assert row["deterministic_guards"] == 1
        assert row["failed_or_truncated"] == 1
        assert row["conditional_advice_pass_rate"] == 0.5
        assert row["both_judges_advice_passes"] == 1
        assert set(row["advice_dimension_means_0_to_2"].values()) == {1.0}


def test_summary_rejects_missing_attempt_even_if_old_packet_and_ratings_exist(completed_fixture):
    manifest, results, _, ratings = completed_fixture
    changed = deepcopy(results)
    del changed["attempts"]["invalid-advice:r0:matched_evidence"]
    with pytest.raises(ValueError):
        evaluation.summarize(manifest, changed, ratings)


def test_summary_rejects_swapped_condition_mapping(completed_fixture, tmp_path):
    manifest, results, _, ratings = completed_fixture
    mapping_path = tmp_path / "condition-key.private.json"
    mapping = shared.read_json(mapping_path)
    keys = [key for key, value in mapping.items() if value.startswith("valid-advice:")]
    mapping[keys[0]], mapping[keys[1]] = mapping[keys[1]], mapping[keys[0]]
    shared.atomic_json(mapping_path, mapping)
    with pytest.raises(ValueError):
        evaluation.summarize(manifest, results, ratings)


def test_summary_rejects_changed_blind_packet_answer(completed_fixture, tmp_path):
    manifest, results, packet, ratings = completed_fixture
    packet["answers"][0]["answer"] = "Different text never presented by the fixture provider."
    shared.atomic_json(tmp_path / "blind-packet.json", packet)
    with pytest.raises(ValueError):
        evaluation.summarize(manifest, results, ratings)


@pytest.mark.parametrize("field,changed", [
    ("case_id", "style"),
    ("condition", "compact_values"),
    ("repeat", 1),
    ("prompt_key", "style:matched_evidence"),
    ("prompt_sha256", "0" * 64),
])
def test_blinding_rejects_rehashed_result_labels_that_disagree_with_schedule(completed_fixture, field, changed):
    manifest, results, _, _ = completed_fixture
    modified = deepcopy(results)
    row = modified["attempts"]["valid-advice:r0:matched_evidence"]
    row[field] = changed
    row["result_sha256"] = shared.digest({key: value for key, value in row.items() if key != "result_sha256"})
    with pytest.raises(ValueError):
        evaluation.blind(manifest, modified)


def test_summary_requires_exactly_two_raters(completed_fixture):
    manifest, results, _, ratings = completed_fixture
    with pytest.raises(ValueError):
        evaluation.summarize(manifest, results, ratings[:1])
    with pytest.raises(ValueError):
        evaluation.summarize(manifest, results, ratings + ratings[:1])


@pytest.mark.parametrize("identity", [None, "", " ", "fixture-judge-a.json"])
def test_summary_rejects_missing_blank_or_duplicate_rater_identity(completed_fixture, identity):
    manifest, results, _, ratings = completed_fixture
    second = shared.read_json(ratings[1])
    if identity is None:
        second.pop("rater_id")
    else:
        second["rater_id"] = identity
    shared.atomic_json(ratings[1], second)
    with pytest.raises(ValueError):
        evaluation.summarize(manifest, results, ratings)


def test_summary_rejects_duplicate_rating_path_or_exact_file_copy(completed_fixture, tmp_path):
    manifest, results, _, ratings = completed_fixture
    with pytest.raises(ValueError):
        evaluation.summarize(manifest, results, [ratings[0], ratings[0]])
    copied = tmp_path / "different-name-same-rating-file.json"
    copied.write_bytes(ratings[0].read_bytes())
    with pytest.raises(ValueError):
        evaluation.summarize(manifest, results, [ratings[0], copied])


def test_resume_uses_exact_saved_requests_and_never_retries_invalid_attempt(tmp_path):
    manifest, provider = synthetic_manifest(), FixtureProvider()
    path = tmp_path / "partial.json"
    first = shared.run_manifest(manifest, path, provider, limit=2)
    before = deepcopy(first["attempts"])
    assert len(before) == 2 and len(provider.calls) == 2
    finished = shared.run_manifest(manifest, path, provider)
    assert len(provider.calls) == 9  # Guards do not invoke provider.
    assert len(finished["attempts"]) == len(manifest["schedule"])
    assert all(finished["attempts"][key] == value for key, value in before.items())
    assert shared.run_manifest(manifest, path, provider) == finished
    assert len(provider.calls) == 9


@pytest.mark.parametrize("corruption", ["result_text", "request_identity", "manifest"])
def test_resume_rejects_corrupted_identity_before_calling_provider(tmp_path, corruption):
    manifest, provider = synthetic_manifest(), FixtureProvider()
    path = tmp_path / "partial.json"
    result = shared.run_manifest(manifest, path, provider, limit=1)
    first = next(iter(result["attempts"].values()))
    if corruption == "result_text":
        first["raw_text"] = "Unrecorded replacement."
    elif corruption == "request_identity":
        first["request_sha256"] = "0" * 64
        first["result_sha256"] = shared.digest({k: v for k, v in first.items() if k != "result_sha256"})
    else:
        manifest["configuration"]["decode"]["temperature"] = 1
    shared.atomic_json(path, result)
    with pytest.raises(ValueError):
        shared.run_manifest(manifest, path, provider)
    assert len(provider.calls) == 1
