"""Synthetic collected outputs only: no real sources, model or network calls."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.adaptation_evaluation.blind_review import main, package_reviews
from tools.adaptation_evaluation.protocol import CONDITIONS, PREFIX, SCHEMA, digest, implementation_hashes, validate_plan
from tools.adaptation_evaluation.run import RATINGS, _expected_request, compare_reports, export_review


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(path: Path):
    return json.loads(path.read_text())


def synthetic_plan(exposure="development", *, empty_second=None):
    cases = []
    for index in (1, 2):
        fact = {"id": f"synthetic-fact-{index}", "citation": f"[source:synthetic-fact-{index}]",
                "text": f"Synthetic factual passage {index}.\nFull second line.", "published_at": "2020-01-01"}
        style = {"example_id": f"synthetic-style-{index}", "question": "A synthetic style question.",
                 "answer": "A complete synthetic wording example.", "role": "wording_only_not_factual_support"}
        question = f"Explain synthetic mechanism {index}."
        facts = [] if index == 2 and empty_second in {"facts", "all"} else [fact]
        styles = [] if index == 2 and empty_second == "all" else [style]
        contexts = {"facts": {"factual_evidence": facts, "style_examples": []},
                    "style": {"factual_evidence": facts, "style_examples": styles}}
        messages = {condition: [{"role": "system", "content": "Use supplied facts and preserve uncertainty."},
            {"role": "user", "content": PREFIX + json.dumps(context, ensure_ascii=False, separators=(",", ":"))},
            {"role": "user", "content": question}] for condition, context in contexts.items()}
        messages["adapter"] = copy.deepcopy(messages["style"])
        reference = f"PRIVATE_REFERENCE_{exposure}_{index}"
        cases.append({"id": f"case-{index}", "question": question, "language": "en",
            "event_family_id": "synthetic-heldout", "reference_answer": reference,
            "reference_sha256": digest(reference), "factual_ids": [source["id"] for source in facts],
            "style_ids": [source["example_id"] for source in styles],
            "factual_sha256": digest(facts), "style_contrast_present": bool(styles),
            "messages": messages, "prompt_sha256": {condition: digest(prompt) for condition, prompt in messages.items()}})
    plan = {"schema_version": SCHEMA, "id": "synthetic-" + exposure, "exposure": exposure,
        "implementation_sha256": implementation_hashes(), "cases": cases,
        "base_identity": {"base_model": "synthetic/base", "base_revision": "b" * 40},
        "policy": {"conditions": list(CONDITIONS), "max_tokens": 160, "context_window": 4096, "seed": 42}}
    plan["plan_sha256"] = digest(plan)
    validate_plan(plan)
    return plan


def synthetic_model(condition):
    identity = {"base_model": "synthetic/base", "base_revision": "b" * 40,
        "base_checkpoint_manifest_sha256": "c" * 64, "tokenizer_contract_sha256": "d" * 64,
        "chat_template_options": {"enable_thinking": False}}
    adapter = {"configured": condition == "adapter", "loaded": condition == "adapter", "verified": condition == "adapter",
        "run_sha256": "a" * 64, "artifact_manifest_sha256": "e" * 64, "dataset_sha256": "f" * 64,
        "training_event_families": ["synthetic-training"], "optimizer_steps": 1, **identity}
    runtime = {"device": "cpu", "dtype": "float32", "backend": "synthetic", "context_window": 4096,
        "threads": 1, "attention": "synthetic", "base_parameters": 1000,
        "versions": {"torch": "synthetic", "transformers": "synthetic"},
        "generation_defaults": {"seed": 42, "top_k": 20, "temperature": 0.0, "top_p": 1.0}}
    return {"identity": identity, "adapter": adapter, "runtime": runtime}


def make_collection(directory: Path, *, exposure="development", failed=False, unattempted=False, empty_second=None):
    directory.mkdir(parents=True)
    plan = synthetic_plan(exposure, empty_second=empty_second)
    plan_path = directory.parent / (exposure + "-plan.json")
    write(plan_path, plan)
    reports = []
    for condition in CONDITIONS:
        rows = []
        for index, case in enumerate(plan["cases"]):
            if unattempted and condition == "style" and index == 1:
                continue
            cut = failed and condition == "adapter" and index == 0
            # Long Unicode/multiline text checks that packaging never clips or normalizes output.
            answer = (f"Synthetic {CONDITIONS.index(condition)} / {index}: 한글\n" * 50) + ",".join(case["factual_ids"])
            generated = {"text": answer, "response_model": "synthetic-served-alias", "completed": not cut,
                "error": "provider_finish_length" if cut else None, "finish_reason": "length" if cut else "stop",
                "wall_ms": 1, "request_sha256": digest(_expected_request(plan, condition, "synthetic-served-alias", case))}
            rows.append({"case_id": case["id"], "condition": condition,
                "prompt_sha256": case["prompt_sha256"][condition], "factual_sha256": case["factual_sha256"],
                "generation": copy.deepcopy(generated), "provider_generation": generated,
                "identity_check": {"before": "verified", "after": "verified"}, "validation_errors": []})
        report = {"schema_version": "text-adaptation-condition-run.v1", "condition": condition,
            "plan_sha256": plan["plan_sha256"], "policy": plan["policy"], "model": synthetic_model(condition),
            "served_model": "synthetic-served-alias", "requested_adapter_run_sha256": "a" * 64 if condition == "adapter" else None,
            "planned_case_ids": [case["id"] for case in plan["cases"]], "runs": rows,
            "status": "deadline_exhausted" if unattempted and condition == "style" else "completed"}
        report["run_sha256"] = digest(report)
        reports.append(report)
    summary = compare_reports(plan, reports)
    review = export_review(plan, reports, directory / "blind-review")
    summary["local_execution"] = {"contract_sha256": "1" * 64, "durability_receipts_file_sha256": "2" * 64,
        "supervisors": [{"condition": condition, "supervisor_sha256": "3" * 64} for condition in CONDITIONS]}
    write(directory / "comparison.json", summary)
    write(directory / "review-summary.json", review)
    return plan_path, directory


def all_keys(value):
    if isinstance(value, dict):
        yield from value
        for child in value.values():
            yield from all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_keys(child)


def test_common_full_source_pool_blind_metadata_and_ungraded_fields(tmp_path):
    inputs = [make_collection(tmp_path / "development"), make_collection(tmp_path / "heldout", exposure="heldout")]
    before = {path: path.read_bytes() for plan, directory in inputs for path in [plan, *directory.rglob("*.json")]}
    output = tmp_path / "package"
    result = package_reviews(inputs, output)
    assert result["review_samples"] == 12 and result["source_pools"] == 4
    assert result["denominators"] == {"planned": 12, "attempted": 12, "completed": 12, "failed": 0, "unattempted": 0}
    samples = read(output / "reviewer/samples.json")["samples"]
    pools = read(output / "reviewer/source-pools.json")["pools"]
    key = read(output / "private/answer-key.json")["key"]
    references = read(output / "reviewer/reference-answers.json")["references"]
    for pool_id, pool in pools.items():
        members = [sample for sample in samples if sample["source_pool_id"] == pool_id]
        assert len(members) == 3
        assert {key[sample["id"]]["condition"] for sample in members} == set(CONDITIONS)
        assert pool["factual_evidence"][0]["text"].endswith("Full second line.")
        assert pool["style_examples"][0]["answer"] == "A complete synthetic wording example."
        for pointer in pool["evidence_pointers"]:
            node = {"pools": pools}
            for component in pointer["json_pointer"].lstrip("/").split("/"):
                node = node[int(component)] if isinstance(node, list) else node[component]
            assert digest(node) == pointer["source_sha256"]
        assert references[pool_id]["reference_answer"].startswith("PRIVATE_REFERENCE_")
    for sample in samples:
        assert len(sample["answer"]) > 800 and "한글\n" in sample["answer"]
        assert hashlib.sha256(sample["answer"].encode()).hexdigest() == key[sample["id"]]["output_sha256"]
        assert sample["ratings"] == {name: None for name in RATINGS}
        assert sample["reviewer"] is None and sample["reviewer_type"] is None
    for path in (output / "reviewer").glob("*.json"):
        assert not {"condition", "model", "runtime", "request_sha256", "original_sample_id", "output_sha256"} & set(all_keys(read(path)))
        assert "synthetic-served-alias" not in path.read_text()
    assert "PRIVATE_REFERENCE_" not in (output / "reviewer/samples.json").read_text()
    assert "PRIVATE_REFERENCE_" not in (output / "reviewer/source-pools.json").read_text()
    assert all(path.read_bytes() == content for path, content in before.items())
    assert all("PRIVATE_REFERENCE_" not in json.dumps(case["messages"]) for plan, _ in inputs for case in read(plan)["cases"])
    assert result["human_review"] == "ungraded" and result["efficacy_claim"] is False
    for name, expected in result["files_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected


def test_length_failure_and_unattempted_case_keep_denominators(tmp_path):
    pair = make_collection(tmp_path / "collection", failed=True, unattempted=True)
    output = tmp_path / "package"
    result = package_reviews([pair], output)
    assert result["denominators"] == {"planned": 6, "attempted": 5, "completed": 4, "failed": 1, "unattempted": 1}
    assert result["review_samples"] == 4
    diagnostic = read(output / "private/execution-diagnostics.json")["inputs"][0]
    assert diagnostic["failed_attempts"] == [{"case_id": "case-1", "condition": "adapter", "error": "provider_finish_length"}]
    assert diagnostic["unattempted_cases"] == [{"condition": "style", "case_id": "case-2"}]


def test_changed_failure_reason_is_not_relabelled_as_verified(tmp_path):
    pair = make_collection(tmp_path / "collection", failed=True)
    path = pair[1] / "blind-review/private/failed-attempts.json"
    failures = read(path)
    failures[0]["error"] = "different_failure"
    write(path, failures)
    with pytest.raises(ValueError, match="denominators"):
        package_reviews([pair], tmp_path / "package")


@pytest.mark.parametrize("empty_second", ["facts", "all"])
def test_empty_facts_and_absent_style_contrast_remain_explicit(tmp_path, empty_second):
    pair = make_collection(tmp_path / "collection", empty_second=empty_second)
    output = tmp_path / "package"
    package_reviews([pair], output)
    pools = read(output / "reviewer/source-pools.json")["pools"]
    pool_id, pool = next((key, value) for key, value in pools.items() if value["case_id"] == "case-2")
    assert pool["factual_evidence"] == []
    assert bool(pool["style_examples"]) == (empty_second == "facts")
    assert all(pointer["role"] == "style_examples" for pointer in pool["evidence_pointers"])
    samples = read(output / "reviewer/samples.json")["samples"]
    assert len([sample for sample in samples if sample["source_pool_id"] == pool_id]) == 3


@pytest.mark.parametrize("change", ["answer", "evidence", "reference", "human_rating", "missing_key", "extra_key", "duplicate_pair", "missing_sample"])
def test_rejects_changed_or_incomplete_sample_package(tmp_path, change):
    pair = make_collection(tmp_path / "collection")
    samples_path = pair[1] / "blind-review/reviewer/samples.json"
    key_path = pair[1] / "blind-review/private/answer-key.json"
    samples, key = read(samples_path), read(key_path)
    first, second = samples["samples"][:2]
    if change == "answer":
        first["answer"] += " altered"
    elif change == "evidence":
        first["factual_evidence"][0]["text"] = "changed"
    elif change == "reference":
        first["reference_answer"] = "changed"
    elif change == "human_rating":
        first["ratings"][RATINGS[0]] = 5
    elif change == "missing_key":
        key.pop(first["id"])
    elif change == "extra_key":
        key["unknown-sample"] = copy.deepcopy(key[first["id"]])
    elif change == "duplicate_pair":
        key[second["id"]] = copy.deepcopy(key[first["id"]])
    else:
        samples["samples"].pop()
    write(samples_path, samples)
    write(key_path, key)
    with pytest.raises(ValueError):
        package_reviews([pair], tmp_path / "package")
    assert not (tmp_path / "package").exists()


@pytest.mark.parametrize("change", ["not_collected", "count", "output_hash", "foreign_case", "running"])
def test_rejects_uncollected_or_inconsistent_denominators(tmp_path, change):
    pair = make_collection(tmp_path / "collection")
    summary_path = pair[1] / "comparison.json"
    private_path = pair[1] / "blind-review/private/comparison.json"
    summary = read(summary_path)
    if change == "not_collected":
        summary.pop("local_execution")
    elif change == "count":
        summary["conditions"]["facts"]["completed"] = 99
    elif change == "output_hash":
        summary["conditions"]["facts"]["output_sha256"]["case-1"] = "0" * 64
    elif change == "foreign_case":
        summary["conditions"]["facts"]["output_sha256"]["foreign-case"] = "0" * 64
    else:
        summary["conditions"]["facts"]["run_status"] = "running"
    write(summary_path, summary)
    write(private_path, {key: value for key, value in summary.items() if key != "local_execution"})
    with pytest.raises(ValueError):
        package_reviews([pair], tmp_path / "package")


def test_cli_refuses_duplicate_inputs_and_never_overwrites(tmp_path, capsys):
    pair = make_collection(tmp_path / "collection")
    output = tmp_path / "package"
    with pytest.raises(ValueError, match="only once"):
        package_reviews([pair, pair], output)
    assert main(["--input", *map(str, pair), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["review_samples"] == 6
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["--input", *map(str, pair), "--output", str(output)])
    assert exc.value.code == 2 and (output / "manifest.json").read_bytes() == before
