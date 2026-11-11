"""Synthetic receipts exercise integrity gates; they are not trained models."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from services.language.adapters import (
    CHAT_TEMPLATE_OPTIONS, AdapterArtifactError, RUN_SCHEMA, artifact_manifest,
    checkpoint_manifest, file_digest, json_digest, tokenizer_contract, validate_adapter_run,
)
from tools.adaptation.train import encode_example
from tools.adaptation.datasets import read_projection, validate_dataset
from services.language.app import create_app
from services.language.engine import LanguageEngine, LanguageError, LanguageSettings, TransformersBackend


MODEL = "example/causal-model"
REVISION = "a" * 40


class ToyTokenizer:
    chat_template = "explicit test template"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None
    unk_token_id = None
    all_special_ids = [1, 2]

    def get_vocab(self):
        return {"begin": 1, "end": 2, "word": 3}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is True
        assert enable_thinking is False
        return [1, 3] if add_generation_prompt else [1, 3, 3, 2]


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def make_run(tmp_path):
    run = tmp_path / "completed-run"
    (run / "adapter").mkdir(parents=True)
    (run / "tokenizer").mkdir()
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": MODEL,
              "revision": REVISION, "inference_mode": True, "r": 8, "lora_alpha": 16, "lora_dropout": 0.05}
    write_json(run / "adapter/adapter_config.json", config)
    (run / "adapter/adapter_model.safetensors").write_bytes(b"synthetic-not-model-weights")
    write_json(run / "tokenizer/tokenizer_config.json", {"chat_template": ToyTokenizer.chat_template})
    write_json(run / "tokenizer/tokenizer.json", {"vocab": ToyTokenizer().get_vocab()})
    base = tmp_path / "base"
    base.mkdir()
    write_json(base / "config.json", {"model_type": "test"})
    (base / "model.safetensors").write_bytes(b"synthetic-not-base-weights")
    contract = tokenizer_contract(ToyTokenizer())
    report = {"schema_version": RUN_SCHEMA, "status": "completed", "base_model": MODEL,
              "base_revision": REVISION, "tokenizer_revision": REVISION, "optimizer_steps": 2,
              "dataset_manifest": {"schema_version": "text-adaptation-dataset.v1", "dataset_sha256": "b" * 64,
                                   "samples": 8, "test_or_validation_samples": 0, "event_counts": {"train-event": 8}},
              "tokenizer_contract": contract, "tokenizer_contract_sha256": json_digest(contract),
              "adapter": {"type": "LoRA", "rank": 8, "alpha": 16, "dropout": 0.05},
              "base_checkpoint_sha256": checkpoint_manifest(base), "artifact_sha256": artifact_manifest(run)}
    write_json(run / "run.json", report)
    return run, base, report


def validate(run, **kwargs):
    return validate_adapter_run(run, base_model=MODEL, base_revision=REVISION, **kwargs)


def test_completed_run_binds_every_artifact_base_and_tokenizer_without_paths(tmp_path):
    run, base, report = make_run(tmp_path)
    verified = validate(run, expected_run_sha256=file_digest(run / "run.json"))
    assert verified.verify_base_checkpoint(base) == report["base_checkpoint_sha256"]
    assert verified.verify_tokenizer(ToyTokenizer()) == report["tokenizer_contract"]
    identity = verified.public_identity()
    assert identity["base_revision"] == REVISION
    assert identity["artifact_manifest_sha256"] == json_digest(artifact_manifest(run))
    assert identity["efficacy_evaluated"] is False
    assert identity["training_event_families"] == ["train-event"]
    assert str(tmp_path) not in json.dumps(identity)


@pytest.mark.parametrize("status", ["prepared", "running", "failed", None])
def test_unfinished_run_cannot_be_loaded(tmp_path, status):
    run, _, report = make_run(tmp_path)
    report["status"] = status
    write_json(run / "run.json", report)
    with pytest.raises(AdapterArtifactError, match="completed"):
        validate(run)


def test_legacy_run_requires_retraining_not_receipt_relabeling(tmp_path):
    run, _, report = make_run(tmp_path)
    report["schema_version"] = "text-adaptation-run.v1"
    write_json(run / "run.json", report)
    with pytest.raises(AdapterArtifactError, match="do not relabel"):
        validate(run)


@pytest.mark.parametrize("change", ["modified", "missing", "extra"])
def test_changed_missing_or_unregistered_artifacts_fail(tmp_path, change):
    run, _, _ = make_run(tmp_path)
    path = run / "adapter/adapter_model.safetensors"
    if change == "modified":
        path.write_bytes(b"different weights")
    elif change == "missing":
        path.unlink()
    else:
        (run / "tokenizer/unregistered.json").write_text("{}")
    with pytest.raises(AdapterArtifactError, match="changed|missing|unregistered"):
        validate(run)


@pytest.mark.parametrize("name", ["../secret", "adapter/../secret", "adapter//weights", "/adapter/weights", "adapter\\weights", "tokenizer/./config.json"])
def test_receipt_paths_cannot_escape_or_alias_package_files(tmp_path, name):
    run, _, report = make_run(tmp_path)
    report["artifact_sha256"][name] = "b" * 64
    write_json(run / "run.json", report)
    with pytest.raises(AdapterArtifactError, match="paths"):
        validate(run)


@pytest.mark.parametrize("target", ["run.json", "adapter/adapter_model.safetensors", "tokenizer"])
def test_local_receipts_and_packages_reject_symlinks(tmp_path, target):
    run, _, _ = make_run(tmp_path)
    path = run / target
    outside = tmp_path / "outside"
    path.rename(outside)
    path.symlink_to(outside, target_is_directory=outside.is_dir())
    with pytest.raises(AdapterArtifactError, match="regular|real|symlinks"):
        validate(run)


def test_pinned_receipt_and_base_revision_cannot_change(tmp_path):
    run, _, _ = make_run(tmp_path)
    with pytest.raises(AdapterArtifactError, match="receipt SHA-256"):
        validate(run, expected_run_sha256="0" * 64)
    for revision in ("main", "c" * 40, None):
        with pytest.raises(AdapterArtifactError, match="base repository"):
            validate_adapter_run(run, base_model=MODEL, base_revision=revision)
    with pytest.raises(AdapterArtifactError, match="base repository"):
        validate_adapter_run(run, base_model="other/model", base_revision=REVISION)


def test_peft_config_must_match_even_if_its_file_hash_is_updated(tmp_path):
    run, _, report = make_run(tmp_path)
    path = run / "adapter/adapter_config.json"
    config = json.loads(path.read_text())
    config["revision"] = "c" * 40
    write_json(path, config)
    report["artifact_sha256"] = artifact_manifest(run)
    write_json(run / "run.json", report)
    with pytest.raises(AdapterArtifactError, match="adapter config"):
        validate(run)


def test_training_evidence_and_template_cannot_be_omitted(tmp_path):
    run, _, report = make_run(tmp_path)
    report["optimizer_steps"] = 0
    write_json(run / "run.json", report)
    with pytest.raises(AdapterArtifactError, match="optimizer steps"):
        validate(run)
    report["optimizer_steps"] = 2
    report["dataset_manifest"]["test_or_validation_samples"] = 1
    write_json(run / "run.json", report)
    with pytest.raises(AdapterArtifactError, match="training-only"):
        validate(run)
    report["dataset_manifest"]["test_or_validation_samples"] = 0
    report["tokenizer_contract"]["chat_template_options"] = {"enable_thinking": True}
    report["tokenizer_contract_sha256"] = json_digest(report["tokenizer_contract"])
    write_json(run / "run.json", report)
    with pytest.raises(AdapterArtifactError, match="tokenizer/template"):
        validate(run)


def test_base_weights_changed_behind_same_revision_are_rejected(tmp_path):
    run, base, _ = make_run(tmp_path)
    verified = validate(run)
    (base / "model.safetensors").write_bytes(b"wrong base model bytes")
    with pytest.raises(AdapterArtifactError, match="base checkpoint differs"):
        verified.verify_base_checkpoint(base)


@pytest.mark.parametrize("change", ["vocabulary", "template", "special_ids", "rendering"])
def test_tokenizer_compatibility_checks_actual_behavior(tmp_path, change):
    run, _, _ = make_run(tmp_path)
    altered = ToyTokenizer()
    if change == "vocabulary":
        altered.get_vocab = lambda: {"begin": 1, "end": 2, "different": 3}
    elif change == "template":
        altered.chat_template = "a changed template"
    elif change == "special_ids":
        altered.eos_token_id = 3
    else:
        altered.apply_chat_template = lambda *args, **kwargs: [1, 3, 2]
    with pytest.raises(AdapterArtifactError, match="tokenizer/template differs"):
        validate(run).verify_tokenizer(altered)


def test_training_and_runtime_use_same_nonreasoning_template_options():
    encoded = encode_example(ToyTokenizer(), [{"role": "user"}, {"role": "assistant"}], 10)
    assert CHAT_TEMPLATE_OPTIONS == {"enable_thinking": False}
    assert encoded["labels"] == [-100, -100, 3, 2]
    assert encode_example(ToyTokenizer(), [{"role": "user"}, {"role": "assistant"}], 3) is None


def test_duplicate_json_receipt_keys_are_rejected(tmp_path):
    run, _, _ = make_run(tmp_path)
    (run / "run.json").write_text('{"status":"completed","status":"failed"}')
    with pytest.raises(AdapterArtifactError, match="Duplicate JSON"):
        validate(run)


def test_hf_snapshot_blob_symlinks_are_supported_with_exact_hashes(tmp_path):
    run, base, _ = make_run(tmp_path)
    weights = base / "model.safetensors"
    blob = tmp_path / "hf-blob"
    weights.rename(blob)
    weights.symlink_to(blob)
    validate(run).verify_base_checkpoint(base)
    blob.write_bytes(b"changed cache blob")
    with pytest.raises(AdapterArtifactError, match="base checkpoint differs"):
        validate(run).verify_base_checkpoint(base)


def install_fake_model_modules(monkeypatch, base, *, changed_saved_tokenizer=False, peft_missing=False):
    """Capture adapter loading policy without pretending to execute neural work."""
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(use_cache=False, max_position_embeddings=8192))
    model.parameters = lambda: iter([SimpleNamespace(device="cpu", dtype="torch.float32", numel=lambda: 42)])
    model.to = lambda device: calls.append(("device", device)) or model
    model.eval = lambda: calls.append(("eval",)) or model

    def tokenizer(path, **kwargs):
        calls.append(("tokenizer", str(path), kwargs))
        value = ToyTokenizer()
        if changed_saved_tokenizer and str(path).endswith("/tokenizer"):
            value.chat_template = "wrong saved tokenizer"
        return value

    def load_model(path, **kwargs):
        calls.append(("base", str(path), kwargs))
        return model

    def load_adapter(selected_model, path, **kwargs):
        assert selected_model is model
        calls.append(("adapter", str(path), kwargs))
        return model

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32", set_num_threads=lambda _: None,
                         cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
                         AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model)))
    monkeypatch.setitem(sys.modules, "peft", None if peft_missing else SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=load_adapter)))
    monkeypatch.setattr("services.language.engine.model_location", lambda _: str(base))
    return calls, model


def test_service_loads_verified_adapter_frozen_on_matching_base(monkeypatch, tmp_path):
    run, base, _ = make_run(tmp_path)
    calls, model = install_fake_model_modules(monkeypatch, base)
    settings = LanguageSettings(model_id=MODEL, revision=REVISION, adapter_run=run)
    backend = TransformersBackend(settings)
    adapter_call = next(call for call in calls if call[0] == "adapter")
    assert adapter_call[1] == str(run / "adapter")
    assert adapter_call[2] == {"is_trainable": False, "local_files_only": True, "use_safetensors": True}
    tokenizers = [call for call in calls if call[0] == "tokenizer"]
    assert tokenizers[-1][1] == str(run / "tokenizer")
    assert tokenizers[-1][2] == {"local_files_only": True, "trust_remote_code": False}
    assert model.config.use_cache is True
    assert backend.adapter_identity["verified"] and backend.adapter_identity["loaded"]
    assert backend.runtime_identity["base_revision"] == REVISION
    assert backend.runtime["device"] == "cpu" and backend.runtime["dtype"] == "float32"
    assert backend.runtime["context_window"] == 8192
    assert backend.runtime["base_parameters"] == 42
    assert backend.runtime_identity["tokenizer_contract_sha256"] == backend.adapter_identity["tokenizer_contract_sha256"]
    assert str(tmp_path) not in json.dumps(backend.runtime_identity)


def test_base_only_service_exposes_the_same_control_identity(monkeypatch, tmp_path):
    run, base, _ = make_run(tmp_path)
    calls, _ = install_fake_model_modules(monkeypatch, base, peft_missing=True)
    backend = TransformersBackend(LanguageSettings(model_id=MODEL, revision=REVISION))
    adapter_identity = validate(run).public_identity()
    assert backend.runtime_identity["base_checkpoint_manifest_sha256"] == adapter_identity["base_checkpoint_manifest_sha256"]
    assert backend.runtime_identity["tokenizer_contract_sha256"] == adapter_identity["tokenizer_contract_sha256"]
    assert backend.adapter_identity == {"configured": False, "loaded": False, "verified": False}
    assert backend.runtime["base_parameters"] == 42
    assert not any(call[0] == "adapter" for call in calls)


def test_incompatible_saved_tokenizer_fails_before_loading_weights(monkeypatch, tmp_path):
    run, base, _ = make_run(tmp_path)
    calls, _ = install_fake_model_modules(monkeypatch, base, changed_saved_tokenizer=True)
    with pytest.raises(LanguageError) as error:
        TransformersBackend(LanguageSettings(model_id=MODEL, revision=REVISION, adapter_run=run))
    assert error.value.code == "adapter_incompatible"
    assert not any(call[0] in {"base", "adapter"} for call in calls)


def test_missing_optional_peft_is_explicit_and_never_silently_base_only(monkeypatch, tmp_path):
    run, base, _ = make_run(tmp_path)
    install_fake_model_modules(monkeypatch, base, peft_missing=True)
    with pytest.raises(LanguageError) as error:
        TransformersBackend(LanguageSettings(model_id=MODEL, revision=REVISION, adapter_run=run))
    assert error.value.code == "adapter_dependencies_missing"


def test_configured_adapter_is_lazy_and_clients_cannot_change_it(tmp_path, monkeypatch):
    run = tmp_path / "not-acquired"
    settings = LanguageSettings(model_id=MODEL, revision=REVISION, adapter_run=run)
    monkeypatch.setitem(sys.modules, "torch", None)
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health["adapter"] == {"configured": True, "loaded": False, "verified": False}
        assert health["identity"]["base_revision"] == REVISION
        assert health["identity"]["tokenizer_contract_sha256"] is None
        assert health["runtime"]["device"] is None and health["runtime"]["dtype"] is None
        assert str(run) not in json.dumps(health)
        request = {"model": settings.served_model, "messages": [{"role": "user", "content": "Hello"}]}
        assert client.post("/v1/chat/completions", json={**request, "adapter": "client-chosen"}).status_code == 422
        response = client.post("/v1/chat/completions", json=request)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "adapter_invalid"
        assert str(run) not in response.text


def test_settings_read_adapter_selection_only_from_server_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BIAS_LANGUAGE_MODEL", MODEL)
    monkeypatch.setenv("BIAS_LANGUAGE_REVISION", REVISION)
    monkeypatch.setenv("BIAS_LANGUAGE_ADAPTER_RUN", str(tmp_path))
    monkeypatch.setenv("BIAS_LANGUAGE_ADAPTER_RUN_SHA256", "b" * 64)
    settings = LanguageSettings.from_env()
    assert settings.adapter_run == tmp_path
    assert settings.adapter_run_sha256 == "b" * 64
    assert str(tmp_path) not in repr(settings)
    with pytest.raises(ValueError, match="immutable"):
        LanguageSettings(adapter_run=tmp_path, revision="main")
    with pytest.raises(ValueError, match="SHA-256"):
        LanguageSettings(adapter_run_sha256="b" * 64)


def test_evaluation_only_case_export_fails_with_explicit_split_boundary(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"schema_version": "persona-heldout-case.v1", "purpose": "evaluation_only", "split": "test"}) + "\n")
    with pytest.raises(ValueError, match="Evaluation-only held-out"):
        read_projection(path)


def test_cli_preserves_declared_low_memory_training_flags(monkeypatch, tmp_path):
    from tools.adaptation import __main__ as cli
    captured = []
    monkeypatch.setattr(cli, "train_text", lambda **kwargs: captured.append(kwargs) or {"status": "not_executed_test"})
    common = ["train-text", "--dataset", str(tmp_path / "data"), "--output", str(tmp_path / "run"),
              "--model", MODEL, "--revision", REVISION]
    assert cli.main(common) == 0
    assert captured[-1]["precision"] == "float32" and captured[-1]["gradient_checkpointing"] is False
    assert cli.main([*common, "--precision", "bfloat16", "--gradient-checkpointing", "--max-steps", "4"]) == 0
    assert captured[-1]["precision"] == "bfloat16" and captured[-1]["gradient_checkpointing"] is True
    assert captured[-1]["max_steps"] == 4


def text_dataset(tmp_path, *, rows=None):
    directory = tmp_path / "dataset"
    directory.mkdir()
    answer = "An original attributed answer."
    from tools.persona_data.parsers import sha_text
    row = {"id": "sample-one", "source_id": "publisher-source", "event_family_id": "actual-event", "split": "train",
           "answer_sha256": sha_text(answer), "messages": [{"role": "system", "content": "Disclosed simulation"},
           {"role": "user", "content": "Original question?"}, {"role": "assistant", "content": answer}]}
    (directory / "train.jsonl").write_text("".join(json.dumps(item) + "\n" for item in (rows if rows is not None else [row])))
    manifest = {"schema_version": "text-adaptation-dataset.v1", "dataset_sha256": file_digest(directory / "train.jsonl"),
                "samples": len(rows) if rows is not None else 1, "test_or_validation_samples": 0,
                "event_counts": {"actual-event": len(rows) if rows is not None else 1}}
    write_json(directory / "manifest.json", manifest)
    return directory, manifest, row


def test_dataset_family_claims_are_recomputed_before_training(tmp_path):
    directory, manifest, _ = text_dataset(tmp_path)
    validate_dataset(directory, "text")
    manifest["event_counts"] = {"unrelated-event": 1}
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="event-family counts"):
        validate_dataset(directory, "text")


@pytest.mark.parametrize("change", ["duplicate", "lineage_missing", "source_crosses_families", "heldout_count"])
def test_dataset_rows_cannot_conceal_duplicate_or_unknown_lineage(tmp_path, change):
    directory, manifest, row = text_dataset(tmp_path)
    rows = [row]
    if change == "duplicate":
        rows.append(dict(row))
        manifest.update(samples=2, event_counts={"actual-event": 2})
    elif change == "lineage_missing":
        del row["event_family_id"]
    elif change == "source_crosses_families":
        rows.append(dict(row, id="sample-two", event_family_id="heldout-event"))
        manifest.update(samples=2, event_counts={"actual-event": 1, "heldout-event": 1})
    else:
        manifest["test_or_validation_samples"] = 1
    (directory / "train.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows))
    manifest["dataset_sha256"] = file_digest(directory / "train.jsonl")
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="unique|lineage|different event families|training-only"):
        validate_dataset(directory, "text")
