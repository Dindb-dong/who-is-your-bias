"""Verified local LoRA artifacts and a shared training/inference token contract.

This module uses only the standard library. Checking a run never downloads or
loads a model. A valid receipt establishes artifact identity, not model quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


CHAT_TEMPLATE_OPTIONS = {"enable_thinking": False}
RUN_SCHEMA = "text-adaptation-run.v2"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_REVISION = re.compile(r"^[a-f0-9]{40}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?$")
_CHECKPOINT_SUFFIXES = {".json", ".safetensors", ".model", ".txt", ".jinja", ".tiktoken"}


class AdapterArtifactError(ValueError):
    """A local training receipt does not establish the requested serving identity."""


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_id(value: str) -> bool:
    return bool(isinstance(value, str) and _REPO.fullmatch(value) and ".." not in value)


def immutable_revision(value: str | None) -> bool:
    return isinstance(value, str) and bool(_REVISION.fullmatch(value))


def _relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise AdapterArtifactError("Artifact paths must be normalized relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts) or path.as_posix() != value:
        raise AdapterArtifactError("Artifact paths must be normalized relative POSIX paths")
    return path


def _hash_map(value: Any, *, roots: set[str] | None = None) -> dict[str, str]:
    if not isinstance(value, dict) or not value or len(value) > 20_000:
        raise AdapterArtifactError("A nonempty bounded artifact hash manifest is required")
    for name, digest in value.items():
        path = _relative_path(name)
        if roots is not None and (len(path.parts) < 2 or path.parts[0] not in roots):
            raise AdapterArtifactError("Only adapter and tokenizer files belong in the run artifact manifest")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise AdapterArtifactError("Every artifact requires an exact SHA-256 digest")
    return value


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise AdapterArtifactError("Receipt/config must be a regular JSON file smaller than 16 MiB")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AdapterArtifactError("Duplicate JSON keys make the artifact receipt ambiguous")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                           parse_constant=lambda value: (_ for _ in ()).throw(AdapterArtifactError("Non-finite JSON values are invalid")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterArtifactError("The artifact receipt/config is not readable JSON") from exc
    if not isinstance(value, dict):
        raise AdapterArtifactError("The artifact receipt/config must be a JSON object")
    return value


def artifact_manifest(directory: Path) -> dict[str, str]:
    """Hash every adapter/tokenizer file; reject links, special files and aliases."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise AdapterArtifactError("The training run must be a real local directory")
    result = {}
    for folder in ("adapter", "tokenizer"):
        root = directory / folder
        if root.is_symlink() or not root.is_dir():
            raise AdapterArtifactError("The run must contain real adapter and tokenizer directories")
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise AdapterArtifactError("Adapter/tokenizer artifacts must not contain symlinks or special files")
            if path.is_file():
                result[path.relative_to(directory).as_posix()] = file_digest(path)
    return result


def checkpoint_manifest(directory: Path) -> dict[str, str]:
    """Hash inference files in a resolved checkpoint, including safe model weights.

    Hugging Face's ordinary snapshot-to-blob symlinks are supported here: their
    target bytes are hashed. They are not allowed in the local adapter package.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise AdapterArtifactError("A resolved local base checkpoint is required")
    result = {path.relative_to(directory).as_posix(): file_digest(path)
              for path in sorted(directory.rglob("*")) if path.is_file() and path.suffix in _CHECKPOINT_SUFFIXES}
    if "config.json" not in result or not any(name.endswith(".safetensors") for name in result):
        raise AdapterArtifactError("The base checkpoint needs config.json and safetensors weights")
    return result


def tokenizer_contract(tokenizer) -> dict:
    """Fingerprint actual vocabulary, special IDs, template and rendered tokens."""
    template = tokenizer.chat_template
    vocabulary = tokenizer.get_vocab()
    if not template or not isinstance(vocabulary, dict) or not vocabulary:
        raise AdapterArtifactError("The tokenizer must expose its chat template and vocabulary")
    if any(not isinstance(key, str) or type(value) is not int or value < 0 for key, value in vocabulary.items()):
        raise AdapterArtifactError("The tokenizer vocabulary has invalid token IDs")
    probes = []
    for question, answer in (("Explain the evidence.", "The evidence is historical."),
                             ("현재 계획은 무엇인가요? 2030년과 비교해 주세요.", "현재 근거는 확인되지 않았습니다.")):
        prompt = [{"role": "system", "content": "You are a disclosed AI simulation."},
                  {"role": "user", "content": question}]
        full = [*prompt, {"role": "assistant", "content": answer}]
        values = [tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, **CHAT_TEMPLATE_OPTIONS),
                  tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False, **CHAT_TEMPLATE_OPTIONS)]
        if any(not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens) for tokens in values):
            raise AdapterArtifactError("The tokenizer chat template must produce a flat token-ID list")
        probes.append(values)
    return {"schema_version": "chat-tokenizer-contract.v1", "chat_template_options": dict(CHAT_TEMPLATE_OPTIONS),
            "chat_template_sha256": json_digest(template), "vocabulary_sha256": json_digest(vocabulary),
            "vocabulary_size": len(vocabulary), "rendered_probes_sha256": json_digest(probes),
            "special_token_ids": {name: getattr(tokenizer, name + "_token_id", None) for name in ("bos", "eos", "pad", "unk")},
            "all_special_ids": sorted(tokenizer.all_special_ids)}


@dataclass(frozen=True)
class VerifiedAdapterRun:
    directory: Path
    run_sha256: str
    report: dict

    def public_identity(self) -> dict:
        return {"schema_version": "text-adapter-identity.v1", "run_sha256": self.run_sha256,
                "artifact_manifest_sha256": json_digest(self.report["artifact_sha256"]),
                "base_model": self.report["base_model"], "base_revision": self.report["base_revision"],
                "base_checkpoint_manifest_sha256": json_digest(self.report["base_checkpoint_sha256"]),
                "dataset_sha256": self.report["dataset_manifest"]["dataset_sha256"],
                "training_event_families": sorted(self.report["dataset_manifest"]["event_counts"]),
                "tokenizer_contract_sha256": self.report["tokenizer_contract_sha256"],
                "optimizer_steps": self.report["optimizer_steps"], "efficacy_evaluated": False}

    def verify_base_checkpoint(self, directory: Path) -> dict[str, str]:
        actual = checkpoint_manifest(directory)
        if actual != self.report["base_checkpoint_sha256"]:
            raise AdapterArtifactError("The served base checkpoint differs from the training checkpoint")
        return actual

    def verify_tokenizer(self, tokenizer) -> dict:
        actual = tokenizer_contract(tokenizer)
        if actual != self.report["tokenizer_contract"]:
            raise AdapterArtifactError("The serving tokenizer/template differs from the training token contract")
        return actual


def validate_adapter_run(directory: Path, *, base_model: str, base_revision: str | None,
                         expected_run_sha256: str | None = None) -> VerifiedAdapterRun:
    directory = Path(directory).expanduser()
    if directory.is_symlink() or not directory.is_dir():
        raise AdapterArtifactError("The adapter run must be a real local directory")
    report = _read_json(directory / "run.json")
    run_sha = file_digest(directory / "run.json")
    if expected_run_sha256 is not None and (not _SHA256.fullmatch(expected_run_sha256) or run_sha != expected_run_sha256):
        raise AdapterArtifactError("The selected adapter run receipt SHA-256 changed")
    if report.get("schema_version") != RUN_SCHEMA:
        raise AdapterArtifactError("Serving requires text-adaptation-run.v2; legacy v1 lacks tokenizer compatibility evidence. Re-run training with the current trainer; do not relabel an old receipt.")
    if report.get("status") != "completed":
        raise AdapterArtifactError("Only a completed text-adaptation run can be served")
    if (not repository_id(base_model) or not immutable_revision(base_revision)
            or report.get("base_model") != base_model or report.get("base_revision") != base_revision
            or report.get("tokenizer_revision") != base_revision):
        raise AdapterArtifactError("The configured base repository and immutable revision must match the completed run")
    if type(report.get("optimizer_steps")) is not int or report["optimizer_steps"] < 1:
        raise AdapterArtifactError("The completed run must record actual optimizer steps")
    dataset = report.get("dataset_manifest", {})
    if (not isinstance(dataset, dict) or dataset.get("schema_version") != "text-adaptation-dataset.v1"
            or not isinstance(dataset.get("dataset_sha256"), str) or not _SHA256.fullmatch(dataset["dataset_sha256"])
            or type(dataset.get("samples")) is not int or dataset["samples"] < 1
            or type(dataset.get("test_or_validation_samples")) is not int or dataset["test_or_validation_samples"] != 0):
        raise AdapterArtifactError("The training run needs a nonempty training-only dataset identity")
    events = dataset.get("event_counts")
    if (not isinstance(events, dict) or not events or len(events) > 10_000
            or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", name)
                   or type(count) is not int or count < 1 for name, count in events.items())
            or sum(events.values()) != dataset["samples"]):
        raise AdapterArtifactError("The training dataset must bind complete source-family sample counts")
    contract = report.get("tokenizer_contract")
    if (not isinstance(contract, dict) or contract.get("schema_version") != "chat-tokenizer-contract.v1"
            or contract.get("chat_template_options") != CHAT_TEMPLATE_OPTIONS
            or json_digest(contract) != report.get("tokenizer_contract_sha256")):
        raise AdapterArtifactError("The completed run does not bind the supported tokenizer/template contract")
    expected = _hash_map(report.get("artifact_sha256"), roots={"adapter", "tokenizer"})
    base_hashes = _hash_map(report.get("base_checkpoint_sha256"))
    if "config.json" not in base_hashes or not any(name.endswith(".safetensors") for name in base_hashes):
        raise AdapterArtifactError("The run must bind the base config and safe weights")
    required = {"adapter/adapter_config.json", "adapter/adapter_model.safetensors", "tokenizer/tokenizer_config.json"}
    if not required.issubset(expected) or any(name.endswith((".bin", ".pt", ".pth", ".pkl", ".py")) for name in expected):
        raise AdapterArtifactError("A safe LoRA package requires its config, safetensors weights and tokenizer config")
    if artifact_manifest(directory) != expected:
        raise AdapterArtifactError("Adapter/tokenizer files changed, are missing, or contain unregistered files")
    config = _read_json(directory / "adapter/adapter_config.json")
    if (config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM"
            or config.get("base_model_name_or_path") != base_model or config.get("revision") != base_revision
            or config.get("inference_mode") is not True):
        raise AdapterArtifactError("The adapter config must identify the same causal-LM LoRA base and revision")
    adapter = report.get("adapter", {})
    if not isinstance(adapter, dict) or adapter.get("type") != "LoRA":
        raise AdapterArtifactError("The completed run must describe its LoRA configuration")
    for key, report_key in (("r", "rank"), ("lora_alpha", "alpha"), ("lora_dropout", "dropout")):
        value = config.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value != adapter.get(report_key)):
            raise AdapterArtifactError("The saved LoRA configuration differs from the completed run")
    return VerifiedAdapterRun(directory.resolve(), run_sha, report)
