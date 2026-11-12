"""Integrity-checked loading for experimental Qwen speech LoRA checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SCHEMA = "voice-lora-checkpoint.v1"
TARGET_MODULES = r"talker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def json_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def base_manifest(directory: Path) -> dict:
    """Snapshot bytes include the acoustic tokenizer as well as the talker."""
    files = {str(path.relative_to(directory)): digest(path) for path in sorted(directory.rglob("*"))
             if path.is_file() and not any(part.startswith(".") for part in path.relative_to(directory).parts)
             and path.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja", ".tiktoken"}}
    if "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Voice base checkpoint needs config and actual safetensors weights")
    return files


def read_checkpoint(directory: Path, *, persona_id: str, expected_sha256: str | None = None) -> dict:
    metadata = directory / "checkpoint.json"
    if metadata.is_symlink() or not metadata.is_file() or metadata.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Voice adapter checkpoint metadata is missing or invalid")
    if expected_sha256 and digest(metadata) != expected_sha256:
        raise ValueError("Voice adapter receipt hash differs")
    receipt = json.loads(metadata.read_text())
    contract = receipt.get("contract", {})
    if receipt.get("schema_version") != SCHEMA or contract.get("persona_id") != persona_id:
        raise ValueError("Voice adapter schema or subject mismatch")
    if (type(receipt.get("optimizer_steps")) is not int or receipt["optimizer_steps"] < 1
            or receipt.get("contract_sha256") != json_digest(contract)):
        raise ValueError("Voice adapter lacks verified training progress or contract")
    if not re.fullmatch(r"[a-f0-9]{40}", contract.get("base_revision", "")):
        raise ValueError("Voice adapter requires an immutable base revision")
    lora = contract.get("lora", {})
    if (lora.get("target_modules") != TARGET_MODULES or type(lora.get("r")) is not int or not 1 <= lora["r"] <= 64
            or lora.get("lora_alpha") != 2*lora["r"] or lora.get("lora_dropout") != 0.05):
        raise ValueError("Voice adapter configuration differs from the supported experiment")
    for name in ("adapter.safetensors", "state.pt"):
        path = directory / name
        if path.is_symlink() or not path.is_file() or digest(path) != receipt.get("artifacts", {}).get(name):
            raise ValueError("Voice adapter artifact hash differs")
    return receipt


def apply_voice_adapter(model, directory: Path, *, persona_id: str, base_directory: Path,
                        expected_sha256: str | None = None) -> dict:
    """Apply learned low-rank deltas to the exact, independently loaded base."""
    receipt = read_checkpoint(directory, persona_id=persona_id, expected_sha256=expected_sha256)
    contract = receipt["contract"]
    if base_manifest(base_directory) != contract.get("base_files_sha256"):
        raise ValueError("Voice adapter base bytes differ from the training snapshot")
    from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    model.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(**contract["lora"]), model)
    weights = load_file(str(directory / "adapter.safetensors"))
    expected = get_peft_model_state_dict(model)
    if set(weights) != set(expected) or any(weights[key].shape != expected[key].shape for key in expected):
        raise ValueError("Voice adapter tensors differ from the configured model")
    set_peft_model_state_dict(model, weights)
    model.eval()
    return {"schema_version": SCHEMA, "persona_id": persona_id, "optimizer_steps": receipt["optimizer_steps"],
            "base_revision": contract["base_revision"], "dataset_sha256": contract["dataset_manifest_sha256"],
            "adapter_sha256": receipt["artifacts"]["adapter.safetensors"], "checkpoint_sha256": digest(directory / "checkpoint.json"),
            "weight_adaptation": True, "synthesis": "adapted_reference_conditioned", "promotion_status": "candidate",
            "speaker_similarity_evaluated": False}
