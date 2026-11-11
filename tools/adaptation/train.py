"""Optional training backends; importing this module does not load model weights."""

from __future__ import annotations

import importlib.metadata
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.persona_data.pipeline import write_json

from .datasets import file_hash, validate_dataset


def encode_example(tokenizer, messages: list[dict], max_length: int) -> dict | None:
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError("Tokenizer chat template does not preserve the assistant prefix; refuse incorrect loss masking")
    if len(full) > max_length or len(full) <= len(prefix):
        return None
    return {"input_ids": full, "labels": [-100]*len(prefix) + full[len(prefix):]}


def _package_versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def train_text(dataset: Path, output: Path, *, model: str, revision: str, epochs=1, seed=42, max_length=1024,
               rank=8, learning_rate=0.0002, gradient_accumulation=8, max_steps: int | None = None, device="cpu") -> dict:
    manifest, rows = validate_dataset(dataset, "text")
    if output.exists() or Path(model).is_dir() or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Choose a new output, Hugging Face repository ID and immutable 40-character base model revision; a local directory cannot verify that repository revision")
    if epochs < 1 or rank < 1 or max_length < 32 or learning_rate <= 0 or gradient_accumulation < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("Invalid bounded training settings")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema_version": "text-adaptation-run.v1", "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "base_model": model, "base_revision": revision, "tokenizer_revision": revision,
              "dataset_manifest": manifest, "seed": seed, "epochs": epochs, "max_length": max_length,
              "adapter": {"type": "LoRA", "rank": rank, "alpha": 2*rank, "dropout": 0.05, "targets": "all-linear"},
              "optimization": {"learning_rate": learning_rate, "gradient_accumulation": gradient_accumulation, "batch_size": 1, "max_steps": max_steps},
              "requested_device": device, "precision": "float32", "versions": _package_versions(["torch", "transformers", "peft", "huggingface-hub"]),
              "loss_history": [], "efficacy_evaluated": False}
    write_json(output / "run.json", report)
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download

        random.seed(seed)
        torch.manual_seed(seed)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("Requested CUDA is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Requested Apple MPS is unavailable")
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        encoded = [(row["id"], encode_example(tokenizer, row["messages"], max_length)) for row in rows]
        report["excluded_whole_examples"] = [key for key, value in encoded if value is None]
        examples = [value for _, value in encoded if value is not None]
        if not examples:
            raise ValueError("No whole examples fit the model context budget")
        base = AutoModelForCausalLM.from_pretrained(model, revision=revision, trust_remote_code=False, torch_dtype=torch.float32)
        checkpoint = Path(snapshot_download(model, revision=revision, local_files_only=True))
        report["base_checkpoint_sha256"] = {str(p.relative_to(checkpoint)): file_hash(p) for p in sorted(checkpoint.rglob("*"))
                                            if p.is_file() and p.suffix in {".json", ".safetensors", ".bin"}}
        adapted = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=rank, lora_alpha=2*rank, lora_dropout=0.05, target_modules="all-linear"))
        adapted.to(device)
        adapted.train()
        adapted.config.use_cache = False
        optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=learning_rate)
        optimizer.zero_grad()
        report["actual_device"] = str(next(adapted.parameters()).device)
        report["hardware"] = torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else device
        report["accepted_examples"] = len(examples)
        report["trainable_parameters"] = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
        steps = 0
        for epoch in range(epochs):
            random.Random(seed+epoch).shuffle(examples)
            for index, example in enumerate(examples):
                inputs = torch.tensor([example["input_ids"]], device=device)
                labels = torch.tensor([example["labels"]], device=device)
                loss = adapted(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels).loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                group_start = (index // gradient_accumulation) * gradient_accumulation
                group_size = min(gradient_accumulation, len(examples) - group_start)
                (loss / group_size).backward()
                report["loss_history"].append(float(loss.detach().cpu()))
                if (index+1) % gradient_accumulation == 0 or index+1 == len(examples):
                    torch.nn.utils.clip_grad_norm_(adapted.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    steps += 1
                    report["optimizer_steps"] = steps
                    write_json(output / "run.json", report)
                    if max_steps is not None and steps >= max_steps:
                        break
            if max_steps is not None and steps >= max_steps:
                break
        adapted.save_pretrained(output / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(output / "tokenizer")
        report["artifact_sha256"] = {str(p.relative_to(output)): file_hash(p) for folder in ("adapter", "tokenizer") for p in sorted((output/folder).rglob("*")) if p.is_file()}
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "run.json", report)
    return report


def train_voice(dataset: Path, output: Path, *, upstream: Path, upstream_revision: str, base_model: Path, tokenizer_model: Path,
                device="cuda:0", batch_size=2, learning_rate=2e-6, epochs=3, speaker="musk_research", execute=False) -> dict:
    manifest, _ = validate_dataset(dataset, "voice")
    if output.exists() or not re.fullmatch(r"[a-f0-9]{40}", upstream_revision):
        raise ValueError("Choose new output and immutable upstream revision")
    if batch_size < 1 or learning_rate <= 0 or epochs < 1 or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", speaker):
        raise ValueError("Invalid voice training settings")
    actual = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if actual != upstream_revision:
        raise ValueError("Qwen3-TTS checkout differs from its pinned revision")
    dirty = subprocess.run(["git", "-C", str(upstream), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("Qwen3-TTS checkout has uncommitted or untracked changes")
    scripts = [upstream / "finetuning" / name for name in ("prepare_data.py", "sft_12hz.py")]
    if not all(path.is_file() for path in scripts) or not base_model.is_dir() or not tokenizer_model.is_dir():
        raise ValueError("Pinned upstream scripts and local base/tokenizer checkpoints are required")
    output.mkdir(parents=True)
    codes = output.resolve() / "train_with_codes.jsonl"
    commands = [[sys.executable, str(scripts[0].resolve()), "--device", device, "--tokenizer_model_path", str(tokenizer_model.resolve()),
                 "--input_jsonl", str((dataset/"train_raw.jsonl").resolve()), "--output_jsonl", str(codes)],
                [sys.executable, str(scripts[1].resolve()), "--init_model_path", str(base_model.resolve()), "--output_model_path", str((output/"checkpoints").resolve()),
                 "--train_jsonl", str(codes), "--batch_size", str(batch_size), "--lr", str(learning_rate), "--num_epochs", str(epochs), "--speaker_name", speaker]]
    report = {"schema_version": "qwen3-tts-adaptation-run.v1", "status": "prepared", "dataset_manifest": manifest,
              "upstream_revision": actual, "script_sha256": {str(p.name): file_hash(p) for p in scripts},
              "model_sha256": {str(p.relative_to(base_model)): file_hash(p) for p in sorted(base_model.rglob("*")) if p.is_file()},
              "tokenizer_sha256": {str(p.relative_to(tokenizer_model)): file_hash(p) for p in sorted(tokenizer_model.rglob("*")) if p.is_file()},
              "commands": commands, "execute": execute, "efficacy_evaluated": False,
              "preparation_device": device,
              "training_device": "Selected by the pinned upstream Accelerate configuration; the preparation --device option does not select SFT hardware.",
              "seed": None, "seed_control": "The inspected upstream CLI does not expose a seed flag; randomness is not claimed controlled.",
              "versions": _package_versions(["torch", "transformers", "qwen-tts", "accelerate", "tensorboard", "huggingface-hub"])}
    write_json(output / "run.json", report)
    started = time.monotonic()
    if execute:
        try:
            for i, command in enumerate(commands):
                with (output / f"stage-{i+1}.log").open("w") as log:
                    subprocess.run(command, cwd=upstream / "finetuning", check=True, stdout=log, stderr=subprocess.STDOUT)
            report["status"] = "completed"
            report["artifact_sha256"] = {str(p.relative_to(output)): file_hash(p) for p in sorted((output/"checkpoints").rglob("*")) if p.is_file()}
        except BaseException as exc:
            report["status"] = "failed"
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic()-started
            write_json(output / "run.json", report)
    return report
