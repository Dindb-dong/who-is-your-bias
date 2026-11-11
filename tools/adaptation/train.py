"""Optional training backends; importing this module does not load model weights."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.persona_data.pipeline import write_json
from services.language.adapters import (
    CHAT_TEMPLATE_OPTIONS, RUN_SCHEMA, artifact_manifest, checkpoint_manifest,
    immutable_revision, json_digest, repository_id, tokenizer_contract,
)

from .datasets import file_hash, validate_dataset


def encode_example(tokenizer, messages: list[dict], max_length: int) -> dict | None:
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, **CHAT_TEMPLATE_OPTIONS)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, **CHAT_TEMPLATE_OPTIONS)
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
               rank=8, learning_rate=0.0002, gradient_accumulation=8, max_steps: int | None = None, device="cpu",
               precision="float32", gradient_checkpointing=False) -> dict:
    manifest, rows = validate_dataset(dataset, "text")
    if output.exists() or Path(model).is_dir() or not repository_id(model) or not immutable_revision(revision):
        raise ValueError("Choose a new output, Hugging Face repository ID and immutable 40-character base model revision; a local directory cannot verify that repository revision")
    if (any(type(value) is not int or value < low for value, low in ((epochs, 1), (rank, 1), (max_length, 32), (gradient_accumulation, 1), (seed, 0)))
            or seed > 2**32 - 1 or isinstance(learning_rate, bool) or not isinstance(learning_rate, (float, int))
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or (max_steps is not None and (type(max_steps) is not int or max_steps < 1))
            or not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", device)
            or precision not in {"float32", "bfloat16"} or type(gradient_checkpointing) is not bool):
        raise ValueError("Invalid bounded training settings")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema_version": RUN_SCHEMA, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "base_model": model, "base_revision": revision, "tokenizer_revision": revision,
              "dataset_manifest": manifest, "seed": seed, "epochs": epochs, "max_length": max_length,
              "adapter": {"type": "LoRA", "rank": rank, "alpha": 2*rank, "dropout": 0.05, "targets": "all-linear"},
              "optimization": {"learning_rate": learning_rate, "gradient_accumulation": gradient_accumulation, "batch_size": 1, "max_steps": max_steps,
                               "gradient_checkpointing": gradient_checkpointing, "checkpoint_use_reentrant": False if gradient_checkpointing else None},
              "requested_device": device, "precision": precision, "versions": _package_versions(["torch", "transformers", "peft", "huggingface-hub"]),
              "loss_history": [], "loss_history_limit": 2048, "loss_history_truncated": False,
              "loss_statistics": {"examples": 0, "sum": 0., "minimum": None, "maximum": None},
              "chat_template_options": dict(CHAT_TEMPLATE_OPTIONS), "safe_serialization": True,
              "implementation_sha256": {"tools/adaptation/train.py": file_hash(Path(__file__)),
                    "tools/adaptation/datasets.py": file_hash(Path(__file__).with_name("datasets.py")),
                    "services/language/adapters.py": file_hash(Path(__file__).resolve().parents[2] / "services/language/adapters.py")},
              "efficacy_evaluated": False}
    write_json(output / "run.json", report)
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download

        report["runtime"] = {"torch_threads": torch.get_num_threads(),
                             "torch_interop_threads": torch.get_num_interop_threads(),
                             "platform": platform.platform(), "python": platform.python_version(),
                             "machine": platform.machine(), "processor": platform.processor() or None,
                             "logical_cpu_count": os.cpu_count(),
                             "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                             "cuda_runtime": torch.version.cuda, "requested_device": device}
        write_json(output / "run.json", report)
        random.seed(seed)
        torch.manual_seed(seed)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("Requested CUDA is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Requested Apple MPS is unavailable")
        dtype = getattr(torch, precision)
        if precision == "bfloat16":
            # Probe the selected runtime rather than promising native BF16
            # acceleration from an architecture name. Unsupported kernels fail
            # before loading the checkpoint; later unsupported ops still fail
            # explicitly and are retained in the run receipt.
            probe = torch.ones((2, 2), dtype=dtype, device=device, requires_grad=True)
            (probe @ probe).float().sum().backward()
            if probe.grad is None or not torch.isfinite(probe.grad).all():
                raise ValueError("The selected device failed the bfloat16 forward/backward probe")
            report["precision_probe"] = {"dtype": precision, "device": device, "finite_forward_backward": True,
                                         "native_acceleration_measured": False}
            del probe
        # Resolve the pinned base once; train and serve from the same local
        # checkpoint bytes, without metadata probes changing revision midway.
        checkpoint = Path(snapshot_download(model, revision=revision,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"]))
        report["base_checkpoint_sha256"] = checkpoint_manifest(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
        report["tokenizer_contract"] = tokenizer_contract(tokenizer)
        report["tokenizer_contract_sha256"] = json_digest(report["tokenizer_contract"])
        encoded = [(row["id"], encode_example(tokenizer, row["messages"], max_length)) for row in rows]
        report["excluded_whole_examples"] = [key for key, value in encoded if value is None]
        examples = [value for _, value in encoded if value is not None]
        if not examples:
            raise ValueError("No whole examples fit the model context budget")
        base = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False,
                                                  use_safetensors=True, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa")
        adapted = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=rank, lora_alpha=2*rank, lora_dropout=0.05,
                                                target_modules="all-linear", revision=revision))
        # PEFT otherwise records the resolved cache path as the base model.
        # The receipt binds the repository, revision and actual file hashes.
        adapted.peft_config["default"].base_model_name_or_path = model
        adapted.peft_config["default"].revision = revision
        adapted.to(device)
        adapted.train()
        adapted.config.use_cache = False
        if gradient_checkpointing:
            adapted.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # PEFT's frozen embeddings need an explicit gradient path on stacks
            # whose checkpoint implementation would otherwise drop the graph.
            adapted.enable_input_require_grads()
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
                observed_loss = float(loss.detach().cpu())
                if len(report["loss_history"]) < report["loss_history_limit"]:
                    report["loss_history"].append(observed_loss)
                else:
                    report["loss_history_truncated"] = True
                stats = report["loss_statistics"]
                stats["examples"] += 1
                stats["sum"] += observed_loss
                stats["minimum"] = observed_loss if stats["minimum"] is None else min(stats["minimum"], observed_loss)
                stats["maximum"] = observed_loss if stats["maximum"] is None else max(stats["maximum"], observed_loss)
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
        # Verify that the serialization used by the serving path preserves the
        # token contract before marking the adapter complete.
        restored_tokenizer = AutoTokenizer.from_pretrained(str(output / "tokenizer"), local_files_only=True, trust_remote_code=False)
        if tokenizer_contract(restored_tokenizer) != report["tokenizer_contract"]:
            raise ValueError("Saved tokenizer changed the training token contract")
        report["artifact_sha256"] = artifact_manifest(output)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
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
