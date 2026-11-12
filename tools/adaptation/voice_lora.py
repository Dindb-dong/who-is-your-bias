"""Bounded Qwen3-TTS codec-objective LoRA experiments with resumable state.

The objective follows QwenLM/Qwen3-TTS's Apache-2.0 finetuning/sft_12hz.py:
first-codebook cross entropy plus 0.3 times sub-talker codebook loss. The base,
speaker encoder and acoustic tokenizer remain frozen. The 0.6B model's text
projection is applied, matching its inference path (2048 -> 1024 dimensions).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from services.voice.adapters import SCHEMA, TARGET_MODULES, base_manifest, digest, json_digest, read_checkpoint
from .voice_data import load_dataset

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def atomic_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def upstream_dataset(upstream: Path, revision: str):
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Pin the immutable Qwen upstream revision")
    actual = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if actual != revision:
        raise ValueError("Qwen upstream revision differs")
    # Ignore Python's local bytecode, but never allow edited training code.
    dirty = subprocess.run(["git", "-C", str(upstream), "diff", "--name-only", "HEAD", "--", "finetuning"], check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("Pinned Qwen training sources have local modifications")
    path = upstream / "finetuning/dataset.py"
    spec = importlib.util.spec_from_file_location("bias_pinned_qwen_dataset", path)
    if spec is None or spec.loader is None:
        raise ValueError("Pinned Qwen dataset implementation is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TTSDataset, digest(path)


def prepare_codes(dataset: Path, output: Path, *, base_model: Path, persona_id: str,
                  allow_candidates: bool = False, threads: int = 4) -> dict:
    """Persist one codec sample at a time; interrupted extraction can resume."""
    import torch
    from qwen_tts import Qwen3TTSTokenizer
    manifest, rows = load_dataset(dataset, persona_id=persona_id, allow_candidates=allow_candidates)
    torch.set_num_threads(threads)
    tokenizer_path = base_model / "speech_tokenizer"
    tokenizer_files = base_manifest(tokenizer_path)
    identity = {"dataset_manifest_sha256": digest(dataset / "manifest.json"),
                "tokenizer_files_sha256": tokenizer_files, "persona_id": persona_id}
    index = output / "codes.json"
    if index.exists() and json.loads(index.read_text()).get("identity") != identity:
        raise ValueError("Codec extraction inputs differ from the interrupted job")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(tokenizer_path.resolve()), device_map="cpu", local_files_only=True)
    completed = []
    started = time.monotonic()
    for ordinal, row in enumerate(rows):
        path = output / f"codes-{ordinal:04d}.json"
        expected = {"audio_sha256": digest(Path(row["audio"])), "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
                    "tokenizer_sha256": json_digest(tokenizer_files)}
        if path.exists():
            item = json.loads(path.read_text())
            if item.get("identity") != expected:
                raise ValueError("Existing codec sample has different inputs")
        else:
            with torch.inference_mode():
                code = tokenizer.encode([row["audio"]]).audio_codes[0].cpu().tolist()
            item = {"identity": expected, "audio_codes": code}
            atomic_json(path, item)
        if not item.get("audio_codes") or any(len(frame) != 16 for frame in item["audio_codes"]):
            raise ValueError("Qwen 12Hz encoding requires sixteen nonempty codebooks")
        completed.append({"path": path.name, "sha256": digest(path), "frames": len(item["audio_codes"])})
        atomic_json(index, {"schema_version": "voice-codes.v1", "identity": identity,
            "samples": completed, "completed": len(completed) == len(rows), "elapsed_seconds": time.monotonic()-started})
        print(json.dumps({"stage": "codes", "completed_samples": len(completed), "total_samples": len(rows)}), flush=True)
    return json.loads(index.read_text())


def codec_rows(directory: Path, rows: list[dict], dataset_manifest_sha256: str) -> tuple[list[dict], dict]:
    index = json.loads((directory / "codes.json").read_text())
    if index.get("schema_version") != "voice-codes.v1" or index.get("completed") is not True:
        raise ValueError("Complete codec preparation before optimization")
    if index.get("identity", {}).get("dataset_manifest_sha256") != dataset_manifest_sha256 or len(index["samples"]) != len(rows):
        raise ValueError("Codec preparation and training dataset differ")
    result = []
    for row, entry in zip(rows, index["samples"]):
        name = entry.get("path", "")
        if not re.fullmatch(r"codes-[0-9]{4,}\.json", name):
            raise ValueError("Invalid codec sample path")
        path = directory / name
        if digest(path) != entry["sha256"]:
            raise ValueError("Codec artifact changed")
        item = json.loads(path.read_text())
        if item.get("identity", {}).get("audio_sha256") != digest(Path(row["audio"])):
            raise ValueError("Codec sample does not belong to its audio")
        codes = item.get("audio_codes")
        if (not isinstance(codes, list) or not codes or len(codes) > 400
                or any(not isinstance(frame, list) or len(frame) != 16
                       or any(type(value) is not int or not 0 <= value < 2048 for value in frame) for frame in codes)):
            raise ValueError("Malformed or oversized codec sample")
        result.append(dict(row, audio_codes=codes))
    return result, index


def speech_loss(model, batch):
    """Original speech-token objective, never a synthetic speaker-label loss."""
    import torch
    device, dtype = model.device, model.dtype
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        speaker = model.speaker_encoder(batch["ref_mels"].to(dtype)).detach()
        text = model.talker.text_projection(model.talker.model.text_embedding(batch["input_ids"][:, :, 0]))
        text = text * batch["text_embedding_mask"]
        codec = model.talker.model.codec_embedding(batch["input_ids"][:, :, 1]) * batch["codec_embedding_mask"]
        codec[:, 6, :] = speaker
        inputs = text + codec
        for codebook in range(1, 16):
            embedding = model.talker.code_predictor.get_input_embeddings()[codebook-1](batch["codec_ids"][:, :, codebook])
            inputs = inputs + embedding * batch["codec_mask"].unsqueeze(-1)
    outputs = model.talker(inputs_embeds=inputs[:, :-1, :], attention_mask=batch["attention_mask"][:, :-1],
        labels=batch["codec_0_labels"][:, 1:], output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[0][-1][batch["codec_mask"][:, :-1]]
    target_codes = batch["codec_ids"][batch["codec_mask"]]
    _, sub_loss = model.talker.forward_sub_talker_finetune(target_codes, hidden)
    return outputs.loss + 0.3 * sub_loss


def save_checkpoint(output: Path, *, model, optimizer, contract: dict, step: int,
                    cursor: int, losses: list[float]) -> Path:
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"step-{step:08d}"
    if target.exists():
        raise ValueError("Do not overwrite a known-good voice checkpoint")
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=output))
    try:
        weights = {key: tensor.detach().cpu().clone() for key, tensor in get_peft_model_state_dict(model).items()}
        save_file(weights, str(staging / "adapter.safetensors"))
        torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(), "cursor": cursor, "step": step, "losses": losses,
            "scheduler": None, "scheduler_policy": "constant_learning_rate"}, staging / "state.pt")
        receipt = {"schema_version": SCHEMA, "contract": contract, "contract_sha256": json_digest(contract),
            "optimizer_steps": step, "cursor": cursor, "last_loss": losses[-1], "losses": losses,
            "artifacts": {name: digest(staging / name) for name in ("adapter.safetensors", "state.pt")},
            "status": "resumable", "promotion_status": "candidate", "efficacy_evaluated": False}
        atomic_json(staging / "checkpoint.json", receipt)
        staging.rename(target)
        atomic_json(output / "latest.json", {"path": target.name, "optimizer_steps": step,
            "checkpoint_sha256": digest(target / "checkpoint.json")})
        return target
    except BaseException:
        shutil.rmtree(staging)
        raise


def restore_checkpoint(directory: Path, *, model, optimizer, contract: dict) -> tuple[int, int, list[float]]:
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    receipt = read_checkpoint(directory, persona_id=contract["persona_id"])
    if receipt["contract"] != contract:
        raise ValueError("Resume contract differs: preserve code, model, dataset and settings")
    weights = load_file(str(directory / "adapter.safetensors"))
    expected = get_peft_model_state_dict(model)
    if set(weights) != set(expected) or any(weights[key].shape != expected[key].shape for key in weights):
        raise ValueError("Resumed adapter tensors differ from the initialized model")
    set_peft_model_state_dict(model, weights)
    state = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
    if state["step"] != receipt["optimizer_steps"] or state["cursor"] != receipt["cursor"]:
        raise ValueError("Optimizer state and checkpoint progress differ")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    return state["step"], state["cursor"], state["losses"]


def train(dataset: Path, codes: Path, output: Path, *, base_model: Path, base_revision: str,
          upstream: Path, upstream_revision: str, persona_id: str, base_repo="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
          allow_candidates=False, rank=4, learning_rate=2e-5, max_updates=16, stop_after=4,
          gradient_accumulation=1, seed=42, threads=4, precision="bfloat16", resume: Path | None = None) -> dict:
    if (not re.fullmatch(r"[a-f0-9]{40}", base_revision) or not 1 <= rank <= 64 or not 1 <= max_updates <= 100000
        or not 1 <= stop_after <= max_updates or not 1 <= gradient_accumulation <= 64 or not 1 <= threads <= 16
        or not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.01 or precision not in {"float32", "bfloat16"}):
        raise ValueError("Invalid bounded voice adaptation settings")
    if output.exists() and resume is None:
        raise ValueError("Use a new run output or explicitly resume a saved checkpoint")
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    from qwen_tts import Qwen3TTSModel
    manifest, rows = load_dataset(dataset, persona_id=persona_id, allow_candidates=allow_candidates)
    dataset_digest = digest(dataset / "manifest.json")
    rows, code_index = codec_rows(codes, rows, dataset_digest)
    Dataset, dataset_code_digest = upstream_dataset(upstream, upstream_revision)
    torch.set_num_threads(threads); torch.manual_seed(seed); random.seed(seed)
    lora = {"r": rank, "lora_alpha": 2*rank, "lora_dropout": 0.05, "target_modules": TARGET_MODULES, "bias": "none"}
    contract = {"persona_id": persona_id, "speaker": manifest["speaker"], "base_repo": base_repo,
        "base_revision": base_revision, "base_files_sha256": base_manifest(base_model),
        "dataset_manifest_sha256": dataset_digest, "codec_index_sha256": digest(codes / "codes.json"),
        "upstream_revision": upstream_revision, "upstream_dataset_sha256": dataset_code_digest,
        "implementation_sha256": digest(Path(__file__)), "lora": lora,
        "learning_rate": learning_rate, "max_updates": max_updates, "gradient_accumulation": gradient_accumulation,
        "seed": seed, "threads": threads, "precision": precision, "device": "cpu", "batch_size": 1,
        "training_event_families": sorted(manifest["event_counts"]), "experimental_candidates": manifest["experimental_candidates"],
        "objective": "codec_0_cross_entropy + 0.3 * sub_talker_codebook_cross_entropy",
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "qwen-tts")}}
    if code_index["identity"]["tokenizer_files_sha256"] != base_manifest(base_model / "speech_tokenizer"):
        raise ValueError("Voice codecs belong to a different acoustic tokenizer")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "voice-lora-run.v1", "status": "loading", "contract": contract,
        "optimizer_steps": 0, "planned_updates": max_updates, "efficacy_evaluated": False}
    atomic_json(output / "run.json", report)
    started = time.monotonic()
    try:
        qwen = Qwen3TTSModel.from_pretrained(str(base_model.resolve()), device_map="cpu", dtype=getattr(torch, precision),
            attn_implementation="sdpa", local_files_only=True, trust_remote_code=False)
        model = qwen.model
        model.requires_grad_(False)
        inject_adapter_in_model(LoraConfig(**lora), model)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("No trainable LoRA parameters were attached")
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
        training = Dataset(rows, qwen.processor, model.config)
        step, cursor, losses = (0, 0, []) if resume is None else restore_checkpoint(resume, model=model, optimizer=optimizer, contract=contract)
        target = min(max_updates, step+stop_after)
        if step >= max_updates:
            raise ValueError("Requested voice run already completed its planned updates")
        # Base stochastic layers stay in evaluation mode; only LoRA dropout is
        # trained. Frozen speaker features remain deterministic between batches.
        model.eval()
        for name, module in model.named_modules():
            if "lora_dropout" in name:
                module.train()
        report.update(status="running", optimizer_steps=step, trainable_parameters=sum(p.numel() for p in parameters))
        atomic_json(output / "run.json", report)
        while step < target:
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(gradient_accumulation):
                epoch, within = divmod(cursor, len(training))
                order = list(range(len(training))); random.Random(seed+epoch).shuffle(order)
                batch = training.collate_fn([training[order[within]]])
                if batch["input_ids"].shape[1] > 768:
                    raise ValueError("Whole speech example exceeds the CPU context budget; reselect without truncation")
                loss = speech_loss(model, batch)
                if not torch.isfinite(loss):
                    raise ValueError("Voice optimization produced a nonfinite loss")
                (loss / gradient_accumulation).backward()
                loss_sum += float(loss.detach()) / gradient_accumulation
                cursor += 1
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step(); step += 1; losses.append(loss_sum)
            checkpoint = save_checkpoint(output, model=model, optimizer=optimizer, contract=contract, step=step, cursor=cursor, losses=losses)
            report.update(optimizer_steps=step, last_checkpoint=checkpoint.name, losses=losses, elapsed_seconds=time.monotonic()-started)
            atomic_json(output / "run.json", report)
            print(json.dumps({"optimizer_steps": step, "loss": loss_sum, "checkpoint": checkpoint.name}), flush=True)
        report["status"] = "completed" if step == max_updates else "paused_for_persistence"
        report["resume"] = {"checkpoint": report["last_checkpoint"], "same_contract_required": True,
                            "remote_readback_required_before_next_block": True}
        atomic_json(output / "run.json", report)
        return report
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, elapsed_seconds=time.monotonic()-started)
        atomic_json(output / "run.json", report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train"):
        item = commands.add_parser(command)
        for name in ("dataset", "output", "base-model"):
            item.add_argument("--"+name, type=Path, required=True)
        item.add_argument("--persona-id", required=True)
        item.add_argument("--allow-candidates", action="store_true")
        item.add_argument("--threads", type=int, default=4)
        if command == "train":
            for name in ("codes", "upstream"):
                item.add_argument("--"+name, type=Path, required=True)
            for name in ("base-revision", "upstream-revision"):
                item.add_argument("--"+name, required=True)
            item.add_argument("--base-repo", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
            for name, default in (("rank", 4), ("max-updates", 16), ("stop-after", 4), ("gradient-accumulation", 1), ("seed", 42)):
                item.add_argument("--"+name, type=int, default=default)
            item.add_argument("--learning-rate", type=float, default=2e-5)
            item.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
            item.add_argument("--resume", type=Path)
    args = vars(parser.parse_args(argv)); command = args.pop("command")
    result = (prepare_codes if command == "prepare" else train)(**args)
    print(json.dumps({key: result[key] for key in ("status", "optimizer_steps", "completed") if key in result}, indent=2))


if __name__ == "__main__":
    main()
