"""Bounded MPS experiment with the same frozen Q8-derived base for both arms."""
import hashlib
import importlib.metadata
import json
import random
import resource
import sys
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from gguf import GGUFReader, dequantize
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.modeling_gguf_pytorch_utils import get_gguf_hf_weights_map

from bias.grounding import audit_citations
from services.language.adapters import CHAT_TEMPLATE_OPTIONS, artifact_manifest, json_digest, tokenizer_contract
from tools.adaptation.datasets import file_hash, validate_dataset
from tools.adaptation.train import encode_example
from tools.adaptation_evaluation.protocol import validate_plan

GGUF_SHA256 = "ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1"
REPOSITORY = "ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF"
REVISION = "e6f794d44f9395d0184a966c27b5ae99ea356fcb"
CONFIG_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SEED = 42
MAX_LENGTH = 512
ACCUMULATION = 4
RANK = 8
LEARNING_RATE = 0.0002
DTYPE = torch.bfloat16

def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

def load_q8(source, config):
    # Map each tensor with Transformers' Qwen3 GGUF mapping. Converting one at
    # a time bounds host memory and avoids another multi-GB checkpoint on disk.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    mapping = get_gguf_hf_weights_map(model)
    expected = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    loaded = set()
    reader = GGUFReader(str(source))
    for tensor in reader.tensors:
        key = mapping[tensor.name]
        assert key in expected and key not in loaded, key
        values = torch.from_numpy(dequantize(tensor.data, tensor.tensor_type).copy()).to(DTYPE)
        assert model.get_parameter(key).shape == values.shape, key
        set_module_tensor_to_device(model, key, "mps", value=values, dtype=DTYPE)
        loaded.add(key)
    assert expected - loaded == {"lm_head.weight"}, expected - loaded
    model.tie_weights()
    model.to("mps")
    assert all(p.device.type == "mps" and p.dtype == DTYPE for p in model.parameters())
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    return model, len(loaded)

def generate(model, tokenizer, plan, case):
    messages = case["messages"]["style"]
    assert messages == case["messages"]["adapter"]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_OPTIONS)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("mps")
    input_len = inputs["input_ids"].shape[-1]
    assert input_len + plan["policy"]["max_tokens"] <= plan["policy"]["context_window"]
    torch.manual_seed(plan["policy"]["seed"])
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=plan["policy"]["max_tokens"], do_sample=False)
    torch.mps.synchronize()
    tokens = output[0, input_len:]
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    finish = "stop" if len(tokens) and int(tokens[-1]) in eos_ids else "length"
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    citation = audit_citations(text, set(case["factual_ids"]), required=True)
    return {"case_id": case["id"], "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "completion_tokens": len(tokens), "prompt_tokens": input_len, "finish": finish,
            "prompt_sha256": case["prompt_sha256"]["style"], "factual_sha256": case["factual_sha256"],
            "citation": citation.as_dict(), "seconds": round(time.monotonic() - started, 3)}

def main():
    source, metadata, dataset, plan_path, out = map(Path, sys.argv[1:])
    out.mkdir()
    assert torch.backends.mps.is_available()
    assert file_hash(source) == GGUF_SHA256
    assert metadata.name == CONFIG_REVISION
    manifest, rows = validate_dataset(dataset, "text")
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    assert plan["base_identity"] == {"base_model": REPOSITORY, "base_revision": REVISION}
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(metadata, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(metadata, local_files_only=True, trust_remote_code=False)
    examples = [(row["id"], encode_example(tokenizer, row["messages"], MAX_LENGTH)) for row in rows]
    eligible = [example for _, example in examples if example is not None]
    random.Random(SEED).shuffle(eligible)
    receipt = {
        "status": "loading", "base_model": REPOSITORY, "base_revision": REVISION, "gguf_sha256": GGUF_SHA256,
        "metadata_revision": CONFIG_REVISION, "metadata_file_sha256": {p.name: file_hash(p) for p in metadata.iterdir() if p.is_file()},
        "script_sha256": file_hash(Path(__file__)), "dataset": manifest, "plan_sha256": plan["plan_sha256"],
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "gguf", "accelerate")},
        "tokenizer_contract_sha256": json_digest(tokenizer_contract(tokenizer)), "precision": "bfloat16",
        "excluded_whole_examples": [key for key, value in examples if value is None], "epochs": 1, "rank": RANK,
        "learning_rate": LEARNING_RATE, "gradient_accumulation": ACCUMULATION, "max_length": MAX_LENGTH,
        "seed": SEED, "optimizer_steps": 0, "loss_history": [],
    }
    write(out/"training.json", receipt)
    started = time.monotonic()
    try:
        base, count = load_q8(source, config)
        base.generation_config = GenerationConfig.from_pretrained(metadata, local_files_only=True)
        receipt.update(status="training", tensors_loaded=count, base_parameters=sum(p.numel() for p in base.parameters()))
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=RANK, lora_alpha=2*RANK,
                               lora_dropout=.05, target_modules="all-linear", revision=REVISION))
        model.peft_config["default"].base_model_name_or_path = REPOSITORY
        model.train()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        trainable = [p for p in model.parameters() if p.requires_grad]
        frozen = {key: p._version for key, p in model.named_parameters() if not p.requires_grad}
        assert trainable and all(p.device.type == "mps" for p in trainable)
        receipt.update(actual_device=str(next(model.parameters()).device), trainable_parameters=sum(p.numel() for p in trainable))
        write(out/"training.json", receipt)
        optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
        optimizer.zero_grad()
        for i, ex in enumerate(eligible):
            inputs = torch.tensor([ex["input_ids"]], device="mps")
            labels = torch.tensor([ex["labels"]], device="mps")
            loss = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels).loss
            assert torch.isfinite(loss).item()
            group = min(ACCUMULATION, len(eligible) - i//ACCUMULATION*ACCUMULATION)
            (loss/group).backward()
            receipt["loss_history"].append(float(loss.detach()))
            if (i+1)%ACCUMULATION == 0 or i+1 == len(eligible):
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()
                receipt["optimizer_steps"] += 1
                write(out/"training.json", receipt)
                print("step", receipt["optimizer_steps"], "examples", i+1, "grad_norm", float(norm), flush=True)
        assert frozen == {key: p._version for key, p in model.named_parameters() if not p.requires_grad}
        assert receipt["optimizer_steps"] > 0
        model.save_pretrained(out/"adapter", safe_serialization=True)
        tokenizer.save_pretrained(out/"tokenizer")
        receipt.update(status="completed", artifact_sha256=artifact_manifest(out), accepted_examples=len(eligible),
                       seconds=round(time.monotonic()-started, 3), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(out/"training.json", receipt)
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write(out/"training.json", receipt)
        raise
    del optimizer
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.load_adapter(out/"adapter", adapter_name="saved", is_trainable=False, local_files_only=True)
    model.set_adapter("saved")
    model.eval()
    reports = {}
    for condition in ("base", "adapter"):
        from contextlib import nullcontext
        reports[condition] = {}
        with (model.disable_adapter() if condition == "base" else nullcontext()), (out/f"{condition}.jsonl").open("x") as log:
            for case in plan["cases"]:
                row = generate(model, tokenizer, plan, case)
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                reports[condition][row["case_id"]] = row
                print(condition, row["case_id"], row["finish"], row["completion_tokens"], row["seconds"], flush=True)
    labels = {}
    blinded = []
    rng = random.Random(814)
    for case in plan["cases"]:
        key = case["id"]
        names = ["base", "adapter"]
        rng.shuffle(names)
        labels[key] = dict(zip(("A","B"),names))
        context = json.loads(case["messages"]["style"][1]["content"].split("\n",1)[1])
        blinded.append({"id": key, "question": case["question"], "evidence": context["factual_evidence"],
                        "answers": {label: reports[condition][key] for label, condition in labels[key].items()}})
    write(out/"blinded.json",blinded)
    write(out/"private-labels.json",labels)

if __name__ == "__main__":
    main()
