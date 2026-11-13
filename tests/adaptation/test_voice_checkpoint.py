"""Tiny neural adapters exercise interrupted optimizer/RNG continuation only."""

from copy import deepcopy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from torch import nn
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model

from services.voice.adapters import TARGET_MODULES, read_checkpoint
from tools.adaptation.voice_lora import restore_checkpoint, save_checkpoint


def tiny_model():
    torch.manual_seed(41)
    model = nn.Module()
    model.talker = nn.Module(); model.talker.model = nn.Module()
    model.talker.model.layers = nn.ModuleList([nn.Module()])
    attention = nn.Module()
    model.talker.model.layers[0].self_attn = attention
    attention.q_proj = nn.Linear(4, 4); attention.v_proj = nn.Linear(4, 4)
    model.requires_grad_(False)
    config = {"r": 2, "lora_alpha": 4, "lora_dropout": 0.05, "target_modules": TARGET_MODULES, "bias": "none"}
    inject_adapter_in_model(LoraConfig(**config), model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    contract = {"persona_id": "synthetic-test", "base_revision": "a"*40, "lora": config,
                "dataset_manifest_sha256": "b"*64}
    return model, optimizer, contract


def update(model, optimizer):
    attention = model.talker.model.layers[0].self_attn
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    loss = (attention.q_proj(x) + attention.v_proj(x)).square().mean()
    loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def test_fresh_process_equivalent_load_restores_weights_optimizer_and_rng(tmp_path):
    full, opt, _ = tiny_model()
    loss1 = update(full, opt); loss2 = update(full, opt)
    expected = deepcopy(get_peft_model_state_dict(full))
    partial, partial_opt, contract = tiny_model()
    first = update(partial, partial_opt)
    checkpoint = save_checkpoint(tmp_path / "run", model=partial, optimizer=partial_opt,
        contract=contract, step=1, cursor=1, losses=[first])
    resumed, resumed_opt, contract = tiny_model()
    step, cursor, losses = restore_checkpoint(checkpoint, model=resumed, optimizer=resumed_opt, contract=contract)
    assert (step, cursor, losses) == (1, 1, [loss1])
    assert update(resumed, resumed_opt) == loss2
    actual = get_peft_model_state_dict(resumed)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)


def test_resume_rejects_changed_contract_or_checkpoint_bytes(tmp_path):
    model, opt, contract = tiny_model()
    loss = update(model, opt)
    checkpoint = save_checkpoint(tmp_path / "run", model=model, optimizer=opt,
        contract=contract, step=1, cursor=1, losses=[loss])
    with pytest.raises(ValueError, match="Resume contract differs"):
        restore_checkpoint(checkpoint, model=model, optimizer=opt, contract=dict(contract, dataset_manifest_sha256="c"*64))
    weights = checkpoint / "adapter.safetensors"
    weights.write_bytes(weights.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact hash"):
        read_checkpoint(checkpoint, persona_id="synthetic-test")


def test_known_good_checkpoint_is_never_overwritten(tmp_path):
    model, opt, contract = tiny_model()
    loss = update(model, opt)
    output = tmp_path / "run"
    checkpoint = save_checkpoint(output, model=model, optimizer=opt, contract=contract, step=1, cursor=1, losses=[loss])
    with pytest.raises(ValueError, match="overwrite"):
        save_checkpoint(output, model=model, optimizer=opt, contract=contract, step=1, cursor=1, losses=[loss])
    latest = json.loads((output / "latest.json").read_text())
    assert latest["path"] == checkpoint.name
