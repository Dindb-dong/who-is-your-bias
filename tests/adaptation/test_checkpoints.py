"""Real optimizer/RNG equivalence; optional Torch dependency stays out of core."""
from pathlib import Path
import random

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from tools.adaptation.checkpoints import capture_state, load_checkpoint, restore_state, save_checkpoint


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.25)
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, value):
        return self.linear(self.dropout(value))

    def save_pretrained(self, path, safe_serialization=True):
        path.mkdir()
        safetensors.save_file(self.state_dict(), str(path / "adapter_model.safetensors"))


def update(model, optimizer):
    optimizer.zero_grad()
    value = torch.tensor([[random.random(), 0.3, 0.9]])
    model(value).square().sum().backward()
    optimizer.step()


def test_pause_restore_matches_uninterrupted_optimizer_and_rng(tmp_path):
    random.seed(7)
    torch.manual_seed(7)
    model = TinyAdapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    update(model, optimizer)
    binding = {"dataset_sha256": "a" * 64, "seed": 7, "base_revision": "b" * 40}
    checkpoint = save_checkpoint(tmp_path, model=model,
        state=capture_state(optimizer, cursor={"epoch": 0, "next_index": 1}, optimizer_steps=1),
        binding=binding, report={"optimizer_steps": 1})
    update(model, optimizer)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    # Deliberately disturb every state restored by a new training process.
    random.seed(123)
    torch.manual_seed(123)
    restored = TinyAdapter()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.03)
    metadata, state = load_checkpoint(checkpoint, binding=binding)
    restored.load_state_dict(safetensors.load_file(str(checkpoint / "adapter/adapter_model.safetensors")))
    restore_state(restored_optimizer, state)
    update(restored, restored_optimizer)
    assert metadata["optimizer_steps"] == 1
    assert state["cursor"] == {"epoch": 0, "next_index": 1}
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name])
    assert restored_optimizer.state_dict()["state"][0]["step"].item() == 2


def test_checkpoint_rejects_changed_binding_payload_and_overwrite(tmp_path):
    model = TinyAdapter()
    optimizer = torch.optim.AdamW(model.parameters())
    update(model, optimizer)
    args = dict(model=model, state=capture_state(optimizer, cursor={"epoch": 0, "next_index": 1}, optimizer_steps=1),
                binding={"dataset": "original"}, report={"optimizer_steps": 1})
    checkpoint = save_checkpoint(tmp_path, **args)
    with pytest.raises(ValueError, match="settings changed"):
        load_checkpoint(checkpoint, binding={"dataset": "different"})
    with pytest.raises(ValueError, match="overwrite"):
        save_checkpoint(tmp_path, **args)
    weights = checkpoint / "adapter/adapter_model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="changed"):
        load_checkpoint(checkpoint)


def test_failed_publisher_retains_local_checkpoint_and_stops(tmp_path):
    import sys
    model = TinyAdapter()
    optimizer = torch.optim.AdamW(model.parameters())
    update(model, optimizer)
    with pytest.raises(ValueError, match="verified receipt"):
        save_checkpoint(tmp_path, model=model,
            state=capture_state(optimizer, cursor={"epoch": 0, "next_index": 1}, optimizer_steps=1),
            binding={"dataset": "original"}, report={"optimizer_steps": 1},
            publisher=[sys.executable, "-c", "print('{}')"])
    metadata, _ = load_checkpoint(tmp_path / "step-00000001")
    assert metadata["optimizer_steps"] == 1
