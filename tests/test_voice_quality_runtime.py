"""Runtime reproducibility and reporting; no actual model is loaded here."""

from __future__ import annotations

import contextlib
import math
import struct
import sys
import types
import wave
from pathlib import Path

import pytest

from services.voice.engine import SeededQwen, VoiceEngine, VoiceSettings, model_location


@pytest.mark.parametrize("settings", [
    {"revision": "main"}, {"local_files_only": "true"}, {"do_sample": "false"},
    {"seed": True}, {"seed": -1}, {"threads": 0}, {"temperature": float("nan")},
    {"temperature": 0}, {"top_p": 2}, {"top_k": 101}, {"repetition_penalty": 0.9},
])
def test_quality_configuration_rejects_invalid_or_unpinned_values(settings):
    with pytest.raises(ValueError):
        VoiceSettings(**settings)


def test_offline_resolver_passes_immutable_revision_and_cache(monkeypatch, tmp_path):
    calls = []
    def snapshot(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path / "snapshot")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot))
    revision = "a" * 40
    settings = VoiceSettings(revision=revision, cache_dir=tmp_path / "cache")
    assert model_location(settings) == str(tmp_path / "snapshot")
    assert calls[0][1] == {"revision": revision, "cache_dir": str(tmp_path / "cache"), "local_files_only": True}
    calls.clear()
    assert model_location(VoiceSettings(model_id=str(tmp_path))) == str(tmp_path)
    assert not calls


def test_production_seed_wrapper_resets_seed_per_generation_without_seeding_prompt():
    calls = []
    model = types.SimpleNamespace(create_voice_clone_prompt=lambda **kw: calls.append(("prompt", kw)),
        generate_voice_clone=lambda **kw: calls.append(("generation", kw)), model=object())
    torch = types.SimpleNamespace(manual_seed=lambda value: calls.append(("seed", value)), inference_mode=contextlib.nullcontext)
    wrapped = SeededQwen(model, torch, 42)
    assert wrapped.model is model.model
    wrapped.create_voice_clone_prompt(reference="test")
    wrapped.generate_voice_clone(text="one")
    wrapped.generate_voice_clone(text="two")
    assert [kind for kind, value in calls] == ["prompt", "seed", "generation", "seed", "generation"]


def test_runtime_passes_explicit_sampling_and_reports_unknown_termination(tmp_path: Path):
    audio = tmp_path / "reference.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b''.join(struct.pack('<h', int(5000 * math.sin(i * .1))) for i in range(16000)))
    calls = []
    model = types.SimpleNamespace(create_voice_clone_prompt=lambda **kw: ["fake"],
        generate_voice_clone=lambda **kw: (calls.append(kw) or [[0.1] * 2400], 24000))
    settings = VoiceSettings(reference_wav=audio, reference_text="Synthetic reference.", max_new_tokens=512)
    engine = VoiceEngine(settings, model_loader=lambda config: model)
    assert engine.synthesize("한국어 테스트", "Korean").startswith(b"RIFF")
    assert calls[0]["language"] == "Korean"
    assert calls[0]["max_new_tokens"] == 512
    assert calls[0]["repetition_penalty"] == 1.05
    assert calls[0]["subtalker_dosample"] is True
    assert calls[0]["non_streaming_mode"] is False
    status = engine.status()
    assert status["streaming"] is False
    assert status["last_generation"]["termination_reason"] == "not_exposed_by_backend"
    assert status["last_generation"]["audio_seconds"] == 0.1
    assert status["last_generation"]["cold_start"] is True
    engine.synthesize("English test", "English")
    assert engine.status()["last_generation"]["cold_start"] is False
    assert "한국어 테스트" not in str(status)


def test_environment_reads_seed_and_strict_offline_booleans(monkeypatch):
    monkeypatch.setenv("BIAS_VOICE_SEED", "7")
    monkeypatch.setenv("BIAS_VOICE_THREADS", "2")
    monkeypatch.setenv("BIAS_VOICE_LOCAL_FILES_ONLY", "false")
    settings = VoiceSettings.from_env()
    assert (settings.seed, settings.threads, settings.local_files_only) == (7, 2, False)
    monkeypatch.setenv("BIAS_VOICE_DO_SAMPLE", "maybe")
    with pytest.raises(ValueError, match="DO_SAMPLE"):
        VoiceSettings.from_env()
