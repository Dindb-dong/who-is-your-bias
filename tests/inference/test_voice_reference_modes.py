"""Configuration boundaries, using a synthetic WAV and a protocol fake only."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.voice.app import create_app
from services.voice.engine import VoiceEngine, VoiceSettings, VoiceUnavailable, read_reference


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic-reference.wav"
    rate = 16000
    frames = b"".join(struct.pack("<h", int(5000 * math.sin(2 * math.pi * 220 * index / rate))) for index in range(rate))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(frames)
    return path


def test_default_reference_mode_still_requires_accurate_transcript(reference: Path):
    for transcript in (None, "", "  "):
        settings = VoiceSettings(reference_wav=reference, reference_text=transcript)
        assert settings.x_vector_only_mode is False
        with pytest.raises(VoiceUnavailable, match="accurate transcript"):
            read_reference(settings)
        assert VoiceEngine(settings).status()["reference_configured"] is False


def test_embedding_mode_does_not_require_or_send_a_transcript(reference: Path):
    calls = []

    class Model:
        def create_voice_clone_prompt(self, **kwargs):
            calls.append(kwargs)
            return ["synthetic-prompt"]

        def generate_voice_clone(self, **kwargs):
            return [[0.0, 0.1, -0.1] * 800], 24000

    settings = VoiceSettings(reference_wav=reference, x_vector_only_mode=True)
    engine = VoiceEngine(settings, model_loader=lambda _: Model())
    assert engine.status()["reference_configured"] is True
    assert engine.status()["reference_mode"] == "speaker_embedding_only"
    assert engine.synthesize("A protocol test.", "English").startswith(b"RIFF")
    assert len(calls) == 1
    assert calls[0]["x_vector_only_mode"] is True
    assert calls[0]["ref_text"] is None
    assert calls[0]["ref_audio"][1] == 16000
    assert engine.status()["busy"] is False


def test_embedding_mode_keeps_wav_validation_and_lock_cleanup(tmp_path: Path):
    settings = VoiceSettings(reference_wav=tmp_path / "missing.wav", x_vector_only_mode=True)
    calls = []
    engine = VoiceEngine(settings, model_loader=lambda value: calls.append(value))
    with pytest.raises(VoiceUnavailable, match="missing"):
        engine.synthesize("Hello.", "English")
    assert not calls
    assert engine.status()["busy"] is False


def test_clients_cannot_opt_in_to_embedding_mode(reference: Path):
    client = TestClient(create_app(VoiceSettings(reference_wav=reference)))
    response = client.post("/v1/audio/speech", json={"input": "Hello.", "x_vector_only_mode": True})
    assert response.status_code == 422


def test_environment_requires_explicit_boolean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BIAS_VOICE_X_VECTOR_ONLY_MODE", raising=False)
    assert VoiceSettings.from_env().x_vector_only_mode is False
    monkeypatch.setenv("BIAS_VOICE_X_VECTOR_ONLY_MODE", "true")
    assert VoiceSettings.from_env().x_vector_only_mode is True
    monkeypatch.setenv("BIAS_VOICE_X_VECTOR_ONLY_MODE", "false")
    assert VoiceSettings.from_env().x_vector_only_mode is False
    monkeypatch.setenv("BIAS_VOICE_X_VECTOR_ONLY_MODE", "maybe")
    with pytest.raises(ValueError, match="BIAS_VOICE_X_VECTOR_ONLY_MODE"):
        VoiceSettings.from_env()
    with pytest.raises(ValueError, match="must be a boolean"):
        VoiceSettings(x_vector_only_mode="false")  # type: ignore[arg-type]
