from __future__ import annotations

import io
import math
import struct
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.voice.app import create_app
from services.voice.engine import VoiceBusy, VoiceEngine, VoiceOutputError, VoiceSettings, VoiceUnavailable, encode_wav, read_reference


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    path = tmp_path / "reference.wav"
    rate = 16000
    frames = b"".join(struct.pack("<h", int(5000 * math.sin(2 * math.pi * 220 * i / rate))) for i in range(rate))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(frames)
    return path


class FakeQwen:
    """Protocol fake; this is not synthesis or a model-quality test."""
    def __init__(self):
        self.prompt_calls = 0
        self.generation_calls = []

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_calls += 1
        assert kwargs["x_vector_only_mode"] is False
        assert kwargs["ref_text"] == "A synthetic test reference."
        assert kwargs["ref_audio"][1] == 16000
        return ["cached-test-prompt"]

    def generate_voice_clone(self, **kwargs):
        self.generation_calls.append(kwargs)
        return [[0.0, 0.1, -0.1] * 800], 24000


def configured(reference: Path, **kwargs) -> VoiceSettings:
    return VoiceSettings(reference_wav=reference, reference_text="A synthetic test reference.", **kwargs)


def test_missing_reference_never_downloads_or_loads_model():
    calls = []
    client = TestClient(create_app(VoiceSettings(), model_loader=lambda config: calls.append(config)))
    assert client.get("/health").json()["model_loaded"] is False
    response = client.post("/v1/audio/speech", json={"input": "Hello", "model": "qwen3-tts", "voice": "reference"})
    assert response.status_code == 503
    assert not calls


def test_service_reuses_prompt_and_returns_valid_wav(reference: Path):
    model = FakeQwen()
    loads = []

    def load(settings):
        loads.append(settings)
        return model

    client = TestClient(create_app(configured(reference), model_loader=load))
    for text in ("First message.", "Second message."):
        response = client.post("/v1/audio/speech", json={"input": text, "language": "English", "response_format": "wav"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["cache-control"] == "no-store"
        with wave.open(io.BytesIO(response.content), "rb") as stream:
            assert stream.getframerate() == 24000
            assert stream.getnchannels() == 1
            assert stream.getnframes() == 2400
    assert len(loads) == model.prompt_calls == 1
    assert len(model.generation_calls) == 2
    assert all(call["voice_clone_prompt"] == ["cached-test-prompt"] for call in model.generation_calls)


@pytest.mark.parametrize("payload", [
    {"input": "hello", "voice": "https://example.com/a.wav"},
    {"input": "hello", "ref_audio": "/etc/passwd"},
    {"input": "hello", "model": "../../another-model"},
    {"input": "hello", "response_format": "mp3"},
    {"input": "hello", "speed": 2},
    {"input": "  "},
    {"input": "a\u0000b"},
])
def test_client_cannot_select_server_files_models_or_unsupported_modes(payload):
    client = TestClient(create_app(VoiceSettings()))
    assert client.post("/v1/audio/speech", json=payload).status_code == 422


def test_authentication_and_input_bounds(reference: Path):
    model = FakeQwen()
    client = TestClient(create_app(configured(reference, api_key="test-secret", max_input_chars=5), model_loader=lambda _: model))
    assert client.post("/v1/audio/speech", json={"input": "Hello"}).status_code == 401
    response = client.post("/v1/audio/speech", json={"input": "Too long"}, headers={"Authorization": "Bearer test-secret"})
    assert response.status_code == 422
    assert model.prompt_calls == 0
    response = client.post("/v1/audio/speech", content=b"x" * 33000, headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_chunked_body_cannot_bypass_input_limit():
    client = TestClient(create_app(VoiceSettings(max_body_bytes=1024)))
    response = client.post("/v1/audio/speech", content=iter([b"x" * 800, b"x" * 800]), headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_nonfinite_and_multichannel_output_is_rejected():
    for values in ([0.0, float("nan")], [[0.0, 0.0]], []):
        with pytest.raises(VoiceOutputError):
            encode_wav(values, 24000)


def test_invalid_reference_is_reported_without_absolute_path(tmp_path: Path):
    path = tmp_path / "private-reference.wav"
    path.write_text("not a wave")
    client = TestClient(create_app(configured(path)))
    response = client.post("/v1/audio/speech", json={"input": "Hello"})
    assert response.status_code == 503
    assert str(tmp_path) not in response.text


def test_concurrent_inference_is_rejected_while_first_call_holds_lock(reference: Path):
    started, release = threading.Event(), threading.Event()

    class BlockingQwen(FakeQwen):
        def generate_voice_clone(self, **kwargs):
            started.set()
            assert release.wait(3)
            return super().generate_voice_clone(**kwargs)

    engine = VoiceEngine(configured(reference), lambda _: BlockingQwen())
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(engine.synthesize, "First", "English")
        assert started.wait(3)
        try:
            with pytest.raises(VoiceBusy):
                engine.synthesize("Second", "English")
        finally:
            release.set()
        assert first.result(timeout=3).startswith(b"RIFF")
    assert engine.status()["busy"] is False
