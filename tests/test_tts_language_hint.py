"""Language propagation with synthetic audio and in-process HTTP transports."""

from __future__ import annotations

import asyncio
import json
import math
import struct
import types
import wave

import httpx
import pytest
from fastapi.testclient import TestClient

from bias.api import create_app
from bias.config import ProviderSettings, Settings
from bias.models import SpeechPackageRequest, SpeechRequest
from bias.providers import SpeechProvider
from services.voice.app import create_app as create_voice_app
from services.voice.engine import VoiceSettings


def test_generic_speech_payload_is_unchanged_even_when_app_language_is_known():
    payloads = []
    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, content=b"RIFF0000WAVEtest", headers={"content-type": "audio/wav"})
    provider = SpeechProvider(ProviderSettings("http://localhost/v1", "generic-voice"), transport=httpx.MockTransport(handle))
    asyncio.run(provider.synthesize("테스트", language="ko"))
    assert payloads == [{"model": "generic-voice", "input": "테스트", "voice": "default", "response_format": "wav"}]


def test_optional_simple_language_and_existing_package_default_are_preserved():
    assert SpeechRequest(text="Hello").language is None
    assert SpeechPackageRequest(text="Hello").language == "en"
    assert SpeechRequest(text="안녕하세요", language="ko").language == "ko"


@pytest.mark.parametrize("endpoint", ["/api/speech", "/api/speech/prepare"])
def test_enabled_korean_hint_reaches_qwen_through_both_application_routes(endpoint, corpus, tmp_path):
    reference = tmp_path / "synthetic-reference.wav"
    with wave.open(str(reference), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b''.join(struct.pack('<h', int(5000 * math.sin(i * 0.1))) for i in range(16000)))
    calls = []
    model = types.SimpleNamespace(create_voice_clone_prompt=lambda **kw: ["fake"],
        generate_voice_clone=lambda **kw: (calls.append(kw) or [[0.1] * 2400], 24000))
    voice_app = create_voice_app(VoiceSettings(reference_wav=reference, reference_text="A synthetic reference."), model_loader=lambda _: model)
    app = create_app(Settings(web_dir=tmp_path / "absent", tts=ProviderSettings("http://voice/v1", "qwen3-tts"),
        tts_voice="reference", tts_language_hint=True), corpus=corpus)
    with TestClient(app) as client:
        app.state.speech.transport = httpx.ASGITransport(app=voice_app)
        response = client.post(endpoint, json={"text": "한국어 음성 테스트입니다.", "language": "ko"})
        assert response.status_code == 200
        assert len(calls) == 1
        assert calls[0]["language"] == "Korean"
        assert client.post(endpoint, json={"text": "Test", "language": "unknown"}).status_code == 422


def test_enabled_provider_preserves_auto_for_unspecified_language_and_maps_english():
    payloads = []
    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, content=b"RIFF0000WAVEtest")
    provider = SpeechProvider(ProviderSettings("http://localhost/v1", "qwen3-tts"), language_hint=True,
        transport=httpx.MockTransport(handle))
    asyncio.run(provider.synthesize("Hello"))
    asyncio.run(provider.synthesize("Hello", language="en"))
    assert "language" not in payloads[0]
    assert payloads[1]["language"] == "English"


def test_language_hint_environment_is_opt_in_and_strict(monkeypatch):
    monkeypatch.delenv("BIAS_TTS_LANGUAGE_HINT", raising=False)
    assert Settings.from_env().tts_language_hint is False
    monkeypatch.setenv("BIAS_TTS_LANGUAGE_HINT", "true")
    assert Settings.from_env().tts_language_hint is True
    monkeypatch.setenv("BIAS_TTS_LANGUAGE_HINT", "maybe")
    with pytest.raises(ValueError, match="BIAS_TTS_LANGUAGE_HINT"):
        Settings.from_env()
    with pytest.raises(ValueError, match="tts_language_hint"):
        Settings(tts_language_hint="true")
