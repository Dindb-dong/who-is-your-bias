"""Mocked Google caption transports and fail-open behavior."""

import asyncio
import json
import os
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient

from bias.api import create_app
from bias.config import Settings
from bias.subtitles import GoogleSubtitleProvider, SubtitleTranslator


ENGLISH = "It can run RAG + TTS on 1 machine."
SOURCE = {"turn_id": "turn-a", "segment_id": "turn-a:1", "sequence": 1,
          "text": ENGLISH, "language": "en", "state": "committed"}


@pytest.mark.parametrize("protocol", ["google-web", "google-cloud"])
def test_google_captions_send_only_the_finalized_sentence(protocol):
    def respond(request):
        assert request.method == "POST"
        if protocol == "google-web":
            assert request.url.host == "translate.googleapis.com"
            assert request.url.path == "/translate_a/single"
            assert dict(request.url.params) == {"client": "gtx", "sl": "en", "tl": "ko", "dt": "t", "dj": "1"}
            assert "x-goog-api-key" not in request.headers
            assert parse_qs(request.content.decode()) == {"q": [ENGLISH]}
            response = {"sentences": [{"trans": "기기 1대에서 "}, {"trans": "RAG + TTS를 실행할 수 있습니다."}]}
        else:
            assert request.url.host == "translation.googleapis.com"
            assert request.url.path == "/language/translate/v2"
            assert not request.url.query
            assert request.headers["x-goog-api-key"] == "fixture"
            assert json.loads(request.content) == {"q": ENGLISH, "source": "en", "target": "ko", "format": "text"}
            response = {"data": {"translations": [{"translatedText": "기기 1대에서 RAG &#43; TTS를 실행할 수 있습니다."}]}}
        return httpx.Response(200, json=response)

    provider = GoogleSubtitleProvider(protocol, api_key="fixture", transport=httpx.MockTransport(respond))
    result = asyncio.run(SubtitleTranslator(google=provider).translate(SOURCE, "ko"))
    assert result == {"turn_id": "turn-a", "segment_id": "turn-a:1", "sequence": 1,
                      "text": "기기 1대에서 RAG + TTS를 실행할 수 있습니다.",
                      "language": "ko", "state": "ready", "basis": protocol.replace("-", "_")}


@pytest.mark.parametrize(("status", "payload", "reason"), [
    (200, {"unrecognized": "shape"}, "invalid_translation_format"),
    (200, {"sentences": [{"trans": "기기 2대에서 실행할 수 있습니다."}]}, "translation_numbers_changed"),
    (200, {"sentences": [{"trans": "English fallback on 1 machine."}]}, "translation_language_mismatch"),
    (200, {"sentences": [{"trans": "한" * 66_000}]}, "translation_output_limit"),
    (403, {}, "translation_provider_failed"),
])
def test_google_failures_signal_english_fallback(status, payload, reason):
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json=payload))
    translator = SubtitleTranslator(google=GoogleSubtitleProvider("google-web", transport=transport))
    result = asyncio.run(translator.translate(SOURCE, "ko"))
    assert result["state"] == "unavailable"
    assert result["text"] == ""
    assert result["reason"] == reason


def test_google_rate_limit_skips_next_attempt():
    attempts = []

    def respond(request):
        attempts.append(request)
        return httpx.Response(429, json={})

    translator = SubtitleTranslator(google=GoogleSubtitleProvider("google-web", transport=httpx.MockTransport(respond)))

    async def translate_twice():
        for _ in range(2):
            assert (await translator.translate(SOURCE, "ko"))["reason"] == "translation_busy"

    asyncio.run(translate_twice())
    assert len(attempts) == 1


def test_google_deadline_cancels_outstanding_translation():
    closed = False

    async def respond(request):
        nonlocal closed
        try:
            await asyncio.Event().wait()
        finally:
            closed = True

    provider = GoogleSubtitleProvider("google-web", transport=httpx.MockTransport(respond))
    translator = SubtitleTranslator(google=provider, timeout=0.02)
    assert asyncio.run(translator.translate(SOURCE, "ko"))["reason"] == "translation_timeout"
    assert closed


def test_google_unfinished_input_and_korean_source_are_never_sent():
    def respond(request):
        pytest.fail("unexpected Google request")

    provider = GoogleSubtitleProvider("google-web", transport=httpx.MockTransport(respond))
    translator = SubtitleTranslator(google=provider)
    assert asyncio.run(translator.translate({**SOURCE, "state": "draft"}, "ko"))["reason"] == "source_not_committed"
    assert asyncio.run(translator.translate({**SOURCE, "language": "ko"}, "en"))["reason"] == "unsupported_translation_pair"


def test_google_cannot_drop_the_spoken_engineering_uncertainty():
    payload = {"sentences": [{"trans": "팀은 첫 음성 대기를 줄이기 위해 이 구조를 골랐습니다."}]}
    provider = GoogleSubtitleProvider("google-web", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    translator = SubtitleTranslator(google=provider)
    source = {**SOURCE, "text": "One plausible engineering reason is to reduce the wait for the first audio."}
    assert asyncio.run(translator.translate(source, "ko"))["reason"] == "translation_uncertainty_lost"


@pytest.mark.parametrize(("protocol", "api_key", "available"), [
    ("google-web", "", True),
    ("google-cloud", "", False),
    ("google-cloud", "fixture", True),
])
def test_google_configuration_reaches_the_studio_capability(monkeypatch, protocol, api_key, available):
    for name in os.environ.copy():
        if name.startswith("BIAS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("BIAS_TRANSLATION_PROTOCOL", protocol)
    monkeypatch.setenv("BIAS_TRANSLATION_API_KEY", api_key)
    settings = Settings.from_env()
    assert settings.translation_timeout == 2.5
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/persona").json()["capabilities"]["translated_subtitles"] == available
