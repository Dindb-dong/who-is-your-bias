import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from bias.api import create_app
from bias.config import ProviderSettings, Settings
from bias.models import ChatTurn
from bias.providers import OpenAICompatibleLLM, ProviderError, SpeechProvider


def collect(provider):
    async def run():
        return "".join([part async for part in provider.stream([{"role": "user", "content": "Test"}])])
    return asyncio.run(run())


def test_llm_compatible_stream_and_secret_header():
    def handle(request):
        assert request.headers["authorization"] == "Bearer local-secret"
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["model"] == "local-model"
        data = 'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\ndata: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}\n\ndata: {"choices":[],"usage":{"total_tokens":2}}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})
    provider = OpenAICompatibleLLM(ProviderSettings("http://localhost:9999/v1", "local-model", "local-secret"), transport=httpx.MockTransport(handle))
    assert collect(provider) == "Hello world"


@pytest.mark.parametrize("body,code", [('data: not-json\n\n', "provider_protocol"), ('data: {"choices":[{"delta":{"content":"Partial"}}]}\n\n', "provider_incomplete"), ('data: [DONE]\n\n', "provider_empty")])
def test_stream_failures_are_not_silent(body, code):
    provider = OpenAICompatibleLLM(ProviderSettings("http://localhost:9999/v1", "test"), transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)))
    with pytest.raises(ProviderError) as error:
        collect(provider)
    assert error.value.code == code


def test_provider_error_body_does_not_leak_credentials():
    provider = OpenAICompatibleLLM(ProviderSettings("http://localhost:9999/v1", "test"), transport=httpx.MockTransport(lambda request: httpx.Response(401, text="secret detail")))
    with pytest.raises(ProviderError) as error:
        collect(provider)
    assert error.value.code == "provider_auth"
    assert "secret detail" not in str(error.value)


def test_audio_mime_is_verified_and_json_rejected():
    wav = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"sample"
    provider = SpeechProvider(ProviderSettings("http://localhost:9999/v1", "voice"), transport=httpx.MockTransport(lambda request: httpx.Response(200, content=wav)))
    assert asyncio.run(provider.synthesize("Hi")) == (wav, "audio/wav")
    invalid = SpeechProvider(ProviderSettings("http://localhost:9999/v1", "voice"), transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"error": "bad"})))
    with pytest.raises(ProviderError, match="audio response"):
        asyncio.run(invalid.synthesize("Hi"))


def test_total_deadline_cancels_stalled_provider_stream():
    closed = []
    class StalledStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(10)
            yield b"never"
        async def aclose(self):
            closed.append(True)
    provider = OpenAICompatibleLLM(ProviderSettings("http://localhost:9999/v1", "test"), timeout=0.02, transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=StalledStream())))
    with pytest.raises(ProviderError) as error:
        collect(provider)
    assert error.value.code == "provider_timeout"
    assert closed == [True]


def test_speech_output_size_is_bounded():
    class LargeAudio(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(3):
                yield b"x" * 8_000_000
    provider = SpeechProvider(ProviderSettings("http://localhost:9999/v1", "test"), transport=httpx.MockTransport(lambda request: httpx.Response(200, headers={"Content-Type": "audio/wav"}, stream=LargeAudio())))
    with pytest.raises(ProviderError) as error:
        asyncio.run(provider.synthesize("Test"))
    assert error.value.code == "audio_output_limit"


def test_http_and_sse_return_same_valid_contract(corpus, tmp_path):
    app = create_app(Settings(web_dir=tmp_path / "absent"), corpus=corpus)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["provider"] == "evidence"
        assert client.get("/api/persona").json()["stats"]["evidence"] == 2
        response = client.post("/api/chat", json={"message": "Why reuse rockets?"})
        assert response.status_code == 200
        turn = ChatTurn.model_validate(response.json())
        stream = client.post("/api/chat/stream", json={"message": "Why reuse rockets?", "session_id": turn.session_id})
        assert stream.headers["content-type"].startswith("text/event-stream")
        records = stream.text.strip().split("\n\n")
        assert records[0].startswith("event: meta")
        assert records[-1].startswith("event: done")
        result = next(record for record in records if record.startswith("event: result"))
        final = ChatTurn.model_validate(json.loads(result.split("\ndata: ")[1])["turn"])
        assert final.session_id == turn.session_id
        assert final.text == turn.text
        assert client.delete(f"/api/sessions/{turn.session_id}").json()["deleted"]
        assert client.post("/api/chat", json={"message": "Hi", "session_id": turn.session_id}).status_code == 404


def test_api_rejects_bad_input_and_reports_unconfigured_audio(corpus, tmp_path):
    with TestClient(create_app(Settings(web_dir=tmp_path / "absent"), corpus=corpus)) as client:
        assert client.post("/api/chat", json={"message": "   "}).status_code == 422
        assert client.post("/api/chat", json={"message": "x", "mode": "unknown"}).status_code == 422
        assert client.post("/api/chat", json={"message": "x" * 40_000}).status_code == 413
        speech = client.post("/api/speech", json={"text": "Hi"})
        assert speech.status_code == 503
        assert speech.json()["error"]["code"] == "speech_unconfigured"
        assert client.post("/api/transcribe", content=b"x", headers={"Content-Type": "text/plain"}).status_code == 415
        chunked = client.post("/api/chat", content=iter([b"x" * 20_000, b"x" * 20_000]), headers={"Content-Type": "application/json"})
        assert chunked.status_code == 413


def test_health_never_serializes_provider_secrets(corpus, tmp_path):
    config = Settings(web_dir=tmp_path / "absent", llm=ProviderSettings("http://localhost:9999/v1", "test", "never-print-this"))
    with TestClient(create_app(config, corpus=corpus)) as client:
        response = client.get("/api/health")
        assert "never-print-this" not in response.text
        assert response.json()["configuration_only"] is True
        assert response.json()["capabilities"]["llm"] is True
