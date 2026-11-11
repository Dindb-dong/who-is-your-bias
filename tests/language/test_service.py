"""Protocol/worker tests use explicit fake backends, never downloaded models."""

import asyncio
import builtins
import json
import os
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from bias.config import ProviderSettings, Settings
from bias.engine import Engine
from bias.providers import OpenAICompatibleLLM, ProviderError
from services.language.app import create_app, output_events
from services.language.engine import GenerationResult, LanguageEngine, LanguageError, LanguageSettings, TransformersBackend, model_location
from services.language.schemas import CompletionRequest


def request(**changes):
    return CompletionRequest(model="qwen3-local", messages=[{"role": "user", "content": "Hello"}], **changes)


class FakeBackend:
    def __init__(self, text="Hello there.", finish="stop"):
        self.text = text
        self.finish = finish
        self.requests = []

    def generate(self, payload, job):
        self.requests.append(payload)
        job.emit(self.text)
        return GenerationResult(10, 3, self.finish)


def test_lazy_loading_auth_and_json_usage_are_honest():
    backend = FakeBackend()
    calls = []

    def load(settings):
        calls.append(settings.served_model)
        return backend

    settings = LanguageSettings(api_key="secret-only-on-server")
    with TestClient(create_app(settings, loader=load)) as client:
        health = client.get("/health")
        assert health.json()["model_loaded"] is False
        assert "secret-only-on-server" not in health.text
        assert calls == []
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={b"Authorization": b"Bearer \xff"}).status_code == 401
        assert client.post("/v1/chat/completions", json=request().model_dump()).status_code == 401
        headers = {"Authorization": "Bearer secret-only-on-server"}
        result = client.post("/v1/chat/completions", json=request().model_dump(), headers=headers)
        assert result.status_code == 200
        assert result.json()["choices"][0]["message"]["content"] == "Hello there."
        assert result.json()["usage"] == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        assert client.post("/v1/chat/completions", json=request().model_dump(), headers=headers).status_code == 200
        assert calls == ["qwen3-local"]


def test_sse_works_through_the_existing_provider_and_preserves_length_error():
    async def run():
        for finish in ("stop", "length"):
            app = create_app(LanguageSettings(), loader=lambda settings: FakeBackend(finish=finish))
            provider = OpenAICompatibleLLM(ProviderSettings("http://test/v1", "qwen3-local"), transport=httpx.ASGITransport(app=app))
            if finish == "length":
                with pytest.raises(ProviderError) as error:
                    _ = [piece async for piece in provider.stream([{"role": "user", "content": "Hello"}])]
                assert error.value.code == "provider_truncated"
            else:
                assert "".join([piece async for piece in provider.stream([{"role": "user", "content": "Hello"}])]) == "Hello there."
    asyncio.run(run())


def test_existing_engine_uses_configured_token_budget_and_keeps_uncited_output_unverified(corpus):
    async def run():
        backend = FakeBackend("Reusable rockets can reduce launch cost.")
        app = create_app(LanguageSettings(max_output_tokens=160), loader=lambda settings: backend)
        settings = Settings(llm=ProviderSettings("http://language/v1", "qwen3-local"), llm_max_tokens=160)
        provider = OpenAICompatibleLLM(settings.llm, max_tokens=settings.llm_max_tokens, transport=httpx.ASGITransport(app=app))
        engine = Engine(settings, corpus, llm=provider)
        from bias.models import ChatRequest
        turn = await engine.chat(ChatRequest(message="Why reuse rockets?", mode="rag"))
        assert backend.requests[0].max_tokens == 160
        assert turn.provider == "openai"
        assert turn.grounding.status == "unverified"
        assert turn.citations == []
        assert Engine(settings, corpus).llm.max_tokens == 160
    asyncio.run(run())


def test_sse_error_has_no_fake_completion_or_done_marker():
    def unavailable(settings):
        raise LanguageError("model_unavailable", "The model is unavailable.", 503)
    with TestClient(create_app(LanguageSettings(), loader=unavailable)) as client:
        response = client.post("/v1/chat/completions", json=request(stream=True).model_dump())
        assert response.status_code == 200  # Headers were sent before lazy loading.
        assert '"code": "model_unavailable"' in response.text
        assert "[DONE]" not in response.text
        response = client.post("/v1/chat/completions", json=request().model_dump())
        assert response.status_code == 503


def test_disconnect_keeps_worker_admission_until_the_forward_pass_returns():
    entered = threading.Event()
    release = threading.Event()

    class BlockingBackend:
        def generate(self, payload, job):
            job.emit("First text")
            entered.set()
            release.wait(2)
            job.check()
            return GenerationResult(10, 2, "stop")

    async def run():
        engine = LanguageEngine(LanguageSettings(), lambda settings: BlockingBackend())
        job = engine.start(request())
        assert entered.wait(1)
        events = output_events(job)
        assert await anext(events) == "First text"
        await events.aclose()
        assert job.cancelled.is_set()
        assert engine.status()["busy"] is True
        assert engine.status()["stage"] == "stopping"
        with pytest.raises(LanguageError) as error:
            engine.start(request())
        assert error.value.status == 429
        release.set()
        assert await asyncio.to_thread(job.finished.wait, 1)
        assert job.error.code == "generation_cancelled"
        assert engine.status()["busy"] is False

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_loading_timeout_does_not_release_admission_early():
    entered = threading.Event()
    release = threading.Event()

    def load(settings):
        entered.set()
        release.wait(2)
        return FakeBackend()

    app = create_app(LanguageSettings(generation_timeout=0.04), loader=load)
    try:
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json=request().model_dump())
            assert response.status_code == 504
            assert entered.is_set()
            assert client.get("/health").json()["busy"] is True
            assert client.post("/v1/chat/completions", json=request().model_dump()).status_code == 429
            active = app.state.engine._active
            release.set()
            assert active.finished.wait(1)
            assert client.get("/health").json()["busy"] is False
    finally:
        release.set()


def test_bounded_queue_can_cancel_a_slow_consumer():
    first = threading.Event()

    class ManyChunks:
        def generate(self, payload, job):
            for _ in range(1000):
                job.emit("chunk ")
                first.set()
            return GenerationResult(1, 1000, "length")

    engine = LanguageEngine(LanguageSettings(queue_size=1), lambda settings: ManyChunks())
    job = engine.start(request())
    assert first.wait(1)
    assert job.queue.qsize() == 1
    job.cancel()
    assert job.finished.wait(1)
    assert job.output_chars <= 12
    assert engine.status()["busy"] is False


def test_output_limit_and_internal_errors_are_safe():
    with TestClient(create_app(LanguageSettings(max_output_chars=3), loader=lambda settings: FakeBackend("Too long"))) as client:
        response = client.post("/v1/chat/completions", json=request().model_dump())
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "output_limit"

    def broken(settings):
        raise RuntimeError("sensitive-model-path-and-secret")
    with TestClient(create_app(LanguageSettings(), loader=broken)) as client:
        response = client.post("/v1/chat/completions", json=request().model_dump())
        assert response.status_code == 502
        assert "sensitive-model-path-and-secret" not in response.text


def test_body_and_request_boundaries_apply_without_content_length():
    calls = []
    with TestClient(create_app(LanguageSettings(max_body_bytes=1024), loader=lambda settings: calls.append(True))) as client:
        response = client.post("/v1/chat/completions", content=iter([b"x" * 700, b"x" * 700]), headers={"Content-Type": "application/json"})
        assert response.status_code == 413
        wrong = request().model_dump()
        wrong["tools"] = [{"private": "unreflected"}]
        response = client.post("/v1/chat/completions", json=wrong)
        assert response.status_code == 422
        assert "unreflected" not in response.text
        wrong = request().model_dump()
        wrong["model"] = "not-served"
        assert client.post("/v1/chat/completions", json=wrong).status_code == 404
        wrong = request().model_dump()
        wrong["max_tokens"] = 1025
        assert client.post("/v1/chat/completions", json=wrong).status_code == 422
        assert calls == []


@pytest.mark.parametrize("value", ["0", "4097", "1.5", "nan", "inf"])
def test_main_application_rejects_invalid_token_budgets(monkeypatch, value):
    monkeypatch.setenv("BIAS_LLM_MAX_TOKENS", value)
    with pytest.raises(ValueError, match="BIAS_LLM_MAX_TOKENS"):
        Settings.from_env()


def test_main_application_token_budget_default_and_override(monkeypatch):
    monkeypatch.delenv("BIAS_LLM_MAX_TOKENS", raising=False)
    assert Settings.from_env().llm_max_tokens == 900
    monkeypatch.setenv("BIAS_LLM_MAX_TOKENS", "160")
    assert Settings.from_env().llm_max_tokens == 160


@pytest.mark.parametrize("value", [0, True, 1.5, 4097])
def test_provider_rejects_invalid_direct_token_budget(value):
    with pytest.raises(ValueError, match="max_tokens"):
        OpenAICompatibleLLM(ProviderSettings(), max_tokens=value)


def test_optional_runtime_telemetry_is_disabled_before_first_model_import(monkeypatch):
    original = builtins.__import__
    observed = []

    def import_module(name, *args, **kwargs):
        if name == "torch":
            observed.append(os.environ.get("ORT_DISABLE_TELEMETRY"))
            raise ImportError("Test intentionally skips optional model dependencies")
        return original(name, *args, **kwargs)

    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(LanguageError, match="optional language"):
        TransformersBackend(LanguageSettings())
    assert observed == ["1"]


def test_offline_loader_resolves_cached_snapshot_before_transformers(monkeypatch, tmp_path):
    calls = []

    def snapshot(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path / "cached-snapshot")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    settings = LanguageSettings(revision="pinned-revision", cache_dir=tmp_path, local_files_only=True)
    assert model_location(settings) == str(tmp_path / "cached-snapshot")
    assert calls == [("Qwen/Qwen3-0.6B", {"revision": "pinned-revision", "cache_dir": str(tmp_path), "local_files_only": True})]
    assert model_location(LanguageSettings(model_id=str(tmp_path))) == str(tmp_path.resolve())
    assert len(calls) == 1  # A configured local directory never contacts the Hub.
