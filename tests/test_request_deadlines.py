"""Request budgets include queued ownership, with no orphaned live workers."""

import asyncio
import io
import wave
from contextlib import aclosing

import httpx
import pytest

from bias.config import ProviderSettings, Settings
from bias.engine import Engine
from bias.live import live_events
from bias.models import ChatRequest
from bias.providers import ProviderError


class Voice:
    settings = ProviderSettings("http://voice.invalid/v1", "test")

    async def synthesize(self, text, **kwargs):
        output = io.BytesIO()
        with wave.open(output, "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\x10" * 512)
        return output.getvalue(), "audio/wav"


async def drain(stream, seen):
    async with aclosing(stream):
        async for event in stream:
            seen.append(event)


async def assert_own_timeout(task, deadline):
    # Shield distinguishes the request's own deadline from this test's watchdog.
    # If the watchdog fires, the unfinished request below makes the test fail.
    remaining = max(0, deadline - asyncio.get_running_loop().time())
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), remaining + 0.3)
    assert task.done() and not task.cancelled()
    assert isinstance(task.exception(), TimeoutError)


@pytest.mark.parametrize("path", ["/api/chat/stream", "/api/chat/live"])
def test_queued_deadline_preserves_other_owner_and_emits_one_error(corpus, monkeypatch, path):
    import bias.api as api

    if path.endswith("live"):
        original = api.live_events

        def bounded_live(*args, **kwargs):
            return original(*args, **{**kwargs, "timeout": 1})

        monkeypatch.setattr(api, "live_events", bounded_live)

    async def run():
        app = api.create_app(Settings(request_timeout=0.01), corpus=corpus)
        async with app.router.lifespan_context(app):
            app.state.speech = Voice()
            engine = app.state.engine
            session = engine.sessions.get_or_create(None)
            await session.turn_lock.acquire()
            task = None
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                    task = asyncio.create_task(client.post(path, json={"message": "Why reuse rockets?", "session_id": session.id}))
                    budget = 1 if path.endswith("live") else 2.01
                    response = await asyncio.wait_for(asyncio.shield(task), budget + 0.3)
                assert response.status_code == 200
                assert response.text.count("event: error\n") == 1
                assert '"code":"request_timeout"' in response.text.replace(" ", "")
                assert "event: meta" not in response.text
                assert "event: done" not in response.text
                assert "event: audio_end" not in response.text
                assert session.turn_lock.locked()  # Still belongs to the earlier turn.
                assert session.in_flight == 0 and not session.history
            finally:
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                if session.turn_lock.locked():
                    session.turn_lock.release()

    asyncio.run(run())


def test_generation_receives_only_budget_remaining_after_queue(corpus):
    async def run():
        started, closed = asyncio.Event(), asyncio.Event()

        class WaitingLLM:
            async def stream(self, messages):
                started.set()
                try:
                    await asyncio.Event().wait()
                    yield "Unreachable response."
                finally:
                    closed.set()

        engine = Engine(Settings(request_timeout=0.01, llm=ProviderSettings("http://llm.invalid/v1", "test")), corpus, llm=WaitingLLM())
        session = engine.sessions.get_or_create(None)
        await session.turn_lock.acquire()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2.01
        release = loop.call_later(0.8, session.turn_lock.release)
        seen = []
        task = asyncio.create_task(drain(engine.events(ChatRequest(message="Why reuse rockets?", session_id=session.id)), seen))
        try:
            await assert_own_timeout(task, deadline)
            assert started.is_set() and closed.is_set()
            assert not session.history and not session.lock.locked()
            assert not session.turn_lock.locked() and session.in_flight == 0
            assert not any(kind in {"result", "done"} for kind, _ in seen)
        finally:
            release.cancel()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if session.turn_lock.locked():
                session.turn_lock.release()

    asyncio.run(run())


@pytest.mark.parametrize("finish_text", [False, True])
def test_live_queue_and_audio_share_deadline_and_reap_workers(corpus, finish_text):
    async def run():
        llm_closed, voice_started, voice_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class LLM:
            async def stream(self, messages):
                try:
                    yield "Reusable rockets reduce costs. [source:rocket-reuse]"
                    if not finish_text:
                        yield " Next"
                        await asyncio.Event().wait()
                finally:
                    llm_closed.set()

        class WaitingVoice(Voice):
            async def synthesize(self, text, **kwargs):
                voice_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    voice_closed.set()

        engine = Engine(Settings(llm=ProviderSettings("http://llm.invalid/v1", "test")), corpus, llm=LLM())
        session = engine.sessions.get_or_create(None)
        await session.turn_lock.acquire()
        baseline = asyncio.all_tasks()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 1
        release = loop.call_later(0.6, session.turn_lock.release)
        seen = []
        task = asyncio.create_task(drain(live_events(engine, WaitingVoice(), ChatRequest(message="Why reuse rockets?", session_id=session.id), timeout=1), seen))
        try:
            await assert_own_timeout(task, deadline)
            assert voice_started.is_set() and voice_closed.is_set() and llm_closed.is_set()
            assert not session.turn_lock.locked() and not session.lock.locked() and session.in_flight == 0
            assert asyncio.all_tasks() == baseline
            assert any(kind == "result" for kind, _ in seen) is finish_text
            assert bool(session.history) is finish_text
            assert not any(kind in {"done", "audio_end"} for kind, _ in seen)
        finally:
            release.cancel()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if session.turn_lock.locked():
                session.turn_lock.release()

    asyncio.run(run())


def test_live_without_session_id_creates_one_reusable_session(corpus):
    async def run():
        class LLM:
            async def stream(self, messages):
                yield "Reusable rockets reduce costs. [source:rocket-reuse]"

        engine = Engine(Settings(llm=ProviderSettings("http://llm.invalid/v1", "test")), corpus, llm=LLM())
        seen = []
        await drain(live_events(engine, Voice(), ChatRequest(message="Why reuse rockets?")), seen)
        meta = next(data for kind, data in seen if kind == "meta")
        result = next(data["turn"] for kind, data in seen if kind == "result")
        assert result["session_id"] == meta["session_id"]
        assert set(engine.sessions.sessions) == {meta["session_id"]}
        session = engine.sessions.sessions[meta["session_id"]]
        assert len(session.history) == 2 and session.in_flight == 0
        assert not session.turn_lock.locked()
        followup = []
        await drain(live_events(engine, Voice(), ChatRequest(message="Why that approach?", session_id=session.id)), followup)
        assert len(engine.sessions.sessions) == 1 and len(session.history) == 4
        assert next(data for kind, data in followup if kind == "meta")["session_id"] == session.id

    asyncio.run(run())


def test_unconfigured_no_id_live_request_does_not_allocate_a_session(corpus):
    async def run():
        voice = Voice()
        voice.settings = ProviderSettings()
        engine = Engine(Settings(), corpus)
        with pytest.raises(ProviderError, match="Configure a speech provider"):
            await drain(live_events(engine, voice, ChatRequest(message="Why reuse rockets?")), [])
        assert not engine.sessions.sessions

    asyncio.run(run())
