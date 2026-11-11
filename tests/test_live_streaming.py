import asyncio
import io
import wave
from contextlib import aclosing
import base64

import pytest

from bias.config import Settings, ProviderSettings
from bias.engine import Engine
from bias.live import decode_pcm, live_events
from bias.models import ChatRequest
from bias.providers import ProviderError
from bias.segments import SegmentCommitter


def wav_data(rate=16000, count=3200):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\x10" * count)
    return output.getvalue()


class Voice:
    settings = ProviderSettings("http://voice.invalid/v1", "test")
    def __init__(self):
        self.calls = []
    async def synthesize(self, text, **kwargs):
        self.calls.append(text)
        return wav_data(), "audio/wav"


def test_segment_commit_waits_for_decimals_abbreviations_and_complete_citations():
    s = SegmentCommitter(turn_id="turn", language="en", allowed_ids={"a"})
    assert s.push("Dr. Musk used 3.") == []
    assert s.push("14 units. [sou") == []
    assert s.push("rce:a]") == []
    emitted = s.push(" Next thought.")
    assert len(emitted) == 1
    assert emitted[0]["text"] == "Dr. Musk used 3.14 units."
    assert emitted[0]["evidence_ids"] == ["a"]
    tail = s.finish()
    assert tail[0]["sequence"] == 1 and tail[0]["text"] == "Next thought."
    assert emitted[0]["char_end"] == tail[0]["char_start"]


@pytest.mark.parametrize("text", ["이 방법은 저렴합니다. 다음 설명입니다.", "第一句。第二句。", "Why? Because it helps."])
def test_multilingual_units_keep_full_text(text):
    s = SegmentCommitter(turn_id="t", language="ko", allowed_ids=set())
    output = []
    for char in text:
        output += s.push(char)
    output += s.finish()
    assert "".join(r["raw_text"] for r in output) == text
    assert [r["sequence"] for r in output] == list(range(len(output)))


def test_first_audio_arrives_before_language_generation_completes(corpus):
    async def run():
        release = asyncio.Event()
        class LLM:
            async def stream(self, messages):
                yield "Reusable rockets lower costs. [source:rocket-reuse] Next"
                await release.wait()
                yield " sentence. [source:rocket-reuse]"
        engine = Engine(Settings(llm=ProviderSettings("http://llm.invalid/v1", "test")), corpus, llm=LLM())
        voice = Voice(); events = []
        async with aclosing(live_events(engine, voice, ChatRequest(message="Why reuse rockets?"), timeout=3, chunk_frames=1024)) as stream:
            async for kind, data in stream:
                events.append((kind, data))
                if kind == "audio" and not release.is_set():
                    assert not any(k == "result" for k, _ in events)
                    assert voice.calls == ["Reusable rockets lower costs."]
                    release.set()
        chunks = [d for k, d in events if k == "audio"]
        assert [d["sequence"] for d in chunks] == list(range(len(chunks)))
        offset = 0
        for chunk in chunks:
            assert chunk["sample_offset"] == offset
            assert len(base64.b64decode(chunk["pcm_base64"])) == chunk["sample_count"] * 2
            offset += chunk["sample_count"]
        ending = next(d for k, d in events if k == "audio_end")
        assert offset == ending["sample_count"] == 6400
        assert ending["spoken_complete"] is False and ending["acoustic_latency_ms"] is None
        assert events[-1][0] == "done"
    asyncio.run(run())


def test_cancellation_reaps_both_workers_and_does_not_store_partial_text(corpus):
    async def run():
        closed = []; voice_closed = []
        class LLM:
            async def stream(self, messages):
                try:
                    yield "First sentence. [source:rocket-reuse] Next"
                    await asyncio.Event().wait()
                finally:
                    closed.append(True)
        class SlowVoice(Voice):
            async def synthesize(self, text, **kwargs):
                try:
                    await asyncio.Event().wait()
                finally:
                    voice_closed.append(True)
        engine = Engine(Settings(llm=ProviderSettings("http://llm.invalid/v1", "test")), corpus, llm=LLM())
        baseline = asyncio.all_tasks()
        stream = live_events(engine, SlowVoice(), ChatRequest(message="Why reuse rockets?"), timeout=3)
        async for kind, _ in stream:
            if kind == "segment":
                await asyncio.sleep(0)
                await stream.aclose()
                break
        assert closed == [True] and voice_closed == [True]
        assert asyncio.all_tasks() == baseline
        assert all(not s.history and not s.lock.locked() and s.in_flight == 0 for s in engine.sessions.sessions.values())
    asyncio.run(run())


def test_invalid_voice_fails_without_a_false_complete_turn(corpus):
    async def run():
        class BadVoice(Voice):
            async def synthesize(self, text, **kwargs):
                return b"not audio", "audio/wav"
        events = []
        with pytest.raises(ProviderError, match="PCM WAV"):
            async for kind, data in live_events(Engine(Settings(), corpus), BadVoice(), ChatRequest(message="Why reuse rockets?")):
                events.append((kind, data))
        assert not any(k in {"done", "audio_end"} for k, _ in events)
        assert any(k == "turn_state" and d["state"] == "failed" for k, d in events)
    asyncio.run(run())


def test_pcm_rejects_truncated_frames_and_non_pcm_wave():
    assert decode_pcm(wav_data())[1:] == (16000, 1)
    with pytest.raises(ProviderError):
        decode_pcm(wav_data()[:-2])


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
@pytest.mark.parametrize("path", ["/api/chat/stream", "/api/chat/live"])
def test_asgi_disconnect_during_silent_provider_wait_cancels_owned_work(corpus, spec, path):
    from bias.api import create_app
    import json

    async def run():
        llm_started, llm_closed, voice_started, voice_closed = (asyncio.Event() for _ in range(4))
        class WaitingLLM:
            async def stream(self, messages):
                try:
                    yield "Rockets can be reused. [source:rocket-reuse] Next"
                    llm_started.set()
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
        app = create_app(Settings(llm=ProviderSettings("http://llm.invalid/v1", "test")), corpus=corpus, llm=WaitingLLM())
        async with app.router.lifespan_context(app):
            app.state.speech = WaitingVoice()
            incoming = asyncio.Queue(); sent = []
            body = json.dumps({"message": "Why reuse rockets?", "language": "en", "mode": "behavior"}).encode()
            incoming.put_nowait({"type": "http.request", "body": body, "more_body": False})
            async def send(message): sent.append(message)
            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": spec},
                     "method": "POST", "path": path, "raw_path": path.encode(), "query_string": b"",
                     "headers": [(b"content-type", b"application/json")], "scheme": "http",
                     "server": ("test", 80), "client": ("test", 50000), "http_version": "1.1", "root_path": ""}
            task = asyncio.create_task(app(scope, incoming.get, send))
            try:
                await asyncio.wait_for(llm_started.wait(), 2)
                if path.endswith("live"):
                    await asyncio.wait_for(voice_started.wait(), 2)
                await incoming.put({"type": "http.disconnect"})
                await asyncio.wait_for(task, 2)
                assert llm_closed.is_set()
                if path.endswith("live"):
                    assert voice_closed.is_set()
                assert all(not s.history and not s.lock.locked() and s.in_flight == 0 for s in app.state.engine.sessions.sessions.values())
                assert b"event: done" not in b"".join(m.get("body", b"") for m in sent)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_live_session_remains_owned_until_audio_cleanup(corpus):
    from bias.sessions import SessionCapacityError
    async def run():
        voice_started = asyncio.Event(); voice_closed = []
        class FastLLM:
            async def stream(self, messages):
                yield "Reusable rockets reduce launch costs. [source:rocket-reuse]"
        class WaitingVoice(Voice):
            async def synthesize(self, text, **kwargs):
                voice_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    voice_closed.append(True)
        engine = Engine(Settings(llm=ProviderSettings("http://llm.invalid/v1", "test")), corpus, llm=FastLLM())
        stream = live_events(engine, WaitingVoice(), ChatRequest(message="Why reuse rockets?"))
        queued = None
        try:
            async for kind, data in stream:
                if kind == "result":
                    session_id = data["turn"]["session_id"]
                    await asyncio.wait_for(voice_started.wait(), 1)
                    with pytest.raises(SessionCapacityError):
                        engine.sessions.delete(session_id)
                    queued = asyncio.create_task(engine.chat(ChatRequest(message="Why reuse rockets?", session_id=session_id)))
                    await asyncio.sleep(0)
                    assert not queued.done()
                    break
        finally:
            await stream.aclose()
        assert voice_closed == [True]
        await asyncio.wait_for(queued, 1)
        assert engine.sessions.delete(session_id)
    asyncio.run(run())


def test_send_failure_closes_suspended_live_iterator_before_response_exits(corpus):
    from bias.api import DisconnectStreamingResponse
    async def run():
        llm_closed, voice_closed, voice_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
        class LLM:
            async def stream(self, messages):
                try:
                    yield "Rockets lower costs. [source:rocket-reuse] Next"
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
        async def body():
            async with aclosing(live_events(engine, WaitingVoice(), ChatRequest(message="Why reuse rockets?"))) as stream:
                async for kind, _ in stream:
                    yield kind.encode()
        async def send(message):
            if message.get("body") == b"segment":
                await voice_started.wait()
                raise OSError("Client socket is closed")
        async def receive():
            await asyncio.Event().wait()
        response = DisconnectStreamingResponse(body())
        await asyncio.wait_for(response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send), 2)
        # Keep `response` strongly referenced: GC must not be doing the cleanup.
        assert response is not None and llm_closed.is_set() and voice_closed.is_set()
        assert all(s.in_flight == 0 and not s.turn_lock.locked() for s in engine.sessions.sessions.values())
    asyncio.run(run())
