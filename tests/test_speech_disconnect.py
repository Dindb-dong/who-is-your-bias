"""ASGI disconnects must reach the owned upstream request and analysis work."""

import asyncio
import io
import json
import os
import wave

import httpx
import pytest

from bias.api import create_app
from bias.config import ProviderSettings, Settings
from bias.lipsync import LipsyncEngine
from bias.providers import SpeechProvider


def audio_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 1600)
    return output.getvalue()


class AudioStream(httpx.AsyncByteStream):
    def __init__(self, *, ready=False):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.closed = asyncio.Event()
        if ready:
            self.release.set()

    async def __aiter__(self):
        self.started.set()
        try:
            await self.release.wait()
            yield audio_bytes()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def aclose(self):
        # A checkpoint makes the test verify awaited resource cleanup, not
        # just that cancellation was requested before the handler returned.
        await asyncio.sleep(0)
        self.closed.set()


class HTTPCall:
    """Drive real ASGI messages; HTTPX's test client does not model disconnect."""

    def __init__(self, app, path, text="Hello"):
        self.messages = []
        self.incoming = asyncio.Queue()
        self.receivers = 0
        body = json.dumps({"text": text, "language": "en"}).encode()
        self.incoming.put_nowait({"type": "http.request", "body": body, "more_body": False})
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
            "method": "POST", "path": path, "raw_path": path.encode(), "query_string": b"",
            "headers": [(b"content-type", b"application/json")], "scheme": "http",
            "server": ("test", 80), "client": ("test", 50000), "http_version": "1.1", "root_path": "",
        }
        self.task = asyncio.create_task(app(scope, self.receive, self.send))

    async def receive(self):
        self.receivers += 1
        try:
            return await self.incoming.get()
        finally:
            self.receivers -= 1

    async def send(self, message):
        self.messages.append(message)

    async def disconnect(self):
        await self.incoming.put({"type": "http.disconnect"})

    async def wait(self):
        await asyncio.wait_for(asyncio.shield(self.task), 2)
        assert self.receivers == 0

    @property
    def status(self):
        return next(message["status"] for message in self.messages if message["type"] == "http.response.start")

    @property
    def body(self):
        return b"".join(message.get("body", b"") for message in self.messages if message["type"] == "http.response.body")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


def provider(stream):
    def handle(request):
        selected = stream[json.loads(request.content)["input"]] if isinstance(stream, dict) else stream
        return httpx.Response(200, headers={"content-type": "audio/wav"}, stream=selected)
    return SpeechProvider(ProviderSettings("http://speech.invalid/v1", "test-voice"), transport=httpx.MockTransport(handle))


@pytest.mark.parametrize("path", ["/api/speech", "/api/speech/prepare"])
def test_http_disconnect_cancels_and_closes_own_upstream_audio_request(corpus, path):
    async def run():
        app = create_app(Settings(), corpus=corpus)
        async with app.router.lifespan_context(app):
            baseline = asyncio.all_tasks()
            stream = AudioStream()
            app.state.speech = provider(stream)
            async with HTTPCall(app, path) as call:
                await asyncio.wait_for(stream.started.wait(), 1)
                assert app.state.speech_packages.active == (path == "/api/speech/prepare")
                await call.disconnect()
                await call.wait()
                assert stream.cancelled.is_set() and stream.closed.is_set()
                assert call.status == 499 and call.body == b""
                assert app.state.speech_packages.active == 0
                assert not app.state.speech_packages.clips
            assert asyncio.all_tasks() == baseline
    asyncio.run(run())


def test_prepare_disconnect_reaps_its_lipsync_process_and_releases_slot(corpus, tmp_path):
    marker = tmp_path / "analysis.pid"
    binary = tmp_path / "analysis-tool"
    binary.write_text("#!/usr/bin/env python3\nimport os,pathlib,time\npathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid()))\ntime.sleep(30)\n")
    binary.chmod(0o755)

    async def run():
        app = create_app(Settings(), corpus=corpus)
        async with app.router.lifespan_context(app):
            baseline = asyncio.all_tasks()
            app.state.speech = provider(AudioStream(ready=True))
            app.state.lipsync = LipsyncEngine(str(binary), timeout=5)
            async with HTTPCall(app, "/api/speech/prepare") as call:
                async with asyncio.timeout(2):
                    while not marker.exists():
                        await asyncio.sleep(.01)
                assert app.state.speech_packages.active == 1
                await call.disconnect()
                await call.wait()
                assert call.status == 499 and call.body == b""
                assert app.state.speech_packages.active == 0
                assert not app.state.speech_packages.clips
                with pytest.raises(ProcessLookupError):
                    os.kill(int(marker.read_text()), 0)
            assert asyncio.all_tasks() == baseline
    asyncio.run(run())


@pytest.mark.parametrize("path", ["/api/speech", "/api/speech/prepare"])
def test_server_task_cancellation_also_waits_for_cleanup(corpus, path):
    async def run():
        app = create_app(Settings(), corpus=corpus)
        async with app.router.lifespan_context(app):
            baseline = asyncio.all_tasks()
            stream = AudioStream()
            app.state.speech = provider(stream)
            async with HTTPCall(app, path) as call:
                await asyncio.wait_for(stream.started.wait(), 1)
                call.task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await call.wait()
                assert stream.cancelled.is_set() and stream.closed.is_set()
                assert not call.messages
                assert call.receivers == 0
                assert app.state.speech_packages.active == 0
                assert not app.state.speech_packages.clips
            assert asyncio.all_tasks() == baseline
    asyncio.run(run())


@pytest.mark.parametrize("path", ["/api/speech", "/api/speech/prepare"])
@pytest.mark.parametrize(("upstream_status", "public_status"), [(200, 200), (503, 502)])
def test_success_and_provider_errors_finish_the_disconnect_monitor(corpus, path, upstream_status, public_status):
    async def run():
        app = create_app(Settings(), corpus=corpus)
        async with app.router.lifespan_context(app):
            baseline = asyncio.all_tasks()
            app.state.lipsync = LipsyncEngine()
            app.state.speech = SpeechProvider(
                ProviderSettings("http://speech.invalid/v1", "test-voice"),
                transport=httpx.MockTransport(lambda request: httpx.Response(upstream_status, headers={"content-type": "audio/wav"}, content=audio_bytes())),
            )
            async with HTTPCall(app, path) as call:
                await call.wait()
                assert call.status == public_status
                assert app.state.speech_packages.active == 0
                if upstream_status == 200:
                    if path.endswith("/prepare"):
                        package = json.loads(call.body)
                        assert app.state.speech_packages.get(package["id"]).data == audio_bytes()
                    else:
                        assert call.body == audio_bytes()
                else:
                    assert "error" in json.loads(call.body)
                    assert not app.state.speech_packages.clips
            assert asyncio.all_tasks() == baseline
    asyncio.run(run())


def test_disconnect_frees_one_preparation_slot_without_cancelling_another(corpus):
    async def run():
        app = create_app(Settings(), corpus=corpus)
        async with app.router.lifespan_context(app):
            baseline = asyncio.all_tasks()
            streams = {"cancel": AudioStream(), "keep": AudioStream(), "retry": AudioStream(ready=True)}
            app.state.speech = provider(streams)
            app.state.lipsync = LipsyncEngine()
            async with HTTPCall(app, "/api/speech/prepare", "cancel") as first, HTTPCall(app, "/api/speech/prepare", "keep") as second:
                await asyncio.wait_for(asyncio.gather(streams["cancel"].started.wait(), streams["keep"].started.wait()), 1)
                assert app.state.speech_packages.active == 2
                await first.disconnect()
                await first.wait()
                assert first.status == 499 and first.body == b""
                assert app.state.speech_packages.active == 1
                assert not second.task.done() and not streams["keep"].cancelled.is_set()
                async with HTTPCall(app, "/api/speech/prepare", "retry") as retry:
                    await retry.wait()
                    assert retry.status == 200
                streams["keep"].release.set()
                await second.wait()
                assert second.status == 200
                assert app.state.speech_packages.active == 0
                assert len(app.state.speech_packages.clips) == 2
            assert asyncio.all_tasks() == baseline
    asyncio.run(run())
