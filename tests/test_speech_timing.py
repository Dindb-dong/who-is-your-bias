import asyncio
import hashlib
import io
import json
import os
import wave

import pytest
from fastapi.testclient import TestClient

from bias.api import create_app
from bias.config import Settings
from bias.lipsync import LipsyncEngine, LipsyncResult, parse_rhubarb, wav_duration_ms
from bias.providers import ProviderError
from bias.speech import SpeechPackages


def wav(seconds=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buffer.getvalue()


def cues():
    return {"metadata": {"duration": 1}, "mouthCues": [{"start": 0, "end": .2, "value": "X"}, {"start": .2, "end": 1, "value": "B"}]}


def test_timing_binds_audio_clock_and_does_not_claim_forced_alignment():
    audio = wav()
    result = parse_rhubarb(cues(), audio_sha256=hashlib.sha256(audio).hexdigest(), duration_ms=wav_duration_ms(audio))
    assert result["timing_basis"] == "audio_recognition"
    assert result["segments"][1] == {"start_ms": 200, "end_ms": 1000, "label": "B"}
    assert result["audio_duration_ms"] == 1000
    assert wav_duration_ms(audio[:-2]) is None
    assert wav_duration_ms(b"not audio") is None


@pytest.mark.parametrize("changes", [
    {"metadata": {"duration": 3}},
    {"mouthCues": [{"start": 0, "end": 2, "value": "B"}]},
    {"mouthCues": [{"start": 0, "end": .5, "value": "B"}, {"start": .4, "end": 1, "value": "C"}]},
    {"mouthCues": [{"start": float("nan"), "end": 1, "value": "B"}]},
    {"mouthCues": [{"start": 0, "end": 1, "value": "invented"}]},
])
def test_timing_rejects_overlaps_unknown_labels_and_wrong_duration(changes):
    with pytest.raises(ValueError):
        parse_rhubarb(cues() | changes, audio_sha256="a" * 64, duration_ms=1000)


def executable(tmp_path, body):
    path = tmp_path / "mouth-tool"
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)
    return str(path)


def test_real_subprocess_boundary_keeps_dialogue_out_of_shell(tmp_path):
    binary = executable(tmp_path, "import sys,json,pathlib\nargs=sys.argv[1:]\nassert '-d' in args\ntext=pathlib.Path(args[args.index('-d')+1]).read_text()\nassert text == '$(touch never); `echo hello`'\npathlib.Path(args[args.index('-o')+1]).write_text(" + repr(json.dumps(cues())) + ")\n")
    result = asyncio.run(LipsyncEngine(binary).analyze(wav(), "$(touch never); `echo hello`"))
    assert result.status == "available"
    assert result.timing["audio_sha256"] == hashlib.sha256(wav()).hexdigest()


def test_optional_analysis_falls_back_without_discarding_audio(tmp_path):
    assert asyncio.run(LipsyncEngine().analyze(wav(), "Hello")).status == "not_configured"
    assert asyncio.run(LipsyncEngine("/missing/tool").analyze(wav(), "Hello")).status == "unavailable"
    binary = executable(tmp_path, "raise SystemExit(7)\n")
    assert asyncio.run(LipsyncEngine(binary).analyze(wav(), "Hello")).status == "failed"
    assert asyncio.run(LipsyncEngine(binary).analyze(b"MP3", "Hello")).status == "unsupported_audio"


def test_analysis_timeout_reaps_its_subprocess(tmp_path):
    marker = tmp_path / "pid"
    binary = executable(tmp_path, "import os,time,pathlib\npathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid()))\ntime.sleep(20)\n")
    result = asyncio.run(LipsyncEngine(binary, timeout=1).analyze(wav(), "Hello"))
    assert result.status == "timeout"
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


def test_analysis_cancellation_reaps_its_subprocess(tmp_path):
    marker = tmp_path / "pid"
    binary = executable(tmp_path, "import os,time,pathlib\npathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid()))\ntime.sleep(20)\n")
    async def run():
        task = asyncio.create_task(LipsyncEngine(binary).analyze(wav(), "Hello"))
        async with asyncio.timeout(3):
            while not marker.exists():
                await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


def test_cache_evicts_by_total_bytes_and_expires_without_refresh():
    now = [0]
    cache = SpeechPackages(ttl=3, max_items=2, max_bytes=5, clock=lambda: now[0])
    first = cache.put(b"123", "audio/wav")
    second = cache.put(b"456", "audio/wav")
    assert cache.get(first) is None
    assert cache.get(second).data == b"456"
    now[0] = 3
    assert cache.get(second) is None
    assert cache.total_bytes == 0
    with pytest.raises(ProviderError):
        cache.put(b"123456", "audio/wav")


def test_concurrent_preparation_is_bounded_and_cancellation_releases_slot():
    async def run():
        started = asyncio.Event()
        class Provider:
            async def synthesize(self, text, voice, *, language=None):
                started.set()
                await asyncio.Event().wait()
        cache = SpeechPackages()
        tasks = [asyncio.create_task(cache.prepare(Provider(), LipsyncEngine(), text="Hello")) for _ in range(2)]
        await started.wait()
        with pytest.raises(ProviderError) as exc:
            await cache.prepare(Provider(), LipsyncEngine(), text="Hello")
        assert exc.value.status == 429
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert cache.active == 0
    asyncio.run(run())


def test_prepared_speech_api_retrieves_exact_bytes_and_clears_on_shutdown(corpus, tmp_path):
    audio = wav()
    class Provider:
        async def synthesize(self, text, voice, *, language=None):
            assert text == "Hello"
            assert language == "ko"
            return audio, "audio/wav"
    class Analysis:
        async def analyze(self, data, text, language):
            assert language == "ko"
            return LipsyncResult("available", parse_rhubarb(cues(), audio_sha256=hashlib.sha256(data).hexdigest(), duration_ms=1000))
    app = create_app(Settings(web_dir=tmp_path / "absent"), corpus=corpus)
    with TestClient(app) as client:
        app.state.speech, app.state.lipsync = Provider(), Analysis()
        result = client.post("/api/speech/prepare", json={"text": "Hello", "language": "ko"})
        assert result.status_code == 200
        package = result.json()
        assert package["timing_status"] == "available"
        assert package["speech_timing"]["audio_sha256"] == package["audio_sha256"]
        clip = client.get(package["audio_url"])
        assert clip.content == audio
        assert clip.headers["cache-control"] == "no-store"
        assert client.get("/api/speech/clips/not-an-id").status_code == 404
        assert client.post("/api/speech/prepare", json={"text": "Hello", "language": "invalid"}).status_code == 422
    assert not app.state.speech_packages.clips
