import asyncio
import base64
import io
import struct
import sys
import wave

import pytest

from bias.config import ProviderSettings, Settings
from bias.engine import Engine
from bias.gesture import POSE_WIDTH, GestureEngine, GestureStreamError
from bias.live import live_events
from bias.models import ChatRequest
from bias.lipsync import LipsyncEngine
from bias.speech import SpeechPackages


def wav(seconds=1.0):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buffer.getvalue()


# Mirrors emage_worker.py's protocol without torch: frames follow the WAV length at 30 fps.
FAKE_WORKER = '''
import json, sys, wave
print("library noise on stdout is ignored", flush=True)
mode = sys.argv[1]
if mode == "never_ready":
    import time; time.sleep(30)
print(json.dumps({"ready": True, "fps": 30, "window": 64, "device": "cpu"}), flush=True)
import base64
sessions = {}
for line in sys.stdin:
    request = json.loads(line)
    if "op" in request:
        sid = request["session"]
        if request["op"] == "open":
            sessions[sid] = [0.0, 0]
            print(json.dumps({"id": request["id"], "ok": True, "window": 64}), flush=True)
            continue
        if mode == "stream_crash":
            raise SystemExit(4)
        state = sessions[sid]
        state[0] += len(base64.b64decode(request["pcm"])) / (2 * request["channels"] * request["rate"])
        ready = int(state[0] * 30) - (0 if request["final"] else 4)
        chunks = []
        while ready - state[1] > 0 and (request["final"] or ready - state[1] >= 30):
            n = min(60, ready - state[1])
            chunks.append({"start": state[1], "frames": n, "poses": base64.b64encode(b"\\0" * (n * 165 * 4)).decode()})
            state[1] += n
        print(json.dumps({"id": request["id"], "ok": True, "chunks": chunks, "compute_ms": 1.0}), flush=True)
        continue
    with wave.open(request["wav"]) as audio:
        frames = round(audio.getnframes() / audio.getframerate() * 30)
    if mode == "slow":
        import time; time.sleep(30)
    if mode == "crash":
        raise SystemExit(3)
    if mode == "wrong_length":
        frames += 90
    with open(request["out"], "wb") as out:
        out.write(b"\\0" * (frames * 165 * 4))
    print(json.dumps({"id": request["id"], "ok": True, "frames": frames, "fps": 30, "compute_ms": 1.5}), flush=True)
'''


def engine(tmp_path, mode="ok", **kwargs):
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    return GestureEngine(sys.executable, mode, worker=worker, **kwargs)


def run(coroutine_factory):
    async def body():
        return await coroutine_factory()
    return asyncio.run(body())


def test_unconfigured_and_unsupported_audio_fall_back():
    assert run(lambda: GestureEngine().generate(wav())).status == "not_configured"
    assert not GestureEngine().configured
    assert run(lambda: GestureEngine("python", "repo").generate(b"MP3")).status == "unsupported_audio"


def test_resident_worker_returns_bounded_motion_and_is_reused(tmp_path):
    async def body():
        gestures = engine(tmp_path)
        try:
            first = await gestures.generate(wav(1))
            process = gestures._process
            second = await gestures.generate(wav(2))
            assert gestures._process is process  # loaded once, reused per sentence
            return first, second
        finally:
            await gestures.close()
    first, second = asyncio.run(body())
    assert first.status == "available" and first.frames == 30 and len(first.motion) == 30 * POSE_WIDTH * 4
    assert second.frames == 60 and second.compute_ms == 1.5


def test_worker_failures_degrade_to_designed_movement(tmp_path):
    async def body(mode, **kwargs):
        gestures = engine(tmp_path, mode, **kwargs)
        try:
            return (await gestures.generate(wav())).status
        finally:
            await gestures.close()
    assert asyncio.run(body("crash")) == "failed"
    assert asyncio.run(body("wrong_length")) == "failed"
    assert asyncio.run(body("slow", timeout=1)) == "timeout"
    assert asyncio.run(body("never_ready", startup_timeout=1)) == "unavailable"
    assert run(lambda: GestureEngine("/missing/python", "repo").generate(wav())).status == "unavailable"


class FakeSpeech:
    async def synthesize(self, text, voice=None, language="en"):
        return wav(1), "audio/wav"


def test_prepare_publishes_motion_beside_audio_only_when_requested(tmp_path):
    async def body():
        packages, gestures = SpeechPackages(), engine(tmp_path)
        try:
            plain = await packages.prepare(FakeSpeech(), LipsyncEngine(), text="Hello")
            with_motion = await packages.prepare(FakeSpeech(), LipsyncEngine(), text="Hello", gesture=gestures)
            return packages, plain, with_motion
        finally:
            await gestures.close()
    packages, plain, with_motion = asyncio.run(body())
    assert "gesture" not in plain
    gesture = with_motion["gesture"]
    assert gesture["status"] == "available" and gesture["frames"] == 30 and gesture["width"] == POSE_WIDTH
    assert gesture["audio_sha256"] == with_motion["audio_sha256"]
    clip = packages.get(gesture["motion_url"].rsplit("/", 1)[1])
    assert clip.mime == "application/octet-stream" and len(clip.data) == 30 * POSE_WIDTH * 4
    assert struct.unpack("<f", clip.data[:4]) == (0.0,)
    assert with_motion["server_timing"]["gesture_ms"] >= 0


def test_prepare_reports_fallback_without_a_motion_url():
    packages = SpeechPackages()
    result = asyncio.run(packages.prepare(FakeSpeech(), LipsyncEngine(), text="Hello", gesture=GestureEngine()))
    assert result["gesture"] == {"status": "not_configured", "message": result["gesture"]["message"], "motion_url": None}


def test_stream_returns_contiguous_frames_on_the_audio_clock(tmp_path):
    async def body():
        gestures = engine(tmp_path)
        try:
            stream = gestures.stream()
            await stream.open()
            pcm = b"\x00\x00" * 24000 * 3          # 3 s at 24 kHz mono
            first = await stream.push(pcm, 24000, 1)
            last = await stream.push(b"", 24000, 1, final=True)
            return stream, first, last
        finally:
            await gestures.close()
    stream, first, last = asyncio.run(body())
    starts = [start for start, _ in first + last]
    sizes = [len(data) // (POSE_WIDTH * 4) for _, data in first + last]
    assert starts == [sum(sizes[:i]) for i in range(len(sizes))] and sum(sizes) == 90
    assert first and stream.window == 64 and stream.finished


def test_stream_failures_raise_a_fallback_status(tmp_path):
    async def body(mode):
        gestures = engine(tmp_path, mode)
        try:
            stream = gestures.stream()
            await stream.open()
            await stream.push(b"\x00\x00" * 16000, 16000, 1)
        finally:
            await gestures.close()
    with pytest.raises(GestureStreamError) as failure:
        asyncio.run(body("stream_crash"))
    assert failure.value.status == "failed"
    with pytest.raises(GestureStreamError) as missing:
        asyncio.run(GestureEngine().stream().open())
    assert missing.value.status == "not_configured"


class LiveVoice:
    settings = ProviderSettings("http://voice.invalid/v1", "test")

    async def synthesize(self, text, **kwargs):
        return wav(1.5), "audio/wav"


def collect_live(corpus, gestures):
    from contextlib import aclosing

    async def body():
        class LLM:
            async def stream(self, messages):
                yield "Reusable rockets lower costs. [source:rocket-reuse] They also fly again. [source:rocket-reuse]"
        live_engine = Engine(Settings(llm=ProviderSettings("http://llm.invalid/v1", "test")), corpus, llm=LLM())
        events = []
        try:
            async with aclosing(live_events(live_engine, LiveVoice(), ChatRequest(message="Why reuse rockets?", gesture=True),
                                            timeout=10, gesture=gestures)) as stream:
                async for kind, data in stream:
                    events.append((kind, data))
        finally:
            await gestures.close()
        return events
    return asyncio.run(body())


def test_live_voice_streams_gestures_beside_pcm_before_audio_end(tmp_path, corpus):
    events = collect_live(corpus, engine(tmp_path))
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "meta" and kinds.index("gesture_state") < kinds.index("gesture") < kinds.index("audio_end")
    state = next(data for kind, data in events if kind == "gesture_state")
    assert state["status"] == "streaming" and state["fps"] == 30 and state["window_frames"] == 64
    turn = events[0][1]["turn_id"]
    chunks = [data for kind, data in events if kind == "gesture"]
    assert all(chunk["turn_id"] == turn for chunk in chunks)
    assert [chunk["sequence"] for chunk in chunks] == list(range(len(chunks)))
    frame = 0
    for chunk in chunks:
        assert chunk["start_frame"] == frame and len(base64.b64decode(chunk["poses_base64"])) == chunk["frame_count"] * POSE_WIDTH * 4
        frame += chunk["frame_count"]
    end = next(data for kind, data in events if kind == "gesture_end")
    assert end == {"turn_id": turn, "chunks": len(chunks), "frames": frame, "status": "complete"}
    audio = next(data for kind, data in events if kind == "audio_end")
    assert frame == audio["sample_count"] * 30 // 16000 == 90     # two 1.5 s sentences on one clock


def test_live_voice_keeps_speaking_when_gestures_fail(tmp_path, corpus):
    events = collect_live(corpus, engine(tmp_path, "stream_crash"))
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "done" and "audio_end" in kinds and "gesture" not in kinds
    states = [data["status"] for kind, data in events if kind == "gesture_state"]
    assert states == ["streaming", "failed"]
    assert next(data for kind, data in events if kind == "gesture_end")["status"] == "stopped"
    unconfigured = collect_live(corpus, GestureEngine())
    assert [data["status"] for kind, data in unconfigured if kind == "gesture_state"] == ["not_configured"]
    assert "audio_end" in [kind for kind, _ in unconfigured]
