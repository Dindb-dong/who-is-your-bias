import asyncio
import io
import struct
import sys
import wave

from bias.gesture import POSE_WIDTH, GestureEngine
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
print(json.dumps({"ready": True, "fps": 30, "device": "cpu"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
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
