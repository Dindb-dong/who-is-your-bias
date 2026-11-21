import asyncio
import base64
import json
import sys

import pytest
from fastapi.testclient import TestClient

from bias.api import create_app
from bias.config import Settings
from bias.gesture import POSE_WIDTH, GestureEngine
from bias.sophon import SophonSettings, bridge, build_instructions
from test_gesture import FAKE_WORKER

SECRET = "sk-sophon-test-secret"


class FakeBrowser:
    def __init__(self, start=None, mic=(b"\x01\x00" * 1600,)):
        self.start = start or {"type": "start", "gesture": True, "language": "en"}
        self.mic = list(mic)
        self.sent = []

    async def receive_text(self):
        return json.dumps(self.start)

    async def receive(self):
        if self.mic:
            return {"type": "websocket.receive", "bytes": self.mic.pop(0)}
        await asyncio.Event().wait()

    async def send_json(self, message):
        self.sent.append(message)


class FakeUpstream:
    def __init__(self, script):
        self.script = script
        self.sent = []
        self.url = self.headers = None

    def __call__(self, url, additional_headers=None, **kwargs):
        self.url, self.headers = url, additional_headers
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, text):
        self.sent.append(json.loads(text))

    def __aiter__(self):
        return self._events()

    async def _events(self):
        await asyncio.sleep(0.05)              # let the microphone frame go up first
        for event in self.script:
            await asyncio.sleep(0)
            yield json.dumps(event)


def audio_delta(samples):
    return {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x00\x01" * samples).decode()}


def gesture_engine(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    return GestureEngine(sys.executable, "ok", worker=worker)


def run_bridge(browser, upstream, gestures=None):
    async def body():
        try:
            await bridge(browser, SophonSettings(url="wss://sophon.invalid/v1/realtime", api_key=SECRET), "persona notes",
                         gesture=gestures, connect=upstream)
        finally:
            if gestures is not None:
                await gestures.close()
    asyncio.run(body())


def test_reply_audio_and_gestures_reach_the_browser_on_one_clock(tmp_path):
    upstream = FakeUpstream([
        {"type": "input_audio_buffer.speech_started"}, {"type": "input_audio_buffer.speech_stopped"},
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.output_audio_transcript.delta", "delta": "Accelerated computing"},
        audio_delta(30000), audio_delta(10000),
        {"type": "response.done"},
    ])
    browser = FakeBrowser()
    run_bridge(browser, upstream, gesture_engine(tmp_path))

    assert upstream.url.endswith("?model=sophon-realtime-tag") and upstream.headers == {"Authorization": f"Bearer {SECRET}"}
    session = upstream.sent[0]["session"]
    assert upstream.sent[0]["type"] == "session.update" and session["instructions"] == "persona notes" and session["voice"] == "ryan_en"
    mic = upstream.sent[1]
    assert mic["type"] == "input_audio_buffer.append" and mic["sample_rate_hz"] == 16000 and base64.b64decode(mic["audio"]) == b"\x01\x00" * 1600

    kinds = [m["type"] for m in browser.sent]
    assert kinds[0] == "ready" and browser.sent[0]["gesture"] is True
    assert SECRET not in json.dumps(browser.sent)
    assert [m["state"] for m in browser.sent if m["type"] == "user_speech"] == ["started", "stopped"]
    audio = [m for m in browser.sent if m["type"] == "audio"]
    assert all(m["turn_id"] == "resp_1" and m["sample_count"] <= 16384 and m["sample_rate"] == 24000 for m in audio)
    assert [m["sequence"] for m in audio] == list(range(len(audio)))
    offset = 0
    for chunk in audio:
        assert chunk["sample_offset"] == offset and len(base64.b64decode(chunk["pcm_base64"])) == chunk["sample_count"] * 2
        offset += chunk["sample_count"]
    assert offset == 40000
    assert next(m for m in browser.sent if m["type"] == "transcript")["delta"] == "Accelerated computing"
    frames = [m for m in browser.sent if m["type"] == "gesture"]
    assert frames and all(len(base64.b64decode(m["poses_base64"])) == m["frame_count"] * POSE_WIDTH * 4 for m in frames)
    end = next(m for m in browser.sent if m["type"] == "gesture_end")
    assert end["frames"] == sum(m["frame_count"] for m in frames) == 40000 * 30 // 24000 and end["status"] == "complete"
    done = browser.sent[-1]
    assert done == {"type": "response", "state": "done", "response_id": "resp_1", "sample_count": 40000}
    assert kinds.index("gesture_end") < len(kinds) - 1


def test_barge_in_cancels_the_reply_and_its_gestures(tmp_path):
    upstream = FakeUpstream([
        {"type": "response.created", "response": {"id": "resp_1"}}, audio_delta(30000),
        {"type": "audio.cancelled"},
        {"type": "response.created", "response": {"id": "resp_2"}}, audio_delta(2000), {"type": "response.done"},
    ])
    browser = FakeBrowser()
    run_bridge(browser, upstream, gesture_engine(tmp_path))
    responses = [(m["state"], m["response_id"]) for m in browser.sent if m["type"] == "response"]
    assert responses == [("started", "resp_1"), ("cancelled", "resp_1"), ("started", "resp_2"), ("done", "resp_2")]
    second = [m for m in browser.sent if m["type"] == "audio" and m["turn_id"] == "resp_2"]
    assert second[0]["sample_offset"] == 0 and second[0]["sequence"] == 0      # each reply has its own clock


def test_without_gestures_and_on_upstream_errors(tmp_path):
    upstream = FakeUpstream([{"type": "error", "error": {"code": "insufficient_quota", "message": "No credit left."}},
                             {"type": "response.created", "response": {"id": "never"}}])
    browser = FakeBrowser(start={"type": "start", "gesture": True})
    run_bridge(browser, upstream, None)
    assert browser.sent[0]["gesture"] is False
    assert browser.sent[-1] == {"type": "error", "code": "insufficient_quota", "message": "No credit left."}
    assert not any(m["type"] == "response" for m in browser.sent)


def test_instructions_carry_persona_rules_and_only_training_notes(corpus):
    text = build_instructions(corpus)
    assert "Research Subject" in text and "Use concise explanations." in text
    assert "Reusable rockets" in text and "flightless birds" not in text     # held-out evidence stays out


def test_endpoint_reports_configuration_and_rejects_foreign_origins(tmp_path, corpus, monkeypatch):
    monkeypatch.delenv("BIAS_SOPHON_URL", raising=False)
    with TestClient(create_app(Settings(web_dir=tmp_path / "absent"), corpus=corpus)) as client:
        assert client.get("/api/persona").json()["capabilities"]["realtime_call"] is False
        with client.websocket_connect("/api/realtime") as ws:
            assert ws.receive_json()["code"] == "realtime_unconfigured"
        with pytest.raises(Exception):
            with client.websocket_connect("/api/realtime", headers={"origin": "https://evil.example"}) as ws:
                ws.receive_json()
