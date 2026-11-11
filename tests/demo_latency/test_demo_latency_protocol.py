import hashlib
import io
import json
import struct
import wave

import httpx
import pytest

from tools.demo_latency.protocol import pcm_info, speech_text, sse_events
from tools.demo_latency.run import collect_turn


def wav(samples, rate=8000):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return output.getvalue()


def test_sse_preserves_multiline_json_comments_and_final_frame():
    lines = [": heartbeat\r", "event: token\r", "data: {", 'data: "text":"Hi"}', "\r",
        "event: done", "data: {}"]
    assert list(sse_events(lines)) == [("token", {"text": "Hi"}), ("done", {})]


@pytest.mark.parametrize("line", ['data: []', 'data: invalid'])
def test_invalid_sse_does_not_become_success(line):
    with pytest.raises(ValueError):
        list(sse_events([line, ""]))


def test_citation_removal_matches_browser_without_rewriting_answer():
    assert speech_text("  Cost falls [source:qa-ted-2017-cost]. [Other]  ") == "Cost falls . [Other]"


def test_pcm_onset_is_waveform_offset_with_declared_resolution():
    data = wav([0] * 160 + [1000] * 80)
    info = pcm_info(data, hashlib.sha256(data).hexdigest())
    assert info["duration_ms"] == 30
    assert info["offline_pcm_onset_offset_ms"] == 20
    assert info["offline_pcm_detector"]["window_ms"] == 10
    assert "first_audio_signal_ms" not in info


def test_silent_wav_has_no_invented_pcm_onset():
    data = wav([0] * 80)
    assert pcm_info(data, hashlib.sha256(data).hexdigest())["offline_pcm_onset_offset_ms"] is None


def test_swapped_audio_and_truncated_wav_are_rejected():
    data = wav([500] * 80)
    with pytest.raises(ValueError, match="does not match"):
        pcm_info(data, "0" * 64)
    truncated = data[:-2]
    with pytest.raises(ValueError, match="truncated"):
        pcm_info(truncated, hashlib.sha256(truncated).hexdigest())


def test_transport_report_keeps_browser_and_acoustic_fields_unmeasured(tmp_path):
    """Protocol fixture only; this unit test is never reported as model execution."""
    data = wav([0] * 80 + [1000] * 80)
    audio_hash = hashlib.sha256(data).hexdigest()
    turn = {"id": "fixture", "provider": "openai", "text": "Smaller tunnels reduce cost. [source:fixture]",
        "metrics": {"retrieval_ms": 1, "generation_ms": 2, "total_ms": 3}}
    events = [("token", {"text": " "}), ("token", {"text": "Smaller"}), ("result", {"turn": turn}), ("done", {})]
    stream = "".join(f"event: {event}\ndata: {json.dumps(payload)}\n\n" for event, payload in events)

    def handle(request):
        if request.url.path == "/api/chat/stream":
            return httpx.Response(200, text=stream, headers={"Content-Type": "text/event-stream"})
        if request.url.path == "/api/speech/prepare":
            assert json.loads(request.content)["text"] == "Smaller tunnels reduce cost."
            return httpx.Response(200, json={"audio_url": "/api/speech/clips/" + "a" * 32,
                "audio_sha256": audio_hash, "audio_duration_ms": 20,
                "server_timing": {"synthesis_ms": 1, "alignment_ms": 0, "total_ms": 1}})
        return httpx.Response(200, content=data)

    row = {"id": "fixture", "request": {"message": "Q", "language": "en", "mode": "behavior"}}
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        collect_turn(client, "http://127.0.0.1", row, tmp_path)
    assert row["status"] == "completed"
    assert row["browser"] == {"first_visible_text_ms": None, "audio_start_ms": None,
        "first_audio_signal_ms": None, "turn_complete_ms": None, "status": "not_measured"}
    assert row["acoustic_first_audible_ms"] is None
    assert row["transport"]["first_token_received_ms"] <= row["transport"]["first_content_text_received_ms"]
    assert row["transport"]["audio_available_ms"] <= row["transport"]["inspection_complete_ms"]
    assert (tmp_path / "fixture.wav").read_bytes() == data
