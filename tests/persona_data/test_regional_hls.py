import hashlib
import io

import pytest

from tools.persona_data.regional_hls import parse_registered_playlist
from tools.persona_data import regional_hls


def registered(raw):
    return {"url": "https://publisher.example/vod/replay.m3u8",
            "publisher_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "expected_fragments": 2, "expected_duration_seconds": 12.5}


VALID = b"#EXTM3U\n#EXTINF:10.0,\n00001.ts\n#EXTINF:2.5,\n00002.ts\n#EXT-X-ENDLIST\n"


def test_accepts_complete_registered_playlist_in_order():
    assert parse_registered_playlist(VALID, registered(VALID)) == [
        "https://publisher.example/vod/00001.ts", "https://publisher.example/vod/00002.ts"]


@pytest.mark.parametrize("raw", [
    VALID.replace(b"#EXT-X-ENDLIST\n", b""),
    VALID.replace(b"00002.ts", b"00001.ts"),
    VALID.replace(b"00002.ts", b"https://different.example/00002.ts"),
    VALID.replace(b"00002.ts", b"http://publisher.example/00002.ts"),
    VALID.replace(b"#EXTINF:2.5,", b"#EXT-X-DISCONTINUITY\n#EXTINF:2.5,"),
    VALID.replace(b"#EXTINF:2.5,", b'#EXT-X-KEY:METHOD=AES-128,URI="key"\n#EXTINF:2.5,'),
])
def test_rejects_incomplete_duplicated_changed_origin_or_nonplain_media(raw):
    with pytest.raises(ValueError):
        parse_registered_playlist(raw, registered(raw))


@pytest.mark.parametrize("field,value", [
    ("publisher_manifest_sha256", "0" * 64),
    ("expected_fragments", 3),
    ("expected_duration_seconds", 13.5),
])
def test_rejects_registration_drift(field, value):
    entry = registered(VALID)
    entry[field] = value
    with pytest.raises(ValueError):
        parse_registered_playlist(VALID, entry)


def tracks():
    return {0: {"packets": 3000, "first_dts": 1.4, "last_dts": 101.366667},
            1: {"packets": 4300, "first_dts": 1.453556, "last_dts": 101.42}}


def test_per_track_coverage_accepts_frame_rounding():
    result = regional_hls.stream_coverage(tracks(), 100)
    assert result["status"] == "passed"
    assert result["streams"]["1"]["first_dts_offset_seconds"] == pytest.approx(.053556)
    assert result["streams"]["0"]["last_dts_minus_expected_end_seconds"] == pytest.approx(-.033333)


@pytest.mark.parametrize("change", [
    {"packets": 1, "last_dts": 1.453556},
    {"last_dts": 50.0},
    {"first_dts": 5.0},
    {"last_dts": 103.0},
])
def test_short_or_shifted_audio_cannot_pass_on_video_duration(change):
    states = tracks()
    states[1].update(change)
    with pytest.raises(ValueError, match="span the registered VOD"):
        regional_hls.stream_coverage(states, 100)


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_coverage_rejects_invalid_duration(duration):
    with pytest.raises(ValueError):
        regional_hls.stream_coverage(tracks(), duration)


@pytest.mark.parametrize("first_wait_times_out", [False, True])
def test_early_packet_parse_failure_reaps_probe(monkeypatch, tmp_path, first_wait_times_out):
    class Probe:
        stdout = io.StringIO("0,N/A,N/A,0.033333\n")
        waits = 0
        terminated = False
        killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waits += 1
            if first_wait_times_out and self.waits == 1:
                raise regional_hls.subprocess.TimeoutExpired("ffprobe", timeout)
            return -9 if self.killed else -15

    process = Probe()
    monkeypatch.setattr(regional_hls.subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(ValueError, match="decode timestamp"):
        regional_hls.packet_timeline(tmp_path / "incomplete.ts", expected_duration=100)
    assert process.stdout.closed and process.terminated
    assert process.killed is first_wait_times_out
    assert process.waits == 1 + first_wait_times_out
