import json
import math
import wave
from pathlib import Path

import pytest

from tools.presentation.common import check_timing, read_plan, subtitles, wav_info
from tools.presentation.musetalk import validate_crop


PLAN = Path(__file__).resolve().parents[2] / "data/presentation/intro.v1.json"


def test_prepared_script_preserves_provenance_and_disclosure():
    plan = read_plan(PLAN)
    assert len(plan["segments"]) == 3
    assert all(s["source_refs"] for s in plan["segments"])
    assert "not the real person" in plan["disclosure"]


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(disclosure="Musk presents"),
    lambda p: p["segments"][0].update(id="../outside"),
    lambda p: p["segments"][0].update(source_refs=[]),
    lambda p: p["segments"][1].update(id=p["segments"][0]["id"]),
])
def test_invalid_prepared_scripts_are_rejected(tmp_path, mutation):
    plan = json.loads(PLAN.read_text())
    mutation(plan)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        read_plan(path)


def test_subtitles_keep_exact_segment_gaps_and_escape_markup():
    rows = [{"start_seconds": 0, "end_seconds": 2.96, "text": "<hello> -->"},
            {"start_seconds": 3.16, "end_seconds": 7, "text": "next"}]
    result = subtitles(rows, vtt=True)
    assert "00:00:02.960" in result and "00:00:03.160" in result
    assert "‹hello› →" in result and result.startswith("WEBVTT")
    rows[1]["start_seconds"] = 2
    with pytest.raises(ValueError):
        subtitles(rows)


def test_nonfinite_subtitle_bounds_are_rejected():
    with pytest.raises(ValueError):
        subtitles([{"start_seconds": 0, "end_seconds": math.inf, "text": "test"}])


def media(video_seconds=12.24, audio_seconds=12.24, start=0):
    return {"streams": [
        {"codec_type": "video", "codec_name": "h264", "duration": str(video_seconds), "start_time": str(start)},
        {"codec_type": "audio", "codec_name": "aac", "duration": str(audio_seconds), "sample_rate": "24000", "start_time": "0"}]}


def test_timing_accepts_a_single_frame_rounding_and_rejects_drift():
    assert check_timing(media(12.28), 12.24, 25)["passed"]
    with pytest.raises(ValueError):
        check_timing(media(12.5), 12.24, 25)
    with pytest.raises(ValueError):
        check_timing(media(start=0.3), 12.24, 25)
    with pytest.raises(ValueError):
        check_timing(media(float("nan")), 12.24, 25)


def test_missing_or_duplicated_streams_do_not_pass():
    value = media()
    value["streams"].append(value["streams"][0])
    with pytest.raises(ValueError):
        check_timing(value, 12.24, 25)


def test_crop_rejects_outside_or_too_small_regions():
    assert validate_crop([266, 250, 454, 460], (720, 900)) == (266, 250, 454, 460)
    for crop in [[0, 0, 10, 10], [-1, 0, 200, 200], [0, 0, 800, 300], [True, 0, 200, 200]]:
        with pytest.raises(ValueError):
            validate_crop(crop, (720, 900))


def test_truncated_audio_does_not_pass(tmp_path):
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000)
        stream.writeframes(b"\1\0" * 24000)
    assert wav_info(path)["seconds"] == 1
    path.write_bytes(path.read_bytes()[:-100])
    with pytest.raises(ValueError):
        wav_info(path)
