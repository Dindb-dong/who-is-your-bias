from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from tools.media.common import MediaError, artifact_path, load_manifest, sha256, verify_artifact
from tools.media.prepare import prepare_clip
from tools.media.transcribe import normalize_segments, transcribe_clip


@pytest.fixture
def tone(tmp_path: Path) -> Path:
    """Synthetic tone exercises file processing, not a claimed human voice."""
    audio = tmp_path / "synthetic.wav"
    rate = 24000
    frames = b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 220 * index / rate))) for index in range(rate * 2))
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(frames)
    return audio


@pytest.fixture
def clip(tone: Path, tmp_path: Path) -> Path:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required for media integration tests")
    target = tmp_path / "review"
    prepare_clip(tone, target, source_id="synthetic-test", source_url="https://example.com/fixture", source_timeline="synthetic_source", start=0.25, duration=1, input_source_offset=120)
    return target / "manifest.json"


def test_ffmpeg_clip_resamples_and_preserves_source_clock(clip: Path):
    path, manifest = load_manifest(clip)
    audio = verify_artifact(path, manifest, "audio")
    with wave.open(str(audio), "rb") as stream:
        assert stream.getframerate() == 16000
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getnframes() / stream.getframerate() == pytest.approx(1, abs=0.002)
    assert manifest["source_start_seconds"] == 120.25
    assert manifest["source_end_seconds"] == pytest.approx(121.25, abs=0.002)
    assert manifest["review"]["speaker_identity"] is None
    assert manifest["review"]["status"] == "unreviewed"
    assert not Path(manifest["input"]["name"]).is_absolute()


def test_preparation_never_overwrites_existing_work(tone: Path, tmp_path: Path):
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe required")
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(MediaError, match="never overwritten"):
        prepare_clip(tone, target, source_id="test", source_url="https://example.com", source_timeline="input", duration=1)
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("start,duration", [(3, 1), (0, 3), (float("nan"), 1), (0, 121)])
def test_preparation_rejects_invalid_clip_bounds(tone: Path, tmp_path: Path, start: float, duration: float):
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe required")
    with pytest.raises(MediaError):
        prepare_clip(tone, tmp_path / "bad", source_id="test", source_url="https://example.com", source_timeline="input", start=start, duration=duration)


def test_manifest_cannot_escape_via_relative_path_or_symlink(clip: Path, tmp_path: Path):
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"private")
    for name in ("../outside.wav", str(outside)):
        with pytest.raises(MediaError):
            artifact_path(clip, name)
    (clip.parent / "escape.wav").symlink_to(outside)
    with pytest.raises(MediaError):
        artifact_path(clip, "escape.wav")


def test_modified_audio_is_not_silently_reused(clip: Path):
    path, manifest = load_manifest(clip)
    audio = artifact_path(path, "audio.wav")
    with audio.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(MediaError, match="changed"):
        verify_artifact(path, manifest, "audio")


def test_transcript_keeps_relative_and_source_clocks_separate():
    segments = normalize_segments([{"start": 0.5, "end": 1.5, "text": "Hello there", "words": [{"word": "Hello", "start": 0.5, "end": 0.9, "probability": 0.8}, {"word": "there"}]}], offset=141, duration=2)
    assert segments[0]["start"] == 0.5
    assert segments[0]["source_start"] == 141.5
    assert segments[0]["speaker"] is None
    assert segments[0]["words"][1]["start"] is None
    assert segments[0]["words"][1]["source_start"] is None


@pytest.mark.parametrize("start,end", [(1, 0), (-1, 0.5), (0, float("inf"))])
def test_invalid_asr_times_fail_instead_of_entering_evidence(start, end):
    with pytest.raises(MediaError):
        normalize_segments([{"start": start, "end": end, "text": "bad"}], offset=0, duration=2)


def test_unknown_asr_engine_has_actionable_failure(clip: Path):
    with pytest.raises(MediaError, match="Unknown ASR"):
        transcribe_clip(clip, engine="unknown")


def test_missing_asr_dependency_is_guarded(clip: Path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(MediaError, match="requirements-asr"):
        transcribe_clip(clip)


def test_face_serialization_does_not_invent_identity_or_emotion():
    from types import SimpleNamespace
    from tools.media.observe import serialize_face_result
    result = SimpleNamespace(face_landmarks=[[SimpleNamespace(x=0.1, y=0.2, z=-0.03)]], face_blendshapes=[[SimpleNamespace(category_name="mouthSmileLeft", score=0.6)]], facial_transformation_matrixes=[])
    face = serialize_face_result(result)[0]
    assert face["identity"] is None
    assert face["blendshapes"]["mouthSmileLeft"] == 0.6
    assert "emotion" not in face
    assert face["facial_transformation_matrix"] is None


def test_video_preparation_resets_frame_clock(tmp_path: Path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    from tools.media.observe import video_timestamps
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=navy:s=128x96:r=25:d=2", "-f", "lavfi", "-i", "sine=frequency=220:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)], check=True, timeout=20)
    target = tmp_path / "video-review"
    manifest = prepare_clip(source, target, source_id="test-video", source_url="https://example.com/video", source_timeline="synthetic_video", start=0.4, duration=1)
    video = verify_artifact(target / "manifest.json", manifest, "video")
    times = video_timestamps(video)
    assert times[0] == pytest.approx(0, abs=0.001)
    assert len(times) >= 24
    assert all(a < b for a, b in zip(times, times[1:]))
    assert manifest["source_start_seconds"] == 0.4
