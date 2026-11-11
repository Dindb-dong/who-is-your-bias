from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.media.common import MediaError, probe
from tools.media.prepare import prepare_clip


@pytest.fixture
def video(tmp_path: Path) -> Path:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg/FFprobe required for the real fixed-ROI test")
    path = tmp_path / "synthetic.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=160x120:r=25:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True, timeout=20)
    return path


def test_fixed_crop_preserves_spatial_provenance_and_source_clock(video: Path, tmp_path: Path):
    target = tmp_path / "prepared"
    manifest = prepare_clip(video, target, source_id="fixture", source_url="https://example.com/fixture", source_timeline="synthetic",
                            start=0.2, duration=0.5, input_source_offset=60, crop=(20, 10, 80, 100))
    stream = next(item for item in probe(target / "video.mp4")["streams"] if item["codec_type"] == "video")
    assert (stream["width"], stream["height"]) == (80, 100)
    assert manifest["source_start_seconds"] == 60.2
    assert manifest["preparation"]["video_crop"] == {"x": 20, "y": 10, "width": 80, "height": 100, "input_width": 160, "input_height": 120,
                                                     "coordinate_basis": "input_pixels_before_scale", "mirrored": False}
    assert manifest["review"]["speaker_identity"] is None


@pytest.mark.parametrize("crop", [(-2, 0, 80, 100), (0, 0, 200, 100), (0, 0, 81, 100), (0, 0, 20, 20), (False, 0, 80, 100)])
def test_invalid_roi_is_rejected_before_creating_output(video: Path, tmp_path: Path, crop):
    target = tmp_path / "bad"
    with pytest.raises(MediaError, match="Crop"):
        prepare_clip(video, target, source_id="fixture", source_url="https://example.com/fixture", source_timeline="synthetic", duration=0.5, crop=crop)
    assert not target.exists()
