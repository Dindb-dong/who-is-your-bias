"""Verify the shipped neural artifact and its independently copied UI assets."""
import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'artifacts/presentation/yaicon-intro-v1'
PUBLIC = ROOT / 'apps/web/public/presentation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_package_and_ui_files_have_exact_bound_content():
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert manifest['generated_frames'] == 306
    for name, expected in manifest['artifacts'].items():
        path = PACKAGE / name
        assert path.stat().st_size == expected['bytes'], name
        assert sha(path) == expected['sha256'], name
    ui = json.loads((PUBLIC / 'manifest.json').read_text())
    assert sha(PACKAGE / 'manifest.json') == ui['source_package_sha256']
    assert ui['duration_seconds'] == manifest['duration_seconds']
    for public, original in [('presentation.mp4', 'presentation.mp4'), ('narration.wav', 'speech.wav'), ('captions.vtt', 'captions.vtt'), ('portrait.png', 'poster.png')]:
        assert sha(PUBLIC / public) == sha(PACKAGE / original)
    with wave.open(str(PUBLIC / 'narration.wav')) as audio:
        assert (audio.getnframes(), audio.getframerate(), audio.getnchannels()) == (293760, 24000, 1)


def test_committed_video_streams_and_full_decode():
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg is required for delivery verification; installed in CI')
    video = PUBLIC / 'presentation.mp4'
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(video)], check=True, capture_output=True, text=True, timeout=20)
    streams = json.loads(result.stdout)['streams']
    picture = next(s for s in streams if s['codec_type'] == 'video')
    sound = next(s for s in streams if s['codec_type'] == 'audio')
    assert picture['codec_name'] == 'h264' and sound['codec_name'] == 'aac'
    assert (picture['width'], picture['height'], int(picture['nb_frames'])) == (1280, 720, 306)
    assert abs(float(picture['duration']) - float(sound['duration'])) < 0.048
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True, capture_output=True, timeout=30)
