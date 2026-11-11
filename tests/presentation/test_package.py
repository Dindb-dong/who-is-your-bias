import json
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from tools.presentation.common import digest
from tools.presentation.package import package


PLAN = Path(__file__).resolve().parents[2] / "data/presentation/intro.v1.json"


def setup(tmp_path, *, speech_status="completed", render_status="completed"):
    speech_dir, render_dir = tmp_path / "speech", tmp_path / "render"
    speech_dir.mkdir(); render_dir.mkdir()
    audio = speech_dir / "speech.wav"; audio.write_bytes(b"synthetic test input")
    speech = {"status": speech_status, "plan_sha256": digest(PLAN), "audio": {"sha256": digest(audio)}}
    render = {"status": render_status, "audio_sha256": digest(audio)}
    (speech_dir / "speech.json").write_text(json.dumps(speech))
    (render_dir / "render.json").write_text(json.dumps(render))
    return speech_dir, render_dir


def test_partial_neural_probe_cannot_be_packaged_as_complete(tmp_path):
    speech, render = setup(tmp_path, render_status="partial_probe")
    with pytest.raises(ValueError, match="Only completed"):
        package(plan_path=PLAN, speech_dir=speech, render_dir=render,
            portrait=tmp_path / "portrait.png", output=tmp_path / "package")


def test_script_changes_after_synthesis_are_rejected(tmp_path):
    speech, render = setup(tmp_path)
    plan = json.loads(PLAN.read_text())
    plan["segments"][0]["text"] = "Changed after generation."
    altered = tmp_path / "new-plan.json"; altered.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="script changed"):
        package(plan_path=altered, speech_dir=speech, render_dir=render,
            portrait=tmp_path / "portrait.png", output=tmp_path / "package")


def test_audio_swap_between_speech_and_render_is_rejected(tmp_path):
    speech, render = setup(tmp_path)
    (speech / "speech.wav").write_bytes(b"replacement audio")
    with pytest.raises(ValueError, match="audio fingerprint mismatch"):
        package(plan_path=PLAN, speech_dir=speech, render_dir=render,
            portrait=tmp_path / "portrait.png", output=tmp_path / "package")
