import pytest
from fastapi.testclient import TestClient

from services.voice.app import create_app
from services.voice.engine import VoiceSettings


def test_server_reads_explicit_utf8_reference_text_file(monkeypatch, tmp_path):
    path = tmp_path / "reference.txt"
    path.write_text(" Accurate reference words. \n", encoding="utf-8")
    monkeypatch.delenv("BIAS_VOICE_REFERENCE_TEXT", raising=False)
    monkeypatch.setenv("BIAS_VOICE_REFERENCE_TEXT_FILE", str(path))
    assert VoiceSettings.from_env().reference_text == "Accurate reference words."


def test_server_rejects_ambiguous_and_invalid_reference_files(monkeypatch, tmp_path):
    path = tmp_path / "reference.txt"
    path.write_text("Words", encoding="utf-8")
    monkeypatch.setenv("BIAS_VOICE_REFERENCE_TEXT_FILE", str(path))
    monkeypatch.setenv("BIAS_VOICE_REFERENCE_TEXT", "Other words")
    with pytest.raises(ValueError, match="only one"):
        VoiceSettings.from_env()
    monkeypatch.delenv("BIAS_VOICE_REFERENCE_TEXT")
    path.write_text("x" * 16001)
    with pytest.raises(ValueError, match="16 KB"):
        VoiceSettings.from_env()
    path.write_text("bad\x00text")
    with pytest.raises(ValueError, match="NUL"):
        VoiceSettings.from_env()


def test_non_ascii_authorization_fails_as_unauthorized_without_model_loading():
    with TestClient(create_app(VoiceSettings(api_key="local-secret"))) as client:
        response = client.get("/v1/models", headers=[(b"authorization", b"Bearer \xff")])
        assert response.status_code == 401
