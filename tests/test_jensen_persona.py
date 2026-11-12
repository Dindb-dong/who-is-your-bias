"""Jensen defaults, source provenance, temporal limits and modality identity."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from bias.api import create_app
from bias.bindings import motion_binding_status, voice_binding
from bias.config import ProviderSettings, Settings, repository_root
from bias.cli import main
from bias.corpus import Corpus, projection_eligible
from bias.engine import Engine
from bias.snapshots import PersonaSnapshot


@pytest.fixture
def jensen():
    return Corpus.load(repository_root() / "data/personas/jensen-huang")


def default_settings(monkeypatch, tmp_path):
    for key in ("BIAS_CORPUS_DIR", "BIAS_PERSONA_SNAPSHOT", "BIAS_LLM_BASE_URL", "BIAS_LLM_MODEL",
                "BIAS_TTS_BASE_URL", "BIAS_TTS_MODEL", "BIAS_TTS_PERSONA_ID", "BIAS_MOTION_CATALOG"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BIAS_WEB_DIR", str(tmp_path / "absent"))
    return Settings.from_env()


def test_default_studio_has_jensen_identity_and_its_exact_snapshot(monkeypatch, tmp_path):
    config = default_settings(monkeypatch, tmp_path)
    assert config.corpus_dir == Settings().corpus_dir == repository_root() / "data/personas/jensen-huang"
    with TestClient(create_app(config)) as client:
        profile = client.get("/api/persona").json()
        assert profile["id"] == profile["bindings"]["persona_id"] == "jensen-huang"
        assert profile["name"] == "Jensen Huang"
        assert profile["presentation"]["initials"] == "JH"
        assert profile["stats"] == {"sources": 3, "evidence": 8, "held_out_sources": 0, "total_evidence": 8}
        assert profile["snapshot"]["id"] == "jensen-huang-2026-11-12-seed.v1"
        assert profile["snapshot"]["current_dated_training_sources"] == 0
        assert profile["bindings"]["voice"]["status"] == "not_configured"
        assert profile["capabilities"]["speech"] is False
        assert client.get("/api/motion/library").json()["available"] is False


def test_seed_is_attributed_editorial_text_with_no_style_speech_or_behavior_eligibility(jensen):
    assert len(jensen.training_evidence) == 8
    assert not jensen.style_evidence
    for source in jensen.sources.values():
        assert source.url.host in {"blogs.nvidia.com", "nvidianews.nvidia.com"}
        assert source.model_extra["persona_id"] == jensen.persona.id
        assert source.published_at.year in {2024, 2025}
    for record in jensen.evidence.values():
        assert record.model_extra["persona_id"] == "jensen-huang"
        assert record.model_extra["text_provenance"] == "editorial_summary"
        assert record.question_answer is None
        assert record.behavioral_observation is None
        assert projection_eligible(record, "factual")
        assert not any(projection_eligible(record, kind) for kind in ("text_style", "aligned_speech", "visible_behavior"))
        assert "musk" not in record.text.lower()


@pytest.mark.parametrize("language,question", [
    ("en", "What did Jensen Huang mean by AI factories at COMPUTEX 2025?"),
    ("ko", "2025년 컴퓨텍스에서 AI 팩토리를 어떻게 설명했어?"),
])
def test_dated_jensen_questions_only_return_jensen_source_records(monkeypatch, tmp_path, language, question):
    with TestClient(create_app(default_settings(monkeypatch, tmp_path))) as client:
        response = client.post("/api/chat", json={"message": question, "language": language})
        assert response.status_code == 200
        turn = response.json()
        assert turn["citations"]
        assert all(row["source_id"].startswith("nvidia-") for row in turn["citations"])
        assert "jensen-2025-ai-factories" in {row["evidence_id"] for row in turn["citations"]}
        assert turn["behavior"]["motion"] is None
        assert "Musk" not in turn["text"]


@pytest.mark.parametrize("question", ["What is the current AI factory plan?", "현재 AI 팩토리 계획은 뭐야?"])
def test_historical_seed_does_not_answer_current_position_questions(monkeypatch, tmp_path, question):
    with TestClient(create_app(default_settings(monkeypatch, tmp_path))) as client:
        turn = client.post("/api/chat", json={"message": question}).json()
        assert turn["citations"] == []
        assert turn["grounding"]["retrieved_ids"] == []
        assert turn["grounding"]["status"] == "insufficient"


def test_jensen_snapshot_cannot_be_applied_to_another_subject(corpus):
    with pytest.raises(ValueError, match="does not match this exact corpus"):
        PersonaSnapshot.load(repository_root() / "data/snapshots/jensen-huang-2026-11-12.json", corpus)


def test_explicit_historical_corpus_does_not_inherit_jensen_snapshot(monkeypatch, tmp_path):
    default_settings(monkeypatch, tmp_path)
    legacy = repository_root() / "data/personas/elon-musk"
    monkeypatch.setenv("BIAS_CORPUS_DIR", str(legacy))
    config = Settings.from_env()
    assert config.persona_snapshot is None
    with TestClient(create_app(config)) as client:
        assert client.get("/api/persona").json()["name"] == "Elon Musk"


def test_default_cli_does_not_evaluate_jensen_against_musk_cases(monkeypatch, tmp_path, capsys):
    default_settings(monkeypatch, tmp_path)
    assert main(["evaluate"]) == 2
    assert "historical Musk corpus" in capsys.readouterr().err
    assert main(["validate"]) == 0
    assert json.loads(capsys.readouterr().out)["persona"] == "jensen-huang"


def fake_library(persona_id=None, sources=()):
    return SimpleNamespace(catalog=SimpleNamespace(persona_id=persona_id, sources=list(sources)))


def test_foreign_motion_is_rejected_by_api_and_direct_engine(jensen, tmp_path):
    library = fake_library("elon-musk")
    config = Settings(web_dir=tmp_path / "absent")
    engine = Engine(config, jensen, motion_library=library)
    assert engine.motion_library is None
    with TestClient(create_app(config, corpus=jensen, motion_library=library)) as client:
        assert client.get("/api/motion/library").json()["status"] == "subject_mismatch"
        assert client.get("/api/motion/clip/anything?preview=true").status_code == 404
        turn = client.post("/api/chat", json={"message": "AI factories", "motion_mode": "geometry_preview"}).json()
        assert turn["behavior"]["motion"] is None


def test_unbound_legacy_motion_needs_exact_source_event_and_media_membership(corpus, jensen):
    source = SimpleNamespace(id="rocket-interview", source_event_id="rocket-interview", url="https://example.org/rocket")
    library = fake_library(sources=[source])
    assert motion_binding_status(library, corpus) == "source_bound"
    assert motion_binding_status(library, jensen) == "subject_mismatch"
    source.source_event_id = "different-recording"
    assert motion_binding_status(library, corpus) == "subject_mismatch"
    source.source_event_id = "rocket-interview"
    source.url = "https://example.org/rocket?recording=other"
    assert motion_binding_status(library, corpus) == "subject_mismatch"


@pytest.mark.parametrize("subject,status,enabled", [(None, "unbound", False), ("elon-musk", "subject_mismatch", False), ("jensen-huang", "matched", True), ("generic", "generic", True)])
def test_configured_voice_is_bound_to_active_subject_before_playback(jensen, tmp_path, subject, status, enabled):
    config = Settings(web_dir=tmp_path / "absent", tts=ProviderSettings("http://voice.invalid/v1", "test"), tts_persona_id=subject, tts_voice="reference")
    assert voice_binding(config, "jensen-huang")["status"] == status
    with TestClient(create_app(config, corpus=jensen)) as client:
        profile = client.get("/api/persona").json()
        assert profile["bindings"]["voice"]["status"] == status
        assert profile["capabilities"]["speech"] is enabled
        assert profile["capabilities"]["live_speech"] is enabled
        if not enabled:
            for endpoint in ("/api/speech", "/api/speech/prepare"):
                response = client.post(endpoint, json={"text": "Hello"})
                assert response.status_code == 503
                assert response.json()["error"]["code"] == "speech_subject_mismatch"
            stream = client.post("/api/chat/live", json={"message": "AI factories"}).text
            assert "event: error" in stream and "event: audio" not in stream


def test_bound_voice_preserves_payload_and_cannot_be_overridden(jensen, tmp_path):
    config = Settings(web_dir=tmp_path / "absent", tts=ProviderSettings("http://voice.invalid/v1", "test"), tts_persona_id="generic", tts_voice="narrator")
    calls = []
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, content=b"RIFF0000WAVEtest")
    app = create_app(config, corpus=jensen)
    with TestClient(app) as client:
        app.state.speech.transport = httpx.MockTransport(handle)
        assert client.post("/api/speech", json={"text": "Hello"}).status_code == 200
        assert calls[0]["voice"] == "narrator"
        for endpoint in ("/api/speech", "/api/speech/prepare"):
            response = client.post(endpoint, json={"text": "Hello", "voice": "musk-reference"})
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "speech_voice_override"
        assert len(calls) == 1


def test_voice_binding_environment_requires_valid_identifier(monkeypatch):
    monkeypatch.setenv("BIAS_TTS_PERSONA_ID", "jensen-huang")
    assert Settings.from_env().tts_persona_id == "jensen-huang"
    monkeypatch.setenv("BIAS_TTS_PERSONA_ID", "a secret URL/invalid binding")
    with pytest.raises(ValueError, match="tts_persona_id"):
        Settings.from_env()
