"""Jensen defaults, source provenance, temporal limits and modality identity."""
from __future__ import annotations

import json
from dataclasses import replace
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
from bias.providers import OpenAICompatibleLLM
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
        assert profile["stats"] == {"sources": 9, "evidence": 32, "held_out_sources": 0, "total_evidence": 32}
        assert profile["snapshot"]["id"] == "jensen-huang-2026-11-12-interviews.v2"
        assert profile["snapshot"]["current_dated_training_sources"] == 0
        assert profile["bindings"]["voice"]["status"] == "not_configured"
        assert profile["capabilities"]["speech"] is False
        assert client.get("/api/motion/library").json()["available"] is False


def test_corpus_is_attributed_editorial_text_with_no_style_speech_or_behavior_eligibility(jensen):
    assert len(jensen.training_evidence) == 32
    assert not jensen.style_evidence
    interview_sources = {source.id for source in jensen.sources.values() if source.source_type == "publisher_interview_transcript"}
    assert len(interview_sources) == 6
    interviews = [record for record in jensen.training_evidence if record.source_id in interview_sources]
    assert len(interviews) == len({record.model_extra["acquisition_id"] for record in interviews}) == 24
    for source in jensen.sources.values():
        assert source.model_extra["persona_id"] == jensen.persona.id
        assert 2023 <= source.published_at.year <= 2026
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


def test_korean_interview_evidence_reaches_normal_chat(monkeypatch, tmp_path):
    with TestClient(create_app(default_settings(monkeypatch, tmp_path))) as client:
        turn = client.post("/api/chat", json={"message": "TSMC 반도체 패키징 납기", "language": "ko"}).json()
        citation = next(item for item in turn["citations"] if item["evidence_id"] == "jensen-lex-494-03")
        assert citation["url"] == "https://lexfridman.com/jensen-huang-transcript/"
        assert "01:10:19" in citation["locator"]["label"]
        source = next(item for item in client.get("/api/sources").json()["sources"] if item["id"] == citation["source_id"])
        assert source["published_at"] == "2026-03-23" and source["recorded_at"] is None


@pytest.mark.parametrize(("draft", "accepted"), [
    ("처리량이 어디에서 막히는지 살필 수 있습니다.", True),
    ("식량 보급을 어떻게提高效率할지 살필 수 있습니다.", False),
    ("그가 특허를 냈다고 말했는지 살필 수 있습니다.", False),
    ("“내가 직접 설계했다”라고 말할지 살필 수 있습니다.", False),
])
def test_missing_evidence_streams_only_reviewed_inference(monkeypatch, tmp_path, draft, accepted):
    config = replace(default_settings(monkeypatch, tmp_path), llm=ProviderSettings("http://model.invalid/v1", "test"))

    async def stream(_self, messages):
        assert "질문의 전제를 사실로 받아들이지 말고" in messages[0]["content"]
        assert '"factual_evidence":[]' in messages[1]["content"]
        yield "  "
        yield draft[:14]
        yield draft[14:]

    monkeypatch.setattr(OpenAICompatibleLLM, "stream", stream)
    with TestClient(create_app(config)) as client:
        stream_body = client.post("/api/chat/stream", json={"message": "화성 기지의 식량 배급을 설계한다고 가정하면 무엇을 고려할까?", "language": "ko"}).text
    events = [block.split("\ndata: ", 1) for block in stream_body.strip().split("\n\n")]
    first_token = next(json.loads(data)["text"] for event, data in events if event == "event: token")
    turn = next(json.loads(data)["turn"] for event, data in events if event == "event: result")
    assert first_token.startswith("가상 추론 (실제 발언 확인 불가)")
    assert (draft in first_token) is accepted
    if not accepted:
        assert draft not in stream_body
        assert "전체 성능이 막히는지" in first_token
    assert turn["text"] == first_token
    assert turn["provider"] == "openai" and turn["citations"] == []
    assert turn["grounding"]["status"] == "insufficient"
    assert turn["grounding"]["citation_status"] == "not_required"


def test_future_quotation_is_unavailable_without_generation(monkeypatch, tmp_path):
    config = replace(default_settings(monkeypatch, tmp_path), llm=ProviderSettings("http://model.invalid/v1", "test"))

    async def stream(_self, _messages):
        pytest.fail("An actual future quotation must not reach generation.")
        yield

    monkeypatch.setattr(OpenAICompatibleLLM, "stream", stream)
    with TestClient(create_app(config)) as client:
        turn = client.post("/api/chat", json={
            "message": "2035년에 황이 비공개 회의에서 실제로 무슨 말을 했어?",
            "language": "ko",
        }).json()
    assert "실제 발언을 확인할 수 없습니다" in turn["text"]
    assert turn["grounding"]["status"] == "insufficient"
    assert turn["citations"] == []


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
