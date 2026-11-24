"""Cross-layer QA/subtitle contracts, with explicitly synthetic models/audio."""
import asyncio
import io
import json
import wave
from contextlib import aclosing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bias.api import create_app
from bias.caption_stream import caption_events
from bias.config import ProviderSettings, Settings
from bias.corpus import Corpus
from bias.engine import Engine
from bias.live import live_events
from bias.models import ChatRequest
from bias.subtitles import SubtitleTranslator
from bias.providers import ProviderError

ROOT = Path(__file__).resolve().parents[1]


class Answer:
    def __init__(self):
        self.messages = []

    async def stream(self, messages):
        self.messages.append(messages)
        yield "The team retrieves reviewed evidence before generating a response. [source:QA-project-pipeline]"


class Translation:
    async def stream(self, messages):
        yield "팀은 답변을 생성하기 전에 검토된 근거를 검색합니다."


def engine(*, configured=True, **kwargs):
    return Engine(Settings(corpus_dir=ROOT / "data/personas/jensen-huang", qa_knowledge_dir=ROOT / "data/qa",
                           llm=ProviderSettings("http://llm.invalid/v1", "synthetic") if configured else ProviderSettings(),
                           **kwargs), Corpus.load(ROOT / "data/personas/jensen-huang"), llm=Answer())


def test_project_qa_has_its_own_sources_and_accepts_korean_input_with_english_answer():
    e = engine()
    turn = asyncio.run(e.chat(ChatRequest(message="우리 프로젝트 파이프라인 설명해 주세요", input_language="ko", language="en")))
    assert turn.language == "en" and turn.citations[0].evidence_id == "QA-project-pipeline"
    assert turn.grounding.semantic_status == "unreviewed"
    assert turn.context["qa_knowledge"]["route"] == "project"
    assert "Answer in English" in e.llm.messages[0][0]["content"]
    assert "make the constraint concrete" in e.llm.messages[0][0]["content"]
    assert "student team in the third person" in e.llm.messages[0][0]["content"]
    assert not any("NV-" in m["content"] for m in e.llm.messages[0])


@pytest.mark.parametrize("message,route", [("엔비디아 주식 살까요 팔까요?", "finance"),
    ("젠슨 황 불륜 뉴스 사실인가요?", "allegation"), ("오늘 엔비디아 주가는 얼마인가요?", "news")])
def test_unsupported_sensitive_current_requests_do_not_invoke_speculative_model(message, route):
    e = engine()
    turn = asyncio.run(e.chat(ChatRequest(message=message, language="en")))
    assert e.llm.messages == []
    assert turn.context["qa_knowledge"]["route"] == route
    assert turn.grounding.status == "insufficient" and turn.citations == []


def test_missing_project_cards_never_fall_through_on_presentation_turn():
    e = engine()
    e.qa_knowledge = None
    with pytest.raises(ProviderError) as error:
        asyncio.run(e.chat(ChatRequest(message="What did you build?", presentation_qa=True)))
    assert error.value.code == "presentation_qa_unavailable"
    assert e.llm.messages == []


def test_disabled_knowledge_and_other_persona_keep_explicit_binding(corpus):
    e = Engine(Settings(), corpus)
    assert e.qa_knowledge is None
    with pytest.raises(ValueError, match="bound"):
        Engine(Settings(qa_knowledge_dir=ROOT / "data/qa"), corpus)


@pytest.mark.parametrize("question,source,required", [
    ("우리 프로젝트에서 Qwen을 선택한 이유와 우위는?", "QA-project-language-model", "does not establish"),
    ("우리 아바타는 MCCCD 방식으로 학습했나요?", "QA-project-alignment", "do not verify"),
])
def test_known_documentation_gaps_do_not_become_invented_results(question, source, required):
    e = engine()
    turn = asyncio.run(e.chat(ChatRequest(message=question, language="en")))
    assert e.llm.messages == []
    assert source in {citation.evidence_id for citation in turn.citations}
    assert required in turn.text
    assert turn.grounding.status == "grounded"


@pytest.mark.parametrize(("answer", "problem"), [
    ("The design supports low latency. [source:QA-unknown]", "citation_check_failed"),
    ("We built it for low latency. [source:QA-project-pipeline]", "team_attribution_check_failed"),
    ("Jensen Huang built the system. [source:QA-project-pipeline]", "team_attribution_check_failed"),
    ("Latency fell by 99 percent. [source:QA-project-pipeline]", "numeric_evidence_check_failed"),
    ("이 프로젝트는 검색을 사용합니다. [source:QA-project-pipeline]", "language_check_failed"),
])
def test_presentation_screen_rejects_project_draft_before_text_and_speech(answer, problem):
    class Draft:
        async def stream(self, messages):
            yield answer

    e = engine()
    e.llm = Draft()
    request = ChatRequest(message="이 프로젝트 파이프라인은?", language="en", presentation_qa=True)

    async def run():
        return [item async for item in e.events(request, committed_segments=True)]

    events = asyncio.run(run())
    visible = " ".join(event["text"] for kind, event in events if kind in {"token", "segment"})
    assert answer.split(" [source:")[0] not in visible
    assert "Reviewed source summary" in visible
    turn = next(event["turn"] for kind, event in events if kind == "result")
    assert turn["provider"] == "evidence"
    assert turn["context"]["qa_knowledge"]["draft_validation"] == problem


def test_presentation_approved_draft_can_be_spoken():
    request = ChatRequest(message="이 프로젝트 파이프라인은?", language="en", presentation_qa=True)

    async def run():
        return [item async for item in engine().events(request, committed_segments=True)]

    events = asyncio.run(run())
    speech = [event for kind, event in events if kind == "segment"]
    assert [event["text"] for event in speech] == ["The team retrieves reviewed evidence before generating a response."]
    turn = next(event["turn"] for kind, event in events if kind == "result")
    assert turn["context"]["qa_knowledge"]["draft_validation"] == "mechanical_checks_passed"
    assert turn["grounding"]["semantic_status"] == "unreviewed"


def test_project_model_reason_is_spoken_as_a_possible_explanation():
    e = engine()
    request = ChatRequest(message="Why did the team choose Qwen for this project?", language="en", presentation_qa=True)

    async def run():
        return [item async for item in e.events(request, committed_segments=True)]

    events = asyncio.run(run())
    spoken = " ".join(event["text"] for kind, event in events if kind == "segment")
    assert spoken.startswith("One plausible engineering reason is ")
    assert "do not establish the team's actual reason" in spoken
    assert e.llm.messages == []
    turn = next(event["turn"] for kind, event in events if kind == "result")
    assert turn["context"]["qa_knowledge"]["rationale"] == {
        "kind": "engineering_inference", "source_id": "QA-project-language-model"}
    assert turn["grounding"]["status"] == "unverified"


def test_stream_api_exposes_separate_translation_and_preserves_english_answer():
    e = engine()
    app = create_app(e.settings, corpus=e.corpus, llm=e.llm, subtitle_translator=SubtitleTranslator(Translation()))
    with TestClient(app) as client:
        assert client.get("/api/persona").json()["capabilities"]["translated_subtitles"]
        response = client.post("/api/chat/stream", json={"message": "이 프로젝트 파이프라인은?", "input_language": "ko",
                                                        "language": "en", "subtitle_language": "ko"})
    records = [(block.splitlines()[0][7:], json.loads(block.splitlines()[1][6:])) for block in response.text.strip().split("\n\n")]
    assert not any(kind == "error" for kind, _ in records)
    subtitle = next(data for kind, data in records if kind == "subtitle")
    turn = next(data["turn"] for kind, data in records if kind == "result")
    assert subtitle["state"] == "ready" and "팀은" in subtitle["text"]
    assert turn["language"] == "en" and "팀은" not in turn["text"]
    assert records[-1] == ("done", {})


def test_slow_caption_does_not_block_english_pcm_and_close_cancels_worker():
    async def run():
        entered, closed = asyncio.Event(), asyncio.Event()
        class SlowTranslation:
            async def stream(self, messages):
                entered.set()
                try:
                    await asyncio.Event().wait()
                    yield "never"
                finally:
                    closed.set()
        class Voice:
            settings = ProviderSettings("http://voice.invalid/v1", "synthetic")
            calls = []
            async def synthesize(self, text, **kwargs):
                self.calls.append((text, kwargs))
                b = io.BytesIO()
                with wave.open(b, "wb") as wav:
                    wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    wav.writeframes(b"\0\x10" * 1600)
                return b.getvalue(), "audio/wav"
        e, voice = engine(), Voice()
        request = ChatRequest(message="이 프로젝트 파이프라인은?", input_language="ko", language="en", subtitle_language="ko")
        async with aclosing(caption_events(live_events(e, voice, request), SubtitleTranslator(SlowTranslation()), "ko")) as stream:
            async for kind, data in stream:
                if kind == "audio":
                    await asyncio.wait_for(entered.wait(), 1)
                    assert not closed.is_set()
                    break
        assert closed.is_set()
        assert voice.calls[0][1] == {"language": "en"}
        assert "팀은" not in voice.calls[0][0]
        assert all(session.in_flight == 0 for session in e.sessions.sessions.values())
    asyncio.run(run())


def test_caption_overflow_is_explicit_and_not_speech_backpressure():
    async def run():
        release = asyncio.Event()
        class Slow:
            timeout = 1
            async def translate(self, segment, language):
                await release.wait()
                return {key: segment[key] for key in ("turn_id", "segment_id", "sequence")} | {
                    "language": language, "text": "번역", "state": "ready"}
        async def source():
            for sequence in range(10):
                yield "segment", {"turn_id": "t", "segment_id": f"t:{sequence}", "sequence": sequence,
                                  "text": "English", "language": "en", "state": "committed"}
            yield "audio_end", {}
            yield "done", {}
        events = []
        async for kind, data in caption_events(source(), Slow(), "ko", queue_size=1):
            events.append((kind, data))
            if kind == "audio_end":
                release.set()
        assert len([data for kind, data in events if kind == "subtitle"]) == 10
        assert any(data.get("reason") == "translation_busy" for kind, data in events if kind == "subtitle")
        assert events[-1] == ("done", {})
    asyncio.run(run())


def test_translation_env_is_independent_of_answer_and_voice(monkeypatch):
    monkeypatch.setenv("BIAS_TRANSLATION_BASE_URL", "http://translation.invalid/v1")
    monkeypatch.setenv("BIAS_TRANSLATION_MODEL", "caption-model")
    monkeypatch.setenv("BIAS_QA_KNOWLEDGE_DIR", str(ROOT / "data/qa"))
    config = Settings.from_env()
    assert config.translation.configured and config.qa_knowledge_dir == ROOT / "data/qa"
    assert config.translation != config.llm and config.translation != config.tts


def test_audio_completion_does_not_wait_for_slow_caption_timeout():
    async def run():
        closed = asyncio.Event()
        class Stalled:
            timeout = 30
            async def translate(self, segment, language):
                try:
                    await asyncio.Event().wait()
                finally:
                    closed.set()
        async def source():
            yield "segment", {"turn_id": "t", "segment_id": "t:0", "sequence": 0,
                              "text": "English", "language": "en", "state": "committed"}
            await asyncio.sleep(0.01)
            yield "audio_end", {}
            yield "done", {}
        async with asyncio.timeout(1):
            events = [item async for item in caption_events(source(), Stalled(), "ko", audio_end_grace=0.01)]
        assert closed.is_set()
        assert events[-1] == ("done", {})
        assert next(data for kind, data in events if kind == "subtitle")["reason"] == "translation_late"
    asyncio.run(run())


@pytest.mark.parametrize("request_fields", [
    {"message": "이 프로젝트 파이프라인은?", "subtitle_language": "ko"},
    {"message": "What did you mean by AI factories at COMPUTEX 2025?", "presentation_qa": True},
])
def test_wrong_language_answer_never_reaches_english_voice(request_fields):
    async def run():
        e = engine()
        class WrongLanguage:
            async def stream(self, messages):
                yield "한국어로 잘못 생성한 답변입니다. [source:QA-project-pipeline]"
        class Voice:
            settings = ProviderSettings("http://voice.invalid/v1", "synthetic")
            async def synthesize(self, *args, **kwargs):
                raise AssertionError("Wrong-language text must not reach speech")
        e.llm = WrongLanguage()
        with pytest.raises(ProviderError) as error:
            async for _ in live_events(e, Voice(), ChatRequest(language="en", **request_fields)):
                pass
        assert error.value.code == "speech_language_mismatch"
    asyncio.run(run())


@pytest.mark.parametrize("question", ["엔비디아 주식 사야 하나요?", "젠슨 황 불륜 뉴스 사실인가요?",
                                    "오늘 엔비디아 주가는 얼마인가요?", "우리 아바타는 MCCCD처럼 학습했나요?"])
def test_fixed_uncertainty_subtitles_preserve_meaning_without_another_model(question):
    from bias.segments import SegmentCommitter
    async def run():
        turn = await engine().chat(ChatRequest(message=question, language="en"))
        committer = SegmentCommitter(turn_id=turn.id, language="en", allowed_ids=set())
        events = [await SubtitleTranslator().translate(segment, "ko")
                  for segment in committer.push(turn.text) + committer.finish()]
        assert events and all(event["state"] == "ready" for event in events)
        assert all(event["basis"] == "reviewed_fixed_phrase" for event in events)
        assert "허위" not in " ".join(event["text"] for event in events)
    asyncio.run(run())
