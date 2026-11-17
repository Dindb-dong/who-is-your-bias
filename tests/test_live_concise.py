"""Protocol tests with fake providers, not measured model speed or quality."""
import asyncio
import io
import wave

import pytest

from bias.config import ProviderSettings, Settings
from bias.context_budget import ByteBudgetCounter, build_snapshot_messages, live_concise_instruction
from bias.corpus import Corpus
from bias.engine import Engine
from bias.live import live_events
from bias.models import ChatRequest
from bias.retrieval import Retriever


class RecordingLLM:
    def __init__(self, text="Reusable rockets lower costs. [source:rocket-reuse]"):
        self.messages = []
        self.text = text
        self.complete = False

    async def stream(self, messages):
        self.messages.append(messages)
        yield self.text
        self.complete = True


class Voice:
    settings = ProviderSettings("http://voice.invalid/v1", "test")

    def __init__(self):
        self.texts = []

    async def synthesize(self, text, **kwargs):
        self.texts.append(text)
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\0\x10" * 320)
        return output.getvalue(), "audio/wav"


@pytest.mark.parametrize("enabled", [False, True])
def test_live_policy_is_opt_in_and_does_not_change_text_only_requests(corpus, enabled):
    async def run():
        llm = RecordingLLM()
        engine = Engine(Settings(llm=ProviderSettings("http://llm.invalid/v1", "test"), live_concise=enabled), corpus, llm=llm)
        request = ChatRequest(message="Why reuse rockets?")
        normal = await engine.chat(request)
        voice = Voice()
        events = [event async for event in live_events(engine, voice, request)]
        live = next(data["turn"] for kind, data in events if kind == "result")
        assert "Live spoken-answer mode" not in llm.messages[0][0]["content"]
        assert ("Live spoken-answer mode" in llm.messages[1][0]["content"]) is enabled
        assert (live["context"].get("response_policy") == "live-concise.v1") is enabled
        assert "response_policy" not in normal.context
        assert live["citations"][0]["evidence_id"] == "rocket-reuse"
        assert voice.texts == ["Reusable rockets lower costs."]
    asyncio.run(run())


def test_live_instruction_is_included_in_snapshot_budget(corpus):
    request = ChatRequest(message="Why reuse rockets?")
    result = Retriever(corpus).search(request.message, limit=4)
    counter = ByteBudgetCounter()
    common = dict(context_window=8192, output_tokens=128, counter=counter)
    baseline = build_snapshot_messages(corpus, request, result, [], None, None, **common)
    concise = build_snapshot_messages(corpus, request, result, [], None, None, live_concise=True, **common)
    assert concise.supplied_ids == baseline.supplied_ids == ("rocket-reuse",)
    assert concise.budget["input_units"] == counter.count(concise.messages)
    assert concise.budget["input_units"] - baseline.budget["input_units"] == len(live_concise_instruction("en").encode("utf-8"))


def test_concise_mode_keeps_korean_review_before_speech(corpus):
    async def run():
        jensen = Corpus(corpus.persona.model_copy(update={"id": "jensen-huang"}), corpus.sources, corpus.evidence)
        llm = RecordingLLM("저는 로켓을 재사용합니다. [source:rocket-reuse]")
        class ReviewedVoice(Voice):
            async def synthesize(self, text, **kwargs):
                assert llm.complete
                return await super().synthesize(text, **kwargs)
        voice = ReviewedVoice()
        engine = Engine(Settings(llm=ProviderSettings("http://llm.invalid/v1", "test"), live_concise=True), jensen, llm=llm)
        events = [event async for event in live_events(engine, voice, ChatRequest(message="로켓 재사용", language="ko"))]
        answer = next(data["turn"] for kind, data in events if kind == "result")
        assert "현장 음성 답변" in llm.messages[0][0]["content"]
        assert "한국어 답변을 확인할 수 없습니다" in answer["text"]
        assert all("저는" not in text for text in voice.texts)
        assert answer["citations"][0]["evidence_id"] == "rocket-reuse"
    asyncio.run(run())


def test_live_flag_environment_is_explicit_and_validated(monkeypatch):
    monkeypatch.delenv("BIAS_LIVE_CONCISE", raising=False)
    assert Settings.from_env().live_concise is False
    monkeypatch.setenv("BIAS_LIVE_CONCISE", "true")
    assert Settings.from_env().live_concise is True
    monkeypatch.setenv("BIAS_LIVE_CONCISE", "sometimes")
    with pytest.raises(ValueError, match="BIAS_LIVE_CONCISE"):
        Settings.from_env()
    with pytest.raises(ValueError, match="live_concise"):
        Settings(live_concise="true")


def test_health_exposes_live_policy_for_reproducible_comparison(corpus):
    from fastapi.testclient import TestClient
    from bias.api import create_app

    with TestClient(create_app(Settings(live_concise=True), corpus=corpus)) as client:
        assert client.get("/api/health").json()["live_response_policy"] == "live-concise.v1"
