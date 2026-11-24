"""Production perspective and no-invented-memory checks with synthetic LLMs.

These are routing, source packing and streaming regression tests. They do not
measure resemblance to Jensen or establish generated factual entailment.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from bias.compact_prompts import CONTEXT_PREFIX, build_compact_messages
from bias.config import ProviderSettings, Settings
from bias.context_budget import ByteBudgetCounter, build_snapshot_messages
from bias.corpus import Corpus
from bias.engine import Engine
from bias.models import ChatRequest, Evidence, Locator, Persona, Source
from bias.persona_voice import VERSION, actual_personal_fact_question, identity_reply
from bias.prompts import build_messages
from bias.qa_knowledge import QAEntry, QAKnowledge
from bias.retrieval import SearchHit, SearchResult


@pytest.fixture
def persona_corpus():
    source = Source(id="fixture-interview", title="Synthetic interview fixture", url="https://example.org/interview",
                    publisher="Synthetic test fixture", published_at=date(2024, 1, 1), source_type="interview",
                    persona_id="jensen-huang")
    evidence = Evidence(id="fixture-dennys", source_id=source.id,
        text="Huang worked at Denny's as a dishwasher, busboy and waiter.",
        topics=["Denny's", "dishwasher", "waiter", "데니스", "웨이터"],
        locator=Locator(label="Synthetic fixture, not collected source text"), kind="reported_experience",
        review_status="reviewed", persona_id="jensen-huang", persona_memory={
            "question_context": "What did early restaurant work teach him?",
            "decision_criterion": "Understand a task before setting expectations for colleagues.",
            "revision_condition": "This fixture makes no claim that every restaurant job teaches leadership.",
            "response_structure": "Start with a specific role, then explain its relevance.",
            "first_person_eligible": True, "evidence_mode": "editorial_paraphrase_with_short_excerpt",
        })
    persona = Persona(id="jensen-huang", name="Jensen Huang", tagline="Test", description="Synthetic test profile.")
    return Corpus(persona, {source.id: source}, {evidence.id: evidence})


class RecordingLLM:
    def __init__(self, text):
        self.text, self.messages = text, []

    async def stream(self, messages):
        self.messages.append(messages)
        yield self.text[:12]
        yield self.text[12:]


def configured(**kwargs):
    return Settings(llm=ProviderSettings("http://llm.invalid/v1", "synthetic"), **kwargs)


def result(corpus):
    return SearchResult((SearchHit(corpus.evidence["fixture-dennys"], 1.0, ("denny",)),), 1.0)


@pytest.mark.parametrize("language", ["en", "ko"])
def test_production_builder_requires_first_person_but_frozen_builder_does_not(persona_corpus, language):
    request = ChatRequest(message="Tell me about Denny's.", language=language)
    args = (persona_corpus.persona, request, result(persona_corpus), [], persona_corpus.sources)
    production, frozen = build_messages(*args), build_compact_messages(*args)
    assert "simulated first-person voice" in production[0]["content"]
    assert "simulated first-person voice" not in frozen[0]["content"]
    context = json.loads(production[1]["content"][len(CONTEXT_PREFIX):])
    memory = context["evidence_records"][0]["persona_memory"]
    assert memory["revision_condition"] == persona_corpus.evidence["fixture-dennys"].model_extra["persona_memory"]["revision_condition"]
    assert "missing evidence" in production[0]["content"].lower()


def test_snapshot_path_counts_first_person_and_memory_inside_budget(persona_corpus):
    request = ChatRequest(message="데니스 웨이터 경험", language="ko")
    common = dict(context_window=12000, output_tokens=256, counter=ByteBudgetCounter())
    production = build_snapshot_messages(persona_corpus, request, result(persona_corpus), [], None, None,
                                         persona_voice=True, **common)
    frozen = build_snapshot_messages(persona_corpus, request, result(persona_corpus), [], None, None, **common)
    assert "그의 입장을 1인칭으로 쓰거나" not in production.messages[0]["content"]
    assert "그의 입장을 1인칭으로 쓰거나" in frozen.messages[0]["content"]
    context = json.loads(production.messages[1]["content"].split("\n", 1)[1])
    assert context["factual_evidence"][0]["persona_memory"]["first_person_eligible"] is True
    assert production.supplied_ids == ("fixture-dennys",)
    assert production.budget["input_units"] == ByteBudgetCounter().count(production.messages)


@pytest.mark.parametrize(("question", "expected"), [
    ("Did you ever wait tables at Denny's?", True),
    ("If I remember correctly, did you wait tables at Denny's?", True),
    ("What was your first job?", True),
    ("데니스에서 웨이터로 일한 적 있어?", True),
    ("What would you do if you ran a restaurant?", False),
    ("If you worked at Denny's today, how would you improve the kitchen?", False),
])
def test_missing_biography_gate_does_not_block_hypothetical_advice(question, expected):
    assert actual_personal_fact_question(question) is expected


@pytest.mark.parametrize("language,question", [("en", "Did you work at an unknown restaurant?"),
                                               ("ko", "데니스에서 일한 적 있어?")])
def test_biography_miss_cannot_stream_an_invented_denial(corpus, language, question):
    empty_biography = Corpus(corpus.persona.model_copy(update={"id": "jensen-huang", "name": "Jensen Huang"}),
                            corpus.sources, corpus.evidence)
    llm = RecordingLLM("I never waited tables at Denny's.")
    engine = Engine(configured(), empty_biography, llm=llm)
    async def run():
        return [event async for event in engine.events(ChatRequest(message=question, language=language), committed_segments=True)]
    events = asyncio.run(run())
    assert llm.messages == []
    assert not any("I never" in json.dumps(payload) for _, payload in events)
    turn = next(payload["turn"] for kind, payload in events if kind == "result")
    assert turn["grounding"]["status"] == "insufficient" and not turn["citations"]
    assert turn["context"]["persona_voice"]["memory_status"] == "unverified_not_denied"
    assert any(kind == "segment" for kind, _ in events)


@pytest.mark.parametrize("language,question", [("en", "Who are you?"), ("ko", "너 누구야?")])
def test_identity_is_jensen_simulation_without_historical_citations(persona_corpus, language, question):
    llm = RecordingLLM("An unsupported identity answer.")
    engine = Engine(configured(), persona_corpus, llm=llm)
    turn = asyncio.run(engine.chat(ChatRequest(message=question, language=language)))
    assert llm.messages == [] and not turn.citations
    assert turn.context["persona_voice"]["version"] == VERSION
    assert "simulation" in turn.text if language == "en" else "시뮬레이션" in turn.text
    assert "Jensen Huang" in turn.text if language == "en" else "젠슨 황" in turn.text
    assert identity_reply("Who are you and what is your private address?", language) is None


@pytest.mark.parametrize("language,text", [
    ("en", "I worked at Denny's as a dishwasher, busboy and waiter. [source:fixture-dennys]"),
    ("ko", "저는 데니스에서 식기 세척, 버스보이, 웨이터 일을 했습니다. [source:fixture-dennys]"),
])
def test_supported_first_person_reaches_stream_and_source_cards(persona_corpus, language, text):
    llm = RecordingLLM(text)
    engine = Engine(configured(), persona_corpus, llm=llm)
    async def run():
        return [event async for event in engine.events(ChatRequest(message="Denny's waiter 데니스 웨이터", language=language), committed_segments=True)]
    events = asyncio.run(run())
    turn = next(payload["turn"] for kind, payload in events if kind == "result")
    assert turn["text"] == text.split(" [source:")[0]
    assert turn["citations"][0]["evidence_id"] == "fixture-dennys"
    assert turn["grounding"]["semantic_status"] == "unreviewed"
    assert "simulated first-person voice" in llm.messages[0][0]["content"]
    spoken = " ".join(payload["text"] for kind, payload in events if kind == "segment")
    assert "[source:" not in spoken


def test_uncited_korean_first_person_is_screened_before_speech(persona_corpus):
    llm = RecordingLLM("저는 데니스에서 경영 철학을 완성했습니다.")
    engine = Engine(configured(), persona_corpus, llm=llm)
    async def run():
        return [event async for event in engine.events(ChatRequest(message="Denny's waiter", language="ko"), committed_segments=True)]
    events = asyncio.run(run())
    assert all("경영 철학을 완성" not in json.dumps(payload, ensure_ascii=False) for _, payload in events)


def test_news_uses_dated_corporate_voice_but_never_personal_attendance(persona_corpus):
    now = datetime.now(timezone.utc)
    entry = QAEntry(id="QA-news-test", scope="news", title="NVIDIA fixture announcement",
                    url="https://nvidianews.nvidia.com/news/fixture", publisher="NVIDIA",
                    summary="NVIDIA announced a test research partnership.", cues=["research partnership"],
                    reviewed_at=now, expires_at=now+timedelta(hours=1), reviewer="test",
                    reviewer_type="agent", review_status="reviewed", published_at=now.date(),
                    fact_kind="company_claim")
    llm = RecordingLLM("We announced a research partnership. [source:QA-news-test]")
    engine = Engine(configured(), persona_corpus, llm=llm)
    engine.qa_knowledge = QAKnowledge((entry,))
    turn = asyncio.run(engine.chat(ChatRequest(message="What is the latest NVIDIA research partnership?")))
    assert turn.context["qa_knowledge"]["route"] == "news"
    assert turn.context["persona_voice"]["scope"] == "dated_company_records_not_personal_memories"
    system = llm.messages[0][0]["content"]
    assert "simulated first-person voice" in system
    assert "Do not claim personal attendance" in system
    assert turn.citations[0].evidence_id == "QA-news-test"


def test_explicit_frozen_builder_remains_unmodified(persona_corpus):
    llm = RecordingLLM("An evidence-grounded fixture. [source:fixture-dennys]")
    engine = Engine(configured(), persona_corpus, llm=llm, prompt_builder=build_compact_messages)
    turn = asyncio.run(engine.chat(ChatRequest(message="Denny's waiter")))
    assert "persona_voice" not in turn.context
    assert "simulated first-person voice" not in llm.messages[0][0]["content"]
