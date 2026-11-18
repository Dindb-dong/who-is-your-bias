"""Routing/integration checks with stub models; not response quality measurements."""
import asyncio

import pytest

from bias.answer_policy import VERSION, is_presentation_followup, select_answer_policy
from bias.config import ProviderSettings, Settings
from bias.context_budget import ByteBudgetCounter, build_snapshot_messages
from bias.corpus import Corpus
from bias.dialogue import DialogueState
from bias.engine import Engine
from bias.models import ChatRequest
from bias.prompts import PROMPT_BUILDERS, build_messages
from bias.retrieval import Retriever


@pytest.mark.parametrize("question,kind,maximum", [
    ("What is an AI factory?", "factual", 2),
    ("AI 팩토리가 무엇인가요?", "factual", 2),
    ("Why did you choose Qwen?", "explanation", 4),
    ("Qwen 모델을 선택한 이유가 뭔가요?", "explanation", 4),
    ("Compare local inference versus an API.", "comparison", 5),
    ("로컬 추론과 API의 장단점은?", "comparison", 5),
    ("Walk me through your pipeline.", "procedure", 5),
    ("프로젝트 파이프라인이 어떻게 되나요?", "procedure", 5),
    ("What advice would you give a student?", "advice", 4),
    ("실패를 두려워하는 학생에게 조언한다면?", "advice", 4),
])
def test_bilingual_question_targets(question, kind, maximum):
    policy = select_answer_policy(question, language="en")
    assert policy.question_type == kind
    assert policy.max_sentences == maximum
    assert policy.as_dict()["enforcement"] == "prompt_guidance"


@pytest.mark.parametrize("question,detail,maximum", [
    ("Compare A and B in one sentence.", "one_sentence", 1),
    ("원리를 한 문장으로 설명해줘", "one_sentence", 1),
    ("Explain the pipeline briefly.", "brief", 2),
    ("파이프라인을 짧게 설명해줘", "brief", 2),
    ("What is an AI factory? Explain in detail.", "detailed", 5),
    ("AI 팩토리를 더 자세히 설명해줘", "detailed", 5),
    ("Explain in detail. Actually, answer in one sentence.", "one_sentence", 1),
    ("한 줄로 말하지 말고 자세히 설명해줘", "detailed", 5),
])
def test_current_explicit_request_controls_detail(question, detail, maximum):
    policy = select_answer_policy(question, language="en")
    assert (policy.detail, policy.max_sentences) == (detail, maximum)


@pytest.mark.parametrize("message", ["더 자세히", "한 문장으로", "한 줄로 말해줘", "짧게", "아니, 짧게", "좀 더 자세히 설명해줘",
                                     "Tell me more", "Answer in one sentence", "Explain in detail, please."])
def test_bare_presentation_followups_preserve_user_topic_only(message):
    dialogue = DialogueState()
    initial = dialogue.resolve("Compare rocket reuse and expendable rockets, briefly.")
    dialogue.append(turn_id="first", resolved=initial, assistant="Unsupported secret claim!",
                    cited_ids=[], grounding="unverified", intent=None, snapshot_id=None)
    resolved = dialogue.resolve(message, presentation_followups=True)
    assert resolved.supporting_turn_ids == ("first",)
    assert resolved.anchor == initial.original
    assert resolved.query == initial.original
    assert "Unsupported secret claim" not in resolved.query
    policy = select_answer_policy(message, language="en", anchor=resolved.anchor)
    assert policy.question_type == "comparison"
    if "detail" in message or "자세" in message or message == "Tell me more":
        assert policy.detail == "detailed"  # Prior "briefly" must not leak.


@pytest.mark.parametrize("message", ["Briefly, what is a GPU?", "다른 질문인데, AI 팩토리를 짧게 설명해줘",
                                     "How are you?", "Why does gravity work?", "What is a detailed balance model?"])
def test_new_questions_are_not_presentation_only_followups(message):
    assert not is_presentation_followup(message)


@pytest.mark.parametrize("language,maximum", [("en", 2), ("ko", 1)])
def test_detailed_request_never_expands_unsupported_answers(language, maximum):
    policy = select_answer_policy("Explain the project pipeline in detail", language=language, speculative=True)
    assert policy.evidence_limited
    assert policy.max_sentences == maximum


class RecordingLLM:
    def __init__(self, text="Reusable rockets can lower costs. [source:rocket-reuse]"):
        self.messages = []
        self.text = text
        self.complete = False

    async def stream(self, messages):
        self.messages.append(messages)
        # Splits are intentional: review must hold back even a complete first
        # sentence until the entire Korean draft has passed the existing guard.
        yield self.text[:15]
        yield self.text[15:]
        self.complete = True


def configured(**kwargs):
    return Settings(llm=ProviderSettings("http://llm.invalid/v1", "stub"), **kwargs)


@pytest.mark.parametrize("streaming", [False, True])
def test_policy_reaches_generation_and_keeps_citations(corpus, streaming):
    async def run():
        llm = RecordingLLM()
        engine = Engine(configured(adaptive_responses=True, live_concise=True), corpus, llm=llm)
        events = [event async for event in engine.events(ChatRequest(message="Why reuse rockets?"), committed_segments=streaming)]
        turn = next(data["turn"] for kind, data in events if kind == "result")
        assert len(llm.messages) == 1  # No LLM classification / rewriting call.
        assert VERSION in llm.messages[0][0]["content"]
        assert "Live spoken-answer mode" not in llm.messages[0][0]["content"]
        assert turn["context"]["answer_policy"]["question_type"] == "explanation"
        assert turn["citations"][0]["evidence_id"] == "rocket-reuse"
        assert turn["context"]["response_policy"] == VERSION
        assert any(kind == "segment" for kind, _ in events) is streaming
    asyncio.run(run())


def test_engine_followup_uses_anchor_and_changes_length(corpus):
    async def run():
        llm = RecordingLLM()
        engine = Engine(configured(adaptive_responses=True), corpus, llm=llm)
        first = await engine.chat(ChatRequest(message="Why reuse rockets?"))
        second = await engine.chat(ChatRequest(message="한 문장으로", language="ko", session_id=first.session_id))
        assert second.context["query_resolution"]["supporting_turn_ids"] == [first.id]
        assert second.context["answer_policy"]["max_sentences"] == 1
        assert second.context["answer_policy"]["question_type"] == "explanation"
        assert second.citations[0].evidence_id == "rocket-reuse"
        third = await engine.chat(ChatRequest(message="더 자세히", language="ko", session_id=first.session_id))
        assert third.context["answer_policy"]["max_sentences"] == 5
        assert third.context["query_resolution"]["query"] == "Why reuse rockets?"
        fourth = await engine.chat(ChatRequest(message="Why an expensive sports car?", session_id=first.session_id))
        assert fourth.context["query_resolution"]["supporting_turn_ids"] == []
        assert fourth.context["answer_policy"]["detail"] == "auto"
    asyncio.run(run())


@pytest.mark.parametrize("builder", [build_messages, *PROMPT_BUILDERS.values()])
def test_disabled_policy_and_explicit_evaluation_builders_stay_frozen(corpus, builder):
    async def run(enabled):
        llm = RecordingLLM()
        engine = Engine(configured(adaptive_responses=enabled), corpus, llm=llm, prompt_builder=builder)
        request = ChatRequest(message="Why reuse rockets?")
        turn = await engine.chat(request)
        return llm.messages[0], turn
    baseline, _ = asyncio.run(run(False))
    if builder is not build_messages:
        candidate, turn = asyncio.run(run(True))
        assert baseline == candidate
        assert "answer_policy" not in turn.context
    assert VERSION not in baseline[0]["content"]


def test_profile_ablation_stays_unchanged(corpus):
    async def run():
        llm = RecordingLLM("A profile-only simulation.")
        engine = Engine(configured(adaptive_responses=True), corpus, llm=llm)
        turn = await engine.chat(ChatRequest(message="Explain in detail", mode="profile"))
        assert "answer_policy" not in turn.context
        assert VERSION not in llm.messages[0][0]["content"]
    asyncio.run(run())


def test_snapshot_counts_policy_and_retains_korean_constraints(corpus):
    jensen = Corpus(corpus.persona.model_copy(update={"id": "jensen-huang"}), corpus.sources, corpus.evidence)
    request = ChatRequest(message="로켓 재사용 이유를 자세히 설명해줘", language="ko")
    result = Retriever(jensen).search("로켓 재사용", limit=4)
    counter = ByteBudgetCounter()
    policy = select_answer_policy(request.message, language="ko")
    common = dict(context_window=8192, output_tokens=900, counter=counter)
    baseline = build_snapshot_messages(jensen, request, result, [], None, None, **common)
    packed = build_snapshot_messages(jensen, request, result, [], None, None, answer_policy=policy, **common)
    system = packed.messages[0]["content"]
    assert "3~5문장" in system
    assert "한국어 한 문장" not in system
    assert "한국어 한 문장" in baseline.messages[0]["content"]
    assert "그의 입장을 1인칭으로 쓰거나" in system
    assert packed.supplied_ids == baseline.supplied_ids == ("rocket-reuse",)
    assert packed.budget["input_units"] == counter.count(packed.messages)
    assert packed.budget["input_units"] <= 8192 - 900


@pytest.mark.parametrize("text,accepted", [
    ("재사용은 발사 비용을 줄일 수 있습니다. [source:rocket-reuse] 반복 운항을 통해 비용을 줄입니다. [source:rocket-reuse]", True),
    ("저는 로켓을 개발했습니다. [source:rocket-reuse]", False),
])
def test_korean_review_still_precedes_any_spoken_segment(corpus, text, accepted):
    async def run():
        jensen = Corpus(corpus.persona.model_copy(update={"id": "jensen-huang"}), corpus.sources, corpus.evidence)
        llm = RecordingLLM(text)
        engine = Engine(configured(adaptive_responses=True), jensen, llm=llm)
        events = []
        async for kind, data in engine.events(ChatRequest(message="로켓 재사용", language="ko"), committed_segments=True):
            if kind in {"token", "segment"}:
                assert llm.complete
            events.append((kind, data))
        turn = next(data["turn"] for kind, data in events if kind == "result")
        assert ("한국어 답변을 확인할 수 없습니다" not in turn["text"]) is accepted
        assert turn["citations"][0]["evidence_id"] == "rocket-reuse"
    asyncio.run(run())


def test_flag_and_health_are_explicit(monkeypatch, corpus):
    from fastapi.testclient import TestClient
    from bias.api import create_app
    monkeypatch.delenv("BIAS_ADAPTIVE_RESPONSES", raising=False)
    assert Settings.from_env().adaptive_responses is False
    monkeypatch.setenv("BIAS_ADAPTIVE_RESPONSES", "true")
    assert Settings.from_env().adaptive_responses is True
    monkeypatch.setenv("BIAS_ADAPTIVE_RESPONSES", "sometimes")
    with pytest.raises(ValueError, match="BIAS_ADAPTIVE_RESPONSES"):
        Settings.from_env()
    with pytest.raises(ValueError, match="adaptive_responses"):
        Settings(adaptive_responses="true")
    with TestClient(create_app(Settings(adaptive_responses=True, live_concise=True), corpus=corpus)) as client:
        health = client.get("/api/health").json()
        assert health["answer_policy"] == health["live_response_policy"] == VERSION
