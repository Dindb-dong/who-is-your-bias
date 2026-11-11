"""Orchestrate retrieval, generation, citation validation, and animation cues."""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing

from .behavior import plan_behavior
from .config import Settings
from .corpus import Corpus
from .models import ChatRequest, ChatTurn, Grounding, TurnMetrics
from .prompts import build_messages
from .providers import OpenAICompatibleLLM
from .retrieval import Retriever, SearchResult, tokenize
from .sessions import Session, SessionStore

CITATION = re.compile(r"\[source:([A-Za-z0-9_.-]+)\]")


def validate_citations(text: str, allowed_ids: set[str]) -> tuple[str, list[str], list[str]]:
    mentioned = list(dict.fromkeys(CITATION.findall(text)))
    valid = [identifier for identifier in mentioned if identifier in allowed_ids]
    invalid = [identifier for identifier in mentioned if identifier not in allowed_ids]
    cleaned = CITATION.sub("", text)
    cleaned = re.sub(r"[ \t]+([.,!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, valid, invalid


def evidence_preview(request: ChatRequest, result: SearchResult) -> str:
    if request.mode == "profile":
        return (
            "프로필만 사용하는 실험 조건입니다. 언어 모델을 연결하면 인물 소개와 표현 지침만으로 응답합니다. 현재는 자료 미리보기 모드이며, 인물의 새 답변을 생성하지 않습니다."
            if request.language == "ko" else
            "This is the profile-only experiment. Connect a language model to generate a reply using the profile and presentation guidance alone. Evidence preview mode does not generate a new persona response."
        )
    if not result.hits:
        return (
            "이 질문에 답할 수 있는 검토된 공개 자료를 찾지 못했습니다. 현재 자료는 과거 인터뷰와 기고문이므로 실시간 정보나 공개되지 않은 경험을 확인할 수 없습니다."
            if request.language == "ko" else
            "I couldn't find reviewed public evidence that answers this question. This archive contains historical interviews and articles; it cannot verify current facts or private experiences."
        )
    prefix = (
        "자료 미리보기입니다. 아래 내용은 공개 자료의 영어 요약이며, 언어 모델이 생성한 한국어 답변은 아닙니다.\n\n"
        if request.language == "ko" else
        "Evidence preview — these are curated summaries of public sources, not a newly generated persona response.\n\n"
    )
    return prefix + "\n\n".join(f"{hit.evidence.text} [source:{hit.evidence.id}]" for hit in result.hits[:2])


class Engine:
    def __init__(self, settings: Settings, corpus: Corpus, *, llm: OpenAICompatibleLLM | None = None):
        self.settings = settings
        self.corpus = corpus
        self.retriever = Retriever(corpus)
        self.sessions = SessionStore(ttl=settings.session_ttl, max_sessions=settings.max_sessions, max_turns=settings.max_turns)
        self.llm = llm or OpenAICompatibleLLM(settings.llm, timeout=settings.request_timeout, max_chars=settings.max_response_chars, max_tokens=settings.llm_max_tokens)

    async def events(self, request: ChatRequest) -> AsyncIterator[tuple[str, dict]]:
        started = time.perf_counter()
        session = self.sessions.get_or_create(request.session_id)
        session.in_flight += 1
        try:
            async with aclosing(self._session_events(request, session, started)) as events:
                async for event in events:
                    yield event
        finally:
            session.in_flight -= 1

    async def _session_events(self, request: ChatRequest, session: Session, started: float) -> AsyncIterator[tuple[str, dict]]:
        yield "meta", {"session_id": session.id}
        async with asyncio.timeout(self.settings.request_timeout + 2):
            async with session.lock:
                yield "status", {"stage": "retrieving"}
                retrieval_started = time.perf_counter()
                if request.mode == "profile":
                    result = SearchResult((), 0, "Profile-only ablation has no retrieved evidence.")
                else:
                    query = request.message
                    if session.history and len(tokenize(query)) < 3:
                        query = session.history[-2]["content"] + "\n" + query
                    result = self.retriever.search(query, limit=self.settings.retrieval_limit, behavior=request.mode == "behavior")
                retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
                yield "status", {"stage": "generating"}
                generation_started = time.perf_counter()
                use_llm = self.settings.llm.configured and (request.mode == "profile" or bool(result.hits))
                fragments: list[str] = []
                if use_llm:
                    messages = build_messages(self.corpus.persona, request, result, list(session.history))
                    async with aclosing(self.llm.stream(messages)) as tokens:
                        async for token in tokens:
                            fragments.append(token)
                            yield "token", {"text": token}
                else:
                    preview = evidence_preview(request, result)
                    fragments.append(preview)
                    yield "token", {"text": preview}
                generation_ms = (time.perf_counter() - generation_started) * 1000
                yield "status", {"stage": "planning"}
                raw = "".join(fragments)
                allowed_ids = {hit.evidence.id for hit in result.hits}
                text, valid, invalid = validate_citations(raw, allowed_ids)
                if request.mode != "profile" and not result.hits:
                    status = "insufficient"
                    reason = result.reason
                elif request.mode == "profile" or not valid or invalid:
                    status = "unverified"
                    reason = "No complete set of valid retrieved citation identifiers accompanies the response."
                else:
                    status = "grounded"
                    reason = "Citation identifiers were validated; factual entailment still requires review." if use_llm else "Curated evidence preview; no language-model inference was performed."
                turn = ChatTurn(
                    id=uuid.uuid4().hex,
                    session_id=session.id,
                    text=text,
                    language=request.language,
                    mode=request.mode,
                    provider="openai" if use_llm else "evidence",
                    citations=[self.corpus.citation(identifier) for identifier in valid],
                    behavior=plan_behavior(text, request.message, mode=request.mode, language=request.language),
                    grounding=Grounding(status=status, retrieved_ids=[hit.evidence.id for hit in result.hits], invalid_citation_ids=invalid, reason=reason),
                    metrics=TurnMetrics(retrieval_ms=round(retrieval_ms, 3), generation_ms=round(generation_ms, 3), total_ms=round((time.perf_counter() - started) * 1000, 3)),
                )
                # Failed or cancelled generations never enter conversation memory.
                self.sessions.append(session, request.message, turn.text)
                yield "result", {"turn": turn.model_dump(mode="json")}
                yield "done", {}

    async def chat(self, request: ChatRequest) -> ChatTurn:
        turn: ChatTurn | None = None
        async for event, payload in self.events(request):
            if event == "result":
                turn = ChatTurn.model_validate(payload["turn"])
        if turn is None:
            raise RuntimeError("The response ended without a result")
        return turn
