"""Orchestrate retrieval, generation, citation validation, and animation cues."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing

from .behavior import plan_behavior
from .bindings import motion_binding_status
from .config import Settings
from .context_budget import ByteBudgetCounter, TokenizerCounter, build_snapshot_messages
from .corpus import Corpus
from .grounding import CITATION, audit_citations, validate_citations
from .models import ChatRequest, ChatTurn, Grounding, TurnMetrics
from .motion import MotionError, MotionLibrary, MotionSelection
from .prompts import PromptBuilder, build_messages
from .providers import OpenAICompatibleLLM, ProviderError
from .retrieval import Retriever, SearchResult, tokenize
from .sessions import Session, SessionStore
from .snapshots import PersonaSnapshot, TemporalIntent, select_temporal, temporal_intent
from .segments import SegmentCommitter
from .dialogue import DEICTIC_PAST

logger = logging.getLogger(__name__)

# Speculation is a proposed decision criterion, with no quotes or claims of past speech.
KOREAN_INFERENCE = re.compile(
    r"[가-힣A-Za-z ,·()–-]+(?:살필|비교할) 수 있습니다\."
)
UNSUPPORTED_ATTRIBUTION = re.compile(
    r"말했|발언했|밝혔|주장했|언급했|설명했|강조했|회고했|증언했|했다는|됐다는|"
    r"실제로|당시|이미|특허|발표한|젠슨 황|Huang|이유는|잘하는 일은|라고 전해|다고 답"
)
KOREAN_EXACT_SPEECH = re.compile(r"무슨 말을 했|어떤 말을 했|뭐라고 (?:말)?했|한 말은|그대로 (?:적|말|알려)|따옴표")
KOREAN_EVIDENCE_VIOLATION = re.compile(
    r"""["“”「」『』]|['‘][^'’\n]{12,}['’]"""
    r"|(?<![가-힣])(?:저는|제가|나는|내가|우리는|우리가)(?![가-힣])"
)
KOREAN_INFERENCE_FALLBACK = "어디에서 전체 성능이 막히는지, 개선이 실제 사용자의 비용과 시간에 어떤 영향을 줄지 비교할 수 있습니다."


def korean_inference(text: str) -> str:
    proposal = text.strip()
    if not KOREAN_INFERENCE.fullmatch(proposal) or UNSUPPORTED_ATTRIBUTION.search(proposal):
        proposal = KOREAN_INFERENCE_FALLBACK
    return (
        "가상 추론 (실제 발언 확인 불가)\n\n"
        f"젠슨 황의 공개 발언에서 드러난 판단 방식을 적용해 보면, {proposal}"
    )


def korean_summary_is_valid(text: str) -> bool:
    return (
        re.search(r"[가-힣]", text) is not None
        and KOREAN_EVIDENCE_VIOLATION.search(text) is None
        and all(char.isascii() or not char.isalpha() or "가" <= char <= "힣" for char in text)
    )


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
    if any(hit.evidence.question_answer for hit in result.hits):
        prefix = (
            "자료 미리보기입니다. 원본 인터뷰 문답의 발췌이며, 새로 생성한 인물의 답변이 아닙니다. 전체 문답 여부는 데이터에 별도 기록되어 있습니다.\n\n"
            if request.language == "ko" else
            "Evidence preview — original interview excerpts, not a newly generated persona response. Exchange completeness is recorded separately.\n\n"
        )
        return prefix + "\n\n".join(f"{hit.evidence.text} [source:{hit.evidence.id}]" for hit in result.hits[:2])
    prefix = (
        "자료 미리보기입니다. 아래 내용은 공개 자료의 영어 요약이며, 언어 모델이 생성한 한국어 답변은 아닙니다.\n\n"
        if request.language == "ko" else
        "Evidence preview — these are curated summaries of public sources, not a newly generated persona response.\n\n"
    )
    return prefix + "\n\n".join(f"{hit.evidence.text} [source:{hit.evidence.id}]" for hit in result.hits[:2])


class Engine:
    def __init__(self, settings: Settings, corpus: Corpus, *, llm: OpenAICompatibleLLM | None = None, prompt_builder: PromptBuilder = build_messages, motion_library: MotionLibrary | None = None):
        self.settings = settings
        self.corpus = corpus
        self.prompt_builder = prompt_builder
        self.motion_library = motion_library if motion_library is not None and motion_binding_status(motion_library, corpus) in {"matched", "source_bound"} else None
        self.snapshot = PersonaSnapshot.load(settings.persona_snapshot, corpus) if settings.persona_snapshot else None
        self.counter = ByteBudgetCounter()
        if settings.llm_tokenizer_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(settings.llm_tokenizer_path), local_files_only=True, trust_remote_code=False)
            self.counter = TokenizerCounter(tokenizer, str(settings.llm_tokenizer_path))
        if settings.retrieval_backend == "hybrid":
            from .hybrid_retrieval import CrossEncoderReranker, HybridPolicy, HybridRetriever, OpenAIEmbeddingProvider, SentenceTransformerEmbedder
            common = dict(revision=settings.embedding_revision, query_prefix=settings.embedding_query_prefix,
                          passage_prefix=settings.embedding_passage_prefix)
            embedder = (OpenAIEmbeddingProvider(settings.embedding_base_url, settings.embedding_model,
                                               api_key=settings.embedding_api_key or "", timeout=settings.request_timeout, **common)
                        if settings.embedding_backend == "http" else SentenceTransformerEmbedder(settings.embedding_model, **common))
            reranker = CrossEncoderReranker(settings.reranker_model, revision=settings.reranker_revision) if settings.reranker_model else None
            self.retriever = HybridRetriever(corpus, embedder, policy=HybridPolicy(dense_min_similarity=settings.dense_min_similarity,
                                               reranker_min_score=settings.reranker_min_score), reranker=reranker)
        else:
            self.retriever = Retriever(corpus)
        self.sessions = SessionStore(ttl=settings.session_ttl, max_sessions=settings.max_sessions, max_turns=settings.max_turns)
        self.llm = llm or OpenAICompatibleLLM(settings.llm, timeout=settings.request_timeout, max_chars=settings.max_response_chars, max_tokens=settings.llm_max_tokens)

    def _select_motion(self, request: ChatRequest, result: SearchResult) -> MotionSelection | None:
        if request.mode != "behavior" or request.motion_mode == "disabled" or not result.hits or self.motion_library is None:
            return None
        preview = request.motion_mode == "geometry_preview"
        try:
            # The question is preferred. Bounded retrieved paraphrases provide
            # a cross-language fallback; generated responses and emotion labels
            # never become claims about the source person's gesture policy.
            selected = self.motion_library.select(request.message, allow_geometry_preview=preview)
            if selected is None:
                context = "\n".join(hit.evidence.text[:600] for hit in result.hits[:2])
                selected = self.motion_library.select(request.message + "\n" + context, allow_geometry_preview=preview)
            return selected
        except MotionError as exc:
            logger.warning("Optional motion selection failed (%s)", type(exc).__name__)
            return None

    async def events(self, request: ChatRequest, *, committed_segments: bool = False, _session_owner: Session | None = None) -> AsyncIterator[tuple[str, dict]]:
        started = time.perf_counter()
        deadline = asyncio.get_running_loop().time() + self.settings.request_timeout + 2
        session = _session_owner or self.sessions.get_or_create(request.session_id)
        if _session_owner is not None:
            # The live transport holds the turn lock/pin until voice cleanup.
            async with aclosing(self._session_events(request, session, started, deadline=deadline, committed_segments=committed_segments)) as events:
                async for event in events:
                    yield event
            return
        session.in_flight += 1
        acquired = False
        try:
            # Lock waiting and generation share one absolute deadline. Leaving
            # the acquisition scope must not grant a fresh generation budget.
            async with asyncio.timeout_at(deadline):
                await session.turn_lock.acquire()
                acquired = True
            async with aclosing(self._session_events(request, session, started, deadline=deadline, committed_segments=committed_segments)) as events:
                async for event in events:
                    yield event
        finally:
            if acquired:
                session.turn_lock.release()
            session.in_flight -= 1

    async def _session_events(self, request: ChatRequest, session: Session, started: float, *, deadline: float, committed_segments: bool = False) -> AsyncIterator[tuple[str, dict]]:
        async with asyncio.timeout_at(deadline):
            turn_id = uuid.uuid4().hex
            yield "meta", {"session_id": session.id, "turn_id": turn_id,
                           "snapshot_id": self.snapshot.id if self.snapshot else None}
            async with session.lock:
                yield "status", {"stage": "retrieving"}
                retrieval_started = time.perf_counter()
                resolved = session.dialogue.resolve(request.message)
                temporal_query = request.message
                inherited_intent = None
                if self.snapshot and resolved.supporting_turn_ids:
                    explicit_intent = temporal_intent(request.message, self.snapshot.as_of)
                    # The last supporting user record can be an explicitly
                    # resumed earlier topic, not necessarily the last turn.
                    prior = next((record for record in reversed(session.dialogue.records)
                                  if record.turn_id == resolved.supporting_turn_ids[-1]), None)
                    if prior is not None and (explicit_intent.mode == "general" or
                            (explicit_intent.mode == "historical" and not explicit_intent.years and
                             DEICTIC_PAST.search(request.message) and prior.temporal_mode == "historical")):
                        temporal_query = resolved.query
                        inherited_intent = TemporalIntent(prior.temporal_mode, prior.temporal_years, prior.temporal_period)
                temporal = select_temporal(self.corpus, self.snapshot, temporal_query, intent_override=inherited_intent) if self.snapshot else None
                style_temporal = select_temporal(self.corpus, self.snapshot, temporal_query, projection="text_style", intent_override=inherited_intent) if self.snapshot and self.settings.style_examples else None
                context = {"snapshot": self.snapshot.summary() if self.snapshot else None,
                           "query_resolution": {"original": request.message, "query": resolved.query,
                                                "method": resolved.method, "supporting_turn_ids": list(resolved.supporting_turn_ids)},
                           "temporal": {"mode": temporal.intent.mode, "years": list(temporal.intent.years), "period": temporal.intent.period,
                                        "excluded": temporal.exclusion_reasons} if temporal else None,
                           "retrieval_backend": self.settings.retrieval_backend}
                if request.mode == "profile":
                    result = SearchResult((), 0, "Profile-only ablation has no retrieved evidence.")
                elif temporal and not temporal.eligible_ids:
                    result = SearchResult((), 0, temporal.reason)
                else:
                    query = resolved.query
                    if self.settings.retrieval_backend == "hybrid":
                        from .hybrid_retrieval import EmbeddingError
                        try:
                            result = await asyncio.to_thread(self.retriever.search, query, limit=self.settings.retrieval_limit,
                                behavior=request.mode == "behavior", original_query=request.message, allow_live=bool(temporal),
                                eligible_ids=temporal.eligible_ids if temporal else None,
                                eligible_style_ids=style_temporal.eligible_ids if style_temporal else None,
                                exclusion_reasons=temporal.exclusion_reasons if temporal else None,
                                temporal_scores=temporal.temporal_scores if temporal else None, include_style=self.settings.style_examples)
                        except EmbeddingError as exc:
                            raise ProviderError("embedding_unavailable", "The configured embedding service failed; retrieval was not silently downgraded.", 503) from exc
                        context["retrieval"] = result.trace
                    else:
                        retriever = self.retriever
                        if temporal and len(temporal.eligible_ids) != len(self.corpus.training_evidence):
                            filtered = Corpus(self.corpus.persona, self.corpus.sources,
                                              {key: value for key, value in self.corpus.evidence.items() if key in temporal.eligible_ids})
                            retriever = Retriever(filtered)
                        # The snapshot already applies exact/before/through/
                        # forecast policy to the eligible corpus. The archived
                        # lexical retriever's independent exact-year gate must
                        # not turn "before 2020" into "in 2020", or a future
                        # target into the publication year of its old forecast.
                        lexical_query = (re.sub(r"(?<!\d)(?:19|20)\d{2}(?!\d)", "", query)
                                         if temporal and temporal.intent.mode != "comparison" else query)
                        context["lexical_search"] = {"query": lexical_query, "temporal_scope_applied_before_search": bool(temporal)}
                        result = retriever.search(lexical_query, limit=self.settings.retrieval_limit, behavior=request.mode == "behavior", allow_live=bool(temporal))
                retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
                yield "status", {"stage": "generating"}
                generation_started = time.perf_counter()
                speculative = (self.settings.llm.configured and self.corpus.persona.id == "jensen-huang"
                               and request.mode != "profile" and not result.hits)
                unverified_quote = (
                    speculative and request.language == "ko" and KOREAN_EXACT_SPEECH.search(request.message) is not None
                    and temporal is not None and temporal.intent.mode == "future"
                )
                use_llm = not unverified_quote and self.settings.llm.configured and (
                    request.mode == "profile" or bool(result.hits) or speculative
                )
                fragments: list[str] = []
                allowed_ids = {hit.evidence.id for hit in result.hits}
                if use_llm:
                    if self.snapshot or self.settings.retrieval_backend == "hybrid" or speculative:
                        package = build_snapshot_messages(self.corpus, request, result, list(session.history), self.snapshot, temporal,
                            context_window=self.settings.llm_context_window, output_tokens=self.settings.llm_max_tokens,
                            counter=self.counter, resolved=resolved, speculative=speculative)
                        messages = package.messages
                        allowed_ids = set(package.supplied_ids)
                        context["prompt_budget"] = package.budget
                        if request.mode != "profile" and not speculative and not allowed_ids:
                            raise ProviderError("evidence_budget", "No complete evidence passage fits this model's context budget.", 422)
                    else:
                        messages = self.prompt_builder(self.corpus.persona, request, result, list(session.history), self.corpus.sources)
                    committer = SegmentCommitter(turn_id=turn_id, language=request.language, allowed_ids=allowed_ids)
                    review_inference = speculative and request.language == "ko"
                    review_evidence = (self.corpus.persona.id == "jensen-huang" and request.language == "ko"
                                       and request.mode != "profile" and bool(result.hits))
                    async with aclosing(self.llm.stream(messages)) as tokens:
                        async for token in tokens:
                            if review_inference or review_evidence:
                                fragments.append(token)
                                continue
                            if speculative and not fragments:
                                if not token.strip():
                                    continue
                                token = f"Hypothetical simulation (no verified statement)\n\n{token.lstrip()}"
                            fragments.append(token)
                            yield "token", {"text": token}
                            if committed_segments:
                                for segment in committer.push(token):
                                    yield "segment", segment
                    if review_inference or review_evidence:
                        text = "".join(fragments).strip()
                        if review_inference:
                            text = korean_inference(text)
                        elif not korean_summary_is_valid(text):
                            source = next(hit.evidence.id for hit in result.hits if hit.evidence.id in allowed_ids)
                            text = f"한국어 답변을 확인할 수 없습니다. 관련 공개 자료를 참고해 주세요. [source:{source}]"
                        fragments = [text]
                        yield "token", {"text": text}
                        if committed_segments:
                            for segment in committer.push(text):
                                yield "segment", segment
                    if committed_segments:
                        for segment in committer.finish():
                            yield "segment", segment
                else:
                    preview = evidence_preview(request, result)
                    if unverified_quote:
                        preview = "요청한 시점은 자료 기준일 이후라 실제 발언을 확인할 수 없습니다."
                    elif temporal and temporal.reason and request.mode != "profile":
                        preview = ("이 스냅샷에는 요청한 시점의 입장을 확인할 수 있는 검토된 자료가 없습니다. 과거 발언을 현재 입장으로 간주하지 않습니다."
                                   if request.language == "ko" else temporal.reason)
                    fragments.append(preview)
                    yield "token", {"text": preview}
                    if committed_segments:
                        committer = SegmentCommitter(turn_id=turn_id, language=request.language, allowed_ids=allowed_ids)
                        for segment in committer.push(preview) + committer.finish():
                            yield "segment", segment
                generation_ms = (time.perf_counter() - generation_started) * 1000
                yield "status", {"stage": "planning"}
                raw = "".join(fragments)
                text, valid, invalid = validate_citations(raw, allowed_ids)
                audit = audit_citations(raw, allowed_ids, required=use_llm and not speculative and request.mode != "profile")
                if request.mode != "profile" and not result.hits:
                    status = "insufficient"
                    reason = ("Hypothetical simulation. No supporting evidence was retrieved." if speculative and use_llm else result.reason)
                elif use_llm:
                    status = "unverified"
                    reason = (
                        "Sentence-level citation markers were checked; factual entailment still requires review."
                        if audit.coverage_status == "complete" else
                        "Profile-only responses have no source support; factual entailment still requires review."
                        if request.mode == "profile" else
                        "Citation coverage is incomplete or invalid; factual entailment still requires review."
                    )
                elif request.mode == "profile" or not valid or invalid:
                    status = "unverified"
                    reason = "No complete set of valid retrieved citation identifiers accompanies the response."
                else:
                    status = "grounded"
                    reason = "Curated evidence preview; no language-model inference was performed."
                behavior = plan_behavior(text, request.message, mode=request.mode, language=request.language)
                selected_motion = self._select_motion(request, result)
                if selected_motion is not None:
                    behavior = behavior.model_copy(update={"motion": selected_motion})
                turn = ChatTurn(
                    id=turn_id,
                    session_id=session.id,
                    text=text,
                    language=request.language,
                    mode=request.mode,
                    provider="openai" if use_llm else "evidence",
                    citations=[self.corpus.citation(identifier) for identifier in valid],
                    behavior=behavior,
                    grounding=Grounding(
                        status=status, retrieved_ids=[hit.evidence.id for hit in result.hits], invalid_citation_ids=invalid, reason=reason,
                        citation_status=audit.coverage_status, citation_unit_count=len(audit.units), cited_unit_count=audit.cited_unit_count,
                        malformed_citations=list(audit.malformed_markers), semantic_status="curated_source" if status == "grounded" else "unreviewed",
                    ),
                    metrics=TurnMetrics(retrieval_ms=round(retrieval_ms, 3), generation_ms=round(generation_ms, 3), total_ms=round((time.perf_counter() - started) * 1000, 3)),
                    context=context,
                )
                # Failed or cancelled generations never enter conversation memory.
                self.sessions.append(session, request.message, turn.text)
                session.dialogue.append(turn_id=turn.id, resolved=resolved, assistant=turn.text, cited_ids=valid,
                                        grounding=status, intent=temporal.intent if temporal else None,
                                        snapshot_id=self.snapshot.id if self.snapshot else None)
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

