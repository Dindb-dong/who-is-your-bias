"""HTTP and SSE API for the research studio."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import anyio
from contextlib import aclosing, asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import ClientDisconnect

from . import __version__
from .config import Settings
from .corpus import Corpus
from .engine import Engine
from .lipsync import LipsyncEngine
from .models import ChatRequest, ChatTurn, SpeechPackageRequest, SpeechRequest
from .motion import MotionError, MotionLibrary
from .providers import OpenAICompatibleLLM, ProviderError, SpeechProvider, TranscriptionProvider
from .request_work import run_until_disconnect
from .sessions import SessionCapacityError, SessionNotFound
from .speech import SpeechPackages
from .live import live_events

logger = logging.getLogger(__name__)


class DisconnectStreamingResponse(StreamingResponse):
    """Observe disconnect while awaiting a provider, including ASGI 2.4.

    The default 2.4 response detects send errors only at its next outgoing
    chunk. A listening task here cancels owned synthesis during silent waits.
    """
    async def __call__(self, scope, receive, send):
        async with anyio.create_task_group() as group:
            async def deliver():
                try:
                    await self.stream_response(send)
                except OSError:
                    pass
                finally:
                    # Cancellation may arrive while send() is applying socket
                    # backpressure, leaving the body generator suspended at a
                    # yield. Explicitly close it before this request exits.
                    with anyio.CancelScope(shield=True):
                        close = getattr(self.body_iterator, "aclose", None)
                        if close is not None:
                            await close()
                    group.cancel_scope.cancel()
            group.start_soon(deliver)
            await self.listen_for_disconnect(receive)
            group.cancel_scope.cancel()
        if self.background is not None:
            await self.background()


class RequestBodyLimit:
    """Bound incoming bytes even when Content-Length is missing or dishonest."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        limit = 10_000_000 if scope.get("path") == "/api/transcribe" else 32_000
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > limit:
                response = JSONResponse(status_code=413, content={"error": {"code": "request_too_large", "message": "Request exceeds the size limit."}})
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        replayed = False

        async def bounded_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


def encode_sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _public_error(exc: Exception) -> tuple[int, dict[str, str]]:
    if isinstance(exc, ProviderError):
        return exc.status, {"code": exc.code, "message": exc.public_message}
    if isinstance(exc, SessionNotFound):
        return 404, {"code": "session_expired", "message": str(exc)}
    if isinstance(exc, SessionCapacityError):
        return 429, {"code": "session_busy", "message": str(exc)}
    if isinstance(exc, TimeoutError):
        return 504, {"code": "request_timeout", "message": "The response timed out. Try again."}
    logger.error("Request failed: %s", type(exc).__name__)
    return 500, {"code": "internal_error", "message": "The response could not be completed."}


def create_app(settings: Settings | None = None, *, corpus: Corpus | None = None, llm: OpenAICompatibleLLM | None = None, motion_library: MotionLibrary | None = None) -> FastAPI:
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        loaded_motion = motion_library
        motion_status = "not_configured"
        if loaded_motion is None and config.motion_catalog is not None:
            try:
                loaded_motion = MotionLibrary.load(config.motion_catalog, config.motion_asset_root)
            except (ValueError, OSError) as exc:
                logger.warning("Optional motion library unavailable (%s)", type(exc).__name__)
                motion_status = "invalid_catalog"
        application.state.motion_library = loaded_motion
        application.state.motion_status = "available" if loaded_motion else motion_status
        application.state.engine = Engine(config, corpus or Corpus.load(config.corpus_dir), llm=llm, motion_library=loaded_motion)
        application.state.retrieval_warmup_ms = None
        if config.retrieval_backend == "hybrid" and config.retrieval_prewarm:
            # Cold local indexing can take longer than a conversation deadline.
            # Finish it before advertising readiness; failures stop startup.
            started = time.perf_counter()
            logger.info("Warming the configured hybrid retrieval index before API readiness")
            await asyncio.to_thread(application.state.engine.retriever.warm_up)
            application.state.retrieval_warmup_ms = round((time.perf_counter() - started) * 1000, 3)
        application.state.speech = SpeechProvider(config.tts, voice=config.tts_voice, timeout=config.request_timeout, language_hint=config.tts_language_hint)
        application.state.transcription = TranscriptionProvider(config.asr, timeout=config.request_timeout)
        application.state.lipsync = LipsyncEngine.from_env()
        application.state.speech_packages = SpeechPackages()
        try:
            yield
        finally:
            application.state.speech_packages.clear()

    application = FastAPI(title="Who Is Your Bias?", version=__version__, lifespan=lifespan)
    application.add_middleware(CORSMiddleware, allow_origins=list(config.cors_origins), allow_credentials=False, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"], expose_headers=["X-Speech-Timing", "X-Voice-Mode"])
    application.add_middleware(RequestBodyLimit)

    @application.middleware("http")
    async def response_headers(request: Request, call_next):
        length = request.headers.get("content-length")
        maximum = 10_000_000 if request.url.path == "/api/transcribe" else 32_000
        if request.method in {"POST", "PUT", "PATCH"} and length:
            try:
                if int(length) > maximum or int(length) < 0:
                    return JSONResponse(status_code=413, content={"error": {"code": "request_too_large", "message": "Request exceeds the size limit."}})
            except ValueError:
                return JSONResponse(status_code=400, content={"error": {"code": "bad_content_length", "message": "Invalid Content-Length header."}})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/api/health")
    async def health(request: Request):
        engine: Engine = request.app.state.engine
        return {
            "status": "ok", "version": __version__, "provider": "openai" if config.llm.configured else "evidence",
            "capabilities": {"llm": config.llm.configured, "speech": config.tts.configured, "transcription": config.asr.configured},
            "providers": {"llm": {"configured": config.llm.configured, "model": config.llm.model}, "speech": {"configured": config.tts.configured, "model": config.tts.model}, "transcription": {"configured": config.asr.configured, "model": config.asr.model}},
            "corpus": engine.corpus.stats(),
            "qa_corpus": engine.corpus.qa_report(),
            "persona_snapshot": engine.snapshot.summary() if engine.snapshot else None,
            "retrieval": {"backend": config.retrieval_backend, "embedding_model": config.embedding_model,
                          "embedding_revision": config.embedding_revision, "style_examples": config.style_examples,
                          "prewarmed": request.app.state.retrieval_warmup_ms is not None,
                          "startup_warmup_ms": request.app.state.retrieval_warmup_ms},
            "live_protocol": "live-segments-pcm.v1",
            "configuration_only": True,
        }

    @application.get("/api/persona")
    async def persona(request: Request):
        loaded: Corpus = request.app.state.engine.corpus
        item = loaded.persona
        snapshot = request.app.state.engine.snapshot
        return {"id": item.id, "name": item.name, "tagline": item.tagline, "description": item.description, "example_questions": item.example_questions, "stats": loaded.stats(), "qa_corpus": loaded.qa_report(), "snapshot": snapshot.summary() if snapshot else None, "capabilities": {"llm": config.llm.configured, "speech": config.tts.configured, "transcription": config.asr.configured, "live_speech": config.tts.configured}, "notice": item.notice}

    @application.get("/api/sources")
    async def sources(request: Request):
        loaded: Corpus = request.app.state.engine.corpus
        records = list(loaded.sources.values())
        return {"sources": [source.model_dump(mode="json") for source in records], "counts": {split: sum(source.split == split for source in records) for split in ("train", "validation", "test")}}

    @application.get("/api/motion/library")
    async def motion_summary(request: Request):
        library: MotionLibrary | None = request.app.state.motion_library
        if library is None:
            return {"available": False, "status": request.app.state.motion_status,
                "reviewed_runtime_primitives": 0, "preview_runtime_primitives": 0}
        return {"available": True, "status": "available", **library.summary()}

    @application.get("/api/motion/clip/{primitive_id}")
    async def motion_clip(primitive_id: str, request: Request, preview: bool = False):
        library: MotionLibrary | None = request.app.state.motion_library
        if library is None or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,119}", primitive_id):
            raise HTTPException(status_code=404, detail="The motion clip is unavailable under the requested review gate.")
        try:
            clip = library.clip(primitive_id, allow_geometry_preview=preview)
        except MotionError as exc:
            raise HTTPException(status_code=404, detail="The motion clip is unavailable under the requested review gate.") from exc
        # Match MotionLibrary.clip_digest byte-for-byte, including Unicode.
        payload = json.dumps(clip, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return Response(content=payload, media_type="application/json")

    @application.post("/api/chat", response_model=ChatTurn)
    async def chat(body: ChatRequest, request: Request):
        try:
            return await request.app.state.engine.chat(body)
        except Exception as exc:
            status, error = _public_error(exc)
            return JSONResponse(status_code=status, content={"error": error})

    @application.post("/api/chat/stream")
    async def stream(body: ChatRequest, request: Request):
        async def events():
            try:
                async with aclosing(request.app.state.engine.events(body)) as engine_events:
                    async for event, data in engine_events:
                        yield encode_sse(event, data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _, error = _public_error(exc)
                yield encode_sse("error", error)
        return DisconnectStreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

    @application.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        try:
            deleted = request.app.state.engine.sessions.delete(session_id)
            return {"deleted": deleted}
        except SessionCapacityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/api/sessions/{session_id}/context")
    async def session_context(session_id: str, request: Request):
        store = request.app.state.engine.sessions
        try:
            session = store.get_or_create(session_id)
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return session.dialogue.inspect()

    @application.post("/api/chat/live")
    async def live_chat(body: ChatRequest, request: Request):
        async def events():
            try:
                async with aclosing(live_events(request.app.state.engine, request.app.state.speech, body,
                                               timeout=min(1800, max(180, config.request_timeout * 4)))) as stream:
                    async for event, data in stream:
                        yield encode_sse(event, data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _, error = _public_error(exc)
                yield encode_sse("error", error)
        return DisconnectStreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})

    @application.post("/api/speech")
    async def speech(body: SpeechRequest, request: Request):
        try:
            data, mime = await run_until_disconnect(
                request, request.app.state.speech.synthesize(body.text, body.voice, language=body.language),
            )
            return Response(content=data, media_type=mime, headers={"X-Speech-Timing": "unavailable", "X-Voice-Mode": "configured-provider"})
        except ClientDisconnect:
            return Response(status_code=499)
        except ProviderError as exc:
            status, error = _public_error(exc)
            return JSONResponse(status_code=status, content={"error": error})

    @application.post("/api/speech/prepare")
    async def prepare_speech(body: SpeechPackageRequest, request: Request):
        try:
            return await run_until_disconnect(
                request, request.app.state.speech_packages.prepare(
                    request.app.state.speech, request.app.state.lipsync,
                    text=body.text, voice=body.voice, language=body.language,
                ),
            )
        except ClientDisconnect:
            return Response(status_code=499)
        except ProviderError as exc:
            status, error = _public_error(exc)
            return JSONResponse(status_code=status, content={"error": error})

    @application.get("/api/speech/clips/{clip_id}")
    async def speech_clip(clip_id: str, request: Request):
        clip = request.app.state.speech_packages.get(clip_id) if re.fullmatch(r"[a-f0-9]{32}", clip_id) else None
        if clip is None:
            raise HTTPException(status_code=404, detail="The speech clip has expired. Generate the voice again.")
        return Response(content=clip.data, media_type=clip.mime)

    @application.post("/api/transcribe")
    async def transcribe(request: Request):
        mime = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if mime not in {"audio/webm", "audio/wav", "audio/ogg", "audio/mp4", "audio/mpeg"}:
            raise HTTPException(status_code=415, detail="Send raw audio/webm, wav, ogg, mp4, or mpeg bytes.")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 10_000_000:
                raise HTTPException(status_code=413, detail="Recording exceeds 10 MB.")
        if not data:
            raise HTTPException(status_code=422, detail="Recording is empty.")
        try:
            text = await request.app.state.transcription.transcribe(bytes(data), mime)
            return {"text": text}
        except ProviderError as exc:
            status, error = _public_error(exc)
            return JSONResponse(status_code=status, content={"error": error})

    if config.web_dir.is_dir():
        application.mount("/", StaticFiles(directory=config.web_dir, html=True), name="studio")
    return application


app = create_app()
