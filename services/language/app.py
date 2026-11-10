"""Run with uvicorn services.language.app:app --host 127.0.0.1 --port 8020."""

from __future__ import annotations

import asyncio
import hmac
import json
import queue
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .engine import GenerationJob, LanguageEngine, LanguageError, LanguageSettings, load_backend
from .schemas import CompletionRequest


def error_payload(error: LanguageError) -> dict:
    return {"error": {"message": error.public_message, "type": "local_language_error", "code": error.code}}


class BodyLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        async def reject(code: str, message: str, status: int) -> None:
            await JSONResponse(error_payload(LanguageError(code, message, status)), status_code=status)(scope, receive, send)

        headers = dict(scope.get("headers", []))
        if b"content-length" in headers:
            try:
                length = int(headers[b"content-length"])
            except ValueError:
                await reject("invalid_length", "Invalid content length.", 400)
                return
            if length < 0 or length > self.max_bytes:
                await reject("body_limit", "The language request body exceeds its byte limit.", 413)
                return
        body = bytearray()
        try:
            async with asyncio.timeout(15):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > self.max_bytes:
                        await reject("body_limit", "The language request body exceeds its byte limit.", 413)
                        return
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await reject("body_timeout", "The language request body timed out.", 408)
            return
        consumed = False

        async def replay() -> dict:
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


async def output_events(job: GenerationJob, disconnected: Callable[[], Awaitable[bool]] | None = None) -> AsyncIterator[str | None]:
    """Yield bounded worker output; None is a transport keepalive, not a token."""
    heartbeat = time.monotonic() + 5
    try:
        while True:
            if disconnected is not None and await disconnected():
                job.cancel()
                raise asyncio.CancelledError
            job.check()
            try:
                yield job.queue.get_nowait()
                continue
            except queue.Empty:
                pass
            if job.finished.is_set():
                if job.error:
                    raise job.error
                if job.result is None:
                    raise LanguageError("missing_result", "The language worker ended without a result.")
                return
            if time.monotonic() >= heartbeat:
                heartbeat = time.monotonic() + 5
                yield None
            await asyncio.sleep(0.02)
    finally:
        if not job.finished.is_set():
            job.cancel()


def create_app(settings: LanguageSettings | None = None, *, loader: Callable = load_backend) -> FastAPI:
    settings = settings or LanguageSettings.from_env()
    engine = LanguageEngine(settings, loader)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await asyncio.to_thread(engine.shutdown)

    app = FastAPI(title="who-is-your-bias optional local language runtime", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)

    @app.exception_handler(LanguageError)
    async def language_error(request: Request, exc: LanguageError) -> JSONResponse:
        return JSONResponse(error_payload(exc), status_code=exc.status, headers={"Retry-After": "2"} if exc.status == 429 else None)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic errors can contain the rejected input. Do not reflect prompts
        # or credentials into error payloads or logs.
        return JSONResponse(error_payload(LanguageError("invalid_request", "The request does not match the supported text-chat schema.", 422)), status_code=422)

    def authenticate(authorization: str | None) -> None:
        if settings.api_key and not hmac.compare_digest((authorization or "").encode("utf-8"), f"Bearer {settings.api_key}".encode("utf-8")):
            raise LanguageError("authentication_required", "Valid language service credentials are required.", 401)

    @app.get("/health")
    def health() -> dict:
        return engine.status()

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)) -> dict:
        authenticate(authorization)
        return {"object": "list", "data": [{"id": settings.served_model, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions", response_model=None)
    async def completion(payload: CompletionRequest, request: Request, authorization: str | None = Header(default=None)) -> JSONResponse | StreamingResponse:
        authenticate(authorization)
        job = engine.start(payload)
        common = {"id": job.id, "created": job.created, "model": settings.served_model}
        headers = {"Cache-Control": "no-store", "X-AI-Simulation": "true"}
        if not payload.stream:
            async for _ in output_events(job, request.is_disconnected):
                pass
            assert job.result is not None
            return JSONResponse({**common, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(job.fragments)}, "finish_reason": job.result.finish_reason}], "usage": job.result.usage()}, headers=headers)

        def chunk(delta: dict, finish: str | None = None) -> str:
            return "data: " + json.dumps({**common, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}, ensure_ascii=False) + "\n\n"

        async def stream() -> AsyncIterator[str]:
            try:
                yield chunk({"role": "assistant"})
                async for fragment in output_events(job):
                    yield ": keepalive\n\n" if fragment is None else chunk({"content": fragment})
                assert job.result is not None
                # Usage precedes the finish chunk so clients that reject length
                # truncation can still account for the work already performed.
                if payload.stream_options.include_usage:
                    yield "data: " + json.dumps({**common, "object": "chat.completion.chunk", "choices": [], "usage": job.result.usage()}) + "\n\n"
                yield chunk({}, job.result.finish_reason)
                yield "data: [DONE]\n\n"
            except LanguageError as exc:
                yield "data: " + json.dumps(error_payload(exc)) + "\n\n"
            finally:
                if not job.finished.is_set():
                    job.cancel()

        return StreamingResponse(stream(), media_type="text/event-stream", headers={**headers, "X-Accel-Buffering": "no"})

    return app


app = create_app()
