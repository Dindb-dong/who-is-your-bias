"""A deliberately narrow OpenAI-compatible speech endpoint.

Run from the checkout: uvicorn services.voice.app:app --host 127.0.0.1 --port 8010
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from typing import Any, Callable, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .engine import VoiceBusy, VoiceEngine, VoiceSettings, VoiceUnavailable, load_qwen_model

logger = logging.getLogger(__name__)


class BodyLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        if b"content-length" in headers:
            try:
                length = int(headers[b"content-length"])
            except ValueError:
                await JSONResponse({"detail": "Invalid content length"}, status_code=400)(scope, receive, send)
                return
            if length < 0 or length > self.max_bytes:
                await JSONResponse({"detail": "Speech request body is too large"}, status_code=413)(scope, receive, send)
                return
        if scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        # Read at most the bounded body before JSON parsing. Raising inside
        # FastAPI's body parser is otherwise converted into a misleading 400.
        size = 0
        chunks = []
        try:
            async with asyncio.timeout(15):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > self.max_bytes:
                        await JSONResponse({"detail": "Speech request body is too large"}, status_code=413)(scope, receive, send)
                        return
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await JSONResponse({"detail": "Speech request body timed out"}, status_code=408)(scope, receive, send)
            return
        consumed = False

        async def replay_receive() -> dict:
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["qwen3-tts"] = "qwen3-tts"
    input: str = Field(min_length=1, max_length=6000)
    voice: Literal["reference"] = "reference"
    response_format: Literal["wav"] = "wav"
    speed: float = Field(default=1.0, ge=1.0, le=1.0)
    language: Literal["Auto", "English", "Korean", "Chinese", "Japanese", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"] = "Auto"

    @field_validator("input")
    @classmethod
    def clean_input(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Speech input must contain text and cannot contain NUL")
        return value


def create_app(settings: VoiceSettings | None = None, *, model_loader: Callable = load_qwen_model) -> FastAPI:
    settings = settings or VoiceSettings.from_env()
    engine = VoiceEngine(settings, model_loader)
    app = FastAPI(title="who-is-your-bias optional reference voice", version="0.1.0")
    app.state.engine = engine
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)

    def authenticate(authorization: str | None) -> None:
        if settings.api_key and not hmac.compare_digest(authorization or "", f"Bearer {settings.api_key}"):
            raise HTTPException(status_code=401, detail="Valid voice service credentials are required")

    @app.get("/health")
    def health() -> dict:
        return engine.status()

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)) -> dict:
        authenticate(authorization)
        return {"object": "list", "data": [{"id": "qwen3-tts", "object": "model", "owned_by": "local"}]}

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest, authorization: str | None = Header(default=None)) -> Response:
        authenticate(authorization)
        if len(request.input) > settings.max_input_chars:
            raise HTTPException(status_code=422, detail=f"Speech input exceeds {settings.max_input_chars} characters")
        try:
            audio = await asyncio.to_thread(engine.synthesize, request.input, request.language)
        except VoiceBusy as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "2"}) from exc
        except VoiceUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Voice generation failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=502, detail="Voice generation failed; inspect the local service and model configuration") from exc
        return Response(audio, media_type="audio/wav", headers={"Cache-Control": "no-store", "X-AI-Simulation": "true"})

    return app


app = create_app()
