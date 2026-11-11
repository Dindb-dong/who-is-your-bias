"""Small OpenAI-compatible adapters with bounded requests and honest failures."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from .config import ProviderSettings


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str, status: int = 502):
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.status = status


def _status_error(status: int) -> ProviderError:
    if status in {401, 403}:
        return ProviderError("provider_auth", "The configured provider rejected its credentials.")
    if status == 429:
        return ProviderError("provider_busy", "The provider is rate limited. Try again later.", 503)
    return ProviderError("provider_failure", f"The provider returned HTTP {status}.")


class OpenAICompatibleLLM:
    def __init__(self, settings: ProviderSettings, *, timeout: float = 60, max_chars: int = 6000, max_tokens: int = 900, transport: httpx.AsyncBaseTransport | None = None):
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be an integer between 1 and 4096")
        self.settings = settings
        self.timeout = timeout
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.transport = transport

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if not self.settings.configured:
            raise ProviderError("provider_unconfigured", "No language model is configured.", 503)
        payload = {"model": self.settings.model, "messages": messages, "stream": True, "max_tokens": self.max_tokens}
        # Retry a connection failure once, before receiving any output. Never
        # replay after partial output or automatically repeat a billable request.
        for attempt in range(2):
            emitted = False
            try:
                async with asyncio.timeout(self.timeout):
                    async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=min(10, self.timeout)), transport=self.transport, follow_redirects=False) as client:
                        async with client.stream("POST", f"{self.settings.base_url}/chat/completions", json=payload, headers=self.settings.headers) as response:
                            if response.status_code != 200:
                                raise _status_error(response.status_code)
                            total = 0
                            completed = False
                            async for line in response.aiter_lines():
                                if not line.startswith("data:"):
                                    continue
                                data = line[5:].strip()
                                if not data:
                                    continue
                                if data == "[DONE]":
                                    completed = True
                                    break
                                try:
                                    chunk = json.loads(data)
                                    if "error" in chunk:
                                        raise ProviderError("provider_stream_error", "The provider interrupted its response.")
                                    choices = chunk.get("choices", [])
                                    if not choices:
                                        continue  # Usage-only stream events are valid.
                                    choice = choices[0]
                                    content = choice.get("delta", {}).get("content")
                                    finish = choice.get("finish_reason")
                                    if finish == "length":
                                        raise ProviderError("provider_truncated", "The language model reached its response limit before finishing.")
                                    if finish in {"content_filter", "tool_calls"}:
                                        raise ProviderError("provider_unsupported_finish", "The language model did not complete a text response.")
                                    if finish == "stop":
                                        completed = True
                                    if content is None:
                                        continue
                                    if not isinstance(content, str):
                                        raise ValueError("Non-text delta")
                                except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
                                    raise ProviderError("provider_protocol", "The provider returned an invalid text stream.") from exc
                                total += len(content)
                                if total > self.max_chars:
                                    raise ProviderError("provider_output_limit", "The provider response exceeded the configured size limit.")
                                if content:
                                    emitted = True
                                    yield content
                            if not emitted:
                                raise ProviderError("provider_empty", "The provider returned no text.")
                            if not completed:
                                raise ProviderError("provider_incomplete", "The provider stream ended before completion.")
                return
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt == 0 and not emitted:
                    continue
                raise ProviderError("provider_unreachable", "The language model endpoint could not be reached.", 503) from exc
            except (httpx.TimeoutException, TimeoutError) as exc:
                raise ProviderError("provider_timeout", "The language model response timed out.", 504) from exc
            except httpx.HTTPError as exc:
                raise ProviderError("provider_network", "The language model connection failed.") from exc


class SpeechProvider:
    def __init__(self, settings: ProviderSettings, *, voice: str = "default", timeout: float = 60, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.voice = voice
        self.timeout = timeout
        self.transport = transport

    async def synthesize(self, text: str, voice: str | None = None) -> tuple[bytes, str]:
        if not self.settings.configured:
            raise ProviderError("speech_unconfigured", "Configure a speech provider to enable server-generated audio.", 503)
        try:
            async with asyncio.timeout(self.timeout):
                async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport, follow_redirects=False) as client:
                    async with client.stream("POST", f"{self.settings.base_url}/audio/speech", headers=self.settings.headers, json={"model": self.settings.model, "input": text, "voice": voice or self.voice, "response_format": "wav"}) as response:
                        if response.status_code != 200:
                            raise _status_error(response.status_code)
                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            data.extend(chunk)
                            if len(data) > 20_000_000:
                                raise ProviderError("audio_output_limit", "The synthesized audio exceeds the size limit.")
                        mime = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if not data:
                raise ProviderError("audio_empty", "The speech provider returned empty audio.")
            if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
                mime = "audio/wav"
            elif data[:4] == b"OggS":
                mime = "audio/ogg"
            elif data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
                mime = "audio/mpeg"
            elif not mime.startswith("audio/"):
                raise ProviderError("audio_protocol", "The speech provider did not return an audio response.")
            return bytes(data), mime
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise ProviderError("speech_timeout", "Speech generation timed out.", 504) from exc
        except httpx.HTTPError as exc:
            raise ProviderError("speech_network", "The speech endpoint could not be reached.", 503) from exc


class TranscriptionProvider:
    def __init__(self, settings: ProviderSettings, *, timeout: float = 60, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.timeout = timeout
        self.transport = transport

    async def transcribe(self, data: bytes, mime: str) -> str:
        if not self.settings.configured:
            raise ProviderError("transcription_unconfigured", "Configure a transcription provider to enable recorded audio input.", 503)
        extension = {"audio/webm": "webm", "audio/wav": "wav", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/mpeg": "mp3"}.get(mime, "audio")
        try:
            async with asyncio.timeout(self.timeout):
                async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport, follow_redirects=False) as client:
                    response = await client.post(f"{self.settings.base_url}/audio/transcriptions", headers=self.settings.headers, data={"model": self.settings.model, "response_format": "json"}, files={"file": (f"recording.{extension}", data, mime)})
            if response.status_code != 200:
                raise _status_error(response.status_code)
            result = response.json()
            text = result.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                raise ProviderError("transcription_protocol", "The provider returned an empty or oversized transcription.")
            return text.strip()
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise ProviderError("transcription_timeout", "Transcription timed out.", 504) from exc
        except httpx.HTTPError as exc:
            raise ProviderError("transcription_network", "The transcription endpoint could not be reached.", 503) from exc
        except (ValueError, AttributeError) as exc:
            raise ProviderError("transcription_protocol", "The provider returned an invalid transcription.") from exc
