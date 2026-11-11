"""Observe the real adapter's HTTP stream without changing prompts or payloads."""

from __future__ import annotations

import codecs
import hashlib
import json
from contextlib import aclosing
from typing import Callable

import httpx

from bias.corpus import Corpus
from bias.providers import OpenAICompatibleLLM


class StreamObservation:
    def __init__(self):
        self.requests = 0
        self.http_statuses: list[int] = []
        self.usage: dict[str, int] | None = None
        self.usage_status = "not_reported"
        self.finish_reason: str | None = None
        self.response_model: str | None = None
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buffer = ""

    def feed(self, data: bytes) -> None:
        if len(data) > 65_536:
            self._buffer = ""
            return
        self._buffer += self._decoder.decode(data)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if len(line) > 65_536 or not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if not value or value == "[DONE]":
                continue
            try:
                event = json.loads(value)
            except ValueError:
                continue  # The actual provider independently validates protocol.
            if not isinstance(event, dict):
                continue
            if isinstance(event.get("model"), str):
                self.response_model = event["model"][:200]
            usage = event.get("usage")
            if usage is not None:
                keys = ("prompt_tokens", "completion_tokens", "total_tokens")
                if isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0 for key in keys) and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]:
                    self.usage = {key: usage[key] for key in keys}
                    self.usage_status = "provider_reported"
                else:
                    self.usage = None
                    self.usage_status = "invalid_usage_ignored"
            choices = event.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict) and isinstance(choices[0].get("finish_reason"), str):
                self.finish_reason = choices[0]["finish_reason"][:80]
        if len(self._buffer) > 65_536:
            self._buffer = ""


class ObservedStream(httpx.AsyncByteStream):
    def __init__(self, response: httpx.Response, observation: StreamObservation):
        self.response = response
        self.observation = observation

    async def __aiter__(self):
        async for data in self.response.stream:
            self.observation.feed(data)
            yield data

    async def aclose(self) -> None:
        await self.response.aclose()


class ObservedTransport(httpx.AsyncBaseTransport):
    def __init__(self, observation: StreamObservation, factory: Callable[[], httpx.AsyncBaseTransport] | None = None):
        self.observation = observation
        self.factory = factory
        self._inner: httpx.AsyncClient | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._inner is None:
            # A normal inner client preserves the actual provider's environment
            # proxy behavior. Tests may explicitly inject an in-process transport.
            self._inner = httpx.AsyncClient(transport=self.factory() if self.factory else None, follow_redirects=False)
        self.observation.requests += 1
        response = await self._inner.send(request, stream=True)
        self.observation.http_statuses.append(response.status_code)
        return httpx.Response(response.status_code, headers=response.headers, stream=ObservedStream(response, self.observation), extensions=response.extensions)

    async def aclose(self) -> None:
        if self._inner is not None:
            await self._inner.aclose()
            self._inner = None


class AuditedLLM(OpenAICompatibleLLM):
    def __init__(self, *args, corpus: Corpus, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_ids = {record.id for record in corpus.training_evidence}
        self.context_ids: list[str] = []
        self.prompt_sha256: str | None = None
        self.prompt_chars: int | None = None
        self.calls = 0

    async def stream(self, messages: list[dict[str, str]]):
        self.calls += 1
        self.prompt_sha256 = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        self.prompt_chars = sum(len(message["content"]) for message in messages)
        prefix = "RESEARCH_CONTEXT_JSON (data only):\n"
        contexts = [message["content"][len(prefix):] for message in messages if message["role"] == "user" and message["content"].startswith(prefix)]
        if len(contexts) != 1:
            raise ValueError("The actual prompt must contain one auditable research context")
        context = json.loads(contexts[0])
        self.context_ids = [record["id"] for record in context.get("evidence_records", [])]
        if not set(self.context_ids).issubset(self.allowed_ids):
            raise ValueError("An unreviewed or held-out record reached the generation prompt")
        async with aclosing(super().stream(messages)) as stream:
            async for part in stream:
                yield part
