"""Bounded ephemeral audio packages bound to their mouth-cue timestamps."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from .lipsync import LipsyncEngine, wav_duration_ms
from .providers import ProviderError, SpeechProvider


@dataclass(frozen=True)
class SpeechClip:
    data: bytes
    mime: str
    expires_at: float


class SpeechPackages:
    def __init__(self, *, ttl: float = 120, max_items: int = 8, max_bytes: int = 40_000_000, clock=time.monotonic):
        if ttl <= 0 or max_items < 1 or max_bytes < 1:
            raise ValueError("Invalid speech-cache bounds")
        self.ttl, self.max_items, self.max_bytes, self.clock = ttl, max_items, max_bytes, clock
        self.clips: OrderedDict[str, SpeechClip] = OrderedDict()
        self.total_bytes = 0
        self.active = 0

    def clear(self) -> None:
        self.clips.clear()
        self.total_bytes = 0

    def _remove(self, identifier: str) -> None:
        clip = self.clips.pop(identifier, None)
        if clip:
            self.total_bytes -= len(clip.data)

    def _purge(self) -> None:
        now = self.clock()
        for identifier, clip in list(self.clips.items()):
            if clip.expires_at <= now:
                self._remove(identifier)

    def get(self, identifier: str) -> SpeechClip | None:
        self._purge()
        return self.clips.get(identifier)

    def put(self, data: bytes, mime: str) -> str:
        if not data or len(data) > self.max_bytes:
            raise ProviderError("speech_cache_limit", "Generated audio exceeds the playback cache limit.", 502)
        self._purge()
        while self.clips and (len(self.clips) >= self.max_items or self.total_bytes + len(data) > self.max_bytes):
            self._remove(next(iter(self.clips)))
        identifier = uuid.uuid4().hex
        self.clips[identifier] = SpeechClip(data, mime, self.clock() + self.ttl)
        self.total_bytes += len(data)
        return identifier

    async def prepare(self, provider: SpeechProvider, lipsync: LipsyncEngine, *, text: str, voice: str | None = None, language: str = "en") -> dict:
        if self.active >= 2:
            raise ProviderError("speech_busy", "Speech preparation is busy. Try again shortly.", 429)
        self.active += 1
        try:
            data, mime = await provider.synthesize(text, voice)
            if len(data) > 20_000_000:
                raise ProviderError("audio_output_limit", "Generated audio exceeds the playback limit.", 502)
            analysis = await lipsync.analyze(data, text, language)
            identifier = self.put(data, mime)
            return {"id": identifier, "audio_url": f"/api/speech/clips/{identifier}", "audio_sha256": hashlib.sha256(data).hexdigest(),
                    "mime_type": mime, "audio_duration_ms": wav_duration_ms(data), "expires_in_seconds": self.ttl,
                    "speech_timing": analysis.timing, "timing_status": analysis.status, "timing_message": analysis.message,
                    "voice_mode": "configured-provider"}
        finally:
            self.active -= 1
