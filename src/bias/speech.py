"""Bounded ephemeral audio packages bound to their mouth-cue timestamps."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from .gesture import FPS as GESTURE_FPS, POSE_WIDTH, GestureEngine
from .lipsync import LipsyncEngine, LipsyncResult, wav_duration_ms
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

    async def prepare(self, provider: SpeechProvider, lipsync: LipsyncEngine, *, text: str, voice: str | None = None, language: str = "en",
                      gesture: GestureEngine | None = None) -> dict:
        if self.active >= 2:
            raise ProviderError("speech_busy", "Speech preparation is busy. Try again shortly.", 429)
        self.active += 1
        try:
            started = time.perf_counter()
            data, mime = await provider.synthesize(text, voice, language=language)
            synthesized = time.perf_counter()
            if len(data) > 20_000_000:
                raise ProviderError("audio_output_limit", "Generated audio exceeds the playback limit.", 502)
            gestured = None
            if gesture is None:
                analysis = await lipsync.analyze(data, text, language)
            else:
                # Both read the same finished audio; running them together keeps the added latency to the slower one.
                async def timed_gesture():
                    nonlocal gestured
                    try:
                        return await gesture.generate(data)
                    finally:
                        gestured = time.perf_counter()
                analysis, motion = await asyncio.gather(lipsync.analyze(data, text, language), timed_gesture())
            aligned = time.perf_counter()
            identifier = self.put(data, mime)
            result = {"id": identifier, "audio_url": f"/api/speech/clips/{identifier}", "audio_sha256": hashlib.sha256(data).hexdigest(),
                    "mime_type": mime, "audio_duration_ms": wav_duration_ms(data), "expires_in_seconds": self.ttl,
                    "speech_timing": analysis.timing, "timing_status": analysis.status, "timing_message": analysis.message,
                    "server_timing": {"synthesis_ms": round((synthesized - started) * 1000, 3),
                                      "alignment_ms": round((aligned - synthesized) * 1000, 3),
                                      "total_ms": round((time.perf_counter() - started) * 1000, 3)},
                    "voice_mode": "configured-provider"}
            if gesture is not None:
                result["server_timing"]["gesture_ms"] = round(((gestured or aligned) - synthesized) * 1000, 3)
                description = {"status": motion.status, "message": motion.message, "motion_url": None}
                if motion.status == "available" and motion.motion:
                    motion_id = self.put(motion.motion, "application/octet-stream")
                    description.update(motion_url=f"/api/speech/clips/{motion_id}", format="smplx-axis-angle-f32le",
                                       fps=GESTURE_FPS, frames=motion.frames, width=POSE_WIDTH, audio_sha256=result["audio_sha256"],
                                       engine="emage-audio (pretrained on BEAT2)", compute_ms=motion.compute_ms)
                result["gesture"] = description
            return result
        finally:
            self.active -= 1
