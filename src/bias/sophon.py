"""Realtime voice conversation through Sophon-Realtime, bridged for the studio.

The browser opens ``/api/realtime`` (this server) and streams microphone PCM16 at 16 kHz. This module holds the
upstream WebSocket to a Sophon-Realtime server (the hosted API or a self-hosted engine; both speak an
OpenAI-Realtime-style protocol), so the API key never reaches the browser. Sophon listens, decides when the user has
finished and speaks the reply. Reply audio is forwarded as PCM chunks on a per-response clock, and the same audio is
streamed to EMAGE for gestures when that is configured.

Browser -> server: binary frames = PCM16LE mono 16 kHz microphone audio (about 100 ms each); text frame
``{"type": "start", "gesture": bool, "language": "en"}`` first, then optionally ``{"type": "stop"}``.
Server -> browser (JSON text frames): ``ready``, ``user_speech``, ``response`` (started / done / cancelled),
``transcript``, ``audio`` (PcmChunk fields), ``gesture_state`` / ``gesture`` / ``gesture_end``, ``error``.

Environment: BIAS_SOPHON_URL (wss://.../v1/realtime), BIAS_SOPHON_MODEL, BIAS_SOPHON_API_KEY, BIAS_SOPHON_VOICE.
Replies come from Sophon's own model with the persona's source notes as instructions; they are not checked
against citations like the text conversation, and the UI labels them that way.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass

import websockets

from .corpus import Corpus
from .gesture import CANCEL, GestureEngine, stream_gestures

INPUT_RATE = 16000
OUTPUT_RATE = 24000
MAX_CHUNK_FRAMES = 16384           # browser PcmSequence bound
MAX_MIC_FRAME = INPUT_RATE * 2     # bytes in one second of mic PCM16; larger frames are rejected


@dataclass(frozen=True)
class SophonSettings:
    url: str = ""
    model: str = "sophon-realtime-tag"
    api_key: str = ""
    voice: str = "ryan_en"

    @classmethod
    def from_env(cls) -> "SophonSettings":
        return cls(url=os.getenv("BIAS_SOPHON_URL", "").strip(), model=os.getenv("BIAS_SOPHON_MODEL", "").strip() or cls.model,
                   api_key=os.getenv("BIAS_SOPHON_API_KEY", "").strip(), voice=os.getenv("BIAS_SOPHON_VOICE", "").strip() or cls.voice)

    @property
    def configured(self) -> bool:
        return self.url.startswith(("wss://", "ws://")) and bool(self.model)

    def endpoint(self) -> str:
        separator = "&" if "?" in self.url else "?"
        return f"{self.url}{separator}model={self.model}"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def build_instructions(corpus: Corpus, language: str = "en") -> str:
    """Persona rules plus every reviewed source note (the Jensen corpus is small enough to send whole)."""
    persona = corpus.persona
    notes = []
    for record in corpus.training_evidence:
        source = corpus.sources[record.source_id]
        notes.append(f"- {record.text} ({source.publisher}, {source.title}, {source.published_at.isoformat()})")
    rules = "\n".join(f"- {rule}" for rule in persona.style_rules)
    reply_language = "Korean" if language == "ko" else "English"
    return (f"You are an AI research simulation of {persona.name}, speaking in the first person as him. {persona.description} "
            f"{persona.notice}\n"
            f"Speak as this simulation in a live voice conversation, in {reply_language}. Keep each reply short: two or "
            "three spoken sentences, under about 20 seconds, unless asked for more. No lists or markdown.\n"
            f"Rules:\n{rules}\n"
            "Use only the source notes below for facts about Jensen Huang or NVIDIA. If they do not cover the question, "
            "say you do not have a source for that.\n"
            "Source notes:\n" + "\n".join(notes))


async def bridge(browser, settings: SophonSettings, instructions: str, *, gesture: GestureEngine | None = None,
                 connect=websockets.connect) -> None:
    """Run one browser conversation until either side closes. ``browser`` is a Starlette WebSocket (accepted)."""
    start = json.loads(await browser.receive_text())
    if start.get("type") != "start":
        await browser.send_json({"type": "error", "code": "protocol", "message": "Send a start message first."})
        return
    language = "ko" if start.get("language") == "ko" else "en"
    use_gesture = bool(start.get("gesture")) and gesture is not None and gesture.configured
    async with connect(settings.endpoint(), additional_headers=settings.headers(), max_size=None, open_timeout=15) as upstream:
        await upstream.send(json.dumps({"type": "session.update", "session": {
            "voice": settings.voice, "instructions": instructions, "input_audio_format": "pcm16", "output_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad"}, "extra_body": {"language": language}}}))
        await browser.send_json({"type": "ready", "model": settings.model, "voice": settings.voice, "sample_rate": OUTPUT_RATE,
                                 "input_rate": INPUT_RATE, "gesture": use_gesture})
        send_lock = asyncio.Lock()

        async def to_browser(message: dict) -> None:
            async with send_lock:
                await browser.send_json(message)

        response = {"id": None, "sequence": 0, "samples": 0, "audio": None, "task": None}

        def finish_response(how) -> None:
            queue = response["audio"]
            if queue is not None:
                queue.put_nowait(how)       # None = complete, CANCEL = barge-in
            response.update(id=None, audio=None, task=None)

        async def mic() -> None:
            while True:
                message = await browser.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                data = message.get("bytes")
                if data is not None:
                    if not data or len(data) > MAX_MIC_FRAME or len(data) % 2:
                        continue
                    await upstream.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(data).decode("ascii"),
                                                    "sample_rate_hz": INPUT_RATE}))
                elif message.get("text") and json.loads(message["text"]).get("type") == "stop":
                    return

        async def replies() -> None:
            async for raw in upstream:
                event = json.loads(raw)
                kind = event.get("type", "")
                if kind == "response.created":
                    finish_response(CANCEL)
                    rid = str((event.get("response") or {}).get("id") or f"resp_{os.urandom(6).hex()}")[:64]
                    response.update(id=rid, sequence=0, samples=0)
                    if use_gesture:
                        queue: asyncio.Queue = asyncio.Queue()
                        response["audio"] = queue
                        task = asyncio.create_task(stream_gestures(gesture, queue, lambda k, d: to_browser({"type": k, **d}),
                                                                   lambda rid=rid: rid))
                        # Never cancelled (that would restart the EMAGE worker); CANCEL ends it after the current window.
                        task.add_done_callback(lambda t: t.cancelled() or t.exception())
                        response["task"] = task
                    await to_browser({"type": "response", "state": "started", "response_id": rid})
                elif kind == "response.output_audio.delta" and response["id"]:
                    pcm = base64.b64decode(event.get("delta", ""))
                    pcm = pcm[:len(pcm) - len(pcm) % 2]
                    if response["audio"] is not None and pcm:
                        response["audio"].put_nowait((pcm, OUTPUT_RATE, 1))
                    for offset in range(0, len(pcm), MAX_CHUNK_FRAMES * 2):
                        part = pcm[offset:offset + MAX_CHUNK_FRAMES * 2]
                        count = len(part) // 2
                        await to_browser({"type": "audio", "turn_id": response["id"], "segment_id": response["id"], "sequence": response["sequence"],
                                          "segment_sequence": 0, "sample_rate": OUTPUT_RATE, "channels": 1, "sample_offset": response["samples"],
                                          "sample_count": count, "encoding": "pcm_s16le", "pcm_base64": base64.b64encode(part).decode("ascii")})
                        response["sequence"] += 1
                        response["samples"] += count
                elif kind == "response.output_audio_transcript.delta" and response["id"]:
                    await to_browser({"type": "transcript", "response_id": response["id"], "delta": str(event.get("delta", ""))})
                elif kind == "audio.cancelled" and response["id"]:
                    rid = response["id"]
                    finish_response(CANCEL)
                    await to_browser({"type": "response", "state": "cancelled", "response_id": rid})
                elif kind == "response.done" and response["id"]:
                    rid, samples = response["id"], response["samples"]
                    task = response["task"]
                    finish_response(None)
                    if task is not None:
                        await asyncio.wait([task])   # gesture_end precedes the done message, like audio_end in the live path
                    await to_browser({"type": "response", "state": "done", "response_id": rid, "sample_count": samples})
                elif kind in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"):
                    await to_browser({"type": "user_speech", "state": kind.rsplit("_", 1)[1]})
                elif kind == "error":
                    error = event.get("error") or {}
                    await to_browser({"type": "error", "code": str(error.get("code", "sophon_error"))[:60],
                                      "message": str(error.get("message", "The realtime voice service reported an error."))[:300]})
                    if error.get("code") in {"idle_timeout", "session_time_limit", "insufficient_quota", "invalid_api_key"}:
                        return

        tasks = [asyncio.create_task(mic()), asyncio.create_task(replies())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            finish_response(CANCEL)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
