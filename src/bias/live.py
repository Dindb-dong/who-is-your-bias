"""Request-owned sentence synthesis and bounded PCM delivery while text continues.

The baseline voice server returns one complete WAV per committed sentence. This
is application-level early playback, not a claim of token-input neural TTS.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import time
import wave
import anyio
from contextlib import aclosing

from .providers import ProviderError


def decode_pcm(data: bytes) -> tuple[bytes, int, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels, rate, frames = wav.getnchannels(), wav.getframerate(), wav.getnframes()
            if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2 or channels not in {1, 2} or not 8000 <= rate <= 48000:
                raise ValueError("Unsupported PCM format")
            if frames > rate * 180:
                raise ValueError("Audio segment exceeds duration limit")
            pcm = wav.readframes(frames)
            if not pcm or len(pcm) != frames * channels * 2:
                raise ValueError("Incomplete PCM frames")
            return pcm, rate, channels
    except (wave.Error, EOFError, ValueError) as exc:
        raise ProviderError("live_audio_format", "Streaming speech requires a complete 16-bit mono/stereo PCM WAV.") from exc


async def live_events(engine, speech, request, *, timeout: float = 180, queue_size: int = 4, chunk_frames: int = 2048):
    if not 1 <= queue_size <= 16 or not 256 <= chunk_frames <= 16384 or not 1 <= timeout <= 1800:
        raise ValueError("Invalid live-stream bounds")
    if not speech.settings.configured:
        raise ProviderError("speech_unconfigured", "Configure a speech provider to enable streaming voice.", 503)
    started = time.perf_counter()
    deadline = asyncio.get_running_loop().time() + timeout
    session = engine.sessions.get_or_create(request.session_id)
    session.in_flight += 1
    acquired = False
    try:
        # Use the same deadline after acquisition, including any time spent
        # waiting for an earlier turn's speech to finish.
        async with asyncio.timeout_at(deadline):
            await session.turn_lock.acquire()
            acquired = True
        async with aclosing(_live_events(engine, speech, request, session=session, deadline=deadline, started=started,
                                         queue_size=queue_size, chunk_frames=chunk_frames)) as events:
            async for event in events:
                yield event
    finally:
        if acquired:
            session.turn_lock.release()
        session.in_flight -= 1


async def _live_events(engine, speech, request, *, session, deadline, started, queue_size, chunk_frames):
    output: asyncio.Queue = asyncio.Queue(maxsize=queue_size * 2)
    segments: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    turn_id = None
    text_complete = False
    sent_chunks = 0
    emitted_chunks = 0
    total_frames = 0
    audio_format = None
    first_pcm_ms = None
    synthesis_ms = []

    async def generate():
        nonlocal turn_id, text_complete
        try:
            async with aclosing(engine.events(request, committed_segments=True, _session_owner=session)) as events:
                async for kind, data in events:
                    if kind == "meta":
                        turn_id = data["turn_id"]
                        data = {**data, "protocol": "live-segments-pcm.v1", "speech_backend": "committed_sentence_wav",
                                "audio_clock": "pcm_sample_offset", "acoustic_latency_ms": None}
                    if kind == "done":
                        text_complete = True
                        continue
                    await output.put((kind, data))
                    if kind == "segment":
                        await segments.put(data)
            await segments.put(None)
            await output.put(("_generation_done", {}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await output.put(("_failure", exc))

    async def synthesize():
        nonlocal sent_chunks, total_frames, audio_format, first_pcm_ms
        try:
            while (segment := await segments.get()) is not None:
                text = segment["text"]
                if len(text) > 4000:
                    raise ProviderError("live_segment_limit", "A committed speech segment exceeds the voice input limit.", 422)
                synthesis_started = time.perf_counter()
                data, _ = await speech.synthesize(text, language=request.language)
                synthesis_ms.append(round((time.perf_counter() - synthesis_started) * 1000, 3))
                pcm, rate, channels = decode_pcm(data)
                if audio_format is not None and audio_format != (rate, channels):
                    raise ProviderError("live_audio_format_changed", "The speech service changed format within a turn.")
                audio_format = rate, channels
                digest = hashlib.sha256(data).hexdigest()
                segment_offset = total_frames
                await output.put(("audio_segment", {"turn_id": turn_id, "segment_id": segment["segment_id"],
                    "sequence": segment["sequence"], "audio_sha256": digest, "sample_rate": rate, "channels": channels,
                    "sample_offset": segment_offset, "sample_count": len(pcm) // (channels * 2),
                    "timing_basis": "pcm_clock_energy_articulation", "speech_timing": None,
                    "synthesis_ms": synthesis_ms[-1], "state": "ready"}))
                step = chunk_frames * channels * 2
                for offset in range(0, len(pcm), step):
                    part = pcm[offset:offset + step]
                    frames = len(part) // (channels * 2)
                    if first_pcm_ms is None:
                        first_pcm_ms = round((time.perf_counter() - started) * 1000, 3)
                    await output.put(("audio", {"turn_id": turn_id, "segment_id": segment["segment_id"],
                        "sequence": sent_chunks, "segment_sequence": segment["sequence"], "sample_rate": rate,
                        "channels": channels, "sample_offset": total_frames, "sample_count": frames,
                        "encoding": "pcm_s16le", "pcm_base64": base64.b64encode(part).decode("ascii")}))
                    sent_chunks += 1
                    total_frames += frames
                await output.put(("audio_segment_end", {"turn_id": turn_id, "segment_id": segment["segment_id"],
                    "sequence": segment["sequence"], "sample_end": total_frames, "state": "delivered"}))
            await output.put(("_audio_done", {}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await output.put(("_failure", exc))

    workers = [asyncio.create_task(generate()), asyncio.create_task(synthesize())]
    complete = set()
    try:
        async with asyncio.timeout_at(deadline):
            while len(complete) < 2:
                kind, data = await output.get()
                if kind == "_failure":
                    yield "turn_state", {"turn_id": turn_id, "state": "failed", "text_complete": text_complete,
                                         "delivered_audio_chunks": emitted_chunks, "spoken_complete": False,
                                         "delivery_basis": "emitted_to_response_iterator_not_client_acknowledgement"}
                    raise data
                if kind.startswith("_"):
                    complete.add(kind)
                    continue
                if kind == "audio":
                    emitted_chunks += 1
                yield kind, data
            yield "audio_end", {"turn_id": turn_id, "chunks": sent_chunks, "sample_count": total_frames,
                                "sample_rate": audio_format[0] if audio_format else None,
                                "first_pcm_ready_ms": first_pcm_ms, "synthesis_ms": synthesis_ms,
                                "total_ms": round((time.perf_counter() - started) * 1000, 3),
                                "state": "delivered", "spoken_complete": False, "acoustic_latency_ms": None}
            yield "done", {}
    finally:
        # Closing an async generator after client disconnect must also close its
        # LLM iterator, provider HTTP stream and any producer blocked on a queue.
        for worker in workers:
            worker.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(*workers, return_exceptions=True)
