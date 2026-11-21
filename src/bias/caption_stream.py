"""Translate committed speech alongside delivery without holding up PCM.

One bounded worker owns the optional translation channel. Overflow/failure is
visible, never a reason to delay speech or silently label English as Korean.
"""
from __future__ import annotations

import asyncio
from contextlib import aclosing

import anyio


def unavailable(segment, language, reason):
    return {key: segment[key] for key in ("turn_id", "segment_id", "sequence")} | {
        "language": language, "text": "", "state": "unavailable", "reason": reason}


async def caption_events(source, translator, language, *, queue_size=4, max_segments=32):
    if language is None:
        async with aclosing(source):
            async for item in source:
                yield item
        return
    if not 1 <= queue_size <= 16 or not 1 <= max_segments <= 128:
        raise ValueError("Invalid caption stream bounds")
    output = asyncio.Queue(maxsize=16)
    jobs = asyncio.Queue(maxsize=queue_size)
    finished = asyncio.Event()

    async def produce():
        seen = 0
        try:
            async with aclosing(source):
                async for kind, data in source:
                    if kind == "done":
                        continue
                    await output.put((kind, data))
                    if kind == "segment":
                        seen += 1
                        if seen > max_segments:
                            await output.put(("subtitle", unavailable(data, language, "caption_limit")))
                        else:
                            try:
                                jobs.put_nowait((data, asyncio.get_running_loop().time()))
                            except asyncio.QueueFull:
                                await output.put(("subtitle", unavailable(data, language, "translation_busy")))
            await output.put(("_source_done", {}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await output.put(("_failure", exc))
        finally:
            finished.set()

    async def translate():
        # Polling only the empty queue keeps completion independent of a full
        # jobs queue; a producer never waits for translation to consume a slot.
        try:
            while not finished.is_set() or not jobs.empty():
                try:
                    segment, queued_at = await asyncio.wait_for(jobs.get(), timeout=0.05)
                except TimeoutError:
                    continue
                # Queue waiting counts against the caption deadline too. Slow
                # translation cannot create minutes of obsolete subtitle work.
                remaining = getattr(translator, "timeout", 12.0) - (asyncio.get_running_loop().time() - queued_at)
                if remaining <= 0:
                    result = unavailable(segment, language, "translation_late")
                else:
                    try:
                        result = await asyncio.wait_for(translator.translate(segment, language), timeout=remaining)
                    except TimeoutError:
                        result = unavailable(segment, language, "translation_timeout")
                    except Exception:
                        result = unavailable(segment, language, "translation_provider_failed")
                await output.put(("subtitle", result))
            await output.put(("_captions_done", {}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await output.put(("_failure", exc))

    tasks = [asyncio.create_task(produce()), asyncio.create_task(translate())]
    complete = set()
    try:
        while len(complete) < 2:
            kind, data = await output.get()
            if kind == "_failure":
                raise data
            if kind.startswith("_"):
                complete.add(kind)
            else:
                yield kind, data
        yield "done", {}
    finally:
        for task in tasks:
            task.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(*tasks, return_exceptions=True)
