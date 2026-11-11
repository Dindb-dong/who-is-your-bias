"""Keep asynchronous work owned by the HTTP client that requested it."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

import anyio
from starlette.requests import ClientDisconnect, Request

T = TypeVar("T")


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        if (await request.receive())["type"] == "http.disconnect":
            return


async def run_until_disconnect(request: Request, operation: Coroutine[Any, Any, T]) -> T:
    """Cancel and await this request's work when its client disconnects.

    The body must already be consumed, and the operation must not read from
    ``request.receive``: the monitor is its sole remaining consumer. Closing an
    upstream HTTP request is cooperative cancellation; a remote service may
    have its own policy for stopping inference already in progress.
    """
    work = asyncio.create_task(operation)
    disconnected = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait((work, disconnected), return_when=asyncio.FIRST_COMPLETED)
        if disconnected in done:
            disconnected.result()
            # A disconnected request never reports a synthetic success or
            # completion timing, including when both tasks finish together.
            raise ClientDisconnect()
        return work.result()
    finally:
        # ASGI middleware can use level cancellation. Shield the cleanup from
        # its cancel scope so provider cleanup and subprocess reaping finish.
        with anyio.CancelScope(shield=True):
            for task in (work, disconnected):
                if not task.done():
                    task.cancel()
            await asyncio.gather(work, disconnected, return_exceptions=True)
