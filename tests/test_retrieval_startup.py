import asyncio
import threading
from types import SimpleNamespace

import pytest

from bias.api import create_app
from bias.config import Settings


def test_hybrid_readiness_waits_for_index_and_records_actual_warmup(monkeypatch, corpus):
    entered, release = threading.Event(), threading.Event()
    def warm_up():
        entered.set()
        assert release.wait(timeout=3)
    monkeypatch.setattr("bias.api.Engine", lambda *args, **kwargs: SimpleNamespace(
        retriever=SimpleNamespace(warm_up=warm_up)))
    app = create_app(Settings(retrieval_backend="hybrid", embedding_model="fixture"), corpus=corpus)

    async def run():
        lifecycle = app.router.lifespan_context(app)
        startup = asyncio.create_task(lifecycle.__aenter__())
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            assert not startup.done()
            release.set()
            await startup
            assert app.state.retrieval_warmup_ms >= 0
        finally:
            release.set()
            if not startup.done():
                await startup
            await lifecycle.__aexit__(None, None, None)
    asyncio.run(run())


def test_failed_hybrid_warmup_does_not_start_a_silently_degraded_api(monkeypatch, corpus):
    def warm_up():
        raise ValueError("fixture index failure")
    monkeypatch.setattr("bias.api.Engine", lambda *args, **kwargs: SimpleNamespace(
        retriever=SimpleNamespace(warm_up=warm_up)))
    app = create_app(Settings(retrieval_backend="hybrid", embedding_model="fixture"), corpus=corpus)
    async def run():
        with pytest.raises(ValueError, match="fixture index failure"):
            async with app.router.lifespan_context(app):
                pytest.fail("API advertised readiness after failed warmup")
    asyncio.run(run())
