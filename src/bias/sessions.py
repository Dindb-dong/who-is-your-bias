"""Bounded process-local conversation memory with per-session serialization."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field


class SessionNotFound(LookupError):
    pass


class SessionCapacityError(RuntimeError):
    pass


@dataclass
class Session:
    id: str
    last_used: float
    history: deque[dict[str, str]]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    in_flight: int = 0


class SessionStore:
    def __init__(self, *, ttl: float = 3600, max_sessions: int = 256, max_turns: int = 12):
        self.ttl = ttl
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self.sessions: dict[str, Session] = {}

    def _prune(self, now: float) -> None:
        expired = [key for key, session in self.sessions.items() if now - session.last_used >= self.ttl and not session.lock.locked() and not session.in_flight]
        for key in expired:
            del self.sessions[key]

    def get_or_create(self, session_id: str | None) -> Session:
        now = time.monotonic()
        self._prune(now)
        if session_id:
            session = self.sessions.get(session_id)
            if session is None:
                raise SessionNotFound("Session expired or does not exist. Start a new conversation.")
        else:
            if len(self.sessions) >= self.max_sessions:
                idle = sorted((s for s in self.sessions.values() if not s.lock.locked() and not s.in_flight), key=lambda s: s.last_used)
                if not idle:
                    raise SessionCapacityError("All conversation slots are busy. Try again shortly.")
                del self.sessions[idle[0].id]
            session = Session(uuid.uuid4().hex, now, deque(maxlen=self.max_turns * 2))
            self.sessions[session.id] = session
        session.last_used = now
        return session

    def append(self, session: Session, user: str, assistant: str) -> None:
        session.history.append({"role": "user", "content": user})
        session.history.append({"role": "assistant", "content": assistant})
        session.last_used = time.monotonic()

    def delete(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if session is None:
            return False
        if session.lock.locked() or session.in_flight:
            raise SessionCapacityError("Cancel the active response before clearing its session.")
        del self.sessions[session_id]
        return True
