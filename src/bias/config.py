"""Environment-only provider configuration. Secrets never enter API responses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _endpoint(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Provider base URLs must be HTTP(S) URLs without embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Provider base URLs cannot include a query or fragment")
    return value.rstrip("/")


@dataclass(frozen=True)
class ProviderSettings:
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _endpoint(self.base_url))
        if bool(self.base_url) != bool(self.model):
            raise ValueError("Configure both a provider base URL and model, or neither")

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = field(default_factory=lambda: repository_root() / "data/personas/elon-musk")
    web_dir: Path = field(default_factory=lambda: repository_root() / "apps/web/dist")
    llm: ProviderSettings = field(default_factory=ProviderSettings)
    tts: ProviderSettings = field(default_factory=ProviderSettings)
    asr: ProviderSettings = field(default_factory=ProviderSettings)
    tts_voice: str = "default"
    request_timeout: float = 60.0
    session_ttl: float = 3_600.0
    max_sessions: int = 256
    max_turns: int = 12
    retrieval_limit: int = 4
    max_response_chars: int = 6_000
    llm_max_tokens: int = 900
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

    @classmethod
    def from_env(cls) -> "Settings":
        def provider(kind: str) -> ProviderSettings:
            return ProviderSettings(
                base_url=os.getenv(f"BIAS_{kind}_BASE_URL"),
                model=os.getenv(f"BIAS_{kind}_MODEL"),
                api_key=os.getenv(f"BIAS_{kind}_API_KEY"),
            )

        def number(name: str, default: float, lower: float, upper: float) -> float:
            value = float(os.getenv(name, str(default)))
            if not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
            return value

        llm_max_tokens = number("BIAS_LLM_MAX_TOKENS", 900, 1, 4096)
        if not llm_max_tokens.is_integer():
            raise ValueError("BIAS_LLM_MAX_TOKENS must be an integer")

        return cls(
            corpus_dir=Path(os.getenv("BIAS_CORPUS_DIR", str(repository_root() / "data/personas/elon-musk"))),
            web_dir=Path(os.getenv("BIAS_WEB_DIR", str(repository_root() / "apps/web/dist"))),
            llm=provider("LLM"),
            tts=provider("TTS"),
            asr=provider("ASR"),
            tts_voice=os.getenv("BIAS_TTS_VOICE", "default"),
            request_timeout=number("BIAS_REQUEST_TIMEOUT", 60, 1, 180),
            session_ttl=number("BIAS_SESSION_TTL", 3600, 1, 86400),
            max_sessions=int(number("BIAS_MAX_SESSIONS", 256, 1, 10000)),
            max_turns=int(number("BIAS_MAX_TURNS", 12, 1, 100)),
            retrieval_limit=int(number("BIAS_RETRIEVAL_LIMIT", 4, 1, 10)),
            llm_max_tokens=int(llm_max_tokens),
            cors_origins=tuple(x.strip() for x in os.getenv("BIAS_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()),
        )
