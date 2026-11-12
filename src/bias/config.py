"""Environment-only provider configuration. Secrets never enter API responses."""

from __future__ import annotations

import os
import re
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
    corpus_dir: Path = field(default_factory=lambda: repository_root() / "data/personas/jensen-huang")
    web_dir: Path = field(default_factory=lambda: repository_root() / "apps/web/dist")
    llm: ProviderSettings = field(default_factory=ProviderSettings)
    tts: ProviderSettings = field(default_factory=ProviderSettings)
    asr: ProviderSettings = field(default_factory=ProviderSettings)
    tts_voice: str = "default"
    # Explicitly bind a server voice to this persona, or declare a generic voice.
    tts_persona_id: str | None = None
    tts_language_hint: bool = False
    motion_catalog: Path | None = None
    motion_asset_root: Path = field(default_factory=lambda: repository_root() / "apps/web/public")
    request_timeout: float = 60.0
    session_ttl: float = 3_600.0
    max_sessions: int = 256
    max_turns: int = 12
    retrieval_limit: int = 4
    max_response_chars: int = 6_000
    llm_max_tokens: int = 900
    persona_snapshot: Path | None = None
    llm_context_window: int = 8192
    llm_tokenizer_path: Path | None = None
    retrieval_backend: str = "lexical"
    retrieval_prewarm: bool = True
    embedding_backend: str = "local"
    embedding_model: str | None = None
    embedding_revision: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "
    dense_min_similarity: float = 0.84
    reranker_model: str | None = None
    reranker_revision: str | None = None
    reranker_min_score: float | None = None
    style_examples: bool = False
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

    def __post_init__(self) -> None:
        if self.tts_persona_id is not None and (not isinstance(self.tts_persona_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", self.tts_persona_id)):
            raise ValueError("tts_persona_id must be a persona identifier or generic")
        if not isinstance(self.tts_language_hint, bool):
            raise ValueError("tts_language_hint must be a boolean")
        if self.retrieval_backend not in {"lexical", "hybrid", "jensen-lexical"} or self.embedding_backend not in {"local", "http"}:
            raise ValueError("Select lexical/hybrid/jensen-lexical retrieval and local/http embeddings")
        if not isinstance(self.retrieval_prewarm, bool):
            raise ValueError("retrieval_prewarm must be a boolean")
        if self.retrieval_backend == "hybrid" and not self.embedding_model:
            raise ValueError("Hybrid retrieval requires an explicit embedding model")
        if self.retrieval_backend == "hybrid" and self.embedding_backend == "http" and not self.embedding_base_url:
            raise ValueError("HTTP embeddings require an explicit base URL")
        object.__setattr__(self, "embedding_base_url", _endpoint(self.embedding_base_url))
        if not -1 <= self.dense_min_similarity <= 1:
            raise ValueError("Dense relevance threshold must be between -1 and 1")
        if self.reranker_min_score is not None:
            import math
            if not self.reranker_model or not math.isfinite(self.reranker_min_score):
                raise ValueError("A finite reranker threshold requires a configured reranker")
        if not 1024 <= self.llm_context_window <= 262144 or self.llm_max_tokens >= self.llm_context_window:
            raise ValueError("Context window must leave an input budget after reserved output tokens")

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
        language_hint = os.getenv("BIAS_TTS_LANGUAGE_HINT", "false").strip().lower()
        if language_hint not in {"true", "false", "1", "0"}:
            raise ValueError("BIAS_TTS_LANGUAGE_HINT must be true, false, 1, or 0")
        prewarm = os.getenv("BIAS_RETRIEVAL_PREWARM", "true").strip().lower()
        if prewarm not in {"true", "false", "1", "0"}:
            raise ValueError("BIAS_RETRIEVAL_PREWARM must be true, false, 1, or 0")

        return cls(
            corpus_dir=Path(os.getenv("BIAS_CORPUS_DIR", str(repository_root() / "data/personas/jensen-huang"))),
            web_dir=Path(os.getenv("BIAS_WEB_DIR", str(repository_root() / "apps/web/dist"))),
            llm=provider("LLM"),
            tts=provider("TTS"),
            asr=provider("ASR"),
            tts_voice=os.getenv("BIAS_TTS_VOICE", "default"),
            tts_persona_id=os.getenv("BIAS_TTS_PERSONA_ID") or None,
            tts_language_hint=language_hint in {"true", "1"},
            motion_catalog=Path(os.environ["BIAS_MOTION_CATALOG"]).expanduser() if os.getenv("BIAS_MOTION_CATALOG") else None,
            motion_asset_root=Path(os.getenv("BIAS_MOTION_ASSET_ROOT", str(repository_root() / "apps/web/public"))).expanduser(),
            request_timeout=number("BIAS_REQUEST_TIMEOUT", 60, 1, 180),
            session_ttl=number("BIAS_SESSION_TTL", 3600, 1, 86400),
            max_sessions=int(number("BIAS_MAX_SESSIONS", 256, 1, 10000)),
            max_turns=int(number("BIAS_MAX_TURNS", 12, 1, 100)),
            retrieval_limit=int(number("BIAS_RETRIEVAL_LIMIT", 4, 1, 10)),
            llm_max_tokens=int(llm_max_tokens),
            persona_snapshot=(Path(os.environ["BIAS_PERSONA_SNAPSHOT"]).expanduser() if os.getenv("BIAS_PERSONA_SNAPSHOT") else
                              None if os.getenv("BIAS_CORPUS_DIR") or os.environ.get("BIAS_PERSONA_SNAPSHOT") == "" else
                              repository_root() / "data/snapshots/jensen-huang-2026-11-12.json"),
            llm_context_window=int(number("BIAS_LLM_CONTEXT_WINDOW", 8192, 1024, 262144)),
            llm_tokenizer_path=Path(os.environ["BIAS_LLM_TOKENIZER_PATH"]).expanduser() if os.getenv("BIAS_LLM_TOKENIZER_PATH") else None,
            retrieval_backend=os.getenv("BIAS_RETRIEVAL_BACKEND", "lexical"),
            retrieval_prewarm=prewarm in {"true", "1"},
            embedding_backend=os.getenv("BIAS_EMBEDDING_BACKEND", "local"),
            embedding_model=os.getenv("BIAS_EMBEDDING_MODEL") or None,
            embedding_revision=os.getenv("BIAS_EMBEDDING_REVISION") or None,
            embedding_base_url=os.getenv("BIAS_EMBEDDING_BASE_URL") or None,
            embedding_api_key=os.getenv("BIAS_EMBEDDING_API_KEY") or None,
            embedding_query_prefix=os.getenv("BIAS_EMBEDDING_QUERY_PREFIX", "query: "),
            embedding_passage_prefix=os.getenv("BIAS_EMBEDDING_PASSAGE_PREFIX", "passage: "),
            dense_min_similarity=number("BIAS_DENSE_MIN_SIMILARITY", 0.84, -1, 1),
            reranker_model=os.getenv("BIAS_RERANKER_MODEL") or None,
            reranker_revision=os.getenv("BIAS_RERANKER_REVISION") or None,
            reranker_min_score=float(os.environ["BIAS_RERANKER_MIN_SCORE"]) if os.getenv("BIAS_RERANKER_MIN_SCORE") else None,
            style_examples=os.getenv("BIAS_STYLE_EXAMPLES", "false").lower() == "true",
            cors_origins=tuple(x.strip() for x in os.getenv("BIAS_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()),
        )

