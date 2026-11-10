"""The deliberately small text-only subset of chat completions we support."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_000)

    @field_validator("content")
    @classmethod
    def plain_text(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Messages must contain text without NUL characters")
        return value


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    include_usage: bool = True


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    model: str = Field(min_length=1, max_length=200)
    messages: list[Message] = Field(min_length=1, max_length=64)
    stream: bool = False
    max_tokens: int = Field(default=256, ge=1, le=4096, strict=True)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1, strict=True)
    n: Literal[1] = 1
    stream_options: StreamOptions = Field(default_factory=StreamOptions)
