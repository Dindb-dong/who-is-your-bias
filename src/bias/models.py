"""Shared, validated contracts for data ingestion, retrieval, and the live API."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")]
Mode = Literal["profile", "rag", "behavior"]
Language = Literal["en", "ko"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Locator(Record):
    label: str = Field(min_length=1, max_length=500)
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    timeline: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def valid_time_range(self) -> "Locator":
        if self.end_seconds is not None:
            if self.start_seconds is None or self.end_seconds <= self.start_seconds:
                raise ValueError("end_seconds requires an earlier start_seconds")
        if self.start_seconds is not None and not self.timeline:
            raise ValueError("Timed locators must identify their media timeline")
        return self


class Source(Record):
    model_config = ConfigDict(extra="allow", frozen=True)
    id: Identifier
    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    publisher: str = Field(min_length=1, max_length=200)
    published_at: date
    recorded_at: date | None = None
    source_type: str = Field(min_length=1, max_length=100)
    split: Literal["train", "validation", "test"] = "train"
    attribution: str = ""
    timeline: str | None = None


class Evidence(Record):
    model_config = ConfigDict(extra="allow", frozen=True)
    id: Identifier
    source_id: Identifier
    text: str = Field(min_length=1, max_length=12_000)
    topics: list[str] = Field(default_factory=list, max_length=30)
    situations: list[str] = Field(default_factory=list, max_length=30)
    locator: Locator
    kind: Literal["stated_view", "reported_experience", "authored_plan", "transcript"]
    confidence: Literal["source_grounded", "unverified_transcript"] = "source_grounded"
    behavioral_observation: dict | None = None
    # Missing review metadata never grants permission to index a new record.
    review_status: Literal["reviewed", "draft"] = "draft"

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Evidence text cannot be blank")
        return value


class Persona(Record):
    model_config = ConfigDict(extra="allow", frozen=True)
    id: Identifier
    name: str
    tagline: str
    description: str
    style_rules: list[str] = Field(default_factory=list)
    example_questions: dict[str, list[str]] = Field(default_factory=lambda: {"en": [], "ko": []})
    notice: str = "AI simulation based on public sources. Not the real person."


class ChatRequest(Record):
    message: str = Field(min_length=1, max_length=4_000)
    session_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    mode: Mode = "behavior"
    language: Language = "en"

    @field_validator("message")
    @classmethod
    def trim_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class Citation(Record):
    evidence_id: str
    source_id: str
    title: str
    url: str
    locator: Locator
    text: str


class BehaviorCue(Record):
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(gt=0, le=120_000)
    channel: Literal["face", "gaze", "head", "gesture"]
    name: str
    intensity: float = Field(ge=0, le=1)
    provenance: Literal["designed", "observed"] = "designed"
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def observation_needs_evidence(self) -> "BehaviorCue":
        if self.provenance == "observed" and not self.evidence_ids:
            raise ValueError("Observed behavior requires evidence identifiers")
        return self


class SpeechStyle(Record):
    rate: float = Field(default=1.0, ge=0.5, le=2)
    pitch: float = Field(default=1.0, ge=0.5, le=2)
    pause_ms: int = Field(default=260, ge=0, le=3_000)


class BehaviorPlan(Record):
    cues: list[BehaviorCue]
    speech: SpeechStyle
    duration_ms: int = Field(ge=0, le=120_000)
    timing_basis: Literal["estimated_text", "audio_aligned"] = "estimated_text"

    @model_validator(mode="after")
    def cues_fit_timeline(self) -> "BehaviorPlan":
        if any(c.start_ms + c.duration_ms > self.duration_ms for c in self.cues):
            raise ValueError("Behavior cues must fit inside the timeline")
        return self


class Grounding(Record):
    status: Literal["grounded", "insufficient", "unverified"]
    retrieved_ids: list[str]
    invalid_citation_ids: list[str] = Field(default_factory=list)
    reason: str | None = None
    # Syntax and coverage do not constitute a semantic grounding verdict.
    citation_status: Literal["complete", "partial", "missing", "invalid", "not_required"] = "not_required"
    citation_unit_count: int = Field(default=0, ge=0)
    cited_unit_count: int = Field(default=0, ge=0)
    malformed_citations: list[str] = Field(default_factory=list)
    semantic_status: Literal["unreviewed", "curated_source"] = "unreviewed"


class TurnMetrics(Record):
    retrieval_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class ChatTurn(Record):
    id: str
    session_id: str
    text: str
    language: Language
    mode: Mode
    provider: Literal["evidence", "openai"]
    citations: list[Citation]
    behavior: BehaviorPlan
    grounding: Grounding
    metrics: TurnMetrics


class SpeechRequest(Record):
    text: str = Field(min_length=1, max_length=4_000)
    voice: str | None = Field(default=None, min_length=1, max_length=100, pattern=r"^[\w.-]+$")


class SpeechPackageRequest(SpeechRequest):
    language: Language = "en"


class EvaluationCase(Record):
    model_config = ConfigDict(extra="allow", frozen=True)
    id: str
    category: Literal["retrieval", "heldout", "abstention", "injection"]
    question: str
    expected_evidence_ids: list[str] = Field(default_factory=list)
    expected_source_ids: list[str] = Field(default_factory=list)
    reference_evidence_ids: list[str] = Field(default_factory=list)
    reference_source_ids: list[str] = Field(default_factory=list)
    forbidden_source_ids: list[str] = Field(default_factory=list)
    expected_action: Literal["answer", "abstain", "ignore_instructions"]
    notes: str | None = None
