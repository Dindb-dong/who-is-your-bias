"""Shared, validated contracts for data ingestion, retrieval, and the live API."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from .motion import MotionSelection

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


class ExchangeReview(Record):
    reviewer_type: Literal["agent", "human"]
    reviewer: str = Field(min_length=1, max_length=200)
    reviewed_at: date
    basis: str = Field(min_length=1, max_length=2000)
    speaker_status: Literal["publisher_labels_checked", "audio_verified", "unreviewed"]
    timing_status: Literal["not_audio_verified", "audio_verified"] = "not_audio_verified"


class ExchangeLocator(Record):
    question_cue_ids: list[Identifier] = Field(min_length=1, max_length=300)
    answer_cue_ids: list[Identifier] = Field(min_length=1, max_length=500)
    context_cue_ids: list[Identifier] = Field(default_factory=list, max_length=300)
    transcript_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class QuestionAnswer(Record):
    """Original publisher wording, distinct from annotations and timing proxies."""

    question: str = Field(min_length=1, max_length=6000)
    answer: str = Field(min_length=1, max_length=12000)
    context_before: str = Field(default="", max_length=3000)
    context_kind: Literal["editorial_summary", "original_text", "none"] = "editorial_summary"
    original_language: Literal["en", "ko"]
    questioner: str = Field(min_length=1, max_length=200)
    respondent: str = Field(min_length=1, max_length=200)
    source_event_id: Identifier
    completeness: Literal["complete_exchange", "excerpt"]
    excerpt_note: str | None = Field(default=None, max_length=2000)
    question_start_seconds: float | None = Field(default=None, ge=0)
    question_end_seconds: float | None = Field(default=None, ge=0)
    answer_start_seconds: float | None = Field(default=None, ge=0)
    answer_end_seconds: float | None = Field(default=None, ge=0)
    # End proxies never masquerade as acoustically verified utterance ends.
    question_end_proxy_seconds: float | None = Field(default=None, ge=0)
    answer_end_proxy_seconds: float | None = Field(default=None, ge=0)
    source_timeline: str | None = None
    timing_basis: Literal["publisher_cue_onsets_only", "publisher_transcript_timestamps_only", "audio_verified", "unavailable"]
    transcript_locator: ExchangeLocator
    review: ExchangeReview
    annotation_basis: str = Field(min_length=1, max_length=2000)
    taxonomy_version: Literal["situations.v1"] = "situations.v1"

    @model_validator(mode="after")
    def valid_exchange(self) -> "QuestionAnswer":
        if not self.question.strip() or not self.answer.strip():
            raise ValueError("Original question and answer must not be blank")
        if self.completeness == "excerpt" and not self.excerpt_note:
            raise ValueError("Excerpted exchanges must explain omitted context")
        times = (self.question_start_seconds, self.question_end_seconds, self.answer_start_seconds, self.answer_end_seconds,
                 self.question_end_proxy_seconds, self.answer_end_proxy_seconds)
        if any(value is not None for value in times) and not self.source_timeline:
            raise ValueError("Timed exchanges require their exact source timeline")
        if self.timing_basis != "audio_verified" and (self.question_end_seconds is not None or self.answer_end_seconds is not None):
            raise ValueError("Publisher onsets do not establish acoustic end boundaries")
        for start, end in ((self.question_start_seconds, self.question_end_seconds), (self.answer_start_seconds, self.answer_end_seconds),
                           (self.question_start_seconds, self.question_end_proxy_seconds), (self.answer_start_seconds, self.answer_end_proxy_seconds)):
            if end is not None and (start is None or end <= start):
                raise ValueError("Exchange end must follow its start")
        if self.question_start_seconds is not None and self.answer_start_seconds is not None and self.answer_start_seconds < self.question_start_seconds:
            raise ValueError("Answer cannot precede its question")
        if self.timing_basis == "unavailable" and any(value is not None for value in times):
            raise ValueError("Unavailable timing must remain null")
        return self


class Evidence(Record):
    model_config = ConfigDict(extra="allow", frozen=True)
    id: Identifier
    source_id: Identifier
    text: str = Field(min_length=1, max_length=12_000)
    topics: list[str] = Field(default_factory=list, max_length=30)
    situations: list[str] = Field(default_factory=list, max_length=30)
    locator: Locator
    kind: Literal["stated_view", "reported_experience", "authored_plan", "transcript", "question_answer"]
    confidence: Literal["source_grounded", "unverified_transcript"] = "source_grounded"
    behavioral_observation: dict | None = None
    # Missing review metadata never grants permission to index a new record.
    review_status: Literal["reviewed", "draft"] = "draft"
    question_answer: QuestionAnswer | None = None

    @model_validator(mode="before")
    @classmethod
    def original_answer_as_citation(cls, value):
        # A JSONL source need not duplicate copyrighted original wording in both
        # fields. Runtime citations retain the same validated answer text.
        if isinstance(value, dict) and "text" not in value and isinstance(value.get("question_answer"), dict):
            value = dict(value, text=value["question_answer"].get("answer"))
        return value

    @model_validator(mode="after")
    def validate_question_answer(self) -> "Evidence":
        if self.kind == "question_answer" and self.question_answer is None:
            raise ValueError("Question-answer evidence requires an original exchange")
        if self.question_answer is not None:
            if self.kind != "question_answer" or self.text != self.question_answer.answer:
                raise ValueError("Question-answer citation text must equal its original answer")
            if self.review_status == "reviewed" and self.question_answer.review.speaker_status == "unreviewed":
                raise ValueError("Reviewed Q&A requires checked speaker attribution")
        return self

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
    motion_mode: Literal["disabled", "reviewed", "geometry_preview"] = "disabled"

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
    # Measured source timing is independent of the designed/audio timeline.
    motion: MotionSelection | None = None

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
    language: Language | None = None


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
