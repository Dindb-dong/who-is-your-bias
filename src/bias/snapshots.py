"""Versioned persona cutoffs and explicit historical/current evidence selection.

Dates and acquisition times have different meanings. A newer publication alone
never makes an undated old recording a current position, or supersedes a claim.
The historical research retriever does not import this runtime policy.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .corpus import Corpus, source_event_id
from .models import Identifier, Record


def corpus_digest(corpus: Corpus) -> str:
    payload = {
        "persona": corpus.persona.model_dump(mode="json"),
        "sources": [s.model_dump(mode="json") for _, s in sorted(corpus.sources.items())],
        "evidence": [e.model_dump(mode="json") for _, e in sorted(corpus.evidence.items())],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


class SourceVersion(Record):
    source_id: Identifier
    event_family: Identifier
    event_at: date | None = None
    published_at: date
    ingested_at: datetime
    split: Literal["train", "validation", "test"]
    source_content_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    # Publication-only metadata cannot establish a current spoken position.
    assertion_period_start: date | None = None
    assertion_period_end: date | None = None
    assertion_period_basis: str | None = Field(default=None, max_length=1000)

    @field_validator("ingested_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Acquisition time must include its UTC offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def period(self):
        if self.assertion_period_start is not None:
            if not self.assertion_period_end or not self.assertion_period_basis:
                raise ValueError("Assertion periods require an end and reviewed attribution basis")
            if self.assertion_period_end < self.assertion_period_start:
                raise ValueError("Assertion period ends before it starts")
        elif self.assertion_period_end is not None or self.assertion_period_basis is not None:
            raise ValueError("Assertion period metadata requires its start")
        return self


class StatementLink(Record):
    earlier_id: Identifier
    later_id: Identifier
    relation: Literal["supersedes", "contradicts"]
    reviewer: str = Field(min_length=1, max_length=200)
    reviewer_type: Literal["human", "agent"]
    basis: str = Field(min_length=1, max_length=2000)


class PersonaSnapshot(Record):
    schema_version: Literal["persona-snapshot.v1"] = "persona-snapshot.v1"
    id: Identifier
    persona_id: Identifier
    as_of: datetime
    current_evidence_since: date
    corpus_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    knowledge_version: str
    style_version: str
    voice_version: str
    appearance_version: str
    sources: list[SourceVersion]
    statement_links: list[StatementLink] = Field(default_factory=list)
    coverage_note: str

    @field_validator("as_of")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Snapshot cutoff must include its UTC offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def consistent(self):
        if self.current_evidence_since > self.as_of.date():
            raise ValueError("Current-evidence period starts after the cutoff")
        if len({s.source_id for s in self.sources}) != len(self.sources):
            raise ValueError("Snapshot has duplicate source versions")
        if any(s.ingested_at > self.as_of for s in self.sources):
            raise ValueError("Snapshot includes an acquisition after its cutoff")
        if any(link.earlier_id == link.later_id for link in self.statement_links):
            raise ValueError("Statement links cannot refer to the same statement")
        return self

    def verify(self, corpus: Corpus) -> None:
        if self.persona_id != corpus.persona.id or self.corpus_sha256 != corpus_digest(corpus):
            raise ValueError("Persona snapshot does not match this exact corpus; build a new snapshot")
        if {s.source_id for s in self.sources} != set(corpus.sources):
            raise ValueError("Snapshot must register every corpus source")
        for row in self.sources:
            source = corpus.sources[row.source_id]
            if (row.event_family, row.published_at, row.split) != (source_event_id(source), source.published_at, source.split):
                raise ValueError(f"Snapshot source lineage differs: {row.source_id}")
            if row.event_at != source.recorded_at:
                raise ValueError(f"Snapshot event date differs: {row.source_id}")
            acquired = (source.model_extra or {}).get("ingested_at")
            if acquired is not None and row.ingested_at != SourceVersion.aware(datetime.fromisoformat(acquired)):
                raise ValueError(f"Snapshot acquisition time differs: {row.source_id}")
        for link in self.statement_links:
            if link.earlier_id not in corpus.evidence or link.later_id not in corpus.evidence:
                raise ValueError("Statement link refers to unavailable evidence")

    @classmethod
    def load(cls, path: Path, corpus: Corpus) -> "PersonaSnapshot":
        value = cls.model_validate_json(path.read_text(encoding="utf-8"))
        value.verify(corpus)
        return value

    def summary(self) -> dict:
        current = [s.source_id for s in self.sources if s.split == "train" and
                   s.event_at is not None and self.current_evidence_since <= s.event_at <= self.as_of.date()]
        return {"id": self.id, "as_of": self.as_of.isoformat(), "corpus_sha256": self.corpus_sha256,
                "current_evidence_since": self.current_evidence_since.isoformat(),
                "current_dated_training_sources": len(current), "coverage_note": self.coverage_note,
                "knowledge_version": self.knowledge_version, "style_version": self.style_version,
                "voice_version": self.voice_version, "appearance_version": self.appearance_version}


CURRENT = re.compile(r"\b(now|today|currently|current|latest|these days|this year|at present)\b|지금|현재|요즘|올해|최근", re.I)
HISTORICAL = re.compile(r"\b(back then|at the time|historically|used to|in the past)\b|당시|예전|과거|그때", re.I)
DATED_STATEMENT = re.compile(r"\b(interview|published|said|what did|did you|did he)\b|인터뷰|발언|말했|기고", re.I)
FORECAST = re.compile(r"\b(plan|plans|forecast|predict|prediction|goal|expect|expected|hope)\b|계획|예상|목표|전망|예측", re.I)
THROUGH_PERIOD = re.compile(r"\b(by|as of|up to|until|through)\s+(?:the year\s+)?(?:19|20)\d{2}|(?:19|20)\d{2}\s*년?\s*까지", re.I)
BEFORE_PERIOD = re.compile(r"\b(before|prior to)\s+(?:the year\s+)?(?:19|20)\d{2}|(?:19|20)\d{2}\s*년?\s*이전", re.I)


@dataclass(frozen=True)
class TemporalIntent:
    mode: Literal["historical", "current", "comparison", "general", "future"]
    years: tuple[int, ...] = ()
    period: Literal["exact", "through", "before", "forecast"] = "exact"


def temporal_intent(query: str, as_of: datetime) -> TemporalIntent:
    years = tuple(sorted({int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", query)}))
    if any(y > as_of.year for y in years):
        if not FORECAST.search(query):
            return TemporalIntent("future", years)
        # A future target date is not the date of the original statement.
        # If an earlier interview year is also named, retain that event filter.
        earlier = tuple(y for y in years if y <= as_of.year)
        return TemporalIntent("historical", earlier or years, "exact" if earlier else "forecast")
    if (len(years) > 1 or (years and max(years) < as_of.year and CURRENT.search(query)) or
            (CURRENT.search(query) and HISTORICAL.search(query))):
        return TemporalIntent("comparison", years)
    if CURRENT.search(query):
        return TemporalIntent("current", years)
    period = "before" if BEFORE_PERIOD.search(query) else "through" if THROUGH_PERIOD.search(query) else "exact"
    if years and (DATED_STATEMENT.search(query) or period != "exact"):
        return TemporalIntent("historical", years, period)
    if as_of.year in years:
        return TemporalIntent("current", years)
    if years or HISTORICAL.search(query):
        return TemporalIntent("historical", years)
    return TemporalIntent("general")


@dataclass(frozen=True)
class TemporalSelection:
    intent: TemporalIntent
    eligible_ids: frozenset[str]
    exclusion_reasons: dict[str, str]
    temporal_scores: dict[str, float]
    reason: str | None


def select_temporal(corpus: Corpus, snapshot: PersonaSnapshot, query: str, *, projection: str = "factual",
                    intent_override: TemporalIntent | None = None) -> TemporalSelection:
    if projection not in {"factual", "text_style"}:
        raise ValueError("Temporal retrieval projection must be factual or text_style")
    intent = intent_override or temporal_intent(query, snapshot.as_of)
    rows = {s.source_id: s for s in snapshot.sources}
    eligible, excluded, scores = set(), {}, {}
    for evidence in (corpus.training_evidence if projection == "factual" else corpus.style_evidence):
        row = rows[evidence.source_id]
        # A date-only release is not known to be available until that day ends.
        release_latest = datetime.combine(row.published_at, time.max, tzinfo=timezone.utc)
        reason = None
        if intent.mode == "future":
            reason = "requested_period_after_snapshot"
        elif release_latest > snapshot.as_of or (row.event_at and row.event_at > snapshot.as_of.date()):
            reason = "publication_or_event_after_snapshot"
        elif intent.mode == "current":
            assertion_current = (row.assertion_period_start is not None and
                                 row.assertion_period_start <= snapshot.as_of.date() <= row.assertion_period_end)
            if not assertion_current and (row.event_at is None or row.event_at < snapshot.current_evidence_since):
                reason = "no_dated_current_position"
        elif intent.mode == "historical" and intent.years:
            source_year = (row.event_at or row.published_at).year
            if intent.period == "exact" and source_year not in intent.years:
                reason = "outside_requested_historical_year"
            elif intent.period == "through" and source_year > max(intent.years):
                reason = "after_requested_historical_period"
            elif intent.period == "before" and source_year >= min(intent.years):
                reason = "not_before_requested_historical_year"
        if reason:
            excluded[evidence.id] = reason
        else:
            eligible.add(evidence.id)
            dated = row.event_at or row.published_at
            scores[evidence.id] = max(0., 1. - (snapshot.as_of.date() - dated).days / 7300) if intent.mode == "current" else 0.
    # Only explicit reviewed supersession links suppress earlier current claims.
    if intent.mode == "current":
        for link in snapshot.statement_links:
            if link.relation == "supersedes" and link.later_id in eligible and link.earlier_id in eligible:
                eligible.remove(link.earlier_id)
                excluded[link.earlier_id] = "reviewed_superseded_statement"
    reason = None
    if not eligible:
        reason = ("The requested period is after this persona snapshot's cutoff." if intent.mode == "future" else
                  "No reviewed, dated evidence establishes a current position in this snapshot." if intent.mode == "current" else
                  "No training evidence is eligible for the requested period.")
    return TemporalSelection(intent, frozenset(eligible), excluded, scores, reason)


def register_snapshot(corpus: Corpus, *, identifier: str, as_of: datetime, ingested_at: datetime,
                      current_evidence_since: date) -> PersonaSnapshot:
    hashes = (corpus.manifest or {}).get("source_content_sha256", {})
    snapshot = PersonaSnapshot(
        id=identifier, persona_id=corpus.persona.id, as_of=as_of, current_evidence_since=current_evidence_since,
        corpus_sha256=corpus_digest(corpus), knowledge_version=(corpus.manifest or {}).get("id", "reviewed-archive"),
        style_version="attributed-examples.v1-no-trained-adapter", voice_version="reference-conditioned-baseline",
        appearance_version="research-human-v2-proxy-not-2026-likeness",
        sources=[SourceVersion(source_id=s.id, event_family=source_event_id(s), event_at=s.recorded_at,
                               published_at=s.published_at,
                               ingested_at=(s.model_extra or {}).get("ingested_at", ingested_at), split=s.split,
                               source_content_sha256=hashes.get(s.id)) for _, s in sorted(corpus.sources.items())],
        coverage_note="Snapshot date limits availability; it does not imply contemporary coverage. Unknown event dates remain unknown.")
    snapshot.verify(corpus)
    return snapshot
