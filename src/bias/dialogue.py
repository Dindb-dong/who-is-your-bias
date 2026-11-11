"""Bounded, inspectable dialogue references without promoting user claims to facts."""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from .snapshots import TemporalIntent


FOLLOWUP = re.compile(r"\b(that|those|this approach|the same|you said|your earlier|what about it|why so|how so)\b|그\s*(?:것|거|런|렇게|렇다면|방법|방식)|아까|앞서|방금|그러면|왜\s*그래", re.I)
CORRECTION = re.compile(r"\b(i meant|i mean|correction|actually,? i|not .{1,80} but|instead of)\b|말한\s*건|말한\s*것은|아니[,\s]|정정|뜻한\s*건", re.I)
# These are entity aliases used to resolve references, not personal beliefs.
ENTITIES = {"Tesla": r"tesla|테슬라|electric (?:car|vehicle)|전기차", "SpaceX": r"spacex|스페이스엑스|rocket|로켓|starship|스타십",
            "Neuralink": r"neuralink|뉴럴링크", "tunneling": r"tunnel|boring company|터널|보링",
            "AI": r"\bai\b|artificial intelligence|인공지능", "Mars": r"\bmars\b|화성"}


def entities(text: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in ENTITIES.items() if re.search(pattern, text, re.I))


@dataclass(frozen=True)
class DialogueRecord:
    turn_id: str
    user_text: str
    assistant_text: str
    entities: tuple[str, ...]
    cited_ids: tuple[str, ...]
    grounding: str
    temporal_mode: str
    snapshot_id: str | None
    is_correction: bool = False
    temporal_years: tuple[int, ...] = ()
    search_anchor: str = ""
    supporting_turn_ids: tuple[str, ...] = ()
    temporal_period: str = "exact"


@dataclass(frozen=True)
class ResolvedQuery:
    original: str
    query: str
    supporting_turn_ids: tuple[str, ...]
    method: str
    entities: tuple[str, ...]
    is_correction: bool
    anchor: str = ""


@dataclass
class DialogueState:
    records: deque[DialogueRecord] = field(default_factory=lambda: deque(maxlen=12))

    def resolve(self, query: str) -> ResolvedQuery:
        current = entities(query)
        correction = bool(CORRECTION.search(query))
        if not self.records or correction:
            return ResolvedQuery(query, query, (), "explicit_correction" if correction else "original", current, correction, query[:1000])
        prior = self.records[-1]
        if current and prior.entities and not set(current).intersection(prior.entities):
            return ResolvedQuery(query, query, (), "topic_switch", current, False, query[:1000])
        # Shortness alone is not evidence of a reference. Never include an
        # assistant's unsupported generated claims in a rewritten search query.
        if FOLLOWUP.search(query):
            anchor = prior.search_anchor or prior.user_text[:1000]
            supporting = tuple(dict.fromkeys((*prior.supporting_turn_ids, prior.turn_id)))[-12:]
            previous = f"\nPrevious user question: {prior.user_text[:500]}" if prior.user_text[:1000] != anchor else ""
            return ResolvedQuery(query, f"{anchor}{previous}\nFollow-up question: {query}", supporting,
                                 "previous_user_reference", current or prior.entities, False, anchor)
        return ResolvedQuery(query, query, (), "original", current, False, query[:1000])

    def append(self, *, turn_id: str, resolved: ResolvedQuery, assistant: str, cited_ids: list[str],
               grounding: str, intent: TemporalIntent | None, snapshot_id: str | None) -> None:
        self.records.append(DialogueRecord(turn_id, resolved.original[:4000], assistant[:6000], resolved.entities,
                                          tuple(cited_ids), grounding, intent.mode if intent else "general",
                                          snapshot_id, resolved.is_correction, intent.years if intent else (),
                                          resolved.anchor, resolved.supporting_turn_ids, intent.period if intent else "exact"))

    def inspect(self) -> dict:
        # A session summary is provenance, never an automatic factual memory.
        return {"schema_version": "dialogue-state.v1", "turn_count": len(self.records),
                "turns": [{"turn_id": r.turn_id, "user_statement": r.user_text, "entities": list(r.entities),
                           "cited_evidence_ids": list(r.cited_ids), "grounding_status": r.grounding,
                           "temporal_mode": r.temporal_mode, "snapshot_id": r.snapshot_id,
                           "temporal_years": list(r.temporal_years), "search_anchor": r.search_anchor,
                           "temporal_period": r.temporal_period,
                           "supporting_user_turn_ids": list(r.supporting_turn_ids),
                           "user_correction": r.is_correction, "claims_semantically_verified": False} for r in self.records]}
