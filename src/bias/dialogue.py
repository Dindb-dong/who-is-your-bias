"""Bounded, inspectable dialogue references without promoting user claims to facts."""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from .snapshots import TemporalIntent


FOLLOWUP = re.compile(r"\b(that|those|this approach|the same|you said|your earlier|what about it|why so|how so)\b|그\s*(?:것|거|건|런|렇게|렇다면|방법|방식)|아까|앞서|방금|그러면|왜\s*그래", re.I)
CORRECTION = re.compile(r"\b(i meant|i mean|correction|actually,? i|not .{1,80} but|instead of)\b|말한\s*건|말한\s*것은|아니[,\s]|정정|뜻한\s*건", re.I)
# Only recognizable elliptical requests inherit context. A short independent
# question ("How are you?", "Why does gravity work?") must remain independent.
ELLIPSIS = re.compile(
    r"^(?:(?:and|but|so|then)\s+)?(?:why|how|why not|tell me more|please explain|"
    r"explain more|can you elaborate|what do you mean)[?!.\s]*$|"
    r"^(?:왜|어떻게|왜요|왜지|왜\s*그런데|(?:좀\s*)?더\s*(?:자세히\s*)?설명해\s*(?:줘|주세요))[?!.\s]*$", re.I)
TEMPORAL_ELLIPSIS = re.compile(
    r"^(?:(?:and|but|so)\s+)?(?:(?:what|how)\s+about\s+)?(?:in\s+|by\s+|before\s+|as of\s+)?"
    r"(?:(?:19|20)\d{2}|now|today|currently|back then|at the time)[?!.\s]*$|"
    r"^(?:(?:그럼|그러면)\s*)?(?:(?:19|20)\d{2}\s*년\s*(?:에는|은|까지는|이전에는)?|"
    r"지금은|현재는|요즘은|그때는|당시에는)[?!.\s]*$", re.I)
EXPLICIT_SWITCH = re.compile(r"\b(different question|switching topics|change (?:the )?topic|separately)\b|다른\s*질문|주제(?:를)?\s*바꿔|별개로", re.I)
EARLIER_REFERENCE = re.compile(r"\b(back to|returning to|earlier.{0,30}(?:question|discussion)|previously.{0,30}(?:question|discussion))\b|아까.{0,30}(?:얘기|질문)|앞서.{0,30}(?:얘기|질문)|다시.{0,30}(?:얘기|질문)", re.I)
DEICTIC_PAST = re.compile(r"\b(back then|at the time)\b|그때|당시", re.I)
TEMPORAL_CORRECTION = re.compile(
    r"^(?:no[,\s]*)?(?:i mean|i meant|correction:?)\s+(?:in\s+|the\s+)?"
    r"(?:(?:19|20)\d{2}|now|today|currently|back then)(?:\s+(?:interview|period|year))?"
    r"(?:[,\s]+(?:not|instead of|rather than)\s+(?:in\s+)?(?:19|20)\d{2})?[?!.\s]*$|"
    r"^(?:아니[,\s]*)?(?:(?:19|20)\d{2}\s*년?\s*(?:이\s*)?(?:말고|아니라|아니고)[,\s]*)?"
    r"(?:19|20)\d{2}\s*년?\s*(?:인터뷰\s*)?(?:(?:얘기|말|시점)(?:였|이었)?(?:어|야|입니다))?[?!.\s]*$", re.I)
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
        if not self.records:
            return ResolvedQuery(query, query, (), "explicit_correction" if correction else "original", current, correction, query[:1000])
        prior = self.records[-1]
        if EXPLICIT_SWITCH.search(query):
            return ResolvedQuery(query, query, (), "topic_switch", current, correction, query[:1000])
        # A specific named earlier USER topic can be resumed. We never recover
        # an assistant claim as an independent source or guess an unnamed topic.
        earlier_reference = bool(current and EARLIER_REFERENCE.search(query))
        if earlier_reference:
            matching_records = [record for record in reversed(self.records) if set(current).intersection(record.entities)]
            explicitly_earlier = bool(re.search(r"\bearlier\b|아까|앞서", query, re.I))
            matching = next((record for record in matching_records
                             if not explicitly_earlier or record.turn_id != self.records[-1].turn_id),
                            matching_records[0] if matching_records else None)
            if matching is not None:
                prior = matching
            else:
                earlier_reference = False
        temporal_correction = bool(correction and not current and TEMPORAL_CORRECTION.fullmatch(query.strip()))
        if correction and not temporal_correction:
            return ResolvedQuery(query, query, (), "explicit_correction", current, True, query[:1000])
        if current and prior.entities and not set(current).intersection(prior.entities):
            return ResolvedQuery(query, query, (), "topic_switch", current, False, query[:1000])
        # Shortness alone is not evidence of a reference. Never include an
        # assistant's unsupported generated claims in a rewritten search query.
        if FOLLOWUP.search(query) or ELLIPSIS.fullmatch(query.strip()) or TEMPORAL_ELLIPSIS.fullmatch(query.strip()) or temporal_correction or earlier_reference:
            anchor = prior.search_anchor or prior.user_text[:1000]
            supporting = tuple(dict.fromkeys((*prior.supporting_turn_ids, prior.turn_id)))[-12:]
            # Search only actual user words. Artificial labels such as
            # "previous user question" dilute the lexical coverage denominator
            # and can turn a supported follow-up into an abstention.
            previous = f"\n{prior.user_text[:500]}" if prior.user_text[:1000] != anchor else ""
            method = "contextual_correction" if temporal_correction else "earlier_user_reference" if earlier_reference else "previous_user_reference"
            return ResolvedQuery(query, f"{anchor}{previous}\n{query}", supporting,
                                 method, current or prior.entities, correction, anchor)
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
