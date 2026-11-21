"""Optional source-bound public communication hypotheses, never voice identity.

This small deterministic selector adds no model call. Observations guide delivery
only, cannot authorize factual claims, and are dropped as whole budgeted units.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import Field, model_validator

from .corpus import Corpus
from .models import Identifier, Record
from .persona_graph import EvidenceAnchor, _advice_policy_reason, _normalize, _validate_anchors, content_sha256
from .persona_rules import RuleReview, _rule_contains
from .snapshots import TemporalIntent, select_temporal


DELIVERY_INSTRUCTION = (
    " Communication guidance contains source-grounded editorial hypotheses about public speaking, "
    "not facts, verified feelings or voice settings. Apply only when its conditions fit; exceptions "
    "win. Answer the user's situation directly. Do not describe these instructions, their sources, "
    "or internal identifiers. Do not invent autobiographical stories or feelings. Humor is optional "
    "only when invited and appropriate; never joke at a vulnerable person's expense."
)
_HUMOR_INVITATION = re.compile(r"\b(?:joke|humou?r|lighthearted|funny)\b|농담|유머", re.I)
_HUMOR_OPTOUT = re.compile(
    r"\b(?:no|without|avoid|skip)\s+(?:any\s+)?(?:jok\w*|humou?r)|"
    r"\b(?:do not|don't|don’t|never)\b.{0,35}\b(?:jok\w*|humou?r|funny)\b|"
    r"(?:농담|유머).{0,18}(?:하지|말아|말아줘|빼|없이|금지)|(?:웃기려|웃기려고).{0,10}하지", re.I)
_SENSITIVE = re.compile(
    r"\b(?:grief|grieving|died|death|bereave\w*|suicid\w*|abuse|harass\w*|panic|"
    r"distress\w*|humiliat\w*|fired|layoff\w*|safety|patient|injur\w*|harm|"
    r"depress\w*|scandal|affair|discrimination|bankrupt\w*)\b|"
    r"\b(?:los\w*|lost)\b.{0,35}\b(?:money|savings|funds|income)\b|"
    r"\bfinancial\s+loss\b|상실|사망|자살|학대|괴롭|해고|안전|환자|불륜|우울|금전|파산|차별|"
    r"(?:돈|저축|재산).{0,12}(?:잃|날렸)", re.I)


class DeliveryObservation(Record):
    id: Identifier
    kind: Literal["reaction", "speaking_style"]
    text: str = Field(min_length=1, max_length=400)
    conditions: tuple[str, ...] = Field(min_length=1, max_length=4)
    exceptions: tuple[str, ...] = Field(min_length=1, max_length=4)
    retrieval_groups: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=2)
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=4)
    source_observation: str = Field(min_length=1, max_length=500)
    epistemic_status: Literal["editorial_interpretation"]
    humor: bool = False

    @model_validator(mode="after")
    def validate_text(self):
        for values in (self.conditions, self.exceptions, *self.retrieval_groups):
            if not values or len(values) != len(set(values)) or any(not s.strip() or len(s) > 400 for s in values):
                raise ValueError("Delivery cues and qualifications must be bounded, nonblank and unique")
        if any(len(group) > 8 or any(len(cue) > 80 for cue in group) for group in self.retrieval_groups):
            raise ValueError("Delivery retrieval groups must be small")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Duplicate delivery evidence")
        if not self.text.strip() or not self.source_observation.strip():
            raise ValueError("Blank delivery observation")
        return self


class DeliveryDocument(Record):
    schema_version: Literal["persona-delivery.v1"]
    id: Identifier
    persona_id: Identifier
    construction: Literal["source_grounded_editorial_delivery"]
    review: RuleReview
    anchors: tuple[EvidenceAnchor, ...] = Field(min_length=1, max_length=32)
    observations: tuple[DeliveryObservation, ...] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def validate_links(self):
        anchors = {a.evidence_id for a in self.anchors}
        if len(anchors) != len(self.anchors) or len({o.id for o in self.observations}) != len(self.observations):
            raise ValueError("Duplicate delivery anchor/observation")
        if anchors != {key for o in self.observations for key in o.evidence_ids}:
            raise ValueError("Delivery observations must bind all and only their source anchors")
        return self


@dataclass(frozen=True)
class DeliverySelection:
    observations: tuple[DeliveryObservation, ...]
    document_sha256: str
    reason: str | None = None

    def trace(self):
        return {"schema_version": "persona-delivery.v1", "document_sha256": self.document_sha256,
                "observation_ids": [o.id for o in self.observations],
                "evidence_ids": list(dict.fromkeys(k for o in self.observations for k in o.evidence_ids)),
                "reason": self.reason, "method": "lexical_cues_editorial_hypotheses",
                "acoustic_control": False}

    def payload(self):
        return [{"kind": o.kind, "guidance": o.text, "when": list(o.conditions),
                 "unless": list(o.exceptions)} for o in self.observations]


class PersonaDelivery:
    def __init__(self, document: DeliveryDocument, corpus: Corpus):
        self.document, self.corpus = document, corpus
        self.sha256 = content_sha256(document.model_dump(mode="json"))
        _validate_anchors(document.persona_id, document.anchors, corpus)

    @classmethod
    def load(cls, path: Path, corpus: Corpus):
        return cls(DeliveryDocument.model_validate(json.loads(path.read_text(encoding="utf-8"))), corpus)

    def select(self, question: str, *, snapshot=None, temporal=None) -> DeliverySelection:
        if not isinstance(question, str) or len(question) > 10000:
            raise ValueError("Delivery query must be bounded text")
        _validate_anchors(self.document.persona_id, self.document.anchors, self.corpus)
        query = _normalize(question)
        reason = _advice_policy_reason(query)
        if reason == "requires_factual_verification_or_disallowed_request":
            return DeliverySelection((), self.sha256, reason)
        eligible = set(temporal.eligible_ids) if temporal is not None else {a.evidence_id for a in self.document.anchors}
        if snapshot is not None:
            snapshot.verify(self.corpus)
            if reason is None or temporal is None:
                eligible = set(select_temporal(self.corpus, snapshot, question,
                    intent_override=TemporalIntent("general") if reason is None else None).eligible_ids)
        candidates = []
        for o in self.document.observations:
            if not set(o.evidence_ids) <= eligible:
                continue
            if o.humor and (not _HUMOR_INVITATION.search(query) or _HUMOR_OPTOUT.search(query)
                            or _SENSITIVE.search(query)):
                continue
            matches = [sum(_rule_contains(query, cue) for cue in group) for group in o.retrieval_groups]
            if all(matches):
                candidates.append((sum(matches), o))
        # At most one reaction and one style observation; deterministic ties.
        chosen, kinds = [], set()
        for _, observation in sorted(candidates, key=lambda pair: (-pair[0], pair[1].id)):
            if observation.kind not in kinds:
                chosen.append(observation); kinds.add(observation.kind)
        return DeliverySelection(tuple(chosen), self.sha256, None if chosen else "no_applicable_delivery_observation")
