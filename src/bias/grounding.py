"""Mechanical citation checks, deliberately separate from factual entailment.

A known identifier can accompany a false claim. These checks find malformed,
unknown, misplaced, and missing markers; they cannot establish semantic support.
Sentence-like units are a conservative formatting heuristic, not extracted facts.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

CITATION = re.compile(r"\[source:([A-Za-z0-9_.-]+)\]")
# Also recognize marker-like attempts that the exact grammar rejects. Stop at a
# new opening bracket so a malformed attempt cannot swallow a later valid one.
CANDIDATE = re.compile(r"\[\s*source\s*:[^\[\]\n]*(?:\]|(?=\[|\n|$))", re.IGNORECASE)
BOUNDARY = re.compile(r"[.!?。！？]+[\"'”’)]*(?:\s+|$)|\n+")
ABBREVIATION = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|e\.g|i\.e|(?:[A-Za-z]\.)+[A-Za-z])\.", re.IGNORECASE)


def validate_citations(text: str, allowed_ids: set[str]) -> tuple[str, list[str], list[str]]:
    """Remove exact markers for speech/display and validate identifier membership.

Kept compatible with the original Engine helper. No citation is ever appended,
corrected, or synthesized from retrieval results.
    """
    mentioned = list(dict.fromkeys(CITATION.findall(text)))
    valid = [identifier for identifier in mentioned if identifier in allowed_ids]
    invalid = [identifier for identifier in mentioned if identifier not in allowed_ids]
    cleaned = CITATION.sub("", text)
    cleaned = re.sub(r"[ \t]+([.,!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, valid, invalid


@dataclass(frozen=True)
class CitationUnit:
    index: int
    text: str
    raw_text: str
    citation_ids: tuple[str, ...]
    trailing_valid_ids: tuple[str, ...]

    @property
    def covered(self) -> bool:
        return bool(self.trailing_valid_ids)

    def as_dict(self) -> dict:
        return {**asdict(self), "citation_ids": list(self.citation_ids), "trailing_valid_ids": list(self.trailing_valid_ids), "mechanically_covered": self.covered, "semantic_support": "ungraded"}


@dataclass(frozen=True)
class CitationAudit:
    valid_ids: tuple[str, ...]
    invalid_ids: tuple[str, ...]
    malformed_markers: tuple[str, ...]
    units: tuple[CitationUnit, ...]
    orphan_marker_count: int
    required: bool

    @property
    def cited_unit_count(self) -> int:
        return sum(unit.covered for unit in self.units)

    @property
    def coverage_status(self) -> Literal["complete", "partial", "missing", "invalid", "not_required"]:
        if self.invalid_ids or self.malformed_markers:
            return "invalid"
        if not self.required:
            return "not_required"
        if not self.units or not self.cited_unit_count:
            return "missing"
        if self.cited_unit_count != len(self.units) or self.orphan_marker_count:
            return "partial"
        return "complete"

    def as_dict(self, *, include_units: bool = True) -> dict:
        result = {
            "status": self.coverage_status,
            "unit_count": len(self.units),
            "cited_unit_count": self.cited_unit_count,
            "uncited_unit_indices": [unit.index for unit in self.units if not unit.covered],
            "orphan_marker_count": self.orphan_marker_count,
            "method": "sentence_like_trailing_markers_v1",
            "semantic_support": "ungraded",
        }
        if include_units:
            result["units"] = [unit.as_dict() for unit in self.units]
        return result


def _unit_slices(text: str) -> list[str]:
    # Preserve offsets while hiding punctuation inside citations. A marker after
    # sentence-ending punctuation belongs to that preceding sentence, including
    # markers on the next line. Decimal numbers and common abbreviations must
    # not create spurious uncited units.
    masked = CANDIDATE.sub(lambda match: " " * len(match.group()), text)
    masked = re.sub(r"(?<=\d)\.(?=\d)", "\u2024", masked)
    masked = ABBREVIATION.sub(lambda match: match.group().replace(".", "\u2024"), masked)
    masked = re.sub(r"(?m)^(\s*(?:[-*•]\s+)?\d+)\.(?=\s)", lambda match: match.group(1) + "\u2024", masked)
    boundaries = [match.end() for match in BOUNDARY.finditer(masked)]
    if not boundaries or boundaries[-1] != len(text):
        boundaries.append(len(text))
    result = []
    start = 0
    for end in boundaries:
        raw = text[start:end].strip()
        if raw:
            result.append(raw)
        start = end
    return result


def audit_citations(text: str, allowed_ids: set[str], *, required: bool) -> CitationAudit:
    """Audit citation syntax and trailing-marker coverage without judging facts."""
    _, valid, invalid = validate_citations(text, allowed_ids)
    candidates = list(CANDIDATE.finditer(text))
    malformed = tuple(dict.fromkeys(match.group() for match in candidates if CITATION.fullmatch(match.group()) is None))
    units = []
    orphan_count = 0
    for raw in _unit_slices(text):
        visible = CANDIDATE.sub(lambda match: " " * len(match.group()), raw)
        positions = [index for index, character in enumerate(visible) if character.isalnum()]
        matches = list(CITATION.finditer(raw))
        if not positions:
            orphan_count += len(matches)
            continue
        # A citation at the start or middle of a sentence is not accepted as
        # coverage for the rest of the sentence. Several trailing records are
        # allowed; exact support by each of them remains a review question.
        trailing = tuple(dict.fromkeys(match.group(1) for match in matches if match.start() > positions[-1] and match.group(1) in allowed_ids))
        orphan_count += sum(match.end() <= positions[0] for match in matches)
        units.append(CitationUnit(
            index=len(units), text=CANDIDATE.sub("", raw).strip(), raw_text=raw,
            citation_ids=tuple(dict.fromkeys(match.group(1) for match in matches)),
            trailing_valid_ids=trailing,
        ))
    return CitationAudit(tuple(valid), tuple(invalid), malformed, tuple(units), orphan_count, required)
