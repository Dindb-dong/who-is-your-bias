"""Source-grounded conditional persona rules, with explicit epistemic limits.

These records are editorial interpretations of public statements, not measured
mental states. Retrieval is deterministic lexical conjunction over situation
cues. It neither proves a condition holds nor resolves factual/current claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Iterable, Literal

from pydantic import Field, model_validator

from .corpus import Corpus, source_event_id
from .models import Identifier, Record
from .persona_graph import (
    EvidenceAnchor, GraphSelection, PersonaGraph, PersonaGraphError,
    _advice_policy_reason, _contains, _normalize, _validate_anchors, content_sha256,
)


class RuleReview(Record):
    reviewer_type: Literal["agent", "human"]
    reviewer: str = Field(min_length=1, max_length=200)
    reviewed_at: date


class RuleAnchor(EvidenceAnchor):
    source_event_id: Identifier
    # Curator-assigned episode identity prevents repeated tellings of one
    # experience from being counted as independent life experiences.
    experience_id: Identifier


class EvidenceLink(Record):
    evidence_id: Identifier
    relation: Literal["supports", "qualifies", "contradicts"]
    rationale: str = Field(min_length=1, max_length=350)


def _check_links(links: tuple[EvidenceLink, ...]) -> None:
    if len({(link.evidence_id, link.relation) for link in links}) != len(links):
        raise ValueError("Duplicate evidence relationship")
    if not any(link.relation == "supports" for link in links):
        raise ValueError("A persona interpretation requires positive source support")
    if any(not link.rationale.strip() for link in links):
        raise ValueError("Evidence relationship must explain its meaning")


class PersonaClaim(Record):
    id: Identifier
    kind: Literal["value", "belief", "interaction_style"]
    text: str = Field(min_length=1, max_length=650)
    epistemic_status: Literal[
        "explicit_statement", "multi_source_interpretation",
        "single_event_interpretation", "contested",
    ]
    review_status: Literal["reviewed", "draft"] = "draft"
    evidence_links: tuple[EvidenceLink, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def valid_claim(self):
        _check_links(self.evidence_links)
        if not self.text.strip():
            raise ValueError("A persona claim must not be blank")
        if self.epistemic_status == "contested" and not any(
            link.relation == "contradicts" for link in self.evidence_links
        ):
            raise ValueError("Contested claims require explicit contrary evidence")
        if any(link.relation == "contradicts" for link in self.evidence_links) and self.epistemic_status != "contested":
            raise ValueError("Contrary evidence must mark a claim contested")
        return self


class ConditionalRule(Record):
    id: Identifier
    kind: Literal["conditional_strategy"] = "conditional_strategy"
    text: str = Field(min_length=1, max_length=650)
    claim_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=5)
    conditions: tuple[str, ...] = Field(min_length=1, max_length=6)
    exceptions: tuple[str, ...] = Field(min_length=1, max_length=6)
    review_status: Literal["reviewed", "draft"] = "draft"
    # Disjunction within a group, conjunction across groups. These select a
    # possibly relevant rule; the LLM must still check conditions/exceptions.
    retrieval_groups: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=4)
    exclusion_terms: tuple[str, ...] = Field(default=(), max_length=30)
    evidence_links: tuple[EvidenceLink, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def valid_rule(self):
        _check_links(self.evidence_links)
        if not self.text.strip() or len(set(self.claim_ids)) != len(self.claim_ids):
            raise ValueError("Rules need nonblank text and unique linked claims")
        for sequence in (self.conditions, self.exceptions):
            if len(set(sequence)) != len(sequence) or any(not text.strip() or len(text) > 500 for text in sequence):
                raise ValueError("Conditions and exceptions must be unique bounded nonblank text")
        if len(set(self.retrieval_groups)) != len(self.retrieval_groups):
            raise ValueError("Duplicate retrieval group")
        for group in (*self.retrieval_groups, self.exclusion_terms):
            if len(group) > 40 or len(set(group)) != len(group):
                raise ValueError("Retrieval groups must contain unique bounded terms")
            if any(not term.strip() or len(term) > 80 or term != term.strip() for term in group):
                raise ValueError("Retrieval terms must be nonblank and bounded")
        if any(not group for group in self.retrieval_groups):
            raise ValueError("Empty retrieval groups are not an applicability rule")
        normalized = [_normalize(term) for group in self.retrieval_groups for term in group]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Retrieval groups must not share normalized terms; conjunction requires distinct cues")
        if set(normalized) & {_normalize(term) for term in self.exclusion_terms}:
            raise ValueError("A retrieval term cannot also be an exclusion term")
        return self


class RulesDocument(Record):
    schema_version: Literal["persona-graph.v2"]
    id: Identifier
    persona_id: Identifier
    construction: Literal["source_grounded_editorial_rules"]
    review: RuleReview
    anchors: tuple[RuleAnchor, ...] = Field(min_length=1, max_length=96)
    claims: tuple[PersonaClaim, ...] = Field(min_length=1, max_length=64)
    rules: tuple[ConditionalRule, ...] = Field(min_length=1, max_length=48)

    @model_validator(mode="after")
    def valid_graph(self):
        anchors = {anchor.evidence_id: anchor for anchor in self.anchors}
        claims = {claim.id: claim for claim in self.claims}
        rules = {rule.id: rule for rule in self.rules}
        if len(anchors) != len(self.anchors) or len(claims) != len(self.claims) or len(rules) != len(self.rules):
            raise ValueError("Duplicate graph anchor, claim or rule ID")
        if (set(claims) & set(rules)) or ((set(claims) | set(rules)) & set(anchors)):
            raise ValueError("Evidence, claim and strategy IDs must be disjoint")
        used_anchors, used_claims = set(), set()
        for item in (*self.claims, *self.rules):
            ids = {link.evidence_id for link in item.evidence_links}
            if not ids <= set(anchors):
                raise ValueError("An interpretation references unknown source evidence")
            used_anchors.update(ids)
        for claim in self.claims:
            if claim.epistemic_status == "multi_source_interpretation":
                supports = [anchors[link.evidence_id] for link in claim.evidence_links if link.relation == "supports"]
                if len({anchor.source_event_id for anchor in supports}) < 2 or len({anchor.experience_id for anchor in supports}) < 2:
                    raise ValueError("Multi-source interpretation needs independent source events and experience groups")
        for rule in self.rules:
            if not set(rule.claim_ids) <= set(claims):
                raise ValueError("Strategy references an unknown value, belief or interaction style")
            used_claims.update(rule.claim_ids)
            if rule.review_status == "reviewed" and any(claims[key].review_status != "reviewed" for key in rule.claim_ids):
                raise ValueError("Reviewed strategies cannot depend on draft interpretations")
        if used_anchors != set(anchors) or used_claims != set(claims):
            raise ValueError("Every anchor and claim must participate in a strategy")
        return self


@dataclass(frozen=True)
class RulePath:
    rule: ConditionalRule
    claims: tuple[PersonaClaim, ...]
    evidence_ids: tuple[str, ...]
    matched_terms: tuple[str, ...]

    @property
    def value_id(self) -> str:
        # Retain the v1 prompt-budget/trace protocol; this is a strategy ID.
        return self.rule.id


@dataclass(frozen=True)
class RuleSelection(GraphSelection):
    paths: tuple[RulePath, ...]
    anchors: tuple[RuleAnchor, ...]

    def payload(self, *, format: Literal["flat", "graph"], supplied_ids: Iterable[str]) -> dict | None:
        if format not in {"flat", "graph"}:
            raise ValueError("Persona memory format must be flat or graph")
        supplied = set(supplied_ids)
        paths = [path for path in self.paths if set(path.evidence_ids) <= supplied]
        if not paths:
            return None
        claims = {claim.id: claim for path in paths for claim in path.claims}
        rules = {path.rule.id: path.rule for path in paths}
        nodes = [
            {"id": claim.id, "kind": claim.kind, "text": claim.text,
             "epistemic_status": claim.epistemic_status}
            for claim in claims.values()
        ] + [
            {"id": rule.id, "kind": rule.kind, "text": rule.text,
             "conditions": list(rule.conditions), "exceptions": list(rule.exceptions)}
            for rule in rules.values()
        ]
        relations = []
        for item in (*claims.values(), *rules.values()):
            for link in item.evidence_links:
                relations.append({"source": link.evidence_id, "relation": link.relation,
                                  "target": item.id, "rationale": link.rationale})
        for rule in rules.values():
            for key in rule.claim_ids:
                relations.append({"source": key, "relation": "informs", "target": rule.id})
        result = {
            "schema_version": "persona-memory.v2", "graph_id": self.graph_id,
            "provenance": "Source-grounded editorial interpretations, not verified mental states. Apply only when conditions hold; exceptions take priority. New advice is hypothetical, not an actual statement or memory.",
            "rule_ids": list(rules), "nodes": nodes,
        }
        if format == "graph":
            result["relations"] = relations
        else:
            result["relation_statements"] = [
                f"{edge['source']} {edge['relation']} {edge['target']}"
                + (f": {edge['rationale']}" if "rationale" in edge else "")
                for edge in relations
            ]
        return result

    def trace(self) -> dict:
        supports = {link.evidence_id for path in self.paths
                    for item in (*path.claims, path.rule) for link in item.evidence_links
                    if link.relation == "supports"}
        return {
            "schema_version": "persona-graph.v2", "graph_id": self.graph_id,
            "graph_sha256": self.graph_sha256,
            "value_ids": [path.value_id for path in self.paths],
            "rule_ids": [path.rule.id for path in self.paths],
            "claim_ids": list(dict.fromkeys(claim.id for path in self.paths for claim in path.claims)),
            "evidence_ids": list(self.evidence_ids),
            "matched_terms": sorted({term for path in self.paths for term in path.matched_terms}),
            "support_event_count": len({a.source_event_id for a in self.anchors if a.evidence_id in supports}),
            "support_experience_count": len({a.experience_id for a in self.anchors if a.evidence_id in supports}),
            "reason": self.reason,
            "selection_method": "editorial_bilingual_condition_groups",
            "condition_check": "lexical_relevance_only; answer_must_check_conditions_and_exceptions",
        }


def _rule_contains(query: str, term: str) -> bool:
    """Match whole words with bounded regular English inflections.

    This is deliberately not prefix or semantic matching: ``test`` does not
    match ``latest`` or ``testament``. Korean uses the existing substring rule
    for agglutinated forms. Multiword cues retain their explicit phrasing.
    """
    if _contains(query, term):
        return True
    word = _normalize(term)
    if not re.fullmatch(r"[a-z]{4,}", word):
        return False
    variants = {word + "s", word + "ed", word + "ing"}
    if word.endswith("e"):
        variants.add(word + "d")
        variants.add((word if word.endswith("ee") else word[:-1]) + "ing")
    if word.endswith("y") and word[-2] not in "aeiou":
        variants.update({word[:-1] + "ies", word[:-1] + "ied"})
    if word[-1] not in "aeiouwxy" and word[-2] in "aeiou" and word[-3] not in "aeiou":
        variants.update({word + word[-1] + "ed", word + word[-1] + "ing"})
    return any(re.search(r"(?<!\w)" + re.escape(form) + r"(?!\w)", query) for form in variants)


class PersonaRuleGraph(PersonaGraph):
    def __init__(self, document: RulesDocument, corpus: Corpus):
        self.document = document
        self._corpus = corpus
        self.sha256 = content_sha256(document.model_dump(mode="json"))
        self._validate_binding()

    def _validate_binding(self) -> None:
        _validate_anchors(self.document.persona_id, self.document.anchors, self._corpus)
        for anchor in self.document.anchors:
            if anchor.source_event_id != source_event_id(self._corpus.sources[anchor.source_id]):
                raise PersonaGraphError("Graph source event does not match bound source metadata")
            experience = (self._corpus.evidence[anchor.evidence_id].model_extra or {}).get("experience_id")
            if experience is not None and experience != anchor.experience_id:
                raise PersonaGraphError("Graph experience group does not match bound evidence metadata")

    def select(self, question: str, *, eligible_ids: Iterable[str] | None = None, limit: int = 2) -> RuleSelection:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 3:
            raise ValueError("Persona path limit must be between 1 and 3")
        if not isinstance(question, str) or len(question) > 10000:
            raise ValueError("Persona query must be text of at most 10000 characters")
        self._validate_binding()
        def result(paths=(), reason=None):
            paths = tuple(paths)
            ids = tuple(dict.fromkeys(key for path in paths for key in path.evidence_ids))
            anchors = tuple(anchor for anchor in self.document.anchors if anchor.evidence_id in ids)
            return RuleSelection(paths, ids, reason, self.document.id, self.sha256, self.document.persona_id, anchors)
        query = _normalize(question)
        reason = _advice_policy_reason(query)
        if reason:
            return result(reason=reason)
        eligible = set(eligible_ids) if eligible_ids is not None else {a.evidence_id for a in self.document.anchors}
        claims = {claim.id: claim for claim in self.document.claims}
        candidates = []
        for rule in self.document.rules:
            if rule.review_status != "reviewed" or any(_rule_contains(query, term) for term in rule.exclusion_terms):
                continue
            matches = [tuple(term for term in group if _rule_contains(query, term)) for group in rule.retrieval_groups]
            if not all(matches):
                continue
            selected_claims = tuple(claims[key] for key in rule.claim_ids)
            ids = tuple(dict.fromkeys(link.evidence_id for item in (rule, *selected_claims) for link in item.evidence_links))
            if not set(ids) <= eligible:
                continue  # Do not strip contrary evidence to keep a path eligible.
            matched = tuple(sorted({term for group in matches for term in group}))
            candidates.append(RulePath(rule, selected_claims, ids, matched))
        # Rank by observed cues, using conjunction specificity only to break a
        # tie. A generic two-group building rule must not displace several
        # explicit mentoring cues merely because its schema has more groups.
        candidates.sort(key=lambda path: (-len(path.matched_terms), -len(path.rule.retrieval_groups), path.rule.id))
        chosen = candidates[:limit]
        return result(chosen, None if chosen else "no_applicable_source_supported_rule")
