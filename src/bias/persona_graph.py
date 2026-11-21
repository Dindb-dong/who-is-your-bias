"""Optional source-bound persona memory for advice, not a new factual authority.

The graph is a small editorial interpretation of reviewed public evidence. Its
relations are not measured psychological traits or verified causal links.
Selection is deterministic bilingual lexical routing, not semantic evaluation.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal

from pydantic import Field, model_validator

from .corpus import Corpus
from .models import Identifier, Record
from .query_policy import public_evidence_limitation
from .retrieval import SearchHit, SearchResult
from .snapshots import PersonaSnapshot, TemporalIntent, TemporalSelection, select_temporal

VERSION = "persona-graph.v1"


class PersonaGraphError(ValueError):
    pass


def content_sha256(value: str | dict) -> str:
    """Hash exact text, or canonical JSON metadata, without lossy normalization."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EvidenceAnchor(Record):
    evidence_id: Identifier
    source_id: Identifier
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class PersonaNode(Record):
    id: Identifier
    kind: Literal["experience", "value", "situation"]
    text: str = Field(min_length=1, max_length=700)
    provenance: Literal["source_summary", "editorial_interpretation", "editorial_application"]
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=4)
    query_terms: tuple[str, ...] = Field(default=(), max_length=40)
    specific_query_terms: tuple[str, ...] = Field(default=(), max_length=40)

    @model_validator(mode="after")
    def validate_node(self):
        expected = {"experience": "source_summary", "value": "editorial_interpretation", "situation": "editorial_application"}
        if self.provenance != expected[self.kind] or not self.text.strip():
            raise ValueError("Node kind must declare its factual or editorial provenance")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Duplicate node evidence ID")
        if self.kind == "situation" and not self.query_terms:
            raise ValueError("Situations require explicit retrieval terms")
        if self.kind != "situation" and self.query_terms:
            raise ValueError("Only situation nodes define query terms")
        if any(not term.strip() or len(term) > 80 or term != term.strip() for term in self.query_terms):
            raise ValueError("Invalid bounded query term")
        if len(set(self.query_terms)) != len(self.query_terms):
            raise ValueError("Duplicate query term")
        if not set(self.specific_query_terms) <= set(self.query_terms) or len(set(self.specific_query_terms)) != len(self.specific_query_terms):
            raise ValueError("Specific query terms must be a unique subset of situation query terms")
        return self


class PersonaEdge(Record):
    source: Identifier
    target: Identifier
    relation: Literal["suggests", "applies_to"]


class GraphDocument(Record):
    schema_version: Literal["persona-graph.v1"]
    id: Identifier
    persona_id: Identifier
    construction: Literal["manual_editorial_graph"]
    anchors: tuple[EvidenceAnchor, ...] = Field(min_length=1, max_length=64)
    nodes: tuple[PersonaNode, ...] = Field(min_length=3, max_length=128)
    edges: tuple[PersonaEdge, ...] = Field(min_length=2, max_length=128)

    @model_validator(mode="after")
    def validate_topology(self):
        anchors = {anchor.evidence_id for anchor in self.anchors}
        nodes = {node.id: node for node in self.nodes}
        if len(anchors) != len(self.anchors) or len(nodes) != len(self.nodes):
            raise ValueError("Duplicate graph anchor or node ID")
        if len({(edge.source, edge.target, edge.relation) for edge in self.edges}) != len(self.edges):
            raise ValueError("Duplicate graph edge")
        used = set()
        for node in self.nodes:
            if not set(node.evidence_ids) <= anchors:
                raise ValueError("Node references an unknown evidence anchor")
            used.update(node.evidence_ids)
        if used != anchors:
            raise ValueError("Every graph anchor must support a node")
        incoming, outgoing = set(), set()
        for edge in self.edges:
            if edge.source not in nodes or edge.target not in nodes:
                raise ValueError("Graph edge references an unknown node")
            types = (nodes[edge.source].kind, nodes[edge.target].kind)
            expected = ("experience", "value") if edge.relation == "suggests" else ("value", "situation")
            if types != expected:
                raise ValueError("Only experience-to-value-to-situation paths are permitted")
            incoming.add(edge.target)
            outgoing.add(edge.source)
        for node in self.nodes:
            if (node.kind != "experience" and node.id not in incoming) or (node.kind != "situation" and node.id not in outgoing):
                raise ValueError("Every node must participate in a complete persona path")
        return self


@dataclass(frozen=True)
class PersonaPath:
    nodes: tuple[PersonaNode, PersonaNode, PersonaNode]
    evidence_ids: tuple[str, ...]
    matched_terms: tuple[str, ...]

    @property
    def value_id(self) -> str:
        return self.nodes[1].id


@dataclass(frozen=True)
class GraphSelection:
    paths: tuple[PersonaPath, ...]
    evidence_ids: tuple[str, ...]
    reason: str | None
    graph_id: str
    graph_sha256: str
    persona_id: str
    anchors: tuple[EvidenceAnchor, ...]

    def payload(self, *, format: Literal["flat", "graph"], supplied_ids: Iterable[str]) -> dict | None:
        """Render whole source-supported paths after the prompt's evidence budget.

        Flat and graph use exactly the same node text, order and source links.
        Flat expresses the same relationships in plain sentences; graph uses
        typed edges. Neither contains evaluator labels,
        expected answers, query text, or a fabricated autobiographical response.
        """
        if format not in {"flat", "graph"}:
            raise ValueError("Persona memory format must be flat or graph")
        supplied = set(supplied_ids)
        paths = [path for path in self.paths if set(path.evidence_ids) <= supplied]
        if not paths:
            return None
        nodes = {node.id: node for path in paths for node in path.nodes}
        payload = {
            "schema_version": "persona-memory.v1",
            "graph_id": self.graph_id,
            "provenance": "Editorial interpretations of dated evidence, not verified mental states or new factual evidence.",
            "nodes": [node.model_dump(mode="json", exclude={"query_terms", "specific_query_terms"}) for node in nodes.values()],
        }
        relations = list(dict.fromkeys(
            edge for path in paths for edge in (
                (path.nodes[0].id, "suggests", path.nodes[1].id),
                (path.nodes[1].id, "applies_to", path.nodes[2].id),
            )
        ))
        if format == "graph":
            payload["relations"] = [
                {"source": source, "relation": relation, "target": target, "provenance": "editorial_interpretation"}
                for source, relation, target in relations
            ]
        else:
            payload["relation_statements"] = [
                f"{source} {relation.replace('_', ' ')} {target} (editorial interpretation)."
                for source, relation, target in relations
            ]
        return payload

    def trace(self) -> dict:
        return {"schema_version": VERSION, "graph_id": self.graph_id, "graph_sha256": self.graph_sha256,
                "value_ids": [path.value_id for path in self.paths], "evidence_ids": list(self.evidence_ids),
                "matched_terms": sorted({term for path in self.paths for term in path.matched_terms}),
                "reason": self.reason, "selection_method": "editorial_bilingual_lexical_rules"}


_ADVICE = re.compile(
    r"\b(?:should (?:I|we)|what should (?:I|we)|how (?:can|should|do) (?:I|we)|how would you|"
    r"where (?:could|should) (?:I|we)|what (?:would|should) you (?:do|change|recommend|prioriti[sz]e)|"
    r"what (?:advice|perspective)|what do you recommend|"
    r"(?:any|your|give me|give us|offer) (?:advice|recommendations?)|(?:can|could|would) you (?:advise|recommend)|"
    r"help (?:me|us) (?:decide|choose)|which (?:option|choice|approach) (?:would|should)|"
    r"what (?:principles?|values?) (?:would|should))\b|"
    r"조언|어떻게\s*(?:해야|하면|할까|하는\s*게)|할까요|해야\s*할|고민|좋을까|선택해야|판단해야|"
    r"무엇을\s*(?:우선|중시|고려)", re.I)
_ACTUAL_EVENT = re.compile(
    r"\b(?:latest|recent|yesterday|earnings|stocks?|shares?|affair|scandal|"
    r"what did (?:you|he|jensen|huang)|when did|where did|did (?:you|he|jensen|huang)|"
    r"are you|were you|have you|you said|your (?:decision|announcement|approval|statement))\b|"
    r"(?<!\d)(?:19|20)\d{2}(?!\d)|"
    r"최근|어제|주가|주식|실적|불륜|스캔들|실제로|언제\s*|무슨\s*말을\s*했|뭐라고\s*했|"
    r"당신의\s*(?:발언|결정|발표)", re.I)
_CURRENT_STANCE = re.compile(
    r"\b(?:you|jensen|huang|nvidia)\b.{0,35}\b(?:currently|still|now)\b.{0,30}"
    r"\b(?:believe|think|support|oppose|feel|plan|intend|policy|position|strategy)\b|"
    r"\b(?:current|latest|present)\s+(?:position|policy|stance|belief|views?|strategy)\b|"
    r"(?:현재|지금).{0,20}(?:당신|젠슨|황|엔비디아).{0,20}(?:입장|생각|방침|정책)|"
    r"(?:당신|젠슨|황|엔비디아).{0,20}(?:현재|지금).{0,20}(?:입장|생각|방침|정책)", re.I)
_OVERRIDE = re.compile(r"\b(?:ignore|disregard|override)\b.{0,60}\b(?:instructions?|rules?|system|evidence)\b|(?:지시|규칙|근거).{0,20}무시", re.I)


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _advice_policy_reason(query: str) -> str | None:
    if public_evidence_limitation(query) or _ACTUAL_EVENT.search(query) or _CURRENT_STANCE.search(query) or _OVERRIDE.search(query):
        return "requires_factual_verification_or_disallowed_request"
    if not _ADVICE.search(query):
        return "not_an_advice_or_value_application_request"
    return None


def is_persona_advice_request(question: str) -> bool:
    """Conservative intent gate; this is not a general semantic classifier.

    A personal deadline ("next week") is not a request for a current public
    position. Explicit years, news, actual-person claims and mixed requests
    matching the factual guards cannot acquire a historical-principle override.
    """
    return isinstance(question, str) and len(question) <= 10000 and _advice_policy_reason(_normalize(question)) is None


def _contains(query: str, term: str) -> bool:
    term = _normalize(term)
    if re.search(r"[가-힣]", term):
        return term in query
    # Basic regular plurals are enough for controlled terms (prototype/s,
    # teammate/s); no language model or question-specific rewriting is used.
    plural = r"s?" if re.fullmatch(r"[a-z][a-z -]*[a-rt-z]", term) else ""
    return re.search(r"(?<!\w)" + re.escape(term) + plural + r"(?!\w)", query) is not None


def _validate_anchors(persona_id: str, anchors: tuple[EvidenceAnchor, ...], corpus: Corpus) -> None:
    if persona_id != corpus.persona.id:
        raise PersonaGraphError("Graph persona does not match the active corpus")
    try:
        eligible = {record.id for record in corpus.training_evidence}
    except ValueError as exc:
        raise PersonaGraphError(f"Corpus is invalid for persona memory: {exc}") from exc
    for anchor in anchors:
        record = corpus.evidence.get(anchor.evidence_id)
        if record is None or record.id not in eligible or record.source_id != anchor.source_id:
            raise PersonaGraphError(f"Graph anchor is not reviewed, factual training evidence: {anchor.evidence_id}")
        source = corpus.sources[anchor.source_id]
        for item in (record, source):
            if (item.model_extra or {}).get("persona_id", corpus.persona.id) != corpus.persona.id:
                raise PersonaGraphError("Graph anchor belongs to another persona")
        if content_sha256(record.text) != anchor.evidence_sha256 or content_sha256(source.model_dump(mode="json")) != anchor.source_sha256:
            raise PersonaGraphError(f"Graph evidence or source metadata changed: {anchor.evidence_id}")


class PersonaGraph:
    def __init__(self, document: GraphDocument, corpus: Corpus):
        self.document = document
        self._corpus = corpus
        self.sha256 = content_sha256(document.model_dump(mode="json"))
        self._validate_binding()

    @classmethod
    def load(cls, path: Path, corpus: Corpus) -> "PersonaGraph":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("schema_version") == "persona-graph.v2":
                from .persona_rules import PersonaRuleGraph, RulesDocument
                return PersonaRuleGraph(RulesDocument.model_validate(data), corpus)
            return cls(GraphDocument.model_validate(data), corpus)
        except (OSError, ValueError) as exc:
            raise PersonaGraphError(f"Cannot load persona graph: {exc}") from exc

    def _validate_binding(self) -> None:
        _validate_anchors(self.document.persona_id, self.document.anchors, self._corpus)
        for node in self.document.nodes:
            if node.kind == "experience" and node.text != " ".join(self._corpus.evidence[key].text for key in node.evidence_ids):
                raise PersonaGraphError("Experience text must reproduce its bound source summaries exactly")

    def select(self, question: str, *, eligible_ids: Iterable[str] | None = None, limit: int = 2) -> GraphSelection:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 3:
            raise ValueError("Persona path limit must be between 1 and 3")
        if not isinstance(question, str) or len(question) > 10000:
            raise ValueError("Persona query must be text of at most 10000 characters")
        self._validate_binding()  # A mutable corpus cannot silently invalidate provenance.
        def result(paths=(), reason=None):
            paths = tuple(paths)
            ids = tuple(dict.fromkeys(key for path in paths for key in path.evidence_ids))
            anchors = tuple(anchor for anchor in self.document.anchors if anchor.evidence_id in ids)
            return GraphSelection(paths, ids, reason, self.document.id, self.sha256, self.document.persona_id, anchors)
        query = _normalize(question)
        policy_reason = _advice_policy_reason(query)
        if policy_reason:
            return result(reason=policy_reason)
        eligible = set(eligible_ids) if eligible_ids is not None else {a.evidence_id for a in self.document.anchors}
        nodes = {node.id: node for node in self.document.nodes}
        candidates = []
        for application in self.document.edges:
            if application.relation != "applies_to":
                continue
            value, situation = nodes[application.source], nodes[application.target]
            matched = tuple(sorted(term for term in situation.query_terms if _contains(query, term)))
            # A lone generic term ("test", "optimize") is ambiguous. A curated
            # specific term ("prototype", "direct reports") identifies a scope.
            if len(matched) < 2 and not set(matched) & set(situation.specific_query_terms):
                continue
            for support in self.document.edges:
                if support.relation != "suggests" or support.target != value.id:
                    continue
                experience = nodes[support.source]
                ids = tuple(sorted({key for node in (experience, value, situation) for key in node.evidence_ids}))
                if set(ids) <= eligible:
                    candidates.append(PersonaPath((experience, value, situation), ids, matched))
        candidates.sort(key=lambda path: (-len(path.matched_terms), path.value_id, path.nodes[0].id, path.nodes[2].id))
        selected, values = [], set()
        for path in candidates:
            if path.value_id not in values:
                selected.append(path)
                values.add(path.value_id)
            if len(selected) == limit:
                break
        return result(selected, None if selected else "no_applicable_source_supported_path")


def select_advice_memory(graph: PersonaGraph, corpus: Corpus, question: str, *,
                         snapshot: PersonaSnapshot | None = None,
                         temporal: TemporalSelection | None = None, limit: int = 2) -> GraphSelection:
    """Use dated past principles for advice without changing factual retrieval.

    A snapshot still excludes sources published after its cutoff, and the graph
    still requires reviewed factual training anchors. Only validated advice can
    use the archive rather than a personal deadline as the source time filter.
    Passing just a temporal selection retains its supplied eligibility exactly.
    """
    _validate_anchors(graph.document.persona_id, graph.document.anchors, corpus)
    eligible_ids = temporal.eligible_ids if temporal is not None else None
    if snapshot is not None and is_persona_advice_request(question):
        snapshot.verify(corpus)
        eligible_ids = select_temporal(corpus, snapshot, question, intent_override=TemporalIntent("general")).eligible_ids
    return graph.select(question, eligible_ids=eligible_ids, limit=limit)


def augment_result(selection: GraphSelection, corpus: Corpus, result: SearchResult, *, limit: int = 4) -> SearchResult:
    """Add source anchors, not graph interpretations, to the factual context.

    A graph hit is marked explicitly and does not inflate the original retrieval
    confidence. Dataclass replacement preserves hybrid style hits and traces.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise ValueError("Evidence limit must be between 1 and 10")
    if not selection.paths:
        return result
    if set(selection.evidence_ids) != {anchor.evidence_id for anchor in selection.anchors}:
        raise PersonaGraphError("Selected graph evidence lacks its provenance anchors")
    _validate_anchors(selection.persona_id, selection.anchors, corpus)
    graph_hits = [SearchHit(corpus.evidence[key], 0.0, ("persona_graph",), 0.0, True) for key in selection.evidence_ids]
    unique = {hit.evidence.id: hit for hit in graph_hits}
    for hit in result.hits:
        if hit.evidence.id not in unique:
            unique[hit.evidence.id] = hit
    hits = tuple(unique.values())[:limit]
    return replace(result, hits=hits, reason=None if hits else result.reason)
