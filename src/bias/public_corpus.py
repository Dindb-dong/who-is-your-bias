"""Overlay public evidence without replacing reviewed private dialogue.

This module is deliberately pure: callers decide where to materialize its result.
An event is the unit of exclusion. A conflicting mirror cannot enter retrieval
through another source belonging to the same incoming event.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Mapping

from .corpus import Corpus, CorpusError, canonical_source_url, source_event_id
from .snapshots import corpus_digest


VERSION = "public-corpus-overlay.v1"


@dataclass(frozen=True)
class PublicCorpusMerge:
    corpus: Corpus
    report: dict


def split_manifest(corpus: Corpus) -> dict:
    """Retain frozen reservations, including sources not materialized locally."""
    assignments = dict(corpus.source_assignments or {})
    assignments.update({key: (source_event_id(source), source.split)
                        for key, source in corpus.sources.items()})
    events = {}
    for sid, (event, split) in sorted(assignments.items()):
        row = events.setdefault(event, {"source_event_id": event, "split": split, "source_ids": []})
        if row["split"] != split:
            raise CorpusError(f"Source event spans multiple splits: {event}")
        row["source_ids"].append(sid)
    return {"schema_version": "source-event-split.v1", "events": list(events.values())}


def _counts(corpus: Corpus) -> dict:
    return {"sources": len(corpus.sources), "records": len(corpus.evidence),
            "training_records": len(corpus.training_evidence),
            "question_answers": sum(row.question_answer is not None for row in corpus.evidence.values()),
            "heldout_sources": sum(source.split != "train" for source in corpus.sources.values())}


def merge_public_corpus(original: Corpus, incoming: Corpus, *,
                        reserved_events: Iterable[str] = (), reserved_urls: Iterable[str] = (),
                        reserved_assignments: Mapping[str, tuple[str, str]] | None = None) -> PublicCorpusMerge:
    """Keep original objects; add distinct reviewed records from eligible events.

    Source identity is an existing ID or canonical URL. When a public alias maps
    to a private source, the private publisher metadata and event identity win.
    Only incoming references are rebound; original Q&A text and metadata never
    change. Conflicting event components are excluded before adding any record.
    """
    original.validate_split_isolation()
    incoming.validate_split_isolation()
    if original.persona.id != incoming.persona.id:
        raise CorpusError("Public corpus persona differs from the private corpus")
    assignments = dict(original.source_assignments or {})
    assignments.update({key: (source_event_id(source), source.split)
                        for key, source in original.sources.items()})
    for sid, assignment in (reserved_assignments or {}).items():
        if assignment[1] not in {"validation", "test"}:
            raise CorpusError("External reservations must be validation or test")
        if sid in assignments and assignments[sid] != assignment:
            raise CorpusError(f"Existing source conflicts with a frozen reservation: {sid}")
        assignments[sid] = assignment
    reserved = set(reserved_events)
    heldout_urls = {canonical_source_url(url) for url in reserved_urls if url}
    for sid, (event, split) in assignments.items():
        if split != "train":
            reserved.update((sid, event))
    by_url = defaultdict(list)
    for source in original.sources.values():
        key = canonical_source_url(str(source.url))
        by_url[key].append(source)
        if source.split != "train":
            heldout_urls.add(key)

    groups = defaultdict(list)
    for source in incoming.sources.values():
        groups[source_event_id(source)].append(source)
    aliases = defaultdict(set)
    reasons = defaultdict(set)
    targets = defaultdict(set)
    matches = {}
    for event, sources in groups.items():
        aliases[("event", event)].add(event)
        for source in sources:
            key = canonical_source_url(str(source.url))
            aliases[("url", key)].add(event)
            aliases[("source", source.id)].add(event)
            if source.split != "train":
                reasons[event].add("incoming_non_train")
            if event in reserved or source.id in reserved or key in heldout_urls:
                reasons[event].add("heldout_event_or_url")
            frozen = assignments.get(source.id)
            if frozen and frozen[1] != source.split:
                reasons[event].add("frozen_split_conflict")
            if frozen and source.id not in original.sources and frozen[0] != event:
                reasons[event].add("frozen_event_conflict")
            candidates = list(by_url.get(key, ()))
            existing = original.sources.get(source.id)
            if existing is not None and all(item.id != existing.id for item in candidates):
                candidates.insert(0, existing)
            if existing is not None and (source_event_id(existing) != event
                                        and canonical_source_url(str(existing.url)) != key):
                reasons[event].add("source_identity_conflict")
            for candidate in candidates:
                target = source_event_id(candidate)
                aliases[("event", target)].add(event)
                targets[event].add(target)
                if candidate.split != source.split:
                    reasons[event].add("existing_split_conflict")
            if event in {assignment[0] for assignment in assignments.values()}:
                targets[event].add(event)
            if candidates:
                matches[source.id] = (existing if existing is not None else
                                      sorted(candidates, key=lambda item: item.id)[0]).id
        if len(targets[event]) > 1:
            reasons[event].add("ambiguous_existing_event")

    # Closure across shared URLs and known existing events prevents a renamed
    # incoming event from bypassing a conflict discovered in another mirror.
    neighbors = defaultdict(set)
    for events in aliases.values():
        for event in events:
            neighbors[event].update(events)
    canonical_events = {}
    for start in sorted(groups):
        if start in canonical_events:
            continue
        component, pending = set(), [start]
        while pending:
            event = pending.pop()
            if event not in component:
                component.add(event)
                pending.extend(neighbors[event] - component)
        existing_events = set().union(*(targets[event] for event in component))
        if len(existing_events) > 1:
            for event in component:
                reasons[event].add("ambiguous_existing_event")
        canonical_event = next(iter(existing_events)) if len(existing_events) == 1 else min(component)
        canonical_events.update({event: canonical_event for event in component})
    excluded = {event for event, values in reasons.items() if values}
    queue = list(excluded)
    while queue:
        event = queue.pop()
        for related in neighbors[event] - excluded:
            reasons[related].add("related_event_excluded")
            excluded.add(related)
            queue.append(related)

    sources = dict(original.sources)
    evidence = dict(original.evidence)
    source_map = {}
    added_sources = []
    url_targets = {key: sorted(values, key=lambda item: item.id)[0].id for key, values in by_url.items()}
    for event, rows in groups.items():
        if event in excluded:
            continue
        canonical_event = canonical_events[event]
        for source in rows:
            key = canonical_source_url(str(source.url))
            target_id = matches.get(source.id) or url_targets.get(key)
            if target_id is None:
                target_id = source.id
                source = source.model_copy(update={"source_event_id": canonical_event})
                sources[target_id] = source
                added_sources.append(target_id)
            url_targets[key] = target_id
            source_map[source.id] = target_id
            actual = sources[target_id]
            assignments[target_id] = (source_event_id(actual), actual.split)
    added_records, duplicates = [], []
    for record in incoming.evidence.values():
        if record.id in evidence:
            duplicates.append(record.id)
            continue
        if record.source_id not in source_map:
            continue
        target_id = source_map[record.source_id]
        event = source_event_id(sources[target_id])
        updates = {"source_id": target_id}
        if "source_event_id" in (record.model_extra or {}):
            updates["source_event_id"] = event
        if record.question_answer is not None and record.question_answer.source_event_id != event:
            updates["question_answer"] = record.question_answer.model_copy(update={"source_event_id": event})
        evidence[record.id] = record.model_copy(update=updates)
        added_records.append(record.id)

    manifest = deepcopy(original.manifest or {})
    public_id = (incoming.manifest or {}).get("id", "reviewed-public-corpus")
    manifest.update(id=f"{manifest.get('id', 'private-corpus')}+{public_id}",
                    materialization="private_reviewed_dialogue_with_public_evidence_overlay",
                    public_overlay={"version": VERSION, "public_corpus_id": public_id,
                                    "original_corpus_sha256": corpus_digest(original),
                                    "public_corpus_sha256": corpus_digest(incoming),
                                    "added_sources": len(added_sources), "added_records": len(added_records)})
    # Never replace the original captured-transcript hash with a public summary.
    hashes = manifest.setdefault("source_content_sha256", {})
    for sid in added_sources:
        value = (incoming.manifest or {}).get("source_content_sha256", {}).get(sid)
        if value:
            hashes[sid] = value
    merged = Corpus(original.persona, sources, evidence, manifest, assignments)
    merged.validate_split_isolation()
    split_manifest(merged)
    manifest["corpus_sha256"] = corpus_digest(merged)
    report = {"schema_version": VERSION, "status": "integrated", "public_corpus_id": public_id,
              "original_corpus_sha256": corpus_digest(original),
              "public_corpus_sha256": corpus_digest(incoming),
              "merged_corpus_sha256": manifest["corpus_sha256"],
              "before": _counts(original), "after": _counts(merged),
              "added_source_ids": sorted(added_sources), "added_evidence_ids": sorted(added_records),
              "preserved_original_evidence_ids": sorted(duplicates),
              "reused_source_ids": {sid: target for sid, target in sorted(source_map.items())
                                    if target in original.sources},
              "excluded_events": [{"source_event_id": event,
                                   "source_ids": sorted(row.id for row in groups[event]),
                                   "evidence_ids": sorted(row.id for row in incoming.evidence.values()
                                                          if source_event_id(incoming.sources[row.source_id]) == event),
                                   "reasons": sorted(reasons[event])} for event in sorted(excluded)],
              "reserved_events": sorted(reserved),
              "scope": "Event/URL/ID boundaries, not semantic duplicate detection or a persona quality score."}
    return PublicCorpusMerge(merged, report)
