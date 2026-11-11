"""Read source-linked records without mixing train and held-out material."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import Citation, Evidence, Persona, Source
from .situations import SITUATION_LABELS, TAXONOMY_VERSION, validate_situations


class CorpusError(ValueError):
    pass


def projection_eligible(record: Evidence, projection: str) -> bool:
    flags = (record.model_extra or {}).get("projection_eligibility")
    if flags is None:
        return True  # Historical corpora retain their frozen eligibility.
    expected = {"factual", "text_style", "aligned_speech", "visible_behavior"}
    if not isinstance(flags, dict) or set(flags) != expected or any(type(value) is not bool for value in flags.values()):
        raise CorpusError(f"Invalid independent projection eligibility: {record.id}")
    return flags[projection]


def source_event_id(source: Source) -> str:
    """Return the reviewed event identity, or a source-local legacy identity.

    Event IDs are curator supplied: a publisher, date, or similar topic is not
    enough to infer that two records came from the same recording.
    """
    value = (source.model_extra or {}).get("source_event_id", source.id)
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value):
        raise CorpusError(f"Invalid source_event_id for source {source.id}")
    return value


def canonical_source_url(value: str) -> str:
    """Compare source URLs without tracking parameters or in-page anchors.

    Content-bearing parameters (for example a video ID or transcript view) are
    preserved; this is a duplicate check, not a guess about URL redirects.
    """
    parsed = urlsplit(value)
    query = [
        (key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"feature", "si"}
    ]
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/") or "/", urlencode(sorted(query)), ""))


@dataclass(frozen=True)
class Corpus:
    persona: Persona
    sources: dict[str, Source]
    evidence: dict[str, Evidence]
    manifest: dict | None = None
    source_assignments: dict[str, tuple[str, str]] | None = None

    @classmethod
    def load(cls, directory: Path, *, check_splits: bool = True) -> "Corpus":
        """Load validated records, enforcing split isolation by default.

        Inspection tools may opt out of the final split guard to report leaks.
        That does not authorize indexing: training_evidence checks it again.
        """
        try:
            source_data = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
            if not isinstance(source_data, list):
                raise CorpusError("sources.json must be an array")
            sources: dict[str, Source] = {}
            for item in source_data:
                source = Source.model_validate(item)
                source_event_id(source)
                if source.id in sources:
                    raise CorpusError(f"Duplicate source id: {source.id}")
                sources[source.id] = source
            evidence: dict[str, Evidence] = {}
            with (directory / "evidence.jsonl").open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        record = Evidence.model_validate_json(line)
                    except ValueError as exc:
                        raise CorpusError(f"Invalid evidence at line {number}: {exc}") from exc
                    if record.id in evidence:
                        raise CorpusError(f"Duplicate evidence id: {record.id}")
                    if record.source_id not in sources:
                        raise CorpusError(f"Unknown source {record.source_id} for evidence {record.id}")
                    if record.question_answer:
                        if record.question_answer.source_event_id != source_event_id(sources[record.source_id]):
                            raise CorpusError(f"Q&A event differs from its source: {record.id}")
                        validate_situations(record.situations)
                    evidence[record.id] = record
            persona = Persona.model_validate_json((directory / "profile.json").read_text(encoding="utf-8"))
            manifest_path = directory / "corpus-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
            if manifest is not None and not isinstance(manifest, dict):
                raise CorpusError("corpus-manifest.json must be an object")
            split_path = directory / "split-manifest.json"
            source_assignments = None
            if split_path.exists():
                split = json.loads(split_path.read_text(encoding="utf-8"))
                if not isinstance(split, dict) or not isinstance(split.get("events"), list) or any(not isinstance(item, dict) or not all(key in item for key in ("source_event_id", "split", "source_ids")) or not isinstance(item["source_ids"], list) or not item["source_ids"] or any(not isinstance(key, str) for key in item["source_ids"]) for item in split["events"]):
                    raise CorpusError("Invalid frozen source-event manifest")
                declared = {item["source_event_id"]: item["split"] for item in split["events"]}
                source_events = {identifier: item["source_event_id"] for item in split["events"] for identifier in item["source_ids"]}
                if len(declared) != len(split["events"]) or sum(len(item["source_ids"]) for item in split["events"]) != len(source_events) or any(declared.get(source_event_id(source)) != source.split or source_events.get(source.id) != source_event_id(source) for source in sources.values()):
                    raise CorpusError("Source splits do not match the frozen event manifest")
                source_assignments = {identifier: (event_id, declared[event_id]) for identifier, event_id in source_events.items()}
        except (OSError, ValueError) as exc:
            if isinstance(exc, CorpusError):
                raise
            raise CorpusError(f"Cannot load corpus from {directory}: {exc}") from exc
        if not sources or not evidence:
            raise CorpusError("Corpus must contain sources and evidence")
        corpus = cls(persona, sources, evidence, manifest, source_assignments)
        if check_splits:
            corpus.validate_split_isolation()
        return corpus

    @property
    def training_evidence(self) -> list[Evidence]:
        # Also guard programmatically constructed or subsequently mutated corpora,
        # rather than relying only on the file loader to enforce split boundaries.
        self.validate_split_isolation()
        return self._eligible_training_evidence()

    def _eligible_training_evidence(self) -> list[Evidence]:
        return [
            record for record in self.evidence.values()
            if self.sources[record.source_id].split == "train"
            and record.review_status == "reviewed"
            and record.confidence == "source_grounded"
            and projection_eligible(record, "factual")
        ]

    @property
    def style_evidence(self) -> list[Evidence]:
        self.validate_split_isolation()
        return [record for record in self.evidence.values()
                if self.sources[record.source_id].split == "train" and record.review_status == "reviewed"
                and record.confidence == "source_grounded" and record.question_answer is not None
                and record.question_answer.completeness == "complete_exchange"
                and projection_eligible(record, "text_style")]

    def quality_report(self) -> dict:
        """Report exact normalized duplicates and known event-family overlap.

        Similar meanings across independent interviews are not automatically
        duplicates or leaks. This report deliberately makes no semantic claim.
        """
        events: dict[str, list[Source]] = defaultdict(list)
        urls: dict[str, list[Source]] = defaultdict(list)
        texts: dict[str, list[Evidence]] = defaultdict(list)
        for source in self.sources.values():
            events[source_event_id(source)].append(source)
            urls[canonical_source_url(str(source.url))].append(source)
        for record in self.evidence.values():
            normalized = " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", record.text).casefold()))
            texts[normalized].append(record)
        event_groups = [
            {"source_event_id": event, "sources": sorted(s.id for s in sources), "splits": sorted({s.split for s in sources})}
            for event, sources in sorted(events.items()) if len(sources) > 1
        ]
        duplicate_urls = [
            {"url": url, "sources": sorted(s.id for s in sources), "splits": sorted({s.split for s in sources})}
            for url, sources in sorted(urls.items()) if len(sources) > 1
        ]
        return {
            "distinct_source_events": len(events),
            "sources_without_event_ids": sorted(s.id for s in self.sources.values() if "source_event_id" not in (s.model_extra or {})),
            "repeated_event_families": event_groups,
            "source_event_split_leaks": [group for group in event_groups if len(group["splits"]) > 1],
            "duplicate_urls": duplicate_urls,
            "url_split_leaks": [group for group in duplicate_urls if len(group["splits"]) > 1],
            "duplicate_evidence": [
                {"evidence_ids": sorted(e.id for e in records), "source_ids": sorted({e.source_id for e in records})}
                for records in texts.values() if len(records) > 1
            ],
            "scope": "Reviewed event IDs and exact normalized URL/text checks; not semantic duplicate detection or a persona-quality score.",
        }

    def validate_split_isolation(self) -> None:
        if self.source_assignments is not None:
            for source in self.sources.values():
                if self.source_assignments.get(source.id) != (source_event_id(source), source.split):
                    raise CorpusError(f"Source {source.id} differs from the frozen event manifest")
        quality = self.quality_report()
        if quality["source_event_split_leaks"]:
            events = ", ".join(group["source_event_id"] for group in quality["source_event_split_leaks"])
            raise CorpusError(f"Source event spans multiple splits: {events}")
        if quality["url_split_leaks"]:
            sources = ", ".join(source for group in quality["url_split_leaks"] for source in group["sources"])
            raise CorpusError(f"Canonical source URL spans multiple splits: {sources}")
        # Programmatic construction must not bypass the same Q&A contract.
        for record in self.evidence.values():
            projection_eligible(record, "factual")
            if record.question_answer:
                if record.question_answer.source_event_id != source_event_id(self.sources[record.source_id]):
                    raise CorpusError(f"Q&A event differs from its source: {record.id}")
                try:
                    validate_situations(record.situations)
                except ValueError as exc:
                    raise CorpusError(str(exc)) from exc

    def qa_report(self) -> dict:
        records = [record for record in self.evidence.values() if record.question_answer]
        training = [record for record in self._eligible_training_evidence() if record.question_answer]
        covered = sorted({label for record in training for label in record.situations})
        return {
            "schema_version": "qa-corpus-report.v1",
            "corpus_id": (self.manifest or {}).get("id"),
            "materialization": (self.manifest or {}).get("materialization", "legacy_summary_corpus"),
            "question_answer_records": len(records),
            "training_question_answers": len(training),
            "complete_exchanges": sum(record.question_answer.completeness == "complete_exchange" for record in records),
            "excerpts": sum(record.question_answer.completeness == "excerpt" for record in records),
            "taxonomy_version": TAXONOMY_VERSION,
            "covered_training_situations": covered,
            "uncovered_situations": sorted(SITUATION_LABELS.difference(covered)),
            "reviewer_types": sorted({record.question_answer.review.reviewer_type for record in records}),
        }

    def citation(self, evidence_id: str) -> Citation:
        record = self.evidence[evidence_id]
        source = self.sources[record.source_id]
        return Citation(
            evidence_id=record.id, source_id=source.id, title=source.title,
            url=str(source.url), locator=record.locator, text=record.text,
        )

    def stats(self) -> dict[str, int]:
        return {
            "sources": len(self.sources),
            # Keep counts inspectable by inventory/reporting tools even if a
            # programmatic edit has made the corpus invalid for indexing.
            "evidence": len(self._eligible_training_evidence()),
            "held_out_sources": sum(s.split != "train" for s in self.sources.values()),
            "total_evidence": len(self.evidence),
        }
