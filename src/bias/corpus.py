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


class CorpusError(ValueError):
    pass


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
                    evidence[record.id] = record
            persona = Persona.model_validate_json((directory / "profile.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            if isinstance(exc, CorpusError):
                raise
            raise CorpusError(f"Cannot load corpus from {directory}: {exc}") from exc
        if not sources or not evidence:
            raise CorpusError("Corpus must contain sources and evidence")
        corpus = cls(persona, sources, evidence)
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
        ]

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
        quality = self.quality_report()
        if quality["source_event_split_leaks"]:
            events = ", ".join(group["source_event_id"] for group in quality["source_event_split_leaks"])
            raise CorpusError(f"Source event spans multiple splits: {events}")
        if quality["url_split_leaks"]:
            sources = ", ".join(source for group in quality["url_split_leaks"] for source in group["sources"])
            raise CorpusError(f"Canonical source URL spans multiple splits: {sources}")

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
