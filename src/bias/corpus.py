"""Read source-linked records without mixing train and held-out material."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import Citation, Evidence, Persona, Source


class CorpusError(ValueError):
    pass


@dataclass(frozen=True)
class Corpus:
    persona: Persona
    sources: dict[str, Source]
    evidence: dict[str, Evidence]

    @classmethod
    def load(cls, directory: Path) -> "Corpus":
        try:
            source_data = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
            if not isinstance(source_data, list):
                raise CorpusError("sources.json must be an array")
            sources: dict[str, Source] = {}
            for item in source_data:
                source = Source.model_validate(item)
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
        return cls(persona, sources, evidence)

    @property
    def training_evidence(self) -> list[Evidence]:
        return [
            record for record in self.evidence.values()
            if self.sources[record.source_id].split == "train"
            and record.review_status == "reviewed"
            and record.confidence == "source_grounded"
        ]

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
            "evidence": len(self.training_evidence),
            "held_out_sources": sum(s.split != "train" for s in self.sources.values()),
            "total_evidence": len(self.evidence),
        }
