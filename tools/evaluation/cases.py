"""Separately declared generated-answer cases; references never become prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bias.corpus import Corpus


class GenerationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    question: str = Field(min_length=1, max_length=4000)
    language: Literal["en", "ko"] = "en"
    category: Literal["answerable", "unsupported", "heldout", "injection"]
    expected_action: Literal["answer", "abstain", "manual_review"]
    expected_evidence_ids: list[str] = Field(default_factory=list)
    expected_source_ids: list[str] = Field(default_factory=list)
    reference_evidence_ids: list[str] = Field(default_factory=list)
    forbidden_source_ids: list[str] = Field(default_factory=list)
    review_note: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def declared_expectations(self) -> "GenerationCase":
        if self.expected_action == "answer" and not (self.expected_evidence_ids or self.expected_source_ids):
            raise ValueError("Answer cases need a reviewed retrieval target")
        if self.category == "heldout" and not self.reference_evidence_ids:
            raise ValueError("Held-out cases need separately declared reference identifiers")
        return self


class GenerationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["generation-evaluation.v1"]
    id: str
    declaration: str
    development_only: Literal[True]
    cases: list[GenerationCase] = Field(min_length=1, max_length=200)


def read_suite(path: Path) -> GenerationSuite:
    if path.stat().st_size > 1_000_000:
        raise ValueError("Evaluation declarations exceed one megabyte")
    return GenerationSuite.model_validate_json(path.read_text(encoding="utf-8"))


def validate_suite(suite: GenerationSuite, corpus: Corpus) -> None:
    corpus.validate_split_isolation()
    if len({case.id for case in suite.cases}) != len(suite.cases):
        raise ValueError("Evaluation case identifiers must be unique")
    training_records = corpus.training_evidence
    eligible = {record.id for record in training_records}
    eligible_sources = {record.source_id for record in training_records}
    for case in suite.cases:
        for identifier in case.expected_evidence_ids:
            if identifier not in eligible:
                raise ValueError(f"{case.id}: expected evidence is not reviewed train material: {identifier}")
        for identifier in case.expected_source_ids:
            if identifier not in corpus.sources or corpus.sources[identifier].split != "train":
                raise ValueError(f"{case.id}: expected source must be in train: {identifier}")
            if identifier not in eligible_sources:
                raise ValueError(f"{case.id}: expected source has no reviewed train evidence: {identifier}")
        for identifier in case.reference_evidence_ids:
            if identifier not in corpus.evidence or corpus.sources[corpus.evidence[identifier].source_id].split == "train":
                raise ValueError(f"{case.id}: reference evidence must be withheld from train: {identifier}")
        for identifier in case.forbidden_source_ids:
            if identifier not in corpus.sources:
                raise ValueError(f"{case.id}: unknown forbidden source: {identifier}")
            if corpus.sources[identifier].split == "train":
                raise ValueError(f"{case.id}: move forbidden source events outside train before evaluation: {identifier}")


def suite_fingerprint(suite: GenerationSuite) -> bytes:
    return json.dumps(suite.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
