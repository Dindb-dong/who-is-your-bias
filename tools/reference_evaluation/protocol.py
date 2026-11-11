"""Freeze reference cases and construct identical-instruction A/B/C conditions."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bias.corpus import Corpus, source_event_id
from bias.retrieval import Retriever, SearchResult
from bias.models import ChatRequest
from bias.compact_prompts import CONTEXT_PREFIX
from bias.qa_prompts import QA_PROMPT_VERSION, build_qa_messages
from bias.retrieval import RETRIEVER_VERSION
from bias.situations import TAXONOMY_VERSION
from tools.evaluation.run import fingerprint_corpus

PROTOCOL_VERSION = "actual-qa-reference.v1"
PROMPT_VERSION = QA_PROMPT_VERSION
PREFIX = CONTEXT_PREFIX
CONDITIONS = {"A": "profile", "B": "rag", "C": "behavior"}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def implementation_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = ["src/bias/models.py", "src/bias/corpus.py", "src/bias/retrieval.py", "src/bias/situations.py", "src/bias/qa_prompts.py", "tools/reference_evaluation/protocol.py", "tools/reference_evaluation/run.py"]
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def qa_data(record) -> dict:
    qa = record.model_dump(mode="json").get("question_answer")
    if not isinstance(qa, dict) or not all(isinstance(qa.get(key), str) and qa[key].strip() for key in ("question", "answer", "original_language", "source_event_id")):
        raise ValueError(f"{record.id}: requires original Q&A, not a summary")
    if record.review_status != "reviewed":
        raise ValueError(f"{record.id}: original Q&A has not been reviewed")
    return qa


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
    reference_id: str
    source_event_id: str
    language: Literal["en", "ko"]
    question_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    answer_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["actual-qa-reference.v1"] = PROTOCOL_VERSION
    id: str
    frozen_at: str
    corpus_sha256: str
    source_manifest_sha256: str
    implementation_sha256: dict[str, str]
    retriever_version: str
    prompt_version: str
    taxonomy_version: str
    evaluation_split: Literal["validation", "test"] = "test"
    selection_rule: Literal["all_reviewed_original_exchanges_in_split"] = "all_reviewed_original_exchanges_in_split"
    cases: list[Case] = Field(min_length=1, max_length=200)
    limitations: list[str]


def freeze_plan(corpus: Corpus, manifest: dict, *, plan_id: str, split: str = "test") -> Plan:
    corpus.validate_split_isolation()
    if split not in {"validation", "test"}:
        raise ValueError("Reference events must be outside training")
    cases = []
    for record in sorted(corpus.evidence.values(), key=lambda item: item.id):
        source = corpus.sources[record.source_id]
        if source.split != split:
            continue
        qa = qa_data(record)
        if qa["source_event_id"] != source_event_id(source):
            raise ValueError("Q&A and publisher source event disagree")
        cases.append(Case(id=record.id, reference_id=record.id, source_event_id=source_event_id(source), language=qa["original_language"], question_sha256=digest(qa["question"]), answer_sha256=digest(qa["answer"])))
    return Plan(id=plan_id, frozen_at=datetime.now(timezone.utc).isoformat(), corpus_sha256=fingerprint_corpus(corpus), source_manifest_sha256=digest(manifest), implementation_sha256=implementation_fingerprints(), retriever_version=RETRIEVER_VERSION, prompt_version=PROMPT_VERSION, taxonomy_version=TAXONOMY_VERSION, evaluation_split=split, cases=cases, limitations=[
        "A small source-event pilot, not evidence of population-level persona fidelity.",
        "Complete events and their declared mirrors are withheld from retrieval; pretraining exposure is unknown.",
        "Source excerpts and full locally materialized exchanges are different experiments and have different corpus hashes.",
        "Lexical overlap is diagnostic only; human and model-assisted judgments are reported separately.",
    ])


def validate_plan(plan: Plan, corpus: Corpus, manifest: dict) -> None:
    corpus.validate_split_isolation()
    if fingerprint_corpus(corpus) != plan.corpus_sha256 or digest(manifest) != plan.source_manifest_sha256:
        raise ValueError("Frozen corpus or split manifest changed; create a new experiment version")
    if implementation_fingerprints() != plan.implementation_sha256:
        raise ValueError("Frozen implementation changed; restore its revision or freeze a new experiment")
    if len({case.id for case in plan.cases}) != len(plan.cases):
        raise ValueError("Duplicate reference case")
    expected = {record.id for record in corpus.evidence.values() if corpus.sources[record.source_id].split == plan.evaluation_split}
    if {case.reference_id for case in plan.cases} != expected:
        raise ValueError("The plan must retain all reviewed exchanges in its frozen evaluation split")
    for case in plan.cases:
        record = corpus.evidence[case.reference_id]
        source = corpus.sources[record.source_id]
        qa = qa_data(record)
        if source.split != plan.evaluation_split or source_event_id(source) != case.source_event_id or qa["source_event_id"] != case.source_event_id:
            raise ValueError("A reference event entered training or changed identity")
        if digest(qa["question"]) != case.question_sha256 or digest(qa["answer"]) != case.answer_sha256 or qa["original_language"] != case.language:
            raise ValueError("Frozen reference text or language changed")
    for record in corpus.training_evidence:
        qa_data(record)


def build_condition(corpus: Corpus, case: Case, condition: str, *, retrieval_limit: int = 4, context_chars: int = 9000) -> tuple[list[dict], dict]:
    if condition not in CONDITIONS or not 1 <= retrieval_limit <= 10 or not 512 <= context_chars <= 20000:
        raise ValueError("Invalid declared condition or context budget")
    reference = qa_data(corpus.evidence[case.reference_id])
    question = reference["question"]
    result = SearchResult((), 0, "profile_only") if condition == "A" else Retriever(corpus).search(question, limit=retrieval_limit, behavior=condition == "C")
    for hit in result.hits:
        source = corpus.sources[hit.evidence.source_id]
        qa_data(hit.evidence)
        if source.split != "train" or source_event_id(source) == case.source_event_id:
            raise ValueError("Held-out event reached prompt construction")
    request = ChatRequest(message=question, mode=CONDITIONS[condition], language=case.language)
    messages = build_qa_messages(corpus.persona, request, result, [], corpus.sources, context_chars=context_chars)
    context = json.loads(messages[1]["content"][len(PREFIX):])
    examples = context["interview_examples"]
    candidates = {hit.evidence.id: hit for hit in result.hits}
    selected = []
    for example in examples:
        if example["id"] not in candidates:
            raise ValueError("Prompt contains a record outside the selected training set")
        hit = candidates[example["id"]]
        qa = qa_data(hit.evidence)
        if example["question"] != qa["question"] or example["answer"] != qa["answer"]:
            raise ValueError("The actual prompt modified the original exchange")
        selected.append({"id": hit.evidence.id, "source_event_id": source_event_id(corpus.sources[hit.evidence.source_id]), "score": hit.score, "exchange_sha256": digest(qa)})
    selected_ids = {item["id"] for item in selected}
    excluded = [{"id": hit.evidence.id, "reason": "whole_exchange_exceeds_context_budget"} for hit in result.hits if hit.evidence.id not in selected_ids]
    audit = {"selected": selected, "budget_exclusions": excluded, "retrieval_reason": result.reason, "prompt_sha256": digest(messages), "common_instructions_sha256": digest(messages[0]["content"]), "profile_sha256": digest(context["profile"]), "question_sha256": digest(question), "prompt_chars": sum(len(message["content"]) for message in messages), "example_chars": len(json.dumps(examples, ensure_ascii=False, separators=(",", ":"))), "reference_answer_in_context": False, "prompt_version": PROMPT_VERSION}
    return messages, audit


def strip_citations(text: str) -> str:
    return re.sub(r"\[source:[^\]]*\]", "", text).strip()


def lexical_diagnostics(answer: str, reference: str) -> dict:
    """No semantic/personality interpretation; preserve language and length effects."""
    a, b = re.findall(r"\w+", strip_citations(answer).casefold()), re.findall(r"\w+", reference.casefold())
    common = sum((Counter(a) & Counter(b)).values())
    precision, recall = common / len(a) if a else 0, common / len(b) if b else 0
    return {"token_unigram_f1": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0, "generated_tokens": len(a), "reference_tokens": len(b), "interpretation": "surface_overlap_only_not_persona_quality"}


def load_inputs(corpus_dir: Path, plan_path: Path) -> tuple[Corpus, Plan, dict]:
    corpus = Corpus.load(corpus_dir)
    manifest = json.loads((corpus_dir / "split-manifest.json").read_text())
    plan = Plan.model_validate_json(plan_path.read_text())
    validate_plan(plan, corpus, manifest)
    return corpus, plan, manifest
