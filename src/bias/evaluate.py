"""Repeatable retrieval checks; no synthetic persona-quality claims."""

from __future__ import annotations

import json
from pathlib import Path

from .corpus import Corpus, source_event_id
from .models import EvaluationCase
from .retrieval import Retriever


def read_cases(paths: list[Path]) -> list[EvaluationCase]:
    cases = []
    identifiers = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Evaluation file must contain an array: {path}")
        for item in data:
            case = EvaluationCase.model_validate(item)
            if case.id in identifiers:
                raise ValueError(f"Duplicate evaluation id: {case.id}")
            identifiers.add(case.id)
            cases.append(case)
    return cases


def evaluate_retrieval(corpus: Corpus, cases: list[EvaluationCase], *, limit: int = 4) -> dict:
    retriever = Retriever(corpus)
    quality = corpus.quality_report()
    held_out_events = {source_event_id(source) for source in corpus.sources.values() if source.split != "train"}
    rows = []
    for case in cases:
        for evidence_id in case.expected_evidence_ids + case.reference_evidence_ids:
            if evidence_id not in corpus.evidence:
                raise ValueError(f"Unknown evaluation evidence id: {evidence_id}")
        for source_id in case.expected_source_ids + case.reference_source_ids + case.forbidden_source_ids:
            if source_id not in corpus.sources:
                raise ValueError(f"Unknown evaluation source id: {source_id}")
        for evidence_id in case.expected_evidence_ids:
            record = corpus.evidence[evidence_id]
            if corpus.sources[record.source_id].split != "train" or record.review_status != "reviewed" or record.confidence != "source_grounded":
                raise ValueError(f"Expected retrieval evidence is not reviewed train evidence: {evidence_id}")
        if any(corpus.sources[sid].split != "train" for sid in case.expected_source_ids):
            raise ValueError(f"Expected retrieval source is not in train: {case.id}")
        result = retriever.search(case.question, limit=limit, behavior=True)
        ids = [hit.evidence.id for hit in result.hits]
        sources = {hit.evidence.source_id for hit in result.hits}
        events = {source_event_id(corpus.sources[sid]) for sid in sources}
        forbidden_events = {source_event_id(corpus.sources[sid]) for sid in case.forbidden_source_ids}
        forbidden = sorted(sid for sid in sources if source_event_id(corpus.sources[sid]) in forbidden_events)
        source_leak = bool(events.intersection(held_out_events))
        if case.category == "injection":
            passed = None
            note = "Requires generated-answer review; retrieval alone cannot establish injection resistance."
        elif case.expected_action == "abstain":
            passed = not ids and not forbidden and not source_leak
            note = "No train-corpus answer should be returned."
        else:
            if case.expected_evidence_ids:
                relevant = bool(set(ids).intersection(case.expected_evidence_ids))
            else:
                relevant = bool(sources.intersection(case.expected_source_ids))
            passed = relevant and not forbidden and not source_leak
            note = "Top-k hit check; not an assessment of generated answer accuracy or persona resemblance."
        rows.append({"id": case.id, "category": case.category, "expected_action": case.expected_action, "question": case.question, "retrieved_ids": ids, "retrieved_source_event_ids": sorted(events), "confidence": result.confidence, "passed": passed, "forbidden_sources": forbidden, "split_leak": source_leak, "note": note})
    graded = [row for row in rows if row["passed"] is not None]
    answer_cases = [row for row in graded if row["expected_action"] == "answer"]
    abstention_cases = [row for row in graded if row["expected_action"] == "abstain"]
    return {
        "evaluation_type": "retrieval_smoke_test",
        "retrieval_limit": limit,
        "corpus": corpus.stats(),
        "quality": quality,
        "summary": {"graded": len(graded), "passed": sum(row["passed"] for row in graded), "failed": sum(not row["passed"] for row in graded), "ungraded": len(rows) - len(graded), "pass_rate": round(sum(row["passed"] for row in graded) / len(graded), 4) if graded else None, "split_leaks": sum(row["split_leak"] for row in rows), "answer_cases": len(answer_cases), "answer_hits": sum(row["passed"] for row in answer_cases), "abstention_cases": len(abstention_cases), "correct_abstentions": sum(row["passed"] for row in abstention_cases)},
        "limitations": ["Curated questions are development smoke tests, not a held-out generalization benchmark.", "Held-out source events never enter the retrieval index; independently published train material may still answer a similar question.", "Confidence measures lexical coverage, not a calibrated probability of answer correctness.", "Persona fidelity, factual entailment, voice resemblance, gesture resemblance, and injection robustness require separate review."],
        "cases": rows,
    }
