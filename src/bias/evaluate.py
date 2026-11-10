"""Repeatable retrieval checks; no synthetic persona-quality claims."""

from __future__ import annotations

import json
from pathlib import Path

from .corpus import Corpus
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
    rows = []
    for case in cases:
        for evidence_id in case.expected_evidence_ids + case.reference_evidence_ids:
            if evidence_id not in corpus.evidence:
                raise ValueError(f"Unknown evaluation evidence id: {evidence_id}")
        result = retriever.search(case.question, limit=limit, behavior=True)
        ids = [hit.evidence.id for hit in result.hits]
        sources = {hit.evidence.source_id for hit in result.hits}
        forbidden = sorted(sources.intersection(case.forbidden_source_ids))
        source_leak = any(corpus.sources[source].split != "train" for source in sources)
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
        rows.append({"id": case.id, "category": case.category, "question": case.question, "retrieved_ids": ids, "confidence": result.confidence, "passed": passed, "forbidden_sources": forbidden, "split_leak": source_leak, "note": note})
    graded = [row for row in rows if row["passed"] is not None]
    return {
        "evaluation_type": "retrieval_smoke_test",
        "retrieval_limit": limit,
        "corpus": corpus.stats(),
        "summary": {"graded": len(graded), "passed": sum(row["passed"] for row in graded), "failed": sum(not row["passed"] for row in graded), "ungraded": len(rows) - len(graded), "pass_rate": round(sum(row["passed"] for row in graded) / len(graded), 4) if graded else None, "split_leaks": sum(row["split_leak"] for row in rows)},
        "limitations": ["Curated questions are small-corpus smoke tests, not a held-out generalization benchmark.", "Held-out source content never enters the retrieval index.", "Persona fidelity, factual entailment, voice resemblance, gesture resemblance, and injection robustness require separate review."],
        "cases": rows,
    }
