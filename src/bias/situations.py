"""One versioned vocabulary for reviewed records and transparent query routing.

These labels describe observable conversational requests, not hidden emotional
states. Rules are a declared baseline, not a trained intention classifier.
"""

from __future__ import annotations

import re

TAXONOMY_VERSION = "situations.v1"
SITUATION_LABELS = frozenset({"explanation", "reflection", "praise", "challenge", "advice", "humor", "disclosure_request", "conversation"})

# Specific requests precede generic "why/how" explanations. English words use
# boundaries: "meanwhile" cannot match "mean", "enterprise" cannot match "praise".
_RULES = (
    ("challenge", r"\b(?:disagree|wrong|critici[sz]e|criticism|critics|objection|unnecessary|skeptical|sceptical|isn't|aren't|doesn't)\b|\b(?:not true|too expensive|waste of money|why not)\b|반대|틀렸|비판|반박|너무 비싸"),
    ("disclosure_request", r"\b(?:spoiler|confidential|reveal|secret)\b|\b(?:who did you make the bet with|can you show us|anything you could show us)\b|스포|비밀|공개해"),
    ("praise", r"\b(?:admire|amazing|inspiring|impressive|spectacular|congratulations|beautiful|brilliant)\b|\b(?:well done|thank you for)\b|멋져|대단|존경|축하"),
    ("reflection", r"\b(?:regret|failure|failed|setback|learned|hardship|lonely)\b|\b(?:look back|looking back|hardest time)\b|실패|배웠|후회|돌이켜"),
    ("advice", r"\b(?:advice|recommend)\b|\b(?:what should i|how should i|can someone learn|can i learn)\b|조언|추천|어떻게 해야"),
    ("humor", r"\b(?:joke|joking|kidding|funny)\b|\bwhy are you boring\b|농담|웃긴|유머"),
    ("explanation", r"\b(?:why|how|explain|what|when|where|which)\b|왜|어떻게|설명|무엇|어떤"),
)
COMPILED_RULES = tuple((label, re.compile(pattern, re.I)) for label, pattern in _RULES)


def classify_situation(text: str) -> str:
    for label, pattern in COMPILED_RULES:
        if pattern.search(text):
            return label
    return "conversation"


def validate_situations(labels: list[str]) -> None:
    if not labels or len(set(labels)) != len(labels) or any(label not in SITUATION_LABELS for label in labels):
        raise ValueError(f"Q&A labels must be unique members of {sorted(SITUATION_LABELS)}")
