"""Frozen classifier for reproducing the published legacy/compact prompt runs.

Preserves their original substring/order behavior, including its limitations.
New original-Q&A retrieval uses bias.situations instead.
"""

_RULES = (
    ("explanation", ("why", "how", "explain", "왜", "어떻게", "설명")),
    ("reflection", ("regret", "failure", "learn", "실패", "배웠", "후회")),
    ("praise", ("admire", "amazing", "inspiring", "멋져", "대단", "존경")),
    ("challenge", ("disagree", "wrong", "critic", "반대", "틀렸", "비판")),
)


def classify_situation(text: str) -> str:
    lower = text.lower()
    for situation, phrases in _RULES:
        if any(phrase in lower for phrase in phrases):
            return situation
    return "conversation"
