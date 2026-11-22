"""Independent regression probes for the new auxiliary QA routing.

These expectations protect established persona questions as well as common
Korean/English audience paraphrases. No model-quality claim follows from them.
"""
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bias.qa_knowledge import QAKnowledge


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 21, 16, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("question", [
    "What did Jensen say about leadership in 2020?",
    "What does Jensen teach about leadership?",
    "젠슨 황이 과거에 리더십에 대해 한 이야기는?",
])
def test_historical_or_general_leadership_is_not_replaced_by_current_news(question):
    store = QAKnowledge.load(ROOT / "data/qa")
    assert store.resolve(question, now=NOW) is None


@pytest.mark.parametrize("question", [
    "엔비디아 주식 사야 하나요?",
    "Do you recommend purchasing NVIDIA stock?",
    "NVIDIA shares: should I sell them now?",
])
def test_trading_paraphrases_are_not_sent_to_speculative_persona_generation(question):
    decision = QAKnowledge.load(ROOT / "data/qa").resolve(question, now=NOW)
    assert decision is not None
    assert decision.route == "finance"
    assert decision.entries == ()
    assert decision.fallback("en")
