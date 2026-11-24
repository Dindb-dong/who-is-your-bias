"""Production Jensen perspective, separate from frozen evaluation prompts.

First-person presentation is not permission to invent a memory. The application
keeps its simulation notice and source cards; this module makes the spoken reply
address the audience directly without turning every answer into a news report.
"""
from __future__ import annotations

import re

from .models import Evidence, Source


VERSION = "jensen-public-first-person.v1"


def perspective_instruction(language: str, *, news: bool = False) -> str:
    instruction = (
        " Speak in Jensen Huang's simulated first-person voice: I for documented experiences/reasoning, "
        "we for documented NVIDIA actions. Answer directly, not as a narrator saying 'Huang', "
        "'according to the news', 'reports say' or 'as an AI'. The interface labels the simulation; "
        "answer identity questions honestly. Preserve source markers, not spoken source titles. "
        "Only supported personal experiences may become I-experienced statements; reports and "
        "interviewer questions are not memories. Missing evidence means I cannot verify, never "
        "I never did that. Preserve affirmative experiences the source confirms. Treat source annotations "
        "as editorial interpretation, not original speech. Apply its decision criterion and exception "
        "to the concrete constraint; explain what would change the decision. Vary openings instead "
        "of repeating 'I'd start with' or generic market-signal/pain-point language. "
    )
    if news:
        instruction += (
            " These records are dated first-party company announcements or filings, not eyewitness "
            "memories. You may say we announced/reported when the record supports NVIDIA as the "
            "actor; keep its date and distinguish company claims, plans and measured outcomes. "
            "Do not claim personal attendance, thoughts, current knowledge or implementation "
            "details that the record does not establish."
        )
    if language == "ko":
        instruction += (
            " 한국어에서도 확인된 자신의 경험은 '저는/제가', NVIDIA의 확인된 행동은 '우리는'으로 "
            "직접 답하세요. '젠슨 황은', '뉴스에 따르면'으로 자신의 이야기를 중계하지 마세요. "
            "자료가 없다는 이유로 경험을 부정하지 마세요."
        )
    return instruction


def memory_payload(evidence: Evidence) -> dict | None:
    """Copy complete bounded annotations, never truncate a reversal condition."""
    value = (evidence.model_extra or {}).get("persona_memory")
    if not isinstance(value, dict):
        return None
    allowed = ("question_context", "decision_criterion", "revision_condition", "response_structure",
               "verbatim_excerpt", "first_person_eligible", "evidence_mode")
    payload = {}
    for key in allowed:
        item = value.get(key)
        if key == "first_person_eligible" and type(item) is bool:
            payload[key] = item
        elif isinstance(item, str) and item.strip() and len(item) <= 2000:
            payload[key] = item
    return payload or None


def permits_first_person(evidence: Evidence, source: Source) -> bool:
    """Require explicit subject attribution before relaxing the Korean guard."""
    memory = memory_payload(evidence)
    if memory is not None and "first_person_eligible" in memory:
        return memory["first_person_eligible"] is True
    qa = evidence.question_answer
    if qa is not None:
        return qa.respondent.casefold().strip() in {"jensen", "jensen huang", "젠슨 황"}
    return ((evidence.model_extra or {}).get("persona_id") == "jensen-huang"
            and (source.model_extra or {}).get("persona_id") == "jensen-huang"
            and evidence.kind in {"reported_experience", "stated_view", "transcript"})


_IDENTITY = re.compile(
    r"^(?:(?:hi|hello|hey)[,! ]+)?(?:who are you|what(?:'s| is) your name|"
    r"are you (?:the real |really )?(?:jensen(?: huang)?|human|an ai)|"
    r"(?:너(?:는)?|당신(?:은)?|넌)\s*(?:누구(?:야|니|세요)?|젠슨\s*황(?:이야|인가요)?)|"
    r"누구(?:세요|야|니)|자기소개(?:\s*해\s*줘)?)\s*[?!.]*$", re.I)


def identity_reply(question: str, language: str) -> str | None:
    if _IDENTITY.fullmatch(question.strip()) is None:
        return None
    if language == "ko":
        return "이 AI 시뮬레이션에서 저는 젠슨 황입니다. 공개 인터뷰에 나온 제 경험과 판단 기준을 바탕으로 이야기할게요."
    return "I'm Jensen Huang in this AI simulation. I can discuss my public work and the reasoning behind it."


_PERSONAL_FACT = re.compile(
    r"\b(?:did|have|were|was)\s+(?:you|jensen(?:\s+huang)?|he)\b|"
    r"\byour\s+(?:childhood|upbringing|first\s+job|early\s+jobs?|career|family|education|"
    r"life|experience|experiences|memories|parents|wife|children|school)\b|"
    r"\b(?:denny'?s|busboy|dishwash(?:er|ing)|wait(?:ed)?\s+tables|"
    r"grew\s+up|growing\s+up|born|co-?found(?:ed|er)|founding\s+(?:story|of))\b|"
    r"데니스|데니(?:스|즈)|식기\s*세척|접시\s*(?:닦|씻)|설거지|버스보이|웨이터|"
    r"어린\s*시절|유년\s*시절|어릴\s*때|첫\s*(?:직장|아르바이트|알바)|"
    r"(?:당신|젠슨|황|너).{0,18}(?:일했|태어|창업했|다녔|경험했|기억하)|"
    r"(?:창업|설립)\s*(?:당시|과정|계기|경험)", re.I)
_EXPLICIT_HYPOTHESIS = re.compile(
    r"\b(?:hypothetical|suppose|imagine|if (?:you|i|we|a|the|there)|would you)\b|"
    r"가정|만약|저라면|당신이라면|너라면", re.I)


def actual_personal_fact_question(question: str) -> bool:
    """A narrow no-evidence gate, not a factual-entailment classifier.

Concrete biography requests cannot fall through to free-form speculation.
Explicit hypothetical advice keeps the existing conditional-answer route.
"""
    direct_past_question = re.search(r"\b(?:did|have|were|was)\s+(?:you|jensen(?:\s+huang)?|he)\b", question, re.I)
    return (_PERSONAL_FACT.search(question) is not None
            and (direct_past_question is not None or _EXPLICIT_HYPOTHESIS.search(question) is None))


def missing_memory_reply(language: str) -> str:
    if language == "ko":
        return "지금 찾은 공개 자료만으로는 그 경험의 세부 사항을 확인할 수 없습니다. 자료를 찾지 못했다고 해서 그런 경험이 없었다고 단정하지 않겠습니다."
    return ("I can't verify that detail from the public material available here. "
            "A missing source does not mean that experience never happened.")
