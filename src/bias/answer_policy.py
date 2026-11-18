"""Deterministic presentation guidance, never a source of facts or a hard cut.

No extra model request is made. Targets are advisory; evidence restrictions and
the existing output review always win. Keep the default evaluation prompts frozen.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

VERSION = "adaptive-answer.v1"

# Only a standalone presentation request inherits the preceding USER topic.
# A new question containing "briefly" must not inherit an unrelated topic.
PRESENTATION_FOLLOWUP = re.compile(
    r"^(?:(?:no|actually)[,\s]+)?(?:please\s+)?(?:tell me more|(?:can you\s+)?(?:explain|elaborate)(?:\s+more|\s+in detail)?|"
    r"(?:make|keep) (?:it|that) (?:shorter|brief|short)|(?:answer|summari[sz]e)(?:\s+(?:it|that))?\s+"
    r"(?:briefly|in (?:one|a single) (?:sentence|line)|in detail)|more detail|briefly)"
    r"(?:,?\s+please)?[?!.\s]*$|"
    r"^(?:아니[,\s]+)?(?:(?:좀|조금)\s*)?(?:(?:더\s*)?(?:자세히|자세하게|구체적으로)|"
    r"(?:더\s*)?짧게|간단히|간략히|한\s*(?:문장|줄)(?:으로|로)|핵심만)"
    r"(?:\s*(?:설명|답변|대답|말|요약)(?:해|해\s*줘|해\s*주세요|해주세요|해줘|해요|해주세요)?)?[?!.\s]*$",
    re.I,
)


def is_presentation_followup(message: str) -> bool:
    return PRESENTATION_FOLLOWUP.fullmatch(message.strip()) is not None


_DETAIL_REQUESTS = (
    ("one_sentence", r"\bin (?:one|a single) (?:sentence|line)\b|한\s*(?:문장|줄)(?:으로|로)|1\s*(?:문장|줄)(?:으로|로)"),
    ("brief", r"\b(?:briefly|concisely)\b|\b(?:keep|make) (?:it|that) (?:short|shorter|brief)\b|짧게|간단히|간략히|핵심만"),
    ("detailed", r"\bin detail\b|\bdetailed (?:answer|explanation)\b|\b(?:elaborate|tell me more|explain more|more detail)\b|자세히|자세하게|구체적으로"),
)

_KINDS = (
    ("comparison", r"\b(?:compar(?:e|ison)|versus|vs\.?|trade[- ]?offs?|pros and cons|difference(?:s)? between)\b|비교|차이|장단점"),
    ("advice", r"\b(?:advice|advise|recommend|should I|should we|how (?:can|do) (?:I|we)|values?|leadership|early jobs?|experience(?:s)?)\b|조언|가치관|리더십|경험|어떻게\s*(?:해야|하면\s*좋)"),
    ("procedure", r"\b(?:pipeline|workflow|step[- ]by[- ]step|steps|procedure)\b|파이프라인|과정|절차|단계별"),
    ("explanation", r"\b(?:why|explain|how (?:does|do|did|is|are|can|would)|reason)\b|왜|이유|원리|어떻게|설명"),
)
_TARGETS = {"factual": (1, 2, 45), "explanation": (2, 4, 95),
            "comparison": (3, 5, 120), "procedure": (3, 5, 120), "advice": (2, 4, 95)}
_STRUCTURE = {
    "factual": ("Give the direct answer, then only a necessary qualification.", "직접 답한 뒤 꼭 필요한 단서만 덧붙이세요."),
    "explanation": ("State the reason or mechanism, then explain its consequence.", "핵심 이유나 원리를 먼저 말하고 그 결과를 설명하세요."),
    "comparison": ("State the main difference, then a benefit, a limitation and when each option fits.", "핵심 차이를 먼저 말하고 장점, 한계와 적합한 상황을 비교하세요."),
    "procedure": ("Give the overall flow first, then the key stages in order.", "전체 흐름을 먼저 요약하고 핵심 단계를 순서대로 설명하세요."),
    "advice": ("Give practical advice, then a relevant sourced experience or decision criterion if available.", "실용적인 조언부터 말하고 관련 근거가 있을 때만 경험이나 판단 기준을 연결하세요."),
}


@dataclass(frozen=True)
class AnswerPolicy:
    question_type: str
    detail: str
    min_sentences: int
    max_sentences: int
    target_words: int
    evidence_limited: bool = False

    def as_dict(self) -> dict:
        return {"version": VERSION, **asdict(self), "enforcement": "prompt_guidance"}

    def instruction(self, language: str) -> str:
        if language == "ko":
            length = ("1문장" if self.max_sentences == 1 else
                      f"{self.min_sentences}~{self.max_sentences}문장")
            # Korean no-evidence inference keeps its existing stricter 15-eojeol
            # format and label; detailed requests do not relax it.
            return (
                f" 이번 답변의 표현 정책 ({VERSION}): 앞의 일반적인 답변 길이 지침 대신 {length}을 목표로 하세요. "
                "첫 문장은 핵심 답을 담은 완결된 문장으로, 가능하면 15어절 이내로 쓰세요. "
                + _STRUCTURE[self.question_type][1] +
                " 인사, 질문 반복, 표와 긴 목록을 생략하세요. 근거가 부족하면 분량을 채우지 마세요. "
                "출처 표기, 시점, 불확실성, 가상 추론 표시와 근거가 없을 때의 더 엄격한 제한은 그대로 지키세요. "
                "자료에 없는 프로젝트 구현, 개인 경험이나 감정을 만들지 마세요."
            )
        length = ("1 sentence" if self.max_sentences == 1 else
                  f"{self.min_sentences}-{self.max_sentences} sentences")
        return (
            f" Turn presentation policy ({VERSION}): replace earlier generic answer-length guidance with "
            f"a target of {length}, about {self.target_words} words or fewer, excluding citation markers. "
            "Start with the key answer in a complete sentence, ideally at most 18 words. "
            + _STRUCTURE[self.question_type][0] +
            " Omit greetings, repeated questions, tables and long lists. Never pad to reach the target. "
            "Keep citations, dates, uncertainty, simulation labels and stricter no-evidence restrictions. "
            "Do not invent project implementation details, personal experiences or feelings."
        )


def select_answer_policy(message: str, *, language: str, anchor: str = "",
                         speculative: bool = False) -> AnswerPolicy:
    topic = anchor if anchor and is_presentation_followup(message) else message
    kind = next((kind for kind, pattern in _KINDS if re.search(pattern, topic, re.I)), "factual")
    # The current user request alone controls detail. The last recognized
    # instruction wins, e.g. "in detail; actually, in one sentence".
    matches = [(match.start(), detail) for detail, pattern in _DETAIL_REQUESTS
               for match in re.finditer(pattern, message, re.I)]
    detail = max(matches)[1] if matches else "auto"
    low, high, words = _TARGETS[kind]
    if detail == "one_sentence":
        low, high, words = 1, 1, 30
    elif detail == "brief":
        low, high, words = 1, 2, 45
    elif detail == "detailed":
        low, high, words = 3, 5, 120
    if speculative:
        low, high, words = 1, min(high, 1 if language == "ko" else 2), min(words, 45)
    return AnswerPolicy(kind, detail, low, high, words, speculative)
