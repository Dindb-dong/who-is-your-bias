"""Opt-in bilingual lexical candidate for the Jensen development corpus.

This is a finite concept dictionary, not a multilingual embedding model or a
claim-entailment classifier. Unknown content words remain in the coverage
denominator, and the public-evidence/current/year guards remain in Retriever.
"""
from __future__ import annotations

import re
import unicodedata

from .corpus import Corpus
from .retrieval import KOREAN_PARTICLES, STOPWORDS, Retriever, SearchHit

VERSION = "jensen-concept-lexical.v1"
NAMED_ANCHORS = frozenset({"cuda", "sega", "tsmc", "tpu", "gpu", "morris"})
NAMES = frozenset({"jensen", "huang", "nvidia", "젠슨", "황", "엔비디아"})
# These grammatical/request words carry no subject evidence. Keep technical or
# unfamiliar Korean terms; silently dropping all untranslated text overmatches.
NEUTRAL = frozenset("""
어떤 어떻게 왜 무엇 무슨 대해 대해서 관해 알려줘 설명 설명해 설명해줘 줘 해줘
말 말해 말해줘 말했어 말한 했어 했을 걸 때 건 것 거 그 그때 이 이런 그런
당시 당시의 당시에는 당시에도 당장 지금은 오래 크지 않았는데도 단순히 이상
이상으로 보다 보다는 것보다 잘하는 일 일은 어떻게는 뜻 뜻이야 이유 뭐야
궁금 궁금해 관점 관점에서 연결해서 기존 당시에는 통해 대해서도 무엇이
않고 않았어 않았나 있었다 있어 있다고 있다는 여러 하나 같은 당시에도
더 각각 각자 가장 직접 다시 처음 처음부터 따로 대신 보다보다 중요하게
거야 것인지 의미 의미인지 말인가요 설명해주세요 알려주세요
""".split())
ENGLISH_NEUTRAL = frozenset("jensen huang nvidia explain explanation mean meant importance prefer preferable especially exactly".split())

# Long roots first, with Korean suffixes handled only after a known concept.
# No document IDs, case IDs, retrieved ranks or held-out answers occur here.
CONCEPTS = {
    "소프트웨어": "software", "개발자": "developer", "생태계": "ecosystem",
    "아키텍처": "architecture", "그래픽": "graphic", "게임기": "game console",
    "병렬": "parallel", "가속": "accelerated", "쿠다": "cuda", "세가": "sega",
    "반도체": "semiconductor", "트랜지스터": "transistor", "공급망": "supply chain",
    "제조": "manufacturing", "양산": "manufacturing", "생산": "production",
    "모리스": "morris", "기술": "technology", "품질": "quality", "수율": "yield",
    "경쟁": "competitor", "차별화": "differentiate", "차별": "differentiate",
    "플랫폼": "platform", "칩": "chip", "투자": "invest", "개발": "develop",
    "매출": "revenue", "수익": "revenue", "시장": "market", "규모": "size",
    "전환": "change", "계약": "contract", "중단": "abandon", "포기": "abandon",
    "잘못": "wrong", "실패": "failure", "시작": "start", "결정": "decision",
    "판단": "reason", "의사결정": "decision", "제1원리": "first principle",
    "원리": "principle", "문제": "problem", "해결": "solve", "계산": "calculate",
    "리더십": "leadership", "투명": "transparent", "공개": "open",
    "직원": "employee", "조직": "organization", "생각": "think", "근거": "reason",
    "나누": "share", "만나": "meeting", "자리": "meeting", "일대일": "one on one",
    "중요": "important", "기회": "opportunity", "창업": "found company",
    "고통": "pain suffering", "어려움": "difficulty", "어려운": "difficulty",
    "회복": "resilience", "책임": "responsibility", "가치": "value",
    "이메일": "email", "메일": "email", "우선순위": "priority",
    "다섯": "five", "가지": "thing", "읽": "read", "보내": "send",
    "방식": "way", "대형": "large", "모델": "model", "학습": "train",
    "인공지능": "artificial intelligence", "데이터센터": "data center",
    "공장": "factory", "토큰": "token", "전력": "power", "컴퓨터": "computer",
    "시스템": "system", "설계": "design", "랙": "rack", "네트워크": "network",
}
# Canonical forms are shared by the English document and Korean query side.
FORMS = {
    "investing": "invest", "investment": "invest", "invested": "invest",
    "developing": "develop", "developed": "develop", "development": "develop",
    "founding": "found", "founded": "found", "founder": "found",
    "starting": "start", "started": "start", "abandoned": "abandon", "abandoning": "abandon",
    "reasoning": "reason", "thinking": "think", "transparency": "transparent",
    "differentiated": "differentiate", "differentiates": "differentiate", "differentiation": "differentiate",
    "difficulty": "difficult", "difficulties": "difficult", "reading": "read",
    "trained": "train", "training": "train", "learning": "learn", "built": "build",
}


def concept_tokens(text: str) -> list[str]:
    normalized = re.sub(r"['’]s\b", "", unicodedata.normalize("NFKC", text).casefold())
    normalized = re.sub(r"첫\s*원리", " first principle ", normalized)
    result = []
    for raw in re.findall(r"[^\W_]+", normalized):
        token = raw
        # CUDA를 and TSMC가 are single Unicode words; keep their ASCII anchor.
        mixed = re.fullmatch(r"([a-z0-9]+)([가-힣]+)", token)
        if mixed and mixed[2] in KOREAN_PARTICLES:
            token = mixed[1]
        if re.search(r"[가-힣]", token):
            if token not in CONCEPTS and token not in NAMES and token not in NEUTRAL:
                for suffix in KOREAN_PARTICLES:
                    if token.endswith(suffix) and len(token) > len(suffix):
                        token = token[:-len(suffix)]
                        break
            if token in NEUTRAL or token in NAMES:
                continue
            matches = [root for root in CONCEPTS if token.startswith(root)]
            words = CONCEPTS[max(matches, key=len)].split() if matches else [token]
        else:
            words = [token]
        for word in words:
            if word in STOPWORDS or word in NAMES or word in ENGLISH_NEUTRAL or word in NEUTRAL or len(word) < 2:
                continue
            word = FORMS.get(word, word)
            if word.isascii() and word.isalpha():
                if len(word) > 4 and word.endswith("ies"):
                    word = word[:-3] + "y"
                elif len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
                    word = word[:-1]
            result.append(FORMS.get(word, word))
    return result


class JensenConceptRetriever(Retriever):
    def __init__(self, corpus: Corpus):
        if corpus.persona.id != "jensen-huang":
            raise ValueError("jensen-lexical requires the Jensen corpus")
        super().__init__(corpus)

    def tokenize(self, text: str) -> list[str]:
        return concept_tokens(text)

    def _eligible_hit(self, hit: SearchHit, terms: set[str]) -> bool:
        anchored = bool(set(hit.matched_terms) & NAMED_ANCHORS)
        return (hit.score >= 0.65
            and len(hit.matched_terms) / max(1, len(terms)) >= 0.28
            and (len(hit.matched_terms) >= 2 or anchored or bool(set(hit.matched_terms) & self.topic_terms[hit.evidence.id]))
            and (hit.evidence.question_answer is None or anchored or self._qa_topic_supported(hit)))
