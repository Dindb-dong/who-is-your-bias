"""Small-corpus BM25 retrieval with transparent lexical confidence and split isolation."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .corpus import Corpus
from .models import Evidence

STOPWORDS = frozenset("a an the and or of to in on for is are was were be been being that this it its as at by with from i you your me my we our he his they their what why how when where which do did does would could should can about tell explain please elon musk think thoughts opinion question have has had will might may need said say described describe back suggest compare".split())
KOREAN_STOPWORDS = frozenset("어떤 어떻게 왜 무엇 무슨 했는데 대해 관해 알려줘 설명해줘 했어 말했어 이유 뭐야 있어 받았어 다뤘어".split())
ALIASES = {
    "화성": "mars multiplanetary", "우주": "space spacex", "스페이스엑스": "spacex rocket",
    "로켓": "rocket", "재사용": "reuse reusable", "전기차": "electric vehicle",
    "테슬라": "tesla", "에너지": "energy", "태양광": "solar", "배터리": "battery",
    "제조": "manufacturing production", "생산": "manufacturing production", "공장": "factory manufacturing",
    "자율주행": "autonomous driving", "인공지능": "artificial intelligence ai",
    "첫 원리": "first principles", "제1원리": "first principles", "문제 해결": "problem solving",
    "실패": "failure", "위험": "risk", "창업": "startup company",
    "엔지니어링": "engineering", "엔지니어": "engineer engineering", "수면": "sleep",
    "게임": "game gaming", "취미": "hobby", "아버지": "father", "행복": "happiness", "의식": "consciousness",
    "스포츠카": "sports car", "비싼": "expensive costly", "공학": "engineering",
    "원리": "principles fundamentals", "가설": "hypothesis hypotheses", "검증": "testable validation",
    "유용": "useful usefulness", "세상": "world", "바꾸": "transform change",
    "친구": "friend", "비판": "criticism critical", "피드백": "feedback",
    "채용": "hiring", "친절": "kindness", "신뢰": "trust trustworthiness",
    "신뢰성": "reliability", "성공": "success succeeding", "가능성": "chance probability",
    "확률": "probability", "도전": "attempt", "낮아": "low",
    "책": "book reading", "독서": "reading book", "어릴": "childhood growing",
    "어린": "childhood growing", "영화": "film movie", "영향": "influence impact",
    "요구사항": "requirements", "합치": "combine combined", "문제": "problem",
    "하이퍼루프": "hyperloop", "제안": "proposal", "오류": "mistake error",
    "두려": "fear", "목표": "goal objective", "불확실": "uncertainty uncertain",
    "일정": "schedule deadline", "예측": "forecast prediction", "낙관": "optimism optimistic",
    "유머": "humor", "농담": "joke humor", "학습": "learning", "배움": "learning",
    "경험": "experience", "실용": "useful utility", "디자인": "design",
    "팀": "team", "기술": "technology", "호기심": "curiosity", "관심": "interest",
    "기억": "memory memorable", "바꿔": "transform change",
}
KOREAN_PARTICLES = ("들에게", "들에서", "이라고", "에서는", "에게", "부터", "라고", "에서", "으로", "적인", "은", "는", "이", "가", "을", "를", "로", "과", "와", "에")
SITUATIONS = {
    "explanation": ("why", "how", "explain", "왜", "어떻게", "설명"),
    "reflection": ("regret", "failure", "learn", "실패", "배웠", "후회"),
    "praise": ("admire", "amazing", "inspiring", "멋져", "대단", "존경"),
    "challenge": ("disagree", "wrong", "critic", "반대", "틀렸", "비판"),
}
LIVE_QUERY = re.compile(r"\b(today|currently|latest|right now|(?:this|next) (?:week|month)|stock price|net worth)\b|오늘|현재|최신|지금 주가|최근 소식|다음 주", re.I)


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    # Translate known concepts, not arbitrary substrings: 책 (book) must not
    # match 정책 (policy) or 책임 (responsibility). Prefer a longer concept such
    # as 신뢰성 (reliability) to its shorter root 신뢰 (trust).
    alias_values = []
    raw_tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    for token in raw_tokens:
        matches = [
            key for key in ALIASES if " " not in key and token.startswith(key)
            and (len(key) > 1 or token == key or token[len(key):] in KOREAN_PARTICLES)
        ]
        if matches:
            alias_values.append(ALIASES[max(matches, key=len)])
    alias_values.extend(value for key, value in ALIASES.items() if " " in key and key in normalized)
    expanded = normalized + " " + " ".join(alias_values)
    expanded = re.sub(r"['’]s\b", "", expanded)
    tokens = re.findall(r"[^\W_]+", expanded, flags=re.UNICODE)
    result = []
    for token in tokens:
        if token in STOPWORDS or token in KOREAN_STOPWORDS or len(token) <= 1:
            continue
        if re.fullmatch(r"[가-힣]+", token):
            for suffix in KOREAN_PARTICLES:
                if token.endswith(suffix) and len(token) - len(suffix) >= 1:
                    token = token[:-len(suffix)]
                    break
        if token in KOREAN_STOPWORDS:
            continue
        if re.fullmatch(r"[a-z]+", token):
            if len(token) > 4 and token.endswith("ies"):
                token = token[:-3] + "y"
            elif len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
                token = token[:-1]
        result.append(token)
    return result


def classify_situation(text: str) -> str:
    lower = text.lower()
    for situation, phrases in SITUATIONS.items():
        if any(phrase in lower for phrase in phrases):
            return situation
    return "conversation"


@dataclass(frozen=True)
class SearchHit:
    evidence: Evidence
    score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    hits: tuple[SearchHit, ...]
    confidence: float
    reason: str | None = None


class Retriever:
    def __init__(self, corpus: Corpus):
        self.records = tuple(corpus.training_evidence)
        self.source_years = {source.id: {source.published_at.year, *([source.recorded_at.year] if source.recorded_at else [])} for source in corpus.sources.values()}
        self.topic_terms = {record.id: set(tokenize(" ".join(record.topics))) for record in self.records}
        self.term_counts = [Counter(tokenize(record.text + " " + " ".join(record.topics + record.topics))) for record in self.records]
        self.lengths = [sum(count.values()) for count in self.term_counts]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.df: Counter[str] = Counter()
        for count in self.term_counts:
            self.df.update(count.keys())

    def search(self, query: str, *, limit: int = 4, behavior: bool = False, allow_live: bool = False) -> SearchResult:
        if not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10")
        if not allow_live and LIVE_QUERY.search(query):
            return SearchResult((), 0, "The curated corpus is historical and cannot verify current information.")
        terms = set(tokenize(query))
        known = terms.intersection(self.df)
        if not known or not self.records:
            return SearchResult((), 0, "No matching evidence in the reviewed training corpus.")
        situation = classify_situation(query)
        requested_years = {int(value) for value in re.findall(r"\b((?:19|20)\d{2})\b", query)}
        n = len(self.records)
        hits = []
        for record, counts, length in zip(self.records, self.term_counts, self.lengths, strict=True):
            mentioned_years = {int(value) for value in re.findall(r"\b((?:19|20)\d{2})\b", record.text)}
            if requested_years and not requested_years.intersection(self.source_years[record.source_id]) and not requested_years.intersection(mentioned_years):
                continue
            matched = tuple(sorted(known.intersection(counts)))
            if not matched:
                continue
            score = 0.0
            for term in matched:
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                tf = counts[term]
                denominator = tf + 1.5 * (1 - 0.75 + 0.75 * length / max(1, self.average_length))
                score += idf * tf * 2.5 / denominator
            if behavior and situation in record.situations:
                score *= 1.15
            hits.append(SearchHit(record, round(score, 6), matched))
        hits.sort(key=lambda hit: (-hit.score, hit.evidence.id))
        # A weak, generic match must not be promoted into a factual response.
        hits = [
            hit for hit in hits
            if hit.score >= 0.65
            and len(hit.matched_terms) / max(1, len(terms)) >= 0.28
            and (len(hit.matched_terms) >= 2 or bool(set(hit.matched_terms).intersection(self.topic_terms[hit.evidence.id])))
        ]
        if not hits:
            return SearchResult((), 0, "Matches are too weak to support an answer.")
        best_terms = set(hits[0].matched_terms)
        coverage = len(best_terms) / max(1, len(terms))
        confidence = min(1.0, 0.35 + min(0.35, hits[0].score / 12) + 0.30 * coverage)
        return SearchResult(tuple(hits[:limit]), round(confidence, 3))
