"""Source-aware bilingual BM25 for the expanded Jensen production corpus.

Reviewed question aliases and decision annotations are searchable metadata, not
invented interview answers. This backend is deterministic and needs no model
download or API key. It does not claim semantic embedding/entailment quality.
The older lexical backends remain frozen for comparison.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .corpus import Corpus, source_event_id
from .jensen_retrieval import JensenConceptRetriever, concept_tokens
from .retrieval import SearchHit, SearchResult

VERSION = "jensen-public-bm25.v1"
_KOREAN = {
    "데니스": "denny", "데니즈": "denny", "데니": "denny", "덴니스": "denny",
    "설거지": "dishwasher", "식기세척": "dishwasher", "접시": "dish",
    "웨이터": "waiter", "서빙": "waiter", "버스보이": "busboy", "식당": "restaurant",
    "알바": "job", "아르바이트": "job", "어린시절": "childhood", "유년": "childhood",
    "고객": "customer", "유료": "paying", "성능": "performance", "속도": "speed",
    "빨라": "speed", "빠른": "speed", "빠르": "speed", "여섯": "six",
    "측정": "measurement", "오류": "error", "실수": "mistake", "가정": "assumption",
    "전제": "premise", "수요": "demand", "개척": "create", "없어": "absent",
    "없다": "absent", "없는": "absent", "없지만": "absent", "없는데": "absent",
    "프로그래밍": "programmability", "프로그래머블": "programmability",
    "어텐션": "attention", "어텐션기법": "attention", "아키텍처": "architecture",
    "전용칩": "specialized chip", "특화": "specialized", "비판": "criticism",
    "반론": "disagreement", "반대": "disagreement", "솔직": "honesty",
    "가죽": "leather", "재킷": "jacket", "자켓": "jacket", "인스타그램": "instagram",
    "인스타": "instagram", "대학": "university", "오리건": "oregon",
    "스탠퍼드": "stanford", "스탠포드": "stanford", "학위": "degree",
    "이민": "immigration", "대만": "taiwan", "회복탄력": "resilience",
    "추론": "inference", "로봇": "robot", "물리": "physical", "소프트웨어": "software",
    "하드웨어": "hardware", "생태계": "ecosystem", "한계": "constraint",
    "병목": "bottleneck", "대역폭": "bandwidth", "지연": "latency",
    "확장": "scale", "코칭": "coaching", "배움": "learn", "채용": "hiring",
    "해고": "dismissal", "위임": "delegation", "실패했": "failure",
    "파트너": "partner", "인프라": "infrastructure", "만들": "build",
    "테이프아웃": "tapeout", "검증": "verification", "검사": "test",
    "현금": "cash", "자금": "funding", "시도": "attempt",
    "판독": "reading", "의사": "doctor", "의료": "medical", "환자": "patient",
    "목적": "purpose", "작업": "task", "업무": "task", "직업": "job",
    "방사선": "radiology", "영상": "image", "자동화": "automation",
    "리더": "leader", "자신감": "confidence", "확신": "conviction",
    "불확실": "uncertainty", "인정": "admit", "소버린": "sovereign",
    "주권": "sovereign", "언어": "language", "문화": "culture",
    "국가": "country", "나라": "country", "주방": "kitchen", "키노트": "keynote",
}
_SYNONYMS = {
    "dennys": "denny", "diner": "restaurant", "dishes": "dish", "dishwashing": "dishwasher",
    "waited": "waiter", "waiting": "waiter", "waitstaff": "waiter", "server": "waiter",
    "worked": "work", "working": "work", "jobs": "job", "workers": "employee",
    "buyers": "customer", "buyer": "customer", "customer": "customer", "pay": "paying",
    "faster": "speed", "speedup": "speed", "sixfold": "six", "6x": "six",
    "6배": "six", "six-times": "six", "six-times-faster": "speed",
    "incorrect": "error", "erroneous": "error", "faulty": "error", "measuring": "measurement",
    "assumption": "premise", "measured": "measurement", "benchmark": "measurement",
    "programmable": "programmability", "programming": "programmability",
    "specialised": "specialized", "specialization": "specialized", "dedicated": "specialized",
    "disagree": "disagreement", "disagreed": "disagreement", "opposition": "disagreement",
    "learning": "learn", "immigrant": "immigration", "immigrated": "immigration",
    "resilient": "resilience", "coat": "jacket", "leather-jacket": "jacket",
    "scaling": "scale", "bottlenecks": "bottleneck", "constraints": "constraint",
}
_NEUTRAL = frozenset("ever really yourself someone somebody something anything explain tell experience experienced detail details".split())
_ANCHORS = frozenset({"denny", "cuda", "sega", "tsmc", "tpu", "attention", "blackwell", "rubin", "oneida", "oregon", "stanford", "instagram", "jacket"})


def public_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    # Keep translation at word starts; a short morpheme inside another word is
    # not evidence. Remaining unknown Korean/English words keep their weight in
    # the confidence denominator rather than disappearing.
    words = []
    for token in re.findall(r"[^\W_]+", normalized):
        matches = [key for key in _KOREAN if token.startswith(key)]
        words.append(_KOREAN[max(matches, key=len)] if matches else token)
    expanded = " ".join(words)
    expanded = re.sub(r"\bdenny\s+s\b", "denny", expanded)
    result = []
    for token in concept_tokens(expanded):
        token = _SYNONYMS.get(token, token)
        if token not in _NEUTRAL:
            result.append(token)
    return result


class JensenPublicRetriever(JensenConceptRetriever):
    def __init__(self, corpus: Corpus):
        super().__init__(corpus)
        self.events = {key: source_event_id(source) for key, source in corpus.sources.items()}
        self.term_counts = []
        for record in self.records:
            extra = record.model_extra or {}
            aliases = extra.get("retrieval_aliases", [])
            if not isinstance(aliases, list) or any(not isinstance(value, str) for value in aliases):
                raise ValueError(f"Invalid retrieval aliases: {record.id}")
            memory = extra.get("persona_memory", {})
            if not isinstance(memory, dict):
                raise ValueError(f"Invalid persona annotations: {record.id}")
            annotations = " ".join(memory.get(key, "") for key in
                                   ("question_context", "decision_criterion", "revision_condition")
                                   if isinstance(memory.get(key, ""), str))
            original_question = record.question_answer.question if record.question_answer else ""
            terms = (self.tokenize(record.text) * 2 + self.tokenize(" ".join(record.topics)) * 3
                     + self.tokenize(" ".join(aliases)) + self.tokenize(annotations)
                     + self.tokenize(original_question))
            self.term_counts.append(Counter(terms))
        self.lengths = [sum(count.values()) for count in self.term_counts]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter()
        for count in self.term_counts:
            self.df.update(count.keys())

    def tokenize(self, text: str) -> list[str]:
        return public_tokens(text)

    def _eligible_hit(self, hit: SearchHit, terms: set[str]) -> bool:
        matched = set(hit.matched_terms)
        coverage = len(matched) / max(1, len(terms))
        # A concrete named entity can support a short question, but not an
        # unrelated long question with just that name appended.
        specific = bool(matched & _ANCHORS) or len(matched) >= 2
        return hit.score >= .75 and coverage >= .28 and specific

    def search(self, query: str, *, limit: int = 4, behavior: bool = False,
               allow_live: bool = False) -> SearchResult:
        if isinstance(limit, bool) or not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10")
        candidates = super().search(query, limit=10, behavior=behavior, allow_live=allow_live)
        if not candidates.hits:
            return candidates
        # Keep adjacent relevant turns but stop one long interview monopolizing
        # the whole small context. This is event diversity, not evidence voting.
        selected, deferred, counts = [], [], Counter()
        for hit in candidates.hits:
            event = self.events[hit.evidence.source_id]
            if counts[event] >= 2:
                deferred.append(hit)
                continue
            selected.append(hit)
            counts[event] += 1
        selected = (selected + deferred)[:limit]
        return SearchResult(tuple(selected), candidates.confidence, candidates.reason)
