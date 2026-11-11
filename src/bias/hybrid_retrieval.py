"""Versioned hybrid retrieval without changing the frozen lexical experiment.

Dense retrieval is an explicitly configured capability. Missing models, invalid
vectors and provider failures raise an error; they never turn into fabricated
embeddings or an unlabelled lexical fallback. All results are *candidates* for
source-supported answers, not an entailment verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import unicodedata
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol
from urllib.parse import urlsplit

import httpx

from .corpus import Corpus, source_event_id
from .models import Evidence
from .retrieval import LIVE_QUERY, Retriever, SearchHit, SearchResult
from .situations import classify_situation

HYBRID_VERSION = "hybrid-rrf.v1"
_CURRENT_QUERY = re.compile(r"\b(now|current|these days|this year|at present)\b|지금|현재|요즘|올해|최근", re.I)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


class EmbeddingError(RuntimeError):
    """A configured embedding capability could not be used faithfully."""


class EmbeddingProvider(Protocol):
    max_tokens: int

    @property
    def identity(self) -> dict: ...

    def count_tokens(self, text: str, *, kind: Literal["query", "passage"]) -> int: ...

    def encode(self, texts: Sequence[str], *, kind: Literal["query", "passage"]) -> Sequence[Sequence[float]]: ...


class RelevanceReranker(Protocol):
    @property
    def identity(self) -> dict: ...

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]: ...


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _vectors(values: Sequence[Sequence[float]], *, expected: int, dimensions: int | None = None) -> tuple[tuple[float, ...], ...]:
    """Validate and normalize vectors, rejecting NaN, zero and shape changes."""
    if not isinstance(values, (list, tuple)) or len(values) != expected:
        raise EmbeddingError("Embedding response has the wrong number of vectors")
    result = []
    for row in values:
        if not isinstance(row, (list, tuple)) or not 1 <= len(row) <= 8192:
            raise EmbeddingError("Embedding vector must have 1–8192 dimensions")
        if dimensions is None:
            dimensions = len(row)
        if len(row) != dimensions or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row):
            raise EmbeddingError("Embedding response has non-finite values or inconsistent dimensions")
        norm = math.sqrt(sum(v * v for v in row))
        if not math.isfinite(norm) or norm <= 0:
            raise EmbeddingError("Embedding response contains a zero or invalid vector")
        result.append(tuple(float(v / norm) for v in row))
    return tuple(result)


class SentenceTransformerEmbedder:
    """Optional local multilingual model with exact tokenizer length checks.

    Prefixes are explicit because E5 expects query/passage prefixes even for
    Korean, whereas other model families use different instructions. Pin a
    revision or use a locally versioned model directory for reproducible runs.
    """

    def __init__(self, model: str, *, revision: str | None = None, device: str = "cpu", query_prefix: str = "query: ", passage_prefix: str = "passage: ", max_tokens: int = 512, batch_size: int = 16):
        if not model.strip() or not 16 <= max_tokens <= 8192 or not 1 <= batch_size <= 128:
            raise ValueError("Invalid local embedding model, token limit or batch size")
        self.model_name, self.revision, self.device = model, revision, device
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.max_tokens, self.batch_size = max_tokens, batch_size
        self._model = None
        self._lock = threading.RLock()
        self._query_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise EmbeddingError("Local embeddings require the optional sentence-transformers package") from exc
                try:
                    self._model = SentenceTransformer(self.model_name, revision=self.revision, device=self.device, trust_remote_code=False)
                except Exception as exc:
                    raise EmbeddingError("The configured local embedding model could not be loaded") from exc
                self.max_tokens = min(self.max_tokens, int(self._model.max_seq_length))
            return self._model

    @property
    def identity(self) -> dict:
        return {"provider": "sentence-transformers", "model": self.model_name, "revision": self.revision,
                "device": self.device, "query_prefix": self.query_prefix, "passage_prefix": self.passage_prefix,
                "max_tokens": self.max_tokens, "token_count_basis": "model_tokenizer"}

    def count_tokens(self, text: str, *, kind: Literal["query", "passage"]) -> int:
        model = self._load()
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        return len(model.tokenizer(prefix + text, add_special_tokens=True, truncation=False)["input_ids"])

    def encode(self, texts: Sequence[str], *, kind: Literal["query", "passage"]) -> Sequence[Sequence[float]]:
        model = self._load()
        if any(self.count_tokens(text, kind=kind) > self.max_tokens for text in texts):
            raise EmbeddingError("Embedding input exceeds the configured tokenizer limit; truncation is disabled")
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        with self._lock:
            cache_key = _digest([prefix, texts[0]]) if kind == "query" and len(texts) == 1 else None
            if cache_key is not None and cache_key in self._query_cache:
                self._query_cache.move_to_end(cache_key)
                return [list(self._query_cache[cache_key])]
            try:
                result = model.encode([prefix + text for text in texts], batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False).tolist()
                if cache_key is not None:
                    self._query_cache[cache_key] = tuple(result[0])
                    if len(self._query_cache) > 128:
                        self._query_cache.popitem(last=False)
                return result
            except Exception as exc:
                raise EmbeddingError("Local embedding inference failed") from exc


class OpenAIEmbeddingProvider:
    """An explicitly selected compatible /embeddings endpoint.

    This adapter uses a conservative UTF-8 byte budget plus prefix/special-token
    allowance when the server tokenizer is unavailable. Its trace never calls
    this an exact token count. Set prefixes to the selected model's documented
    values. The upstream service must not silently truncate inputs.
    """

    def __init__(self, base_url: str, model: str, *, api_key: str = "", revision: str | None = None, timeout: float = 30, max_tokens: int = 512, batch_size: int = 16, query_prefix: str = "query: ", passage_prefix: str = "passage: ", transport: httpx.BaseTransport | None = None):
        url = urlsplit(base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Embedding URL must be HTTP(S) without credentials, query or fragment")
        if not model.strip() or not 16 <= max_tokens <= 8192 or not 1 <= batch_size <= 128 or not 0 < timeout <= 600:
            raise ValueError("Invalid embedding model, token limit, batch size or timeout")
        self.base_url, self.model, self.revision = base_url.rstrip("/"), model, revision
        self.api_key, self.timeout, self.transport = api_key, timeout, transport
        self.max_tokens, self.batch_size = max_tokens, batch_size
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix

    @property
    def identity(self) -> dict:
        return {"provider": "openai-compatible", "model": self.model, "revision": self.revision,
                "query_prefix": self.query_prefix, "passage_prefix": self.passage_prefix,
                "max_tokens": self.max_tokens, "token_count_basis": "utf8_byte_upper_budget"}

    def count_tokens(self, text: str, *, kind: Literal["query", "passage"]) -> int:
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        return len((prefix + text).encode("utf-8")) + 8

    def encode(self, texts: Sequence[str], *, kind: Literal["query", "passage"]) -> Sequence[Sequence[float]]:
        if any(self.count_tokens(text, kind=kind) > self.max_tokens for text in texts):
            raise EmbeddingError("Embedding input exceeds its declared conservative byte budget")
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        output = []
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport, follow_redirects=False) as client:
                for start in range(0, len(texts), self.batch_size):
                    batch = texts[start:start + self.batch_size]
                    response = client.post(self.base_url + "/embeddings", headers=headers,
                                           json={"model": self.model, "input": [prefix + text for text in batch], "encoding_format": "float"})
                    response.raise_for_status()
                    data = response.json()
                    entries = data.get("data") if isinstance(data, dict) else None
                    if not isinstance(entries, list) or len(entries) != len(batch):
                        raise EmbeddingError("Embedding endpoint returned an invalid batch")
                    indexed = {}
                    for entry in entries:
                        if not isinstance(entry, dict) or type(entry.get("index")) is not int or entry["index"] not in range(len(batch)) or entry["index"] in indexed:
                            raise EmbeddingError("Embedding endpoint returned invalid or duplicate indices")
                        indexed[entry["index"]] = entry.get("embedding")
                    output.extend(indexed[index] for index in range(len(batch)))
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("The configured embedding endpoint failed or returned invalid JSON") from exc
        return _vectors(output, expected=len(texts))


class CrossEncoderReranker:
    """Optional independently versioned pair scorer; thresholds are model-specific."""

    def __init__(self, model: str, *, revision: str | None = None, device: str = "cpu", max_tokens: int = 512):
        self.model_name, self.revision, self.device, self.max_tokens = model, revision, device, max_tokens
        self._model = None
        self._lock = threading.RLock()
        self._score_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()

    @property
    def identity(self) -> dict:
        return {"provider": "sentence-transformers-cross-encoder", "model": self.model_name, "revision": self.revision,
                "device": self.device, "max_tokens": self.max_tokens, "scoring": "raw_model_output"}

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        with self._lock:
            cache_key = _digest([query, list(passages)])
            if cache_key in self._score_cache:
                self._score_cache.move_to_end(cache_key)
                return self._score_cache[cache_key]
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(self.model_name, revision=self.revision, device=self.device, max_length=self.max_tokens, trust_remote_code=False)
                except Exception as exc:
                    raise EmbeddingError("The configured relevance reranker could not be loaded") from exc
            # Rerank the matching bounded subspan, never a silently truncated
            # full exchange. The complete parent remains the supplied evidence.
            for passage in passages:
                count = len(self._model.tokenizer(query, passage, add_special_tokens=True, truncation=False)["input_ids"])
                if count > self.max_tokens:
                    raise EmbeddingError("Reranker pair exceeds its tokenizer limit; truncation is disabled")
            try:
                import torch
                scores = self._model.predict([(query, text) for text in passages], show_progress_bar=False, activation_fn=torch.nn.Identity())
                result = tuple(float(value) for value in scores)
                self._score_cache[cache_key] = result
                if len(self._score_cache) > 64:
                    self._score_cache.popitem(last=False)
                return result
            except Exception as exc:
                raise EmbeddingError("Relevance reranking failed") from exc


@dataclass(frozen=True)
class HybridPolicy:
    # E5-specific initial development value, not a universal quality threshold.
    dense_min_similarity: float = 0.84
    candidate_limit: int = 24
    rrf_constant: int = 60
    max_per_event: int = 2
    max_style_examples: int = 2
    situation_multiplier: float = 1.15
    temporal_weight: float = 0.10
    reranker_min_score: float | None = None
    max_chunk_tokens: int = 240

    def __post_init__(self):
        if not math.isfinite(self.dense_min_similarity) or not -1 <= self.dense_min_similarity <= 1:
            raise ValueError("dense_min_similarity must be finite and between -1 and 1")
        if not 1 <= self.candidate_limit <= 100 or not 1 <= self.rrf_constant <= 1000 or not 1 <= self.max_per_event <= 10:
            raise ValueError("Invalid candidate, RRF or event limits")
        if not 0 <= self.max_style_examples <= 10 or not 1 <= self.situation_multiplier <= 2 or not 0 <= self.temporal_weight <= 1:
            raise ValueError("Invalid style, situation or temporal policy")
        if not 16 <= self.max_chunk_tokens <= 4096:
            raise ValueError("max_chunk_tokens must be between 16 and 4096")
        if self.reranker_min_score is not None and not math.isfinite(self.reranker_min_score):
            raise ValueError("reranker_min_score must be finite")


@dataclass(frozen=True)
class HybridSearchResult(SearchResult):
    style_hits: tuple[SearchHit, ...] = ()
    trace: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Unit:
    id: str
    evidence_id: str
    view: Literal["answer", "question_context"]
    text: str
    char_start: int
    char_end: int


def _chunks(text: str, embedder: EmbeddingProvider, budget: int) -> list[tuple[int, int]]:
    """Cover every character; chunking changes the index, never the parent text."""
    spans = []
    start = 0
    while start < len(text):
        if embedder.count_tokens(text[start:], kind="passage") <= budget:
            spans.append((start, len(text)))
            break
        low, high = start + 1, len(text)
        best = start
        while low <= high:
            middle = (low + high) // 2
            if embedder.count_tokens(text[start:middle], kind="passage") <= budget:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        if best == start:
            raise EmbeddingError("Embedding budget cannot fit one source character and its prefix")
        # Prefer sentence/word edges; offsets still reconstruct the exact text.
        boundaries = list(re.finditer(r"(?:[.!?]\s+|\n+|\s+)", text[start:best]))
        if boundaries and boundaries[-1].end() >= (best - start) // 2:
            best = start + boundaries[-1].end()
        spans.append((start, best))
        start = best
    return spans


class HybridRetriever:
    """Reviewed-training-only lexical + two-view dense retrieval with an audit.

    ``eligible_ids`` is an additional temporal/project gate, never a way to add
    held-out records. ``temporal_scores`` only adjusts already relevant records.
    A high RRF rank or a matching situation cannot open the topical gate.
    """

    def __init__(self, corpus: Corpus, embedder: EmbeddingProvider, *, policy: HybridPolicy | None = None, reranker: RelevanceReranker | None = None):
        self.corpus, self.embedder = corpus, embedder
        self.policy, self.reranker = policy or HybridPolicy(), reranker
        if self.policy.reranker_min_score is not None and reranker is None:
            raise ValueError("A reranker threshold requires a configured reranker")
        self.lexical = Retriever(corpus)
        self.records = self.lexical.records
        self._factual_ids = {record.id for record in self.records}
        self.style_records = tuple(corpus.style_evidence)
        self._style_ids = {record.id for record in self.style_records}
        # Style eligibility is independent of factual eligibility. Embed each
        # parent once, but do not let style-only records occupy factual ranks.
        self._records = {record.id: record for record in (*self.records, *self.style_records)}
        self._events = {record.id: source_event_id(corpus.sources[record.source_id]) for record in self._records.values()}
        self.corpus_sha256 = _digest({"sources": [source.model_dump(mode="json") for source in sorted(corpus.sources.values(), key=lambda item: item.id)],
                                     "training_evidence": [record.model_dump(mode="json") for record in sorted(self.records, key=lambda item: item.id)],
                                     "style_evidence": [record.model_dump(mode="json") for record in sorted(self.style_records, key=lambda item: item.id)],
                                     "manifest": corpus.manifest})
        self._units: tuple[_Unit, ...] = ()
        self._matrix: tuple[tuple[float, ...], ...] | None = None
        self._index_lock = threading.RLock()
        self._warmed = False

    def warm_up(self) -> None:
        """Prepare configured providers before the application reports ready.

        The fixed initialization probe only loads the pair scorer; it is never
        evidence, a relevance decision, or a user-visible retrieval result.
        Failures propagate so startup cannot claim a usable hybrid capability.
        """
        with self._index_lock:
            if self._warmed:
                return
            self._ensure_index()
            if self.reranker is not None:
                scores = self.reranker.score("Retrieval initialization check.", ["Retrieval initialization check."])
                if len(scores) != 1 or isinstance(scores[0], bool) or not math.isfinite(scores[0]):
                    raise EmbeddingError("Reranker initialization returned an invalid score")
            self._warmed = True

    def _ensure_index(self):
        with self._index_lock:
            if self._matrix is not None:
                return
            units = []
            budget = min(self.policy.max_chunk_tokens, self.embedder.max_tokens)
            for record in self._records.values():
                views = [("answer", record.text)]
                if record.question_answer:
                    qa = record.question_answer
                    context = qa.context_before if qa.context_kind == "original_text" else ""
                    views.append(("question_context", (context + "\n" + qa.question).strip()))
                for view, text in views:
                    for index, (start, end) in enumerate(_chunks(text, self.embedder, budget)):
                        units.append(_Unit(f"{record.id}:{view}:{index}", record.id, view, text[start:end], start, end))
            matrix = self.embedder.encode([unit.text for unit in units], kind="passage") if units else []
            self._matrix = _vectors(matrix, expected=len(units))
            self._units = tuple(units)

    def search(self, query: str, *, limit: int = 4, behavior: bool = False, allow_live: bool = False,
               original_query: str | None = None, eligible_ids: set[str] | frozenset[str] | None = None,
               eligible_style_ids: set[str] | frozenset[str] | None = None,
               exclusion_reasons: Mapping[str, str] | None = None, temporal_scores: Mapping[str, float] | None = None,
               include_style: bool = False) -> HybridSearchResult:
        if not 1 <= limit <= 10 or not query.strip():
            raise ValueError("A nonblank query and a limit between 1 and 10 are required")
        if temporal_scores is not None and any(isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1 for value in temporal_scores.values()):
            raise ValueError("Temporal scores must be finite numbers between 0 and 1")
        trace = {"schema_version": "hybrid-retrieval-trace.v1", "retriever_version": HYBRID_VERSION,
                 "original_query": original_query if original_query is not None else query, "search_query": query,
                 "corpus_sha256": self.corpus_sha256, "corpus_id": (self.corpus.manifest or {}).get("id"),
                 "embedding": self.embedder.identity, "reranker": self.reranker.identity if self.reranker else None,
                 "policy": asdict(self.policy), "training_records": len(self.records), "style_records": len(self.style_records),
                 "candidates": [], "style_candidates": [],
                 "factual_ids": [], "style_ids": [], "semantic_status": "unreviewed_candidates"}
        if not allow_live and (LIVE_QUERY.search(query) or _CURRENT_QUERY.search(query)):
            trace["abstention_reason"] = "current_information_requires_snapshot_policy"
            return HybridSearchResult((), 0, "The curated corpus cannot verify current information without a temporal policy.", trace=trace)
        if not self.records:
            trace["abstention_reason"] = "empty_reviewed_training_corpus"
            return HybridSearchResult((), 0, "No reviewed training evidence is available.", trace=trace)
        self._ensure_index()
        trace["embedding"] = self.embedder.identity
        trace["indexed_units"] = len(self._units)
        if self.embedder.count_tokens(query, kind="query") > self.embedder.max_tokens:
            raise EmbeddingError("Search query exceeds the embedding tokenizer limit; shorten resolved conversation context")
        query_vector = _vectors(self.embedder.encode([query], kind="query"), expected=1, dimensions=len(self._matrix[0]))[0]
        similarities = [sum(a * b for a, b in zip(query_vector, row, strict=True)) for row in self._matrix]
        view_scores: dict[str, dict[str, tuple[float, _Unit]]] = {"answer": {}, "question_context": {}}
        for unit, score in zip(self._units, similarities, strict=True):
            prior = view_scores[unit.view].get(unit.evidence_id)
            if prior is None or score > prior[0]:
                view_scores[unit.view][unit.evidence_id] = (score, unit)
        lexical = self.lexical.search(query, limit=10, behavior=False, allow_live=True)
        lexical_hits = {hit.evidence.id: hit for hit in lexical.hits}
        ranks = {"lexical": {hit.evidence.id: rank for rank, hit in enumerate(lexical.hits, 1)}}
        for view, scores in view_scores.items():
            ordered = sorted(self._factual_ids.intersection(scores), key=lambda identifier: (-scores[identifier][0], identifier))[:self.policy.candidate_limit]
            ranks[view] = {identifier: rank for rank, identifier in enumerate(ordered, 1)}
        candidates = set().union(*(set(values) for values in ranks.values()))
        situation = classify_situation(query)
        trace["situation"] = situation
        eligible = set(self._factual_ids) if eligible_ids is None else self._factual_ids.intersection(eligible_ids)
        reasons = dict(exclusion_reasons or {})
        # With no external temporal gate, retain the lexical baseline's explicit
        # year constraint for dense-only candidates as well.
        if eligible_ids is None:
            years = {int(value) for value in _YEAR.findall(query)}
            for identifier in candidates:
                record = self._records[identifier]
                mentioned = {int(value) for value in _YEAR.findall(record.text)}
                if years and not years.intersection(self.lexical.source_years[record.source_id] | mentioned):
                    eligible.discard(identifier)
                    reasons[identifier] = "requested_year_not_supported"
        topical = []
        rows = {}
        for identifier in sorted(candidates):
            record = self._records[identifier]
            answer_score, unit = view_scores["answer"][identifier]
            question_score = view_scores["question_context"].get(identifier, (None, None))[0]
            lexical_hit = lexical_hits.get(identifier)
            rrf = sum(1 / (self.policy.rrf_constant + values[identifier]) for values in ranks.values() if identifier in values)
            row = {"evidence_id": identifier, "source_id": record.source_id, "event_id": self._events[identifier],
                   "answer_sha256": hashlib.sha256(record.text.encode()).hexdigest(),
                   "lexical_score": lexical_hit.topical_score if lexical_hit else None,
                   "answer_similarity": round(answer_score, 7), "question_context_similarity": round(question_score, 7) if question_score is not None else None,
                   "rrf_score": round(rrf, 7), "ranks": {view: values.get(identifier) for view, values in ranks.items()},
                   "matching_unit": {"id": unit.id, "view": unit.view, "char_start": unit.char_start, "char_end": unit.char_end},
                   "situation_match": situation in record.situations, "temporal_score": (temporal_scores or {}).get(identifier, 0.0),
                   "accepted_topically": False, "selection": None, "filter_reason": None}
            rows[identifier] = row
            trace["candidates"].append(row)
            if identifier not in eligible:
                row["filter_reason"] = reasons.get(identifier, "excluded_by_snapshot_or_project_policy")
                continue
            # Question similarity and RRF do not prove that its answer addresses
            # this question. Require answer-passage relevance or a strict BM25
            # gate, and optionally an independently calibrated pair scorer.
            if lexical_hit is None and answer_score < self.policy.dense_min_similarity:
                row["filter_reason"] = "below_topical_relevance_threshold"
                continue
            topical.append((identifier, unit, answer_score, rrf, lexical_hit))
        reranked = None
        if self.reranker is not None and topical:
            reranked = self.reranker.score(query, [unit.text for _, unit, *_ in topical])
            if len(reranked) != len(topical) or any(isinstance(value, bool) or not math.isfinite(value) for value in reranked):
                raise EmbeddingError("Reranker returned invalid scores")
        hits = []
        for index, (identifier, unit, dense, rrf, lexical_hit) in enumerate(topical):
            row = rows[identifier]
            rerank_score = float(reranked[index]) if reranked is not None else None
            row["reranker_score"] = rerank_score
            if rerank_score is not None and self.policy.reranker_min_score is not None and rerank_score < self.policy.reranker_min_score:
                row["filter_reason"] = "below_reranker_relevance_threshold"
                continue
            row["accepted_topically"] = True
            # Transparent feature rerank: cosine/strict lexical support and fused
            # rank, followed by bounded temporal and (C only) situation weights.
            relevance = (dense + 1) / 2
            fused = rrf / (3 / (self.policy.rrf_constant + 1))
            topical_score = 0.65 * relevance + 0.35 * fused
            if rerank_score is not None:
                topical_score = 0.75 * (0.5 + math.atan(rerank_score) / math.pi) + 0.25 * topical_score
            final_score = topical_score * (1 + self.policy.temporal_weight * row["temporal_score"])
            if behavior and row["situation_match"]:
                final_score *= self.policy.situation_multiplier
            row["topical_score"], row["final_score"] = round(topical_score, 7), round(final_score, 7)
            terms = lexical_hit.matched_terms if lexical_hit else ()
            hits.append(SearchHit(self._records[identifier], round(final_score, 7), terms, round(topical_score, 7), row["situation_match"]))
        hits.sort(key=lambda hit: (-hit.score, hit.evidence.id))
        selected = []
        seen_text = set()
        event_counts = Counter()
        for hit in hits:
            identifier = hit.evidence.id
            key = " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", hit.evidence.text).casefold()))
            event = self._events[identifier]
            if key in seen_text:
                rows[identifier]["filter_reason"] = "duplicate_answer_content"
            elif event_counts[event] >= self.policy.max_per_event:
                rows[identifier]["filter_reason"] = "event_diversity_cap"
            elif len(selected) >= limit:
                rows[identifier]["filter_reason"] = "factual_context_limit"
            else:
                selected.append(hit)
                seen_text.add(key)
                event_counts[event] += 1
                rows[identifier]["selection"] = "factual_candidate"
        selected_ids = {hit.evidence.id for hit in selected}
        style = []
        if include_style and selected and self.policy.max_style_examples:
            style_allowed = self._style_ids
            if eligible_style_ids is not None:
                style_allowed = style_allowed.intersection(eligible_style_ids)
            elif eligible_ids is not None:
                # A caller supplying a temporal gate must explicitly opt into
                # any separately eligible historical style references.
                style_allowed = style_allowed.intersection(eligible_ids)
            style = self._select_style(query, view_scores, style_allowed, selected_ids, situation, behavior, trace)
        trace["factual_ids"] = [hit.evidence.id for hit in selected]
        trace["style_ids"] = [hit.evidence.id for hit in style]
        trace["candidate_count"] = len(candidates)
        if not selected:
            trace["abstention_reason"] = "no_candidate_passed_all_gates"
            return HybridSearchResult((), 0, "No reviewed candidate passed topical and temporal eligibility gates.", tuple(style), trace)
        confidence = min(1.0, max(0.0, selected[0].topical_score or 0))
        return HybridSearchResult(tuple(selected), round(confidence, 3), None, tuple(style), trace)

    def _select_style(self, query: str, view_scores: dict, allowed: set[str] | frozenset[str], factual_ids: set[str], situation: str, behavior: bool, trace: dict) -> list[SearchHit]:
        """Select original style independently; its topic is not factual support."""
        ranks = {}
        for view, scores in view_scores.items():
            ids = sorted(self._style_ids.intersection(scores), key=lambda identifier: (-scores[identifier][0], identifier))[:self.policy.candidate_limit]
            ranks[view] = {identifier: rank for rank, identifier in enumerate(ids, 1)}
        candidates = set().union(*(set(values) for values in ranks.values()))
        pending, rows = [], {}
        for identifier in sorted(candidates):
            record = self._records[identifier]
            options = [view_scores[view][identifier] for view in view_scores if identifier in view_scores[view]]
            similarity, unit = max(options, key=lambda item: item[0])
            row = {"evidence_id": identifier, "source_id": record.source_id, "event_id": self._events[identifier],
                   "similarity": round(similarity, 7), "matching_view": unit.view,
                   "matching_unit": {"id": unit.id, "char_start": unit.char_start, "char_end": unit.char_end},
                   "selection": None, "filter_reason": None, "semantic_status": "style_only_not_factual_support",
                   "situation_match": situation in record.situations}
            rows[identifier] = row
            trace["style_candidates"].append(row)
            if identifier not in allowed:
                row["filter_reason"] = "excluded_by_style_temporal_or_project_policy"
            elif identifier in factual_ids:
                row["filter_reason"] = "already_supplied_as_factual_candidate"
            elif similarity < self.policy.dense_min_similarity:
                row["filter_reason"] = "below_style_topic_threshold"
            else:
                rrf = sum(1 / (self.policy.rrf_constant + values[identifier]) for values in ranks.values() if identifier in values)
                pending.append((identifier, similarity, unit, rrf))
        scores = self.reranker.score(query, [unit.text for _, _, unit, _ in pending]) if self.reranker is not None and pending else None
        if scores is not None and (len(scores) != len(pending) or any(isinstance(value, bool) or not math.isfinite(value) for value in scores)):
            raise EmbeddingError("Style reranker returned invalid scores")
        ranked = []
        for index, (identifier, similarity, unit, rrf) in enumerate(pending):
            row = rows[identifier]
            pair_score = float(scores[index]) if scores is not None else None
            row["reranker_score"] = pair_score
            if pair_score is not None and self.policy.reranker_min_score is not None and pair_score < self.policy.reranker_min_score:
                row["filter_reason"] = "below_style_reranker_threshold"
                continue
            topical = 0.65 * (similarity + 1) / 2 + 0.35 * rrf / (2 / (self.policy.rrf_constant + 1))
            if pair_score is not None:
                topical = 0.75 * (0.5 + math.atan(pair_score) / math.pi) + 0.25 * topical
            score = topical * (self.policy.situation_multiplier if behavior and row["situation_match"] else 1)
            ranked.append(SearchHit(self._records[identifier], round(score, 7), (), round(topical, 7), row["situation_match"]))
        ranked.sort(key=lambda hit: (-hit.score, hit.evidence.id))
        selected, events, answers = [], set(), set()
        for hit in ranked:
            row = rows[hit.evidence.id]
            answer = " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", hit.evidence.text).casefold()))
            if row["event_id"] in events:
                row["filter_reason"] = "style_event_diversity_cap"
            elif answer in answers:
                row["filter_reason"] = "duplicate_style_answer"
            elif len(selected) >= self.policy.max_style_examples:
                row["filter_reason"] = "style_context_limit"
            else:
                selected.append(hit)
                events.add(row["event_id"])
                answers.add(answer)
                row["selection"] = "style_only_not_factual_support"
        return selected
