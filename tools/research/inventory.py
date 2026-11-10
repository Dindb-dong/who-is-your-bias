from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from bias.corpus import Corpus, canonical_source_url, source_event_id


def canonical_url(url: str) -> str:
    return canonical_source_url(url)


def document_urls(source) -> list[str]:
    extra = source.model_extra or {}
    plural = extra.get("transcript_urls", [])
    components = extra.get("component_urls", [])
    if not isinstance(plural, list) or not isinstance(components, list):
        raise ValueError(f"Source {source.id} component URLs must be lists")
    supplied = ([extra["transcript_url"]] if extra.get("transcript_url") else []) + plural
    supplied = supplied or components or [str(source.url)]
    if any(not isinstance(url, str) or not url for url in supplied):
        raise ValueError(f"Source {source.id} document URLs must be nonempty strings")
    return list(dict.fromkeys(supplied))


def inventory(corpus: Corpus) -> dict:
    quality = corpus.quality_report()
    by_url: dict[str, list[str]] = defaultdict(list)
    families: dict[str, list] = defaultdict(list)
    source_counts = Counter(item.source_id for item in corpus.evidence.values())
    topics = Counter(topic for item in corpus.evidence.values() for topic in item.topics)
    situations = Counter(situation for item in corpus.evidence.values() for situation in item.situations)
    text_groups: dict[str, list[str]] = defaultdict(list)
    for item in corpus.evidence.values():
        normalized = " ".join(item.text.lower().split())
        text_groups[hashlib.sha256(normalized.encode()).hexdigest()].append(item.id)
    sources = []
    for source in corpus.sources.values():
        extra = source.model_extra or {}
        family = source_event_id(source)
        acquisition_urls = document_urls(source)
        by_url[canonical_url(str(source.url))].append(source.id)
        families[family].append(source)
        sources.append({
            "id": source.id, "source_event_id": family, "title": source.title,
            "url": str(source.url), "published_at": source.published_at.isoformat(),
            "recorded_at": source.recorded_at.isoformat() if source.recorded_at else None,
            "publisher": source.publisher, "split": source.split,
            "evidence_records": source_counts[source.id], "timeline": source.timeline,
            "acquisition_url": acquisition_urls[0],
            "acquisition_urls": acquisition_urls,
            "transcript_urls": extra.get("transcript_urls", []),
            "component_urls": extra.get("component_urls", []),
            "video_url": extra.get("video_url"), "audio_url": extra.get("audio_url"),
            "audio_episode_url": extra.get("audio_episode_url"),
            "media_links": extra.get("media_links", []),
        })
    leaks = [{"source_event_id": family, "sources": [source.id for source in records]} for family, records in families.items() if len({source.split for source in records}) > 1]
    return {
        "schema_version": 1,
        "summary": {**corpus.stats(), "distinct_source_events": len(families), "publishers": len({s.publisher for s in corpus.sources.values()}), "earliest_publication": min(s.published_at for s in corpus.sources.values()).isoformat(), "latest_publication": max(s.published_at for s in corpus.sources.values()).isoformat()},
        "sources": sorted(sources, key=lambda source: (source["published_at"], source["id"])),
        "coverage": {"topics": dict(topics.most_common()), "situations": dict(situations.most_common())},
        "duplicate_urls": [ids for ids in by_url.values() if len(ids) > 1],
        "duplicate_evidence": [ids for ids in text_groups.values() if len(ids) > 1],
        "source_event_split_leaks": leaks,
        "url_split_leaks": quality["url_split_leaks"],
        "limitations": ["Inventory breadth is not persona fidelity.", "Text-only evidence does not establish facial, vocal, or gesture traits.", "Publication date and recording date are distinct.", "Only supplied, verified media URLs are emitted; media locators are never guessed."],
    }
