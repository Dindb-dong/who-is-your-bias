"""Build the reviewable public Jensen RAG corpus without copying full transcripts.

Acquisition shards contain bounded paraphrases and explicitly identified short
original excerpts. Discovery-only social links never become runtime evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from bias.corpus import Corpus, CorpusError, source_event_id
from bias.models import Evidence, Source
from bias.snapshots import register_snapshot

ROOT = Path(__file__).resolve().parents[1]
ACQUISITION = ROOT / "data/acquisition/jensen-public-2026-11-24"
OUTPUT = ROOT / "data/personas/jensen-huang-public-v3"
AS_OF = datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(acquisition: Path = ACQUISITION, output: Path = OUTPUT) -> dict:
    baseline = ROOT / "data/personas/jensen-huang-values-v2"
    original = Corpus.load(baseline)
    sources = {key: value.model_dump(mode="json") for key, value in original.sources.items()}
    evidence = {key: value.model_dump(mode="json") for key, value in original.evidence.items()}
    inherited_ids = set(evidence)
    shard_hashes = {}
    for path in sorted(acquisition.glob("*-sources.json")):
        shard_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for item in read_json(path):
            source = Source.model_validate(item)
            if source.published_at > AS_OF.date():
                raise ValueError(f"Future source date: {source.id}")
            if source.id in sources:
                previous = Source.model_validate(sources[source.id])
                if source_event_id(source) != source_event_id(previous) or source.split != previous.split:
                    raise ValueError(f"An existing event/split cannot be relabeled: {source.id}")
                # Keep inherited provenance fields unless explicitly re-reviewed.
                sources[source.id] = sources[source.id] | item
            else:
                sources[source.id] = item
    added_ids = set()
    excerpts: dict[str, set[str]] = {}
    for path in sorted(acquisition.glob("*-evidence.jsonl")):
        shard_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            record = Evidence.model_validate(item)
            if record.id in evidence:
                raise ValueError(f"Duplicate record {record.id} at {path.name}:{number}")
            if record.source_id not in sources:
                raise ValueError(f"Unknown source: {record.source_id}")
            aliases = item.get("retrieval_aliases", [])
            if not isinstance(aliases, list) or any(not isinstance(x, str) or len(x) > 600 for x in aliases):
                raise ValueError(f"Invalid retrieval aliases: {record.id}")
            memory = item.get("persona_memory", {})
            if not isinstance(memory, dict) or type(memory.get("first_person_eligible")) is not bool:
                raise ValueError(f"Explicit speaker eligibility is required: {record.id}")
            excerpt = memory.get("verbatim_excerpt", "")
            if not isinstance(excerpt, str):
                raise ValueError(f"Invalid original excerpt: {record.id}")
            if excerpt:
                excerpts.setdefault(record.source_id, set()).add(excerpt)
            evidence[record.id] = item
            added_ids.add(record.id)
    for source_id, values in excerpts.items():
        if sum(len(value.split()) for value in values) > 25:
            raise ValueError(f"Short-excerpt budget exceeded for {source_id}")
    profile = original.persona.model_dump(mode="json")
    profile.update({
        "tagline": "Jensen Huang — public-source AI simulation",
        "description": "A clearly labeled AI simulation conversing in Jensen Huang's first person, grounded in reviewed public experiences and decision criteria. Historical statements retain their dates; new advice is an application of those criteria, not an authentic quotation or private memory.",
        "persona_contract": "jensen-public-first-person.v1",
        "knowledge_version": "jensen-public-rag.v3",
        "style_rules": [
            "Speak as the simulated Jensen in first person when the retrieved source actually describes his experience or stated view. Keep the simulation label in the interface and answer identity questions honestly.",
            "Lead with the answer or a concrete premise check, then a sourced experience or a system constraint. Do not repeatedly introduce answers with I'd start with.",
            "Distinguish an absent market from a disproven technical premise. Apply a criterion only when the supporting interview establishes it.",
            "Use citations as source metadata; do not default to news-reader framing. Corporate announcements are dated company claims, not personal eyewitness memories.",
            "A retrieval miss is missing evidence, never proof an experience did not happen. Do not invent, affirm, or deny private/unsupported biography.",
            "Do not pretend the student team's implementation is Jensen's or NVIDIA's work. Preserve team attribution for project questions.",
        ],
    })
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "profile.json", profile)
    write_json(output / "sources.json", [sources[key] for key in sorted(sources)])
    (output / "evidence.jsonl").write_text("".join(json.dumps(evidence[key], ensure_ascii=False, separators=(",", ":")) + "\n" for key in sorted(evidence)), encoding="utf-8")
    events = {}
    for item in sources.values():
        source = Source.model_validate(item)
        event = source_event_id(source)
        group = events.setdefault(event, {"source_event_id": event, "split": source.split, "source_ids": []})
        if group["split"] != source.split:
            raise CorpusError(f"Event spans splits: {event}")
        group["source_ids"].append(source.id)
    for group in events.values():
        group["source_ids"].sort()
    write_json(output / "split-manifest.json", {"schema_version": "source-event-split.v1", "events": [events[key] for key in sorted(events)]})
    manifest = {"id": "jensen-public-rag.v3", "built_as_of": AS_OF.isoformat(),
                "baseline": "jensen-huang-values-v2", "inherited_records": len(inherited_ids),
                "added_records": len(added_ids), "acquisition_shard_sha256": shard_hashes,
                "source_content_sha256": (original.manifest or {}).get("source_content_sha256", {}),
                "content_note": "Bounded editorial paraphrases, source-linked decision annotations and explicitly marked short verbatim excerpts; no newly acquired full transcripts, audio, weights or acoustic alignment.",
                "review_note": "Agent review against public publisher pages; not human certification or a measured likeness score."}
    write_json(output / "corpus-manifest.json", manifest)
    corpus = Corpus.load(output)
    snapshot = register_snapshot(corpus, identifier="jensen-public-2026-11-24.v3", as_of=AS_OF,
                                 ingested_at=AS_OF, current_evidence_since=date(2026, 1, 1))
    snapshot.verify(corpus)
    write_json(ROOT / "data/snapshots/jensen-public-2026-11-24.json", snapshot.model_dump(mode="json"))
    report = {"sources": len(corpus.sources), "event_families": len(events),
              "records": len(corpus.evidence), "retrieval_records": len(corpus.training_evidence),
              "added_records": len(added_ids), "new_records_with_short_original_excerpt": sum(bool((evidence[key].get("persona_memory") or {}).get("verbatim_excerpt")) for key in added_ids),
              "source_types": dict(sorted(Counter(s.source_type for s in corpus.sources.values()).items())),
              "splits": dict(Counter(s.split for s in corpus.sources.values())),
              "quality": corpus.quality_report(), "corpus_sha256": snapshot.corpus_sha256}
    write_json(output / "build-report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, default=ACQUISITION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.acquisition, args.output), ensure_ascii=False, indent=2))
