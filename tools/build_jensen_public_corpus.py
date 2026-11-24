"""Build the reviewable public Jensen RAG corpus without copying full transcripts.

Acquisition shards contain bounded paraphrases and explicitly identified short
original excerpts. Discovery-only social links never become runtime evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from uuid import uuid4
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from bias.corpus import Corpus, CorpusError, source_event_id
from bias.models import Evidence, Source
from bias.snapshots import register_snapshot

ROOT = Path(__file__).resolve().parents[1]
ACQUISITION = ROOT / "data/acquisition/jensen-public-2026-11-24"
OUTPUT = ROOT / "data/personas/jensen-huang-public-v3"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_staged(acquisition: Path, output: Path, *, as_of: datetime | None,
                  snapshot_path: Path) -> dict:
    as_of = as_of or datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        raise ValueError("Snapshot cutoff requires a timezone")
    as_of = as_of.astimezone(timezone.utc)
    baseline = ROOT / "data/personas/jensen-huang-values-v2"
    original = Corpus.load(baseline)
    sources = {key: value.model_dump(mode="json") for key, value in original.sources.items()}
    evidence = {key: value.model_dump(mode="json") for key, value in original.evidence.items()}
    inherited_ids = set(evidence)
    registration = read_json(ROOT / "data/acquisition/jensen-huang/source-families.json")
    reserved = {item.get("source_event_id", item["id"]) for item in registration["sources"]
                if item.get("split") in {"test", "validation"}}
    shard_hashes = {}
    for path in sorted(acquisition.glob("*-sources.json")):
        shard_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for item in read_json(path):
            source = Source.model_validate(item)
            if source.published_at > as_of.date():
                raise ValueError(f"Future source date: {source.id}")
            if source.id in sources:
                previous = Source.model_validate(sources[source.id])
                if source_event_id(source) != source_event_id(previous) or source.split != previous.split:
                    raise ValueError(f"An existing event/split cannot be relabeled: {source.id}")
                # Keep inherited provenance fields unless explicitly re-reviewed.
                sources[source.id] = sources[source.id] | item
            else:
                sources[source.id] = item
    # A historical expanded experiment already exposed No Priors 2023 even
    # though the original acquisition register reserved that event. Preserve
    # that experiment on disk, but do not repeat the conflict in the new index.
    excluded_sources = {key for key, item in sources.items()
                        if source_event_id(Source.model_validate(item)) in reserved}
    sources = {key: item for key, item in sources.items() if key not in excluded_sources}
    evidence = {key: item for key, item in evidence.items() if item["source_id"] not in excluded_sources}
    excluded_records = sorted(inherited_ids - evidence.keys())
    added_ids = set()
    excerpts: dict[str, set[str]] = {}
    for path in sorted(acquisition.glob("*-evidence.jsonl")):
        shard_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            record = Evidence.model_validate(item)
            if record.source_id in excluded_sources:
                excluded_records.append(record.id)
                continue
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
        "corpus_start": min(item.get("recorded_at") or item["published_at"] for item in sources.values()),
        "corpus_cutoff": as_of.date().isoformat(),
        "corpus_date_basis": "Start is the earliest known event/publication date; cutoff is source availability, not a claim of current-position coverage. Each source retains separate event/publication/capture metadata.",
        "style_provenance": "agent-reviewed public statements, editorial annotations and explicitly marked short original excerpts",
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
    manifest = {"id": "jensen-public-rag.v3", "built_as_of": as_of.isoformat(),
                "baseline": "jensen-huang-values-v2", "inherited_records": len(inherited_ids),
                "inherited_retained_records": len(inherited_ids) - len(set(excluded_records) & inherited_ids),
                "reserved_event_exclusions": {"source_ids": sorted(excluded_sources), "evidence_ids": sorted(excluded_records)},
                "added_records": len(added_ids), "acquisition_shard_sha256": shard_hashes,
                "source_content_sha256": {key: item.get("raw_content_sha256") or (original.manifest or {}).get("source_content_sha256", {}).get(key)
                                          for key, item in sources.items()
                                          if item.get("raw_content_sha256") or (original.manifest or {}).get("source_content_sha256", {}).get(key)},
                "content_note": "Runtime records are bounded editorial paraphrases, source-linked decision annotations and explicitly marked short verbatim excerpts. Eight original source documents are preserved in a separate verified archive; no newly prepared complete original Q&A, audio, weights or acoustic alignment.",
                "review_note": "Agent review against public publisher pages; not human certification or a measured likeness score."}
    write_json(output / "corpus-manifest.json", manifest)
    corpus = Corpus.load(output)
    snapshot = register_snapshot(corpus, identifier="jensen-public-2026-11-24.v3", as_of=as_of,
                                 ingested_at=as_of, current_evidence_since=date(2026, 1, 1))
    snapshot = snapshot.model_copy(update={
        "style_version": "source-linked-editorial-cues-and-short-excerpts.v1",
        "voice_version": "runtime-provider-unchanged-not-evaluated",
        "appearance_version": "runtime-avatar-unchanged-not-evaluated",
    })
    snapshot.verify(corpus)
    write_json(snapshot_path, snapshot.model_dump(mode="json"))
    report = {"sources": len(corpus.sources), "event_families": len(events),
              "records": len(corpus.evidence), "retrieval_records": len(corpus.training_evidence),
              "added_records": len(added_ids), "new_records_with_short_original_excerpt": sum(bool((evidence[key].get("persona_memory") or {}).get("verbatim_excerpt")) for key in added_ids),
              "source_types": dict(sorted(Counter(s.source_type for s in corpus.sources.values()).items())),
              "splits": dict(Counter(s.split for s in corpus.sources.values())),
              "quality": corpus.quality_report(), "corpus_sha256": snapshot.corpus_sha256}
    write_json(output / "build-report.json", report)
    return report


def build(acquisition: Path = ACQUISITION, output: Path = OUTPUT, *, as_of: datetime | None = None,
          snapshot_path: Path | None = None) -> dict:
    """Validate a complete staged bundle before replacing the usable corpus.

Publication is an offline preparation step; restart a server after building.
Existing non-generated files such as the README are retained. A failed schema
or snapshot check cannot corrupt the previous corpus, and publication failures
restore its backup.
    """
    output = output.resolve()
    if snapshot_path is None:
        snapshot_path = (ROOT / "data/snapshots/jensen-public-2026-11-24.json"
                         if output == OUTPUT.resolve() else output / "snapshot.json")
    snapshot_path = snapshot_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".jensen-build-", dir=output.parent) as temp:
        scratch = Path(temp)
        staged = scratch / "corpus"
        if output.exists():
            shutil.copytree(output, staged)
        staged_snapshot = scratch / "snapshot.json"
        report = _build_staged(acquisition, staged, as_of=as_of, snapshot_path=staged_snapshot)
        contained_snapshot = snapshot_path.is_relative_to(output)
        if contained_snapshot:
            destination = staged / snapshot_path.relative_to(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staged_snapshot, destination)
        else:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        # Last-good data must survive interruption or a failed restoration; an
        # automatically cleaned temporary directory cannot own that backup.
        backup = output.with_name(f".{output.name}.last-good-{uuid4().hex}")
        had_output = output.exists()
        snapshot_backup = snapshot_path.with_name(f".{snapshot_path.name}.last-good-{uuid4().hex}")
        had_snapshot = not contained_snapshot and snapshot_path.exists()
        if had_snapshot:
            shutil.copyfile(snapshot_path, snapshot_backup)
        published = False
        snapshot_changed = False
        try:
            if had_output:
                os.replace(output, backup)
            os.replace(staged, output)
            published = True
            if not contained_snapshot:
                # Snapshot is installed last, after all its corpus files exist.
                descriptor, pending_name = tempfile.mkstemp(prefix=".jensen-snapshot-", dir=snapshot_path.parent)
                os.close(descriptor)
                pending = Path(pending_name)
                try:
                    shutil.copyfile(staged_snapshot, pending)
                    os.replace(pending, snapshot_path)
                    snapshot_changed = True
                finally:
                    pending.unlink(missing_ok=True)
        except BaseException:
            if published:
                shutil.rmtree(output)
            if backup.exists():
                os.replace(backup, output)
            if snapshot_changed:
                if not had_snapshot:
                    snapshot_path.unlink(missing_ok=True)
                else:
                    os.replace(snapshot_backup, snapshot_path)
            elif snapshot_backup.exists():
                snapshot_backup.unlink()
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup)
            snapshot_backup.unlink(missing_ok=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, default=ACQUISITION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--as-of", type=datetime.fromisoformat,
                        help="Timezone-aware cutoff for exact reproduction; defaults to the actual build time.")
    parser.add_argument("--snapshot", type=Path, help="Explicit output snapshot path; custom corpora otherwise keep it beside their files.")
    args = parser.parse_args()
    print(json.dumps(build(args.acquisition, args.output, as_of=args.as_of, snapshot_path=args.snapshot), ensure_ascii=False, indent=2))
