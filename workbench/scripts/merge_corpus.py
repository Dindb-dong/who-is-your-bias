"""Merge reviewed data, with shared event IDs and independent media status."""

from collections import Counter, defaultdict
from datetime import datetime
import re
from hashlib import sha256
from zoneinfo import ZoneInfo

from pipeline import dataset, media
from pipeline.common import PipelineError, digest, read_json, read_jsonl, text_hash, write_json


def url_key(url):
    from bias.corpus import canonical_source_url

    if not url:
        return None
    try:
        return "yt-" + media.youtube_video_id(url)
    except PipelineError:
        return canonical_source_url(url)


class Merge:
    def __init__(self, root, seed, upstream, local_sources, evaluations):
        self.root = root
        self.upstream = upstream
        self.sources = {s["id"]: s for s in read_json(seed / "sources.json")}
        self.records = {r["id"]: r for r in read_jsonl(seed / "evidence.jsonl")}
        self.content_hashes = {}
        self.skips = []
        self.imported = defaultdict(Counter)
        self.reserved = set()
        self.event_aliases = {}
        self.heldout_urls = set()
        self.heldout_hashes = set()
        self.heldout_answers = {text_hash(r["answer_verbatim"]) for r in evaluations}
        self.answers = {evidence_key(r) for r in self.records.values()}
        self.restore = []
        for item in self.sources.values():
            self.register_event(item)
        registry = upstream / "data/acquisition/jensen-huang"
        if (registry / "source-families.json").exists():
            for item in read_json(registry / "source-families.json")["sources"]:
                self.register_event(item)
            for item in read_json(registry / "heldout-documents.json")["documents"]:
                self.heldout_urls.add(url_key(item["acquisition_url"]))
        for item in local_sources:
            if item["split"] != "train":
                self.register_event({**item, "id": item["source_id"], "source_event_id": item["recording_id"]})
        for sid, source in self.sources.items():
            if source["split"] != "train" or self.event_aliases.get(url_key(source["url"])) in self.reserved:
                raise PipelineError(f"기본 시연 자료에 평가 전용 출처가 있습니다: {sid}")

    def register_event(self, source):
        sid, event = source["id"], source.get("source_event_id", source["id"])
        key = url_key(source.get("url"))
        for alias in (sid, event, key):
            if alias:
                existing = self.event_aliases.get(alias)
                if existing in self.reserved and source["split"] == "train":
                    raise PipelineError(f"검색용과 평가용이 같은 원본을 사용합니다: {sid}")
                self.event_aliases[alias] = event
        if source["split"] != "train":
            self.reserved.update((sid, event))
            if key:
                self.heldout_urls.add(key)
            self.heldout_hashes.update(source[k] for k in ("media_sha256", "subtitle_sha256") if source.get(k))

    def skip(self, identifier, reason, count=1):
        self.skips.append({"id": identifier, "reason": reason, "rows": count})

    def add(self, source, record, origin, workbench=None):
        from bias.models import Evidence, Source

        sid, rid = source["id"], record["id"]
        supplied_event = source.get("source_event_id", sid)
        key = url_key(source["url"])
        event = self.event_aliases.get(key) or self.event_aliases.get(supplied_event, supplied_event)
        # These titles identify the reserved interviews even if a mirror uses
        # a different platform or uploader. Other edited copies use the event ID.
        title = re.sub(r"[^a-z0-9]+", " ", source["title"].lower())
        reserved_title = (
            ("jensen" in title and ("no priors" in title or "computing platform underlying ai" in title))
            or ("jensen" in title and ("davos" in title or "world economic forum" in title) and "2026" in title)
            or (re.search(r"\bgtc\s*(?:20)?26\b", title) and "analyst" in title)
        )
        hashes = {workbench[k] for k in ("media_sha256", "subtitle_sha256") if workbench and workbench.get(k)}
        if (source["split"] != "train" or {sid, supplied_event, event} & self.reserved
                or key in self.heldout_urls or hashes & self.heldout_hashes or reserved_title):
            self.skip(rid, "heldout_event")
            return False
        qa = record.get("question_answer")
        fingerprint = text_hash(qa["answer"] if qa else record["text"])
        if fingerprint in self.heldout_answers:
            self.skip(rid, "heldout_answer_overlap")
            return False
        evidence_fingerprint = evidence_key(record)
        if rid in self.records or evidence_fingerprint in self.answers:
            self.skip(rid, "duplicate_evidence")
            return False
        existing = self.sources.get(sid)
        if existing and (url_key(existing["url"]) != key or existing.get("source_event_id", sid) != event):
            self.skip(rid, "source_identity_conflict")
            return False
        source = {**source, "source_event_id": event}
        record = {**record, "question_answer": {**qa, "source_event_id": event}} if qa else record
        try:
            Source.model_validate(source)
            Evidence.model_validate(record)
        except ValueError as exc:
            self.skip(rid, f"invalid_record: {exc.errors(include_input=False)[0]['msg']}")
            return False
        # The PR's original dialogue and editorial summary share a source ID.
        # Preserve the publisher metadata of the original when both are present.
        if not existing or origin == "pr19":
            self.sources[sid] = source
        self.register_event(source)
        self.records[rid] = record
        self.answers.add(evidence_fingerprint)
        self.imported[origin]["records"] += 1
        self.imported[origin]["qa"] += bool(qa)
        self.imported[origin]["factual_qa"] += bool(qa and record["projection_eligibility"]["factual"])
        if workbench and workbench.get("media_sha256"):
            self.content_hashes[sid] = workbench["media_sha256"]
        if source.get("transcript_sha256"):
            self.content_hashes[sid] = source["transcript_sha256"]
        return True

    def import_originals(self):
        recipe_path = self.upstream / "reports/jensen-recovery/corpus-v4/materialization-recipes.json"
        pointer = self.root / "integration/pr19-corpus.json"
        if not recipe_path.exists() or not pointer.exists():
            self.restore.append({"status": "original_corpus_unavailable"})
            return
        base = (self.root / "integration").resolve()
        directory = (base / read_json(pointer)["relative_path"]).resolve()
        if not directory.is_relative_to(base):
            raise PipelineError("복원 자료는 이 프로젝트의 integration 데이터 폴더 안에 있어야 합니다.")
        recipe = read_json(recipe_path)
        sources = {s["id"]: s for s in read_json(directory / "sources.json")}
        records = defaultdict(list)
        for item in read_jsonl(directory / "evidence.jsonl"):
            records[item["source_id"]].append(item)
        expected = defaultdict(dict)
        for item in recipe["selections"]:
            expected[item["source_id"]][item["id"]] = item
        for reference in recipe["sources"]:
            sid = reference["source_id"]
            source = sources.get(sid)
            actual = records[sid]
            expected_count = len(expected[sid])
            if not source:
                self.restore.append({"source_id": sid, "status": "unavailable", "expected": expected_count})
                continue
            valid = (source.get("transcript_sha256") == reference["transcript_sha256"]
                     and len(actual) == expected_count and {r["id"] for r in actual} == expected[sid].keys()
                     and all(matches_selection(r, expected[sid][r["id"]]) for r in actual))
            if not valid:
                self.restore.append({"source_id": sid, "status": "hash_mismatch", "expected": expected_count})
                continue
            count = sum(self.add(source, r, "pr19") for r in actual)
            self.restore.append({"source_id": sid, "status": "verified", "expected": expected_count, "imported": count})

    def split_manifest(self):
        events = {}
        for item in self.sources.values():
            event = item.get("source_event_id", item["id"])
            entry = events.setdefault(event, {"source_event_id": event, "split": item["split"], "source_ids": []})
            if entry["split"] != item["split"]:
                raise PipelineError(f"촬영분 검색·평가 분리 위반: {event}")
            entry["source_ids"].append(item["id"])
        return {"events": list(events.values())}

    def summary(self):
        return {"imports": {key: dict(counts) for key, counts in self.imported.items()}, "skipped": self.skips,
                "reserved_events": sorted(self.reserved), "pr19_restore": self.restore,
                "total_sources": len(self.sources), "total_records": len(self.records)}


def matches_selection(record, expected):
    qa = record["question_answer"]
    return (all(sha256(qa[k].encode()).hexdigest() == expected[f"{k}_sha256"] for k in ("question", "answer"))
            and qa["completeness"] == expected["completeness"] and all(
                qa["transcript_locator"].get(k) == v for k, v in expected["transcript_locator"].items()))


def evidence_key(record):
    qa = record.get("question_answer")
    if not qa:
        return text_hash(record["text"])
    # Keep repeated short acknowledgements attached to their distinct questions.
    return text_hash(qa["answer"]) if len(qa["answer"].split()) >= 8 else text_hash(qa["question"] + " " + qa["answer"])


def workbench_review(snapshot, built_at):
    proof = snapshot.get("provenance") or {}
    stamp = proof.get("reviewed_at") or built_at
    review = {"reviewer_type": proof.get("reviewer_type", "human" if proof.get("reviewer") else "agent"),
              "reviewer": proof.get("reviewer") or ("Codex" if proof.get("reviewer_type") == "agent" else "Legacy workbench review"),
              "reviewed_at": datetime.fromisoformat(stamp).astimezone(ZoneInfo("Asia/Seoul")).date().isoformat() if "T" in stamp else stamp,
              "basis": proof.get("review_basis") or "Checked speaker and text in the workbench review table; acoustic boundaries not independently verified.",
              "speaker_status": "publisher_labels_checked", "timing_status": "not_audio_verified"}
    return {item["segment_id"]: item for item in proof.get("segments", [])}, review


def media_alignment(root, source, row, review_rows, snapshot):
    path = dataset.source_dir(root, source["source_id"])
    filename = source.get("media_file")
    asset = path / filename if filename else None
    available = bool(asset and asset.is_file())
    is_video = available and asset.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
    target_rows = [review_rows[sid] for sid in row["segment_ids"] if review_rows[sid]["speaker"] == "target"]
    return {
        "dialogue_id": row["dialogue_id"], "source_id": row["source_id"], "source_event_id": source["source_event_id"],
        "source_url": source["url"], "timeline": source["timeline"], "time_basis": "transcript_proxies",
        "answer_sha256": sha256(row["answer_verbatim"].encode()).hexdigest(),
        "review_sha256": snapshot["review_sha256"], "transcript_sha256": snapshot["transcript_sha256"],
        "media_file": str(asset.relative_to(root)) if available else None, "media_sha256": source.get("media_sha256"),
        "utterance_segments": [{"segment_id": s["segment_id"], "speaker": s["speaker"], "start": s["start"],
                                "end": s["end"]} for s in (review_rows[sid] for sid in row["segment_ids"])],
        "audio": {"status": "candidate" if available else "missing_media", "start": row["answer_start"],
                  "end": row["answer_end"], "voice_clean_segment_ids": [r["segment_id"] for r in target_rows if r["voice_clean"]]},
        "face": {"status": "candidate" if is_video else "missing_video", "start": row["answer_start"], "end": row["answer_end"]},
    }


def prepare_media(root, run, dialogue_id, start=None, end=None):
    from tools.media.prepare import prepare_clip
    from pipeline.common import run_id

    root = root.resolve()
    row = next((r for r in read_jsonl(run / "aligned-media.jsonl") if r["dialogue_id"] == dialogue_id), None)
    if row is None or not row["media_file"]:
        raise PipelineError("해당 문답에 검수 데이터와 연결된 로컬 미디어가 없습니다.")
    path = (root / row["media_file"]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or digest(path) != row["media_sha256"]:
        raise PipelineError("원본 미디어가 변경됐습니다. 새 데이터셋을 만들어 주세요.")
    output = run / "media-candidates" / run_id("clip")
    audio = row["audio"]
    if (start is None) != (end is None):
        raise PipelineError("시작과 끝을 함께 지정하세요.")
    if start is None:
        start, end = audio["start"], audio["end"]
    if not audio["start"] <= start < end <= audio["end"] or not 0.1 <= end - start <= 120:
        raise PipelineError("답변 범위 안에서 0.1~120초 사이로 시작과 끝을 지정하세요.")
    source_id = row["source_id"]
    tool_source_id = source_id if source_id == source_id.lower() else "yaicon-" + sha256(source_id.encode()).hexdigest()[:20]
    prepare_clip(path, output, source_id=tool_source_id, source_url=row["source_url"],
                 source_timeline=row["timeline"], start=start, duration=end - start,
                 include_video=row["face"]["status"] == "candidate")
    write_json(output / "dialogue-link.json", {**row, "media_tool_source_id": tool_source_id,
                                              "selected_clip": {"start": start, "end": end}})
    return output


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Export source audio and video for one reviewed dialogue ID")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dialogue-id", required=True)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    from tools.media.common import MediaError

    try:
        print(prepare_media(args.root, args.run, args.dialogue_id, args.start, args.end))
    except (PipelineError, MediaError, OSError) as exc:
        parser.exit(2, f"미디어 후보를 만들 수 없습니다: {exc}\n")
