from __future__ import annotations

import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from . import captions, media
from .common import (PipelineError, boolean, digest, identifier, interval, now, read_csv, read_json,
                     run_id, text_hash, write_csv, write_json, write_jsonl)

FIELDS = ["segment_id", "start", "end", "text", "speaker", "turn_id", "kind", "topic", "situation",
          "context", "translation_ko", "reviewed", "voice_clean", "notes"]


def source_folder_name(source_id, title=""):
    safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title)
    safe_title = " ".join(safe_title.split()).encode("utf-8")[:140].decode("utf-8", "ignore").strip(" .")
    return f"{safe_title} [{source_id}]" if safe_title else source_id


def source_paths(root):
    base = Path(root) / "sources"
    return [p for p in sorted(base.glob("*/source.json")) if not p.parent.name.startswith(".")]


def source_dir(root, source_id):
    sid = identifier(source_id)
    base = Path(root) / "sources"
    legacy = base / sid
    if (legacy / "source.json").exists():
        return legacy
    for metadata in source_paths(root):
        if metadata.parent.name.endswith(f" [{sid}]") and read_json(metadata)["source_id"] == sid:
            return metadata.parent
    raise PipelineError(f"등록되지 않은 출처: {sid}")


def sources(root):
    return sorted((read_json(p) for p in source_paths(root)), key=lambda source: source["source_id"])


def save_source_attribution(root, source_id, url, publisher, published_at):
    path = source_dir(root, source_id) / "source.json"
    source = read_json(path)
    if not all((url, publisher, published_at)):
        raise PipelineError("출처 URL, 게시 채널·기관, 게시일을 입력하세요.")
    write_json(path, {**source, "url": url, "channel": publisher, "upload_date": published_at.isoformat()})
    return path


def organize_sources(root):
    """Move titled legacy folders when the caller has exclusive access to the sources."""
    moved = []
    for metadata in source_paths(root):
        source = read_json(metadata)
        sid = identifier(source["source_id"])
        if metadata.parent.name != sid:
            continue
        dest = metadata.parent.with_name(source_folder_name(sid, source.get("title") or ""))
        if dest != metadata.parent and not dest.exists():
            metadata.parent.rename(dest)
            moved.append(str(dest))
    return moved


def register(root, source_id, recording_id, split="train", target="jensen-huang", language="en",
             media_path=None, subtitle_path=None, url="", notes="", synthetic=False, title=None,
             publisher="", published_at=None):
    source_id, recording_id, target = map(identifier, (source_id, recording_id, target))
    if split not in {"train", "eval"}:
        raise PipelineError("split은 train 또는 eval입니다.")
    base = Path(root) / "sources"
    base.mkdir(parents=True, exist_ok=True)
    if any(old["source_id"] == source_id for old in sources(root)) or (base / source_id).exists():
        raise PipelineError("같은 출처 ID가 있습니다. 기존 파일은 덮어쓰지 않습니다.")
    if not media_path and not subtitle_path and not url:
        raise PipelineError("영상·음성·자막 파일 또는 URL을 입력하세요.")
    for old in sources(root):
        if old["recording_id"] == recording_id and old["split"] != split:
            raise PipelineError("같은 촬영분을 검색용과 평가용으로 나눌 수 없습니다.")
    tmp = Path(tempfile.mkdtemp(prefix=".ingest-", dir=base))
    try:
        extra = {}
        if url and not media_path and not subtitle_path:
            media_path, subtitle_path, extra = media.download_youtube(url, tmp / "download", language)
        info = {"schema_version": "1.0", "source_id": source_id, "recording_id": recording_id,
                "target_id": target, "split": split, "language": language, "url": url,
                "notes": notes, "synthetic": synthetic, "created_at": now(), **extra}
        original = media_path or subtitle_path
        if title is None:
            title = Path(original).stem if original and not synthetic else ""
        info["title"] = str(info.get("title") or title)
        if publisher:
            info["channel"] = publisher
        if published_at:
            info["upload_date"] = published_at.isoformat()
        dest = base / source_folder_name(source_id, info["title"])
        for label, file in [("media", media_path), ("subtitle", subtitle_path)]:
            if not file:
                info[label + "_file"] = None
                continue
            file = Path(file).expanduser().resolve()
            if not file.is_file():
                raise PipelineError(f"파일이 없습니다: {file}")
            if label == "media" and file.suffix.lower() not in media.MEDIA_SUFFIXES:
                raise PipelineError("지원하는 영상·음성 파일을 선택하세요.")
            fingerprint = digest(file)
            if any(old.get(label + "_sha256") == fingerprint for old in sources(root)):
                raise PipelineError(f"이미 등록된 {label} 파일입니다. 중복 수집을 중단했습니다.")
            name = label + file.suffix.lower()
            shutil.copy2(file, tmp / name)
            info[label + "_file"] = name
            info[label + "_sha256"] = fingerprint
        if subtitle_path:
            segments = captions.parse(tmp / info["subtitle_file"])
            write_json(tmp / "transcript.raw.json", {"method": "provided_subtitles", "segments": segments})
            write_csv(tmp / "review.csv", review_template(segments), FIELDS)
        write_json(tmp / "source.json", info)
        shutil.rmtree(tmp / "download", ignore_errors=True)
        tmp.rename(dest)
        return dest
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def review_template(segments):
    return [{**{k: "" for k in FIELDS}, **{k: s[k] for k in ("segment_id", "start", "end", "text")},
             "speaker": "unknown", "kind": "qa", "reviewed": False, "voice_clean": False} for s in segments]


def transcribe_source(root, source_id, model="small", device="cpu", language="en"):
    path = source_dir(root, source_id)
    source = read_json(path / "source.json")
    if (path / "review.csv").exists():
        raise PipelineError("검수표가 이미 있습니다. 새 출처 ID로 다른 전사 방법을 비교하세요.")
    if not source["media_file"]:
        raise PipelineError("전사할 영상·음성이 없습니다.")
    segments, metadata = media.transcribe(path / source["media_file"], model, device, language,
                                         Path(root) / "models")
    write_json(path / "transcript.raw.json", {**metadata, "segments": segments})
    write_csv(path / "review.csv", review_template(segments), FIELDS)
    return path / "review.csv"


def validate_review(rows, duration=None):
    seen = set()
    cleaned = []
    for row in rows:
        row = {k: row.get(k, "") for k in FIELDS}
        segment_id = identifier(row["segment_id"])
        if segment_id in seen:
            raise PipelineError(f"중복 segment_id: {segment_id}")
        seen.add(segment_id)
        row["start"], row["end"] = interval(row["start"], row["end"], duration)
        for k in FIELDS:
            if k not in {"start", "end", "reviewed", "voice_clean"}:
                row[k] = str(row[k] or "").strip()
        row["reviewed"], row["voice_clean"] = boolean(row["reviewed"]), boolean(row["voice_clean"])
        if not row["text"]:
            raise PipelineError(f"{segment_id}: 전사 텍스트가 비었습니다.")
        if row["speaker"] not in {"target", "interviewer", "other", "unknown"}:
            raise PipelineError(f"{segment_id}: speaker는 target/interviewer/other/unknown 중 하나입니다.")
        if row["kind"] not in {"qa", "style", "exclude"}:
            raise PipelineError(f"{segment_id}: kind는 qa/style/exclude 중 하나입니다.")
        if row["turn_id"]:
            identifier(row["turn_id"])
        if row["reviewed"] and row["speaker"] == "unknown":
            raise PipelineError(f"{segment_id}: 화자를 확인한 뒤 reviewed를 체크하세요.")
        if row["voice_clean"] and (not row["reviewed"] or row["speaker"] != "target"):
            raise PipelineError(f"{segment_id}: 음성 참조는 검수된 대상 화자만 선택할 수 있습니다.")
        cleaned.append(row)
    return sorted(cleaned, key=lambda r: (r["start"], r["end"], r["segment_id"]))


def save_review(root, source_id, rows, *, reviewer=None):
    path = source_dir(root, source_id)
    duration = media.audio_info(path / "audio.wav")["duration"] if (path / "audio.wav").exists() else None
    rows = validate_review(rows, duration)
    if not rows:
        raise PipelineError("검수표가 비었습니다.")
    history = path / "review_history"
    history.mkdir(exist_ok=True)
    if (path / "review.csv").exists():
        shutil.copy2(path / "review.csv", history / (run_id("review") + ".csv"))
    write_csv(path / "review.csv", rows, FIELDS)
    if reviewer:
        write_json(path / "review_provenance.json", {
            "source_id": source_id, "review_sha256": digest(path / "review.csv"),
            "reviewed_at": now(), "reviewer": reviewer,
        })
    return path / "review.csv"


def prepare_audio(root, source_id):
    path = source_dir(root, source_id)
    source = read_json(path / "source.json")
    if not source["media_file"]:
        raise PipelineError("이 출처에는 영상·음성 파일이 없습니다.")
    output = path / "audio.wav"
    if not output.exists():
        media.to_wav(path / source["media_file"], output)
    return output


def collect_source(source, rows):
    groups, skipped = defaultdict(list), []
    for r in rows:
        if r["turn_id"] and r["kind"] != "exclude":
            groups[r["turn_id"]].append(r)
    records = []
    for tid, group in groups.items():
        if any(not r["reviewed"] for r in group):
            skipped.append({"source_id": source["source_id"], "turn_id": tid, "reason": "unreviewed"})
            continue
        kinds = {r["kind"] for r in group}
        if len(kinds) != 1:
            raise PipelineError(f"{tid}: 한 문답의 kind를 통일하세요.")
        targets = [r for r in group if r["speaker"] == "target"]
        questions = [r for r in group if r["speaker"] == "interviewer"]
        kind = next(iter(kinds))
        if not targets or (kind == "qa" and not questions):
            raise PipelineError(f"{tid}: 문답은 질문과 대상 답변이 필요합니다. 독백은 style로 지정하세요.")
        if kind == "qa" and max(r["end"] for r in questions) > min(r["start"] for r in targets) + 0.05:
            raise PipelineError(f"{tid}: 질문·답변이 겹치거나 여러 교환이 섞였습니다. turn_id와 구간을 나눠주세요.")
        ordered = sorted(targets, key=lambda r: r["start"])
        if any(b["start"] < a["end"] - 0.05 for a, b in zip(ordered, ordered[1:])):
            raise PipelineError(f"{tid}: 대상 화자의 자막 시간이 겹칩니다. 롤링 자막을 병합하고 중복을 제거하세요.")
        def joined(field, selected=group):
            return " | ".join(dict.fromkeys(r[field] for r in selected if r[field]))
        question = " ".join(r["text"] for r in questions)
        answer = " ".join(r["text"] for r in ordered)
        records.append({"schema_version": "1.0", "dialogue_id": source["source_id"] + "__" + tid,
                        "source_id": source["source_id"], "recording_id": source["recording_id"],
                        "target_id": source["target_id"], "split": source["split"], "kind": kind,
                        "language": source["language"], "synthetic": source["synthetic"],
                        "question": question, "answer_verbatim": answer,
                        "translation_ko": joined("translation_ko", ordered), "topic": joined("topic"),
                        "situation": joined("situation"), "context": joined("context"),
                        "search_text": question + " " + joined("topic"),
                        "start": min(r["start"] for r in group), "end": max(r["end"] for r in group),
                        "answer_start": ordered[0]["start"], "answer_end": ordered[-1]["end"],
                        "segment_ids": [r["segment_id"] for r in group], "review_status": "approved"})
    return records, skipped


def build(root):
    root = Path(root)
    registered = sources(root)
    if not registered:
        raise PipelineError("출처를 먼저 등록하세요.")
    if len({s["target_id"] for s in registered}) != 1 or len({s["synthetic"] for s in registered}) != 1:
        raise PipelineError("한 작업 폴더에는 같은 인물의 데이터만 넣으세요. 예제와 실제 자료도 분리합니다.")
    assignments, hashes, answer_hashes = {}, {}, {}
    all_rows, skipped, duplicates, snapshots = [], [], [], []
    reviews = {}
    for source in registered:
        rid, split = source["recording_id"], source["split"]
        if assignments.setdefault(rid, split) != split:
            raise PipelineError(f"촬영분 분리 위반: {rid}")
        for key in ("media_sha256", "subtitle_sha256"):
            if source.get(key):
                if hashes.setdefault(source[key], split) != split:
                    raise PipelineError("동일 원본 파일이 검색용과 평가용에 동시에 들어 있습니다.")
        path = source_dir(root, source["source_id"])
        if not (path / "review.csv").exists():
            skipped.append({"source_id": source["source_id"], "reason": "no_transcript"})
            continue
        duration = None
        if source["media_file"]:
            duration = media.audio_info(prepare_audio(root, source["source_id"]))["duration"]
        review_path = path / "review.csv"
        review_hash = digest(review_path)
        rows = validate_review(read_csv(review_path), duration)
        if review_hash != digest(review_path):
            raise PipelineError(f"{source['source_id']}: 검수표가 저장되는 중입니다. 다시 만들어 주세요.")
        provenance = None
        for name in ("expanded_review_notes.json", "pilot_review_notes.json", "review_provenance.json"):
            candidate = path / name
            if candidate.exists():
                entry = read_json(candidate)
                if entry.get("source_id") == source["source_id"] and entry.get("review_sha256") == review_hash:
                    provenance = entry
                    break
        raw_path = path / "transcript.raw.json"
        transcript_hash = digest(raw_path) if raw_path.exists() else None
        raw = read_json(raw_path) if transcript_hash else {}
        if provenance and provenance.get("source_transcript_sha256") not in {None, transcript_hash}:
            raise PipelineError(f"{source['source_id']}: 검수한 원본 자막이 변경됐습니다.")
        records, incomplete = collect_source(source, rows)
        skipped.extend(incomplete)
        snapshots.append({**source, "review_sha256": review_hash,
                          "unassigned_segments": sum(not r["turn_id"] for r in rows),
                          "pending_segments": sum(not r["reviewed"] for r in rows)})
        reviews[source["source_id"]] = {
            "review_sha256": review_hash, "transcript_sha256": transcript_hash,
            "transcript_method": raw.get("method") or raw.get("engine"),
            "provenance": provenance, "segments": [row for row in rows if row["reviewed"]],
        }
        for record in records:
            fingerprint = text_hash(record["answer_verbatim"])
            old = answer_hashes.get(fingerprint)
            if old and old["split"] != split:
                raise PipelineError("검색용·평가용에 같은 답변이 있습니다: " + old["dialogue_id"] + " / " + record["dialogue_id"])
            if old:
                duplicates.append(record["dialogue_id"])
                continue
            answer_hashes[fingerprint] = record
            all_rows.append(record)
    train = [r for r in all_rows if r["kind"] == "qa" and r["split"] == "train"]
    evaluation = [r for r in all_rows if r["kind"] == "qa" and r["split"] == "eval"]
    if not all_rows:
        raise PipelineError("내보낼 검수 자료가 없습니다. 화자·turn_id·reviewed를 확인하세요.")
    output = root / "datasets" / run_id("dataset")
    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "eval.jsonl", evaluation)
    write_jsonl(output / "style.jsonl", [r for r in all_rows if r["kind"] == "style" and r["split"] == "train"])
    write_jsonl(output / "style.eval.jsonl", [r for r in all_rows if r["kind"] == "style" and r["split"] == "eval"])
    write_json(output / "sources.json", snapshots)
    write_json(output / "reviews.json", reviews)
    write_json(output / "build_report.json", {"created_at": now(), "train_qa": len(train), "eval_qa": len(evaluation),
               "skipped": skipped, "duplicates_removed": duplicates, "warnings":
               ([] if train and evaluation else ["A/B/C 평가를 위해 검색용과 별도 촬영분의 평가용 문답이 모두 필요합니다."]),
               "source_review_hashes": {s["source_id"]: s["review_sha256"] for s in snapshots}})
    write_json(root / "latest_dataset.json", {"path": str(output.resolve())})
    return output


def reference(root, source_id, segment_id):
    path = source_dir(root, source_id)
    source = read_json(path / "source.json")
    if source["split"] != "train":
        raise PipelineError("평가 촬영분은 음성 참조로 사용할 수 없습니다.")
    wav = prepare_audio(root, source_id)
    rows = validate_review(read_csv(path / "review.csv"), media.audio_info(wav)["duration"])
    row = next((r for r in rows if r["segment_id"] == segment_id), None)
    if not row or not row["reviewed"] or not row["voice_clean"] or row["speaker"] != "target" or row["kind"] == "exclude":
        raise PipelineError("검수한 대상 단독 음성에서 voice_clean을 체크하세요. exclude 구간은 사용할 수 없습니다.")
    output = Path(root) / "references" / run_id(source_id + "-" + identifier(segment_id))
    output.mkdir(parents=True)
    audio = media.cut_wav(wav, output / "reference.wav", row["start"], row["end"])
    write_json(output / "reference.json", {"schema_version": "1.0", "source_id": source_id,
               "recording_id": source["recording_id"], "target_id": source["target_id"], "split": "train",
               "segment_id": segment_id, "start": row["start"], "end": row["end"], "language": source["language"],
               "reference_audio": "reference.wav", "reference_text": row["text"], "reviewed": True,
               "voice_clean": True, "synthetic": source["synthetic"], "review_sha256": digest(path / "review.csv"),
               "audio_sha256": digest(output / "reference.wav"), **audio})
    return output / "reference.json"
