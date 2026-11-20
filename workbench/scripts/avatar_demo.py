"""Connect the latest workbench dataset to the local avatar studio."""

import re
import subprocess
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from pipeline import dataset
from pipeline.common import PipelineError, read_json, read_jsonl, run_id, text_hash, write_json, write_jsonl

PROJECT = Path(__file__).resolve().parents[1]
_STANDALONE_STUDIO = PROJECT / "integrations" / "who-is-your-bias"
_SHARED_STUDIO = PROJECT.parent if (PROJECT.parent / "src/bias/api.py").is_file() else None
UPSTREAM = _SHARED_STUDIO or _STANDALONE_STUDIO
PR19 = _STANDALONE_STUDIO.with_name("who-is-your-bias-pr19")
GESTURE_STUDIO = _SHARED_STUDIO or _STANDALONE_STUDIO.with_name("who-is-your-bias-gesture")


def source_record(source, split):
    published = source.get("upload_date")
    url = urlparse(source.get("url") or "")
    if not published or not source.get("channel") or url.scheme not in {"http", "https"} or not url.hostname:
        raise PipelineError("시연 출처 정보에 원본 URL, 게시 채널·기관, 게시일이 필요합니다.")
    suffix = Path(source.get("media_file") or "").suffix.lower()
    timeline = "source_audio" if suffix in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aiff"} else "source_video"
    return {
        "id": source["source_id"], "persona_id": source["target_id"],
        "title": source["title"], "url": source["url"], "publisher": source["channel"],
        "published_at": date.fromisoformat(source.get("original_published_at") or str(published)).isoformat(),
        "video_upload_date": date.fromisoformat(str(published)).isoformat(),
        "recorded_at": date.fromisoformat(source["event_date"]).isoformat() if source.get("event_date") else None,
        "event_date_source": source.get("event_date_source"), "source_type": "reviewed_workbench_recording",
        "split": split, "source_event_id": source["recording_id"],
        "timeline": timeline, "ingested_at": source["created_at"],
        "attribution": "Transcribed text and speaker roles accepted in the YAICON workbench. See exchange review.",
    }


def exchange(row, review_by_id, transcript_sha, review, label, annotations, method, timeline):
    from bias.situations import SITUATION_LABELS, classify_situation

    groups = {
        role: [review_by_id[i] for i in row["segment_ids"] if review_by_id[i]["speaker"] == role]
        for role in ("interviewer", "target")
    }
    question, answer = groups["interviewer"], groups["target"]
    situation = row["situation"]
    primary = annotations.get(answer[0]["segment_id"], {})
    excerpt = primary.get("completeness") != "complete_exchange"
    def cue_ids(rows):
        return [cue for r in rows for cue in annotations.get(r["segment_id"], {}).get("source_cue_ids", [r["segment_id"]])]
    return {
        "id": row["dialogue_id"], "source_id": row["source_id"], "persona_id": row["target_id"],
        "kind": "question_answer", "confidence": "source_grounded", "review_status": "reviewed",
        "text_provenance": "reviewed_workbench_transcript", "transcript_method": primary.get("transcript_method", method),
        "topics": [topic.strip() for topic in row["topic"].split("|") if topic.strip()],
        "situations": [situation if situation in SITUATION_LABELS else classify_situation(row["question"])],
        "locator": {"label": label,
                    "start_seconds": row["answer_start"], "end_seconds": row["answer_end"], "timeline": timeline},
        "projection_eligibility": {"factual": True, "text_style": False,
                                   "aligned_speech": False, "visible_behavior": False},
        "question_answer": {
            "question": row["question"], "answer": row["answer_verbatim"],
            "context_before": row["context"], "context_kind": "editorial_summary",
            "original_language": row["language"], "questioner": primary.get("questioner") or "Interviewer", "respondent": "Jensen Huang",
            "source_event_id": row["recording_id"], "completeness": "excerpt" if excerpt else "complete_exchange",
            "excerpt_note": "Selected transcript passage. Full exchange boundaries have not been established." if excerpt else None,
            "question_start_seconds": min(float(r["start"]) for r in question),
            "answer_start_seconds": row["answer_start"],
            "question_end_proxy_seconds": max(float(r["end"]) for r in question),
            "answer_end_proxy_seconds": row["answer_end"],
            "source_timeline": timeline, "timing_basis": "publisher_transcript_timestamps_only",
            "transcript_locator": {
                "question_cue_ids": cue_ids(question)[:300],
                "answer_cue_ids": cue_ids(answer)[:500],
                "transcript_sha256": transcript_sha,
            },
            "review": review, "annotation_basis": review["basis"],
        },
    }


def export(root: Path):
    from bias.corpus import Corpus
    from bias.snapshots import register_snapshot
    from scripts.merge_corpus import Merge, media_alignment, workbench_review

    selected = Path(read_json(root / "latest_dataset.json")["path"])
    if not (selected / "reviews.json").exists():
        selected = dataset.build(root)
    report = read_json(selected / "build_report.json")
    seed = UPSTREAM / "data" / "personas" / "jensen-huang"
    workbench_sources = read_json(selected / "sources.json")
    evaluations = read_jsonl(selected / "eval.jsonl") + read_jsonl(selected / "style.eval.jsonl")
    merge = Merge(root, seed, PR19 if PR19.is_dir() else UPSTREAM, dataset.sources(root), evaluations)
    imported = []
    review_hashes = {}
    alignments = []
    workbench_rows = read_jsonl(selected / "train.jsonl") + read_jsonl(selected / "style.jsonl")
    reviews = read_json(selected / "reviews.json")

    for metadata in workbench_sources:
        sid = metadata["source_id"]
        if metadata["split"] != "train":
            continue
        source_rows = [r for r in workbench_rows if r["source_id"] == sid and r["split"] == "train"]
        if not source_rows:
            continue
        if metadata["synthetic"] or metadata["target_id"] != "jensen-huang" or metadata["language"] not in {"en", "ko"}:
            merge.skip(sid, "unsupported_subject_or_language", len(source_rows))
            continue
        try:
            source = source_record(metadata, "train")
        except ValueError as exc:
            merge.skip(sid, str(exc), len(source_rows))
            continue
        snapshot = reviews[sid]
        review_hash = snapshot["review_sha256"]
        if review_hash != report["source_review_hashes"][sid]:
            raise PipelineError(f"{sid}: 검수 기록과 데이터셋이 일치하지 않습니다.")
        raw_hash = snapshot["transcript_sha256"]
        annotations, review = workbench_review(snapshot, report["created_at"])
        review_rows = {r["segment_id"]: r for r in snapshot["segments"]}
        review_hashes[sid] = review_hash
        for row in source_rows:
            label = f"{metadata['title']} · {row['dialogue_id'].split('__')[-1]}"
            if row["kind"] == "qa":
                record = exchange(row, review_rows, raw_hash or review_hash, review, label, annotations,
                                  snapshot["transcript_method"], source["timeline"])
            else:
                record = {
                    "id": row["dialogue_id"], "source_id": sid, "kind": "transcript",
                    "text": row["answer_verbatim"], "review_status": "reviewed",
                    "topics": [row["topic"]],
                    "locator": {"label": label, "start_seconds": row["start"], "end_seconds": row["end"],
                                "timeline": source["timeline"]},
                    "projection_eligibility": {"factual": False, "text_style": False,
                                               "aligned_speech": False, "visible_behavior": False},
                }
            if not merge.add(source, record, "workbench", metadata):
                continue
            alignments.append(media_alignment(root, {**metadata, "source_event_id": merge.sources[sid]["source_event_id"],
                                                     "timeline": source["timeline"]},
                                              row, review_rows, snapshot))
            imported.append(row)
    merge.import_originals()
    sources, records = list(merge.sources.values()), list(merge.records.values())
    profile = read_json(seed / "profile.json")
    profile["description"] = (
        "A labeled research simulation using public NVIDIA materials and reviewed workbench captions. "
        "Imported Q&A timings are approximate."
    )
    profile["corpus_cutoff"] = max(s["published_at"] for s in sources if s["split"] == "train")
    question = "What did Jensen Huang say about RTX Spark and personal AI at Microsoft Build?"
    profile["example_questions"]["en"] = [question, *profile["example_questions"]["en"]]
    parent = root / "avatar_runs"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / run_id("avatar")
    with tempfile.TemporaryDirectory(prefix=".avatar-", dir=parent) as tmp:
        stage = Path(tmp) / "result"
        corpus_dir = stage / "corpus"
        corpus_dir.mkdir(parents=True)
        write_json(corpus_dir / "profile.json", profile)
        write_json(corpus_dir / "sources.json", sources)
        write_jsonl(corpus_dir / "evidence.jsonl", records)
        write_json(corpus_dir / "split-manifest.json", merge.split_manifest())
        write_json(corpus_dir / "corpus-manifest.json", {
            "id": "jensen-yaicon-interview.v3",
            "materialization": "merged_reviewed_workbench_and_pr19_original_dialogue",
            "source_content_sha256": merge.content_hashes,
        })
        corpus = Corpus.load(corpus_dir)
        stamp = datetime.now(timezone.utc)
        snapshot = register_snapshot(
            corpus, identifier=output.name, as_of=stamp, ingested_at=stamp,
            current_evidence_since=date(2026, 1, 1),
        ).model_copy(update={
            "voice_version": "macos-generic-demo",
            "style_version": "editorial-guidance-no-trained-adapter",
            "appearance_version": "designed-prototype-not-qualified-likeness",
            "coverage_note": "Editorial seed plus reviewed workbench sources. Video publication date does not assert a recording date.",
        })
        snapshot.verify(corpus)
        write_json(stage / "snapshot.json", snapshot.model_dump(mode="json"))
        write_json(stage / "integration.json", {
            **merge.summary(), "workbench_dataset": selected.name, "source_review_hashes": review_hashes,
            "imported_evidence_ids": [r["dialogue_id"] for r in imported], "sample_question": question,
        })
        write_jsonl(stage / "aligned-media.jsonl", alignments)
        stage.rename(output)
    return output.resolve()


def create_tts():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel, Field

    class SpeechInput(BaseModel):
        input: str = Field(min_length=1, max_length=4000)
        language: str | None = None

    app = FastAPI(title="YAICON macOS generic preview voice")

    @app.post("/v1/audio/speech")
    def synthesize(request: SpeechInput):
        voice = "Yuna" if request.language == "Korean" else "Samantha"
        text = re.sub(r"\[\[.*?\]\]", "", request.input, flags=re.S).strip()
        if not text:
            raise HTTPException(422, "Empty speech text.")
        with tempfile.TemporaryDirectory(prefix="yaicon-voice-") as tmp:
            path = Path(tmp)
            source, aiff, wav = (path / name for name in ("text.txt", "voice.aiff", "voice.wav"))
            source.write_text(text, encoding="utf-8")
            try:
                subprocess.run(["/usr/bin/say", "-v", voice, "-f", str(source), "-o", str(aiff)],
                               check=True, capture_output=True, timeout=25)
                subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1",
                                str(aiff), str(wav)], check=True, capture_output=True, timeout=10)
            except (OSError, subprocess.SubprocessError) as exc:
                raise HTTPException(503, "macOS voice generation failed.") from exc
            return Response(wav.read_bytes(), media_type="audio/wav")

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT / "data/local")
    try:
        print(export(parser.parse_args().root))
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        parser.exit(2, f"아바타 자료를 만들 수 없습니다: {exc}\n")
