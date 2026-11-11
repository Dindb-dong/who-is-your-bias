"""Materialize pinned, reviewed exchanges locally without distributing transcripts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from bias.corpus import Corpus
from bias.models import Evidence

from .acquire import AcquisitionError, acquire


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def selected_text(transcript: dict, cue_ids: list[str], speaker: str, reviewed_continuations: list[str] | None = None) -> str:
    by_id = {row["id"]: (index, row) for index, row in enumerate(transcript["segments"])}
    if not cue_ids or len(set(cue_ids)) != len(cue_ids) or any(key not in by_id for key in cue_ids):
        raise AcquisitionError("A selection needs unique, existing cue identifiers")
    indices = [by_id[key][0] for key in cue_ids]
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise AcquisitionError("A source turn selection must be contiguous")
    approved = set(reviewed_continuations or [])
    if not approved.issubset(cue_ids):
        raise AcquisitionError("Continuation approval must refer to selected cues")
    result = []
    anchored = False
    for cue_id in cue_ids:
        _, segment = by_id[cue_id]
        if segment.get("timestamp_order_anomaly"):
            raise AcquisitionError("Reviewed exchanges cannot use a nonmonotonic publisher timestamp")
        attributed = segment.get("speaker")
        if attributed == speaker:
            anchored = True
        elif not (attributed is None and anchored and cue_id in approved):
            raise AcquisitionError(f"Unreviewed or mismatched speaker in {cue_id}")
        if "speaker_spans" in segment:
            spans = [span for span in segment["speaker_spans"] if span["kind"] == "caption_text"]
            if not spans or any(span["speaker"] != speaker for span in spans):
                raise AcquisitionError(f"Mixed or unknown speaker span in {cue_id}")
            result.extend(span["text"] for span in spans)
        else:
            result.append(segment["text"])
    return " ".join(" ".join(result).split())


def materialize_record(selection: dict, transcript: dict, *, bounded: bool = False) -> Evidence:
    if transcript["source_id"] != selection["source_id"] or transcript["publisher_caption_sha256"] != selection["transcript_sha256"]:
        raise AcquisitionError(f"Publisher transcript changed for {selection['id']}; review a new recipe version")
    q_ids, a_ids = selection["question_cue_ids"], selection["answer_cue_ids"]
    question = selected_text(transcript, q_ids, selection["questioner"], selection.get("question_reviewed_continuations"))
    answer = selected_text(transcript, a_ids, selection["respondent"], selection.get("answer_reviewed_continuations"))
    if text_sha(question) != selection["question_sha256"] or text_sha(answer) != selection["answer_sha256"]:
        raise AcquisitionError(f"Original text changed for {selection['id']}")
    segments = {segment["id"]: segment for segment in transcript["segments"]}
    if segments[a_ids[0]]["start"] < segments[q_ids[-1]]["start"]:
        raise AcquisitionError("Answer precedes the end of selected question cues")
    completeness = selection["completeness"]
    excerpt_note = selection.get("excerpt_note")
    if bounded:
        ranges = selection.get("bundled_excerpt")
        if not ranges:
            raise AcquisitionError("No bounded excerpt was declared for this selection")
        for label, original in (("question", question), ("answer", answer)):
            start, end = ranges[label]
            if not 0 <= start < end <= len(original):
                raise AcquisitionError("Invalid excerpt character range")
            if label == "question":
                question = original[start:end]
            else:
                answer = original[start:end]
        completeness = "excerpt"
        excerpt_note = "Bounded verbatim excerpt. The full question, answer and nearby context remain at the publisher; locally hydrate the pinned selection for the experiment. Cue times locate the containing original turns, not individual excerpt words."
    context = selection["context_before"]
    context_kind = selection.get("context_kind", "editorial_summary")
    if context_kind == "original_text":
        context = selected_text(transcript, selection["context_cue_ids"], selection["context_source_speaker"])
        if text_sha(context) != selection["context_sha256"]:
            raise AcquisitionError(f"Original preceding context changed for {selection['id']}")
    qa = {"question": question, "answer": answer, "context_before": context, "context_kind": context_kind,
          "original_language": "en", "questioner": selection["questioner"], "respondent": selection["respondent"],
          "source_event_id": selection["source_event_id"], "completeness": completeness, "excerpt_note": excerpt_note,
          "question_start_seconds": segments[q_ids[0]]["start"], "answer_start_seconds": segments[a_ids[0]]["start"],
          "question_end_seconds": None, "answer_end_seconds": None,
          "question_end_proxy_seconds": segments[q_ids[-1]].get("end"), "answer_end_proxy_seconds": segments[a_ids[-1]].get("end"),
          "source_timeline": transcript["source_timeline"],
          "timing_basis": "publisher_cue_onsets_only" if selection["source_id"].startswith("ted-") else "publisher_transcript_timestamps_only",
          "transcript_locator": {"question_cue_ids": q_ids, "answer_cue_ids": a_ids, "context_cue_ids": selection.get("context_cue_ids", []), "transcript_sha256": selection["transcript_sha256"]},
          "review": {"reviewer_type": "agent", "reviewer": "OpenAI coding agent", "reviewed_at": "2026-11-11",
                     "basis": selection["review_basis"], "speaker_status": "publisher_labels_checked", "timing_status": "not_audio_verified"},
          "annotation_basis": selection["annotation_basis"], "taxonomy_version": "situations.v1"}
    return Evidence.model_validate({"id": selection["id"], "source_id": selection["source_id"], "kind": "question_answer",
                                   "question_answer": qa, "topics": selection["topics"], "situations": selection["situations"],
                                   "review_status": "reviewed", "confidence": "source_grounded", "behavioral_observation": None,
                                   "locator": {"label": f"Original question {q_ids[0]}–{q_ids[-1]}; answer {a_ids[0]}–{a_ids[-1]}. Publisher onset, not acoustic alignment.",
                                               "start_seconds": segments[a_ids[0]]["start"], "timeline": transcript["source_timeline"]}})


def hydrate(recipe_dir: Path, output: Path, *, raw_inputs: dict[str, Path] | None = None, transcripts: dict[str, dict] | None = None, bounded: bool = False, selection_path: Path | None = None) -> dict:
    """Fail closed on changed source text, splits, speakers, or existing output."""
    if output.exists():
        raise AcquisitionError("Output already exists; choose a new directory to preserve the prior corpus")
    selection_path = selection_path or recipe_dir / "selection-recipes.json"
    recipes = json.loads(selection_path.read_text())
    if recipes.get("schema_version") != "qa-selection-recipes.v1":
        raise AcquisitionError("Unsupported Q&A recipe version")
    sources = json.loads((recipe_dir / "sources.json").read_text())
    source_map = {source["id"]: source for source in sources}
    if len(source_map) != len(sources) or not set(raw_inputs or {}).issubset(source_map):
        raise AcquisitionError("Duplicate source or unknown --raw source identifier")
    selections = recipes["selections"]
    if len({selection["id"] for selection in selections}) != len(selections):
        raise AcquisitionError("Duplicate Q&A selection identifier")
    events = json.loads((recipe_dir / "split-manifest.json").read_text())["events"]
    split_map = {event["source_event_id"]: event["split"] for event in events}
    if len(split_map) != len(events) or any(source["source_event_id"] not in split_map or source["split"] != split_map[source["source_event_id"]] for source in sources):
        raise AcquisitionError("Sources disagree with the frozen source-event split manifest")
    loaded = dict(transcripts or {})
    records = []
    for selection in selections:
        if bounded and not selection.get("bundled_excerpt"):
            continue
        source = source_map.get(selection["source_id"])
        if not source or source["source_event_id"] != selection["source_event_id"]:
            raise AcquisitionError("Selection has an unknown source or a different source event")
        if source["id"] not in loaded:
            loaded[source["id"]] = acquire(source, raw_path=(raw_inputs or {}).get(source["id"]))
        records.append(materialize_record(selection, loaded[source["id"]], bounded=bounded))
    if not records:
        raise AcquisitionError("No selected exchanges")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".qa-hydrate-", dir=output.parent))
    try:
        for name in ("profile.json", "sources.json", "split-manifest.json"):
            shutil.copyfile(recipe_dir / name, temporary / name)
        shutil.copyfile(selection_path, temporary / "selection-recipes.json")
        profile = json.loads((temporary / "profile.json").read_text())
        profile["qa_corpus_id"] = recipes["corpus_id"]
        (temporary / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
        (temporary / "evidence.jsonl").write_text("".join(record.model_dump_json(exclude={"text"}) + "\n" for record in records), encoding="utf-8")
        manifest = {"schema_version": "original-qa-corpus.v1", "id": recipes["corpus_id"],
                    "materialization": "bounded_excerpts" if bounded else "full_selected_exchanges_local",
                    "selection_recipe_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                    "split_manifest_sha256": hashlib.sha256((recipe_dir / "split-manifest.json").read_bytes()).hexdigest(),
                    "source_content_sha256": {key: value["publisher_caption_sha256"] for key, value in loaded.items()},
                    "record_count": len(records), "raw_transcripts_distributed": False,
                    "review_note": "Publisher text and speakers checked by an AI agent. No independent human, acoustic alignment or persona fidelity judgment is implied."}
        (temporary / "corpus-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        validated = Corpus.load(temporary)
        report = {**validated.qa_report(), "split_counts": {split: sum(validated.sources[record.source_id].split == split for record in records) for split in ("train", "validation", "test")}}
        (temporary / "materialization-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.rename(output)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
