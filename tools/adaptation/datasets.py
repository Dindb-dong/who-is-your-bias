from __future__ import annotations

import hashlib
import json
import random
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path

from tools.persona_data.parsers import sha_text
from tools.persona_data.pipeline import write_json, write_jsonl


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_projection(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(row.get("purpose") == "evaluation_only" or row.get("schema_version") == "persona-heldout-case.v1" for row in rows):
        raise ValueError("Evaluation-only held-out cases cannot enter adaptation exports")
    ids = [row["segment_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate projection segment")
    for row in rows:
        if row["split"] != "train":
            raise ValueError("Adaptation accepts training projections only")
        if row["completeness"] != "complete_exchange":
            raise ValueError("Adaptation cannot use an incomplete exchange")
        if row["answer_sha256"] != sha_text(row["text"]):
            raise ValueError("Projection original text hash differs")
    return rows


def export_text(projection: Path, output: Path, *, seed: int = 42, max_per_event: int = 150) -> dict:
    if output.exists() or max_per_event < 1:
        raise ValueError("Choose new output and a positive per-event cap")
    rows = read_projection(projection)
    groups = defaultdict(list)
    for row in rows:
        groups[row["event_family_id"]].append(row)
    rng = random.Random(seed)
    selected = []
    for event in sorted(groups):
        group = sorted(groups[event], key=lambda row: row["segment_id"])
        rng.shuffle(group)
        selected.extend(group[:max_per_event])
    selected.sort(key=lambda row: row["segment_id"])
    examples = []
    for row in selected:
        # Factual retrieval is deliberately outside the adaptation dataset. The
        # historical date and simulation instruction remain in every example.
        system = ("You are a disclosed AI simulation using source-attributed dialogue. "
                  f"The following exchange was published on {row['published_at']}; it is historical evidence, not a guarantee of a current opinion. "
                  "Respond to the interviewer using the original speaking style without inventing new personal facts.")
        prompt = row["question"]
        if row.get("context"):
            prompt = "Previous source context:\n" + row["context"] + "\n\nInterviewer:\n" + prompt
        examples.append({"id": row["segment_id"], "event_family_id": row["event_family_id"], "source_id": row["source_id"],
                         "split": "train", "answer_sha256": row["answer_sha256"],
                         "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt},
                                      {"role": "assistant", "content": row["text"]}]})
    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", examples)
    manifest = {"schema_version": "text-adaptation-dataset.v1", "projection_sha256": file_hash(projection),
                "dataset_sha256": file_hash(output / "train.jsonl"), "samples": len(examples), "seed": seed,
                "max_per_event": max_per_event, "event_counts": dict(Counter(row["event_family_id"] for row in examples)),
                "full_source_wording_local_only": True, "generated_augmentation": False,
                "test_or_validation_samples": 0,
                "limitations": ["Historical factual content can still be memorized; retrieval-only facts are a design intention, not a guarantee.",
                                "Training loss does not measure likeness or held-out answer quality."]}
    write_json(output / "manifest.json", manifest)
    return manifest


def _crop_wave(source: Path, target: Path, start: float, end: float):
    with wave.open(str(source), "rb") as inp:
        rate = inp.getframerate()
        first, final = round(start*rate), round(end*rate)
        if not 0 <= first < final <= inp.getnframes():
            raise ValueError("Accepted speech bounds exceed their actual WAV")
        inp.setpos(first)
        pcm = inp.readframes(final-first)
        with wave.open(str(target), "wb") as out:
            out.setnchannels(inp.getnchannels())
            out.setsampwidth(inp.getsampwidth())
            out.setframerate(rate)
            out.writeframes(pcm)


def export_voice(projection: Path, output: Path, *, asset_root: Path, reference_id: str | None = None,
                 speaker: str | None = None, persona_id: str | None = None) -> dict:
    if output.exists():
        raise ValueError("Choose a new voice export directory")
    rows = read_projection(projection)
    if speaker is not None and (not isinstance(speaker, str) or not speaker.strip() or len(speaker) > 200):
        raise ValueError("Supply an explicit publisher speaker label")
    if persona_id is not None and not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,119}", persona_id):
        raise ValueError("Supply a valid persona ID")
    if not rows:
        output.mkdir(parents=True)
        write_jsonl(output / "train_raw.jsonl", [])
        manifest = {"schema_version": "qwen3-tts-dataset.v1", "ready": False, "samples": 0,
                    "reason": "No independently accepted, aligned original target-speaker recordings; publisher text is insufficient.",
                    "projection_sha256": file_hash(projection), "dataset_sha256": file_hash(output / "train_raw.jsonl")}
        write_json(output / "manifest.json", manifest)
        return manifest
    if reference_id not in {row["segment_id"] for row in rows}:
        raise ValueError("Choose one accepted segment as the consistent reference")
    speakers = {row.get("media", {}).get("speaker") for row in rows}
    # Legacy callers can infer the one explicitly reviewed publisher label;
    # neither the active persona nor historical Musk fixtures supply a default.
    if len(speakers) != 1 or None in speakers or "" in speakers:
        raise ValueError("Voice export requires exactly one explicitly attributed speaker")
    inferred_speaker = next(iter(speakers))
    if speaker is not None and inferred_speaker != speaker:
        raise ValueError("Reviewed speech does not match the requested speaker")
    speaker = inferred_speaker
    verified = []
    for row in rows:
        media = row["media"]
        if media.get("review_status") != "accepted" or media.get("alignment_basis") != "acoustically_reviewed":
            raise ValueError("Voice export requires accepted acoustic speaker review")
        if persona_id is not None and row.get("persona_id") not in {None, persona_id}:
            raise ValueError("Voice projection belongs to a different persona")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", row["segment_id"]):
            raise ValueError("Voice segment identifiers must be safe filenames")
        source = (asset_root / media["file"]).resolve()
        if not source.is_relative_to(asset_root.resolve()) or file_hash(source) != media["sha256"]:
            raise ValueError("Accepted speech source hash/path changed")
        verified.append((row, source))
    # Validate the entire selection before producing any reusable output.
    output.mkdir(parents=True)
    clips = {}
    for row, source in verified:
        media = row["media"]
        clip = output / (row["segment_id"] + ".wav")
        _crop_wave(source, clip, media["start_seconds"], media["end_seconds"])
        clips[row["segment_id"]] = clip.resolve()
    # The same actual audio reference is used for every training example, per
    # the upstream Base single-speaker SFT guide.
    reference = str(clips[reference_id])
    examples = [{"audio": str(clips[row["segment_id"]]), "text": row["text"], "ref_audio": reference} for row in rows]
    write_jsonl(output / "train_raw.jsonl", examples)
    manifest = {"schema_version": "qwen3-tts-dataset.v1", "ready": True, "samples": len(examples),
                "speaker": speaker, "persona_id": persona_id,
                "event_counts": dict(Counter(row["event_family_id"] for row in rows)),
                "source_intervals": [{"segment_id": row["segment_id"], "source_id": row["source_id"],
                    "event_family_id": row["event_family_id"], "split": row["split"],
                    "source_sha256": row["media"]["sha256"], "source_timeline": row["media"].get("source_timeline"),
                    "start_seconds": row["media"]["start_seconds"], "end_seconds": row["media"]["end_seconds"],
                    "text_sha256": row["answer_sha256"]} for row in rows],
                "projection_sha256": file_hash(projection), "dataset_sha256": file_hash(output / "train_raw.jsonl"),
                "reference_segment_id": reference_id, "reference_sha256": file_hash(Path(reference)),
                "audio_sha256": {key: file_hash(path) for key, path in clips.items()},
                "upstream_guide": "https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/README.md",
                "evaluation_required": ["English intelligibility", "Korean cross-language intelligibility", "timbre", "prosody", "independent listening comparison"]}
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_dataset(directory: Path, kind: str) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    filename = "train.jsonl" if kind == "text" else "train_raw.jsonl"
    path = directory / filename
    expected_schema = "text-adaptation-dataset.v1" if kind == "text" else "qwen3-tts-dataset.v1"
    if manifest.get("schema_version") != expected_schema or file_hash(path) != manifest.get("dataset_sha256"):
        raise ValueError("Frozen training dataset manifest/hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if type(manifest.get("samples")) is not int or len(rows) != manifest["samples"] or not rows:
        raise ValueError("Training requires a nonempty frozen dataset")
    if kind == "text":
        if type(manifest.get("test_or_validation_samples")) is not int or manifest["test_or_validation_samples"] != 0:
            raise ValueError("Text adaptation requires a training-only dataset manifest")
        identifiers, source_families = set(), {}
        for row in rows:
            if (not isinstance(row, dict) or any(not isinstance(row.get(key), str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", row[key])
                    for key in ("id", "source_id", "event_family_id"))):
                raise ValueError("Text adaptation rows require original sample, source and event-family lineage")
            if row["id"] in identifiers:
                raise ValueError("Text adaptation sample IDs must be unique")
            identifiers.add(row["id"])
            prior_family = source_families.setdefault(row["source_id"], row["event_family_id"])
            if prior_family != row["event_family_id"]:
                raise ValueError("One text source cannot claim different event families")
            messages = row.get("messages", [])
            if (not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages)
                    or [item.get("role") for item in messages] != ["system", "user", "assistant"]
                    or any(not isinstance(item.get("content"), str) or not item["content"].strip() for item in messages)
                    or row.get("split") != "train" or row.get("answer_sha256") != sha_text(messages[-1]["content"])):
                raise ValueError("Training data includes held-out, malformed or changed answer text")
        counts = manifest.get("event_counts")
        if (not isinstance(counts, dict) or any(type(count) is not int or count < 1 for count in counts.values())
                or dict(Counter(row["event_family_id"] for row in rows)) != counts):
            raise ValueError("Training event-family counts differ from the actual exported rows")
    if kind == "voice":
        if manifest.get("ready") is not True:
            raise ValueError("Voice training data is not accepted")
        for row in rows:
            path = Path(row["audio"])
            if file_hash(path) != manifest["audio_sha256"].get(path.stem) or file_hash(Path(row["ref_audio"])) != manifest.get("reference_sha256"):
                raise ValueError("Frozen voice audio changed")
    return manifest, rows
