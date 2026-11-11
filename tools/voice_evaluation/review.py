"""Build condition-blind local listening pages and validate explicit review records."""

from __future__ import annotations

import html
import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

from .metrics import EvaluationError, wav_metrics
from .run import checkpoint, digest, read_json, synthesis_rows, timestamp, verified_audio

RATINGS = ("intelligibility", "naturalness", "reference_similarity")
FLAGS = ("unexpected_repetition", "audible_cutoff")


def create_bundle(synthesis_paths: list[str | Path], *, output: str | Path,
    seed: int = 42, reference: str | Path | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise EvaluationError("Randomization seed must be a 32-bit unsigned integer")
    if not 1 <= len(synthesis_paths) <= 8:
        raise EvaluationError("Supply 1–8 synthesis reports")
    items = []
    for path in synthesis_paths:
        file = Path(path).expanduser().resolve()
        report = read_json(file)
        for row in synthesis_rows(report):
            if row.get("status") == "succeeded":
                audio = verified_audio(file, row)
                wav_metrics(audio.read_bytes())
                items.append({"audio": audio, "condition": report.get("condition"), "case_id": row["id"],
                    "language": row["language"], "expected_text": row["text"], "synthesis_sha256": digest(file)})
    if not 1 <= len(items) <= 64:
        raise EvaluationError("A listening bundle needs 1–64 successful synthesis samples")
    reference_path = Path(reference).expanduser().resolve() if reference else None
    if reference_path:
        if not reference_path.is_file() or reference_path.stat().st_size > 20_000_000:
            raise EvaluationError("Comparison reference is missing or oversized")
        wav_metrics(reference_path.read_bytes(), max_seconds=30)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise EvaluationError("Choose a new listening bundle directory")
    listener = destination / "listener"
    private = destination / "private"
    listener.mkdir(parents=True)
    private.mkdir()
    random.Random(seed).shuffle(items)
    manifest = {"schema_version": "voice_listening.v1", "created_at": timestamp(),
        "status": "unrated", "reference": None, "samples": [],
        "instructions": "Listen before reading any answer key. Rate only languages you understand; leave other fields null. Scores are subjective self-reports, not identity verification. Share the listener directory only."}
    key = {"schema_version": "voice_listening_key.v1", "randomization_seed": seed, "samples": []}
    if reference_path:
        shutil.copyfile(reference_path, listener / "comparison.wav")
        manifest["reference"] = {"path": "comparison.wav", "sha256": digest(listener / "comparison.wav"),
            "note": "Acoustic comparison only; the bundle does not establish the recording's speaker identity or suitability."}
    for index, item in enumerate(items, 1):
        identifier = f"sample{index:03d}"
        name = identifier + ".wav"
        shutil.copyfile(item["audio"], listener / name)
        manifest["samples"].append({"id": identifier, "path": name, "sha256": digest(listener / name), "language": item["language"]})
        key["samples"].append({name: value for name, value in item.items() if name != "audio"} | {"id": identifier})
    checkpoint(listener / "manifest.json", manifest)
    fingerprint = digest(listener / "manifest.json")
    key["listener_manifest_sha256"] = fingerprint
    checkpoint(private / "answer-key.json", key)
    template = {"schema_version": "voice_listening_review.v1", "listener_manifest_sha256": fingerprint,
        "reviewer": None, "created_at": None, "human_review_performed": False, "reference_listened": False,
        "samples": [dict(id=sample["id"], audio_sha256=sample["sha256"], listened=False,
            understood_language=None, heard_text=None, **{name: None for name in RATINGS + FLAGS}, notes=None) for sample in manifest["samples"]]}
    checkpoint(listener / "review-template.json", template)
    (listener / "index.html").write_text(_page(manifest, template), encoding="utf-8")
    (destination / "README.txt").write_text("Open listener/index.html locally. Share only the listener directory with reviewers; private/answer-key.json reveals conditions. All fields begin unrated. Export a completed review from the page, then validate it with python -m tools.voice_evaluation review. No listening or likeness assessment is performed by bundle creation.\n", encoding="utf-8")
    return {"listener_manifest_sha256": fingerprint, "samples": len(items), "status": "unrated", "human_review_performed": False}


def _page(manifest: dict, template: dict) -> str:
    sections = []
    options = '<option value="">Unrated</option>' + ''.join(f'<option value="{i}">{i}</option>' for i in range(1, 6))
    yesno = '<option value="">Unknown / unrated</option><option value="true">Yes</option><option value="false">No</option>'
    for sample in manifest["samples"]:
        identifier = sample["id"]
        ratings = ''.join(f'<label>{html.escape(name.replace("_", " ").capitalize())} (1–5)<select data-field="{name}">{options}</select></label>' for name in RATINGS if name != "reference_similarity" or manifest["reference"])
        flags = ''.join(f'<label>{html.escape(name.replace("_", " ").capitalize())}<select data-field="{name}">{yesno}</select></label>' for name in FLAGS)
        sections.append(f'<section data-id="{identifier}"><h2>{identifier} · {html.escape(sample["language"])}</h2><audio controls preload="none" src="{sample["path"]}"></audio><label><input type="checkbox" data-field="listened"> I listened to this complete clip</label><label>I understand this language<select data-field="understood_language">{yesno}</select></label><label>What I heard<textarea data-field="heard_text"></textarea></label>{ratings}{flags}<label>Notes<textarea data-field="notes"></textarea></label></section>')
    comparison = '<h2>Comparison recording</h2><audio controls preload="none" src="comparison.wav"></audio><label><input type="checkbox" id="reference-listened"> I listened to the comparison recording</label>' if manifest["reference"] else ''
    payload = json.dumps(template, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Blind voice listening review</title><style>body{{max-width:780px;margin:2rem auto;padding:0 1rem;font:16px/1.6 system-ui;background:#f7f8fa;color:#17202b}}section{{background:white;padding:1.5rem;margin:1.5rem 0;border:1px solid #ccd4df;border-radius:10px}}label{{display:block;margin:.7rem 0}}select,textarea,input[type=text]{{display:block;box-sizing:border-box;width:100%;font:inherit;padding:.4rem}}textarea{{min-height:4rem}}audio{{max-width:100%;width:100%}}button{{font:inherit;padding:.8rem 1rem;cursor:pointer}}.note{{color:#4d5867}}input[type=checkbox]{{margin-right:.4rem}}</style><h1>Blind voice listening review</h1><p>This is AI-generated research audio. Listen before consulting any answer key. No scores have been assigned. Rate only languages you understand.</p><p class="note">Intelligibility: 1 = mostly unintelligible, 5 = clear throughout. Naturalness: 1 = very unnatural, 5 = natural throughout. Reference similarity, when offered: 1 = very different, 5 = very similar acoustic impression. This is a subjective comparison, not identity verification. Leave uncertain ratings blank.</p><label>Reviewer initials or identifier<input type="text" maxlength="100" id="reviewer"></label><label><input type="checkbox" id="human"> I personally listened and entered these observations</label>{comparison}{''.join(sections)}<button id="export">Download review JSON</button><p id="message" role="status"></p><script>const template={payload};document.getElementById('export').addEventListener('click',()=>{{const result=structuredClone(template);result.reviewer=document.getElementById('reviewer').value.trim()||null;result.created_at=new Date().toISOString();result.human_review_performed=document.getElementById('human').checked;result.reference_listened=document.getElementById('reference-listened')?.checked||false;for(const section of document.querySelectorAll('section[data-id]')){{const row=result.samples.find(x=>x.id===section.dataset.id);for(const field of section.querySelectorAll('[data-field]')){{const name=field.dataset.field;if(field.type==='checkbox')row[name]=field.checked;else if(['intelligibility','naturalness','reference_similarity'].includes(name))row[name]=field.value?Number(field.value):null;else if(['understood_language','unexpected_repetition','audible_cutoff'].includes(name))row[name]=field.value===''?null:field.value==='true';else row[name]=field.value.trim()||null;}}}}if(!result.human_review_performed||!result.reviewer||!result.samples.some(x=>x.listened)){{document.getElementById('message').textContent='Enter a reviewer identifier, listen to at least one clip, and attest to personal review before exporting.';return;}}const blob=new Blob([JSON.stringify(result,null,2)+'\\n'],{{type:'application/json'}});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='listening-review.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('message').textContent='Review downloaded. Validate it against this bundle before recording results.';}});</script></html>'''


def validate_review(manifest_path: str | Path, review_path: str | Path) -> dict:
    path = Path(manifest_path).expanduser().resolve()
    manifest, review = read_json(path), read_json(review_path)
    if manifest.get("schema_version") != "voice_listening.v1" or review.get("schema_version") != "voice_listening_review.v1" or review.get("listener_manifest_sha256") != digest(path):
        raise EvaluationError("Review must be bound to this listening manifest")
    if review.get("human_review_performed") is not True or not isinstance(review.get("reviewer"), str) or not 1 <= len(review["reviewer"].strip()) <= 100:
        raise EvaluationError("A named reviewer must explicitly attest to personal listening")
    if not isinstance(review.get("created_at"), str) or not review["created_at"].strip():
        raise EvaluationError("Review timestamp is required")
    try:
        if datetime.fromisoformat(review["created_at"].replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("Missing timezone")
    except ValueError as exc:
        raise EvaluationError("Review timestamp must be ISO 8601 with a timezone") from exc
    if not isinstance(review.get("reference_listened"), bool) or (review["reference_listened"] and not manifest.get("reference")):
        raise EvaluationError("Reference listening must match the available comparison recording")
    reference = manifest.get("reference")
    if reference:
        file = path.parent / "comparison.wav"
        if not isinstance(reference, dict) or reference.get("path") != "comparison.wav" or file.is_symlink() or not file.is_file() or file.stat().st_size > 20_000_000 or digest(file) != reference.get("sha256"):
            raise EvaluationError("Comparison recording is missing or has changed")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 64:
        raise EvaluationError("Expected 1–64 listening samples")
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("id"), str) or not re.fullmatch(r"sample[0-9]{3}", sample["id"]) or sample.get("path") != sample["id"] + ".wav":
            raise EvaluationError("Invalid listening sample identifier or path")
    expected = {sample["id"]: sample for sample in manifest["samples"]}
    rows = review.get("samples")
    if len(expected) != len(samples) or not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows) or len(rows) != len(expected) or {row.get("id") for row in rows} != set(expected):
        raise EvaluationError("Review must include each sample exactly once")
    values = {name: [] for name in RATINGS}
    listened = 0
    for row in rows:
        sample = expected[row["id"]]
        audio = path.parent / sample["path"]
        if not audio.is_file() or audio.is_symlink() or audio.stat().st_size > 20_000_000 or audio.parent != path.parent or digest(audio) != sample["sha256"] or row.get("audio_sha256") != sample["sha256"]:
            raise EvaluationError("Listening audio or review hash changed")
        if not isinstance(row.get("listened"), bool) or (row.get("understood_language") is not None and not isinstance(row["understood_language"], bool)):
            raise EvaluationError("Listening and language-understanding fields must be explicit")
        listened += row["listened"]
        for name in RATINGS:
            rating = row.get(name)
            if rating is not None:
                if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5 or not row["listened"]:
                    raise EvaluationError("Scores require actual listening and integer ratings from 1 to 5")
                if name in {"intelligibility", "naturalness"} and row.get("understood_language") is not True:
                    raise EvaluationError("Language-dependent scores require understanding that language")
                if name == "reference_similarity" and not review["reference_listened"]:
                    raise EvaluationError("Reference similarity requires listening to the comparison recording")
                values[name].append(rating)
        for name in FLAGS:
            value = row.get(name)
            if value is not None and (not isinstance(value, bool) or not row["listened"]):
                raise EvaluationError("Perceptual flags require listening and a boolean or null")
        for name in ("heard_text", "notes"):
            value = row.get(name)
            if value is not None and (not isinstance(value, str) or len(value) > 12000 or not row["listened"]):
                raise EvaluationError("Listening text fields must be bounded and attached to a listened sample")
    if not listened:
        raise EvaluationError("At least one sample must have been listened to")
    return {"schema_version": "voice_review_summary.v1", "review_sha256": digest(Path(review_path)),
        "listener_manifest_sha256": digest(path), "reviewer_attestation": "self_reported_human_listening",
        "reviewer": review["reviewer"], "listened_samples": listened, "total_samples": len(rows),
        "ratings": {name: {"count": len(scores), "mean": round(sum(scores) / len(scores), 3) if scores else None} for name, scores in values.items()},
        "limitations": "One reviewer is not a reliable aggregate quality benchmark. Attestation is self-reported. Ratings do not verify identity, grant approval, or change reference-candidate status."}
