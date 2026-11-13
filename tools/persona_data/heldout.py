"""Isolated, hash-bound original references for evaluation, never retrieval.

The ordinary acquisition/build commands deliberately reject reserved test
events. This separate artifact has neither a runtime corpus nor training
projections. Loading it verifies the frozen inputs and the exact references.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from bias.corpus import Corpus, canonical_source_url, source_event_id

from .parsers import PARSER_VERSION, _canonical_turns, _lex_pdf, _pdf_text, canonical_lex, digest, parse, sha_text
from .pipeline import _normal, collect_source, extract_exchanges, load_registry, read_json, target_speaker, write_json, write_jsonl


DATASET_SCHEMA = "persona-heldout-dataset.v1"
CASE_SCHEMA = "persona-heldout-case.v1"
ACQUISITION_SCHEMA = "persona-heldout-acquisition.v1"
PURPOSE = "evaluation_only"
SPAN_WORDS = 12
REGISTRY_FILES = ("source-families.json", "split-registration.json")
FILES = (*REGISTRY_FILES, "cases.jsonl", "recipe.json", "exposure-index.json")
PROJECTIONS = {name: False for name in ("factual", "text_style", "aligned_speech", "visible_behavior")}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identifier(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value):
        raise ValueError("Held-out source/case identifier is invalid")
    return value


def _inside(root: Path, relative: str) -> Path:
    """Only regular, non-symlink descendants of the declared artifact root."""
    path = Path(relative)
    if root.is_symlink() or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("Held-out artifact path escapes its root or uses a symlink")
    candidate = root / path
    if any(item.is_symlink() for item in (candidate, *candidate.parents)):
        raise ValueError("Held-out artifact paths cannot contain symlinks")
    if not candidate.resolve().is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError("Held-out artifact is missing or outside its root")
    return candidate


def _new_directory(path: Path) -> None:
    if path.exists() or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Choose a new held-out directory without symlink ancestors")


def _projections_disabled(value) -> bool:
    return isinstance(value, dict) and set(value) == set(PROJECTIONS) and all(flag is False for flag in value.values())


def _registry_hashes(directory: Path) -> dict:
    return {name: _hash(_inside(directory, name)) for name in REGISTRY_FILES}


def _utc(value: str) -> None:
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Acquisition time must be an exact timezone-aware timestamp") from exc
    if stamp.tzinfo is None:
        raise ValueError("Acquisition time must include its timezone")


def _documents(path: Path | None, registry: dict, registration: dict) -> dict:
    if path is None:
        return {}
    value = read_json(path)
    if (value.get("schema_version") != "persona-heldout-documents.v1"
            or value.get("source_registry_sha256") != digest(registry)
            or value.get("split_registration_sha256") != digest(registration)):
        raise ValueError("Held-out document registration does not bind the frozen registry")
    _utc(value.get("registered_at"))
    sources = {row["id"]: row for row in registry["sources"]}
    result = {}
    for item in value.get("documents", []):
        identifier = _identifier(item.get("source_id"))
        source = sources.get(identifier)
        if not source or source["split"] != "test" or identifier in result:
            raise ValueError("Document registration requires unique reserved test sources")
        if item.get("adapter") not in {"lex_publisher_html", "lex_publisher_pdf", "jensen_no_priors_html"}:
            raise ValueError("No reviewed adapter for this held-out document format")
        url = urlsplit(item.get("acquisition_url", ""))
        evidence = urlsplit(item.get("discovery_url", ""))
        if item["adapter"] == "jensen_no_priors_html":
            allowed = (source["id"] == "no-priors-2023" and source.get("persona_id") == "jensen-huang"
                       and url.hostname == evidence.hostname == "sarah-guo.herokuapp.com"
                       and url.path == evidence.path == "/blog/jensenhuang"
                       and urlsplit(source["url"]).hostname == "podcasts.apple.com")
        else:
            allowed = (url.hostname == evidence.hostname == "lexfridman.com"
                       and urlsplit(source["url"]).hostname == "lexfridman.com")
        if (not allowed or url.scheme != "https" or evidence.scheme != "https"
                or url.username or url.password or url.port not in {None, 443}
                or evidence.username or evidence.password or evidence.port not in {None, 443}):
            raise ValueError("Held-out documents require their reviewed original publisher URLs")
        if not isinstance(item.get("basis"), str) or not item["basis"].strip():
            raise ValueError("Document registration needs a source-identity verification basis")
        result[identifier] = item
    return result


def acquire_heldout(registry_dir: Path, output: Path, identifiers: list[str], *, documents_path: Path | None = None) -> dict:
    """Download reserved documents into a separately marked acquisition root."""
    registry, registration = load_registry(registry_dir)
    sources = {row["id"]: row for row in registry["sources"]}
    if (not identifiers or len(identifiers) != len(set(identifiers))
            or any(key not in sources or sources[key]["split"] != "test" for key in identifiers)):
        raise ValueError("Held-out acquisition accepts unique registered test sources only")
    for identifier in identifiers:
        _identifier(identifier)
    documents = _documents(documents_path, registry, registration)
    if set(documents) - set(identifiers):
        raise ValueError("Document registration includes unselected test sources")
    _new_directory(output)
    output.mkdir(parents=True)
    for name in REGISTRY_FILES:
        shutil.copyfile(_inside(registry_dir, name), output / name)
    manifest = {"schema_version": ACQUISITION_SCHEMA, "purpose": PURPOSE,
                "created_at": datetime.now(timezone.utc).isoformat(), "registry_files": _registry_hashes(registry_dir),
                "source_registry_sha256": digest(registry), "split_registration_sha256": digest(registration),
                "documents": documents, "attempts": []}
    if documents_path is not None:
        shutil.copyfile(documents_path, output / "heldout-documents.json")
        manifest["document_registration_sha256"] = _hash(output / "heldout-documents.json")
    for identifier in identifiers:
        source = dict(sources[identifier])
        if identifier in documents:
            source["acquisition_url"] = documents[identifier]["acquisition_url"]
        try:
            record = collect_source(source, output / "raw")
            manifest["attempts"].append({"source_id": identifier, "status": "acquired", "record": record})
        except Exception as exc:
            manifest["attempts"].append({"source_id": identifier, "status": "failed", "error": str(exc)})
        write_json(output / "heldout-acquisition.json", manifest)
    return manifest


def _parse_heldout(source: dict, artifact: Path, document: dict | None) -> dict:
    adapter = (document or {}).get("adapter")
    if adapter == "lex_publisher_pdf":
        rows = _lex_pdf(_pdf_text(artifact))
        return {"schema_version": "persona-original-turns.v1", "parser_version": PARSER_VERSION,
                "source_id": source["id"], "input_sha256": _hash(artifact), "transcript_sha256": digest(rows),
                "source_timeline": "publisher_transcript_video", "turns": rows, "audio_alignment_verified": False}
    if adapter == "lex_publisher_html" or source["id"] in {"lex-018-2019", "lex-252-2021"}:
        canonical = canonical_lex(artifact.read_bytes(), source_id=source["id"], source_url=(document or {}).get("acquisition_url", source["url"]))
        return {"schema_version": "persona-original-turns.v1", "parser_version": PARSER_VERSION,
                "source_id": source["id"], "input_sha256": _hash(artifact),
                "transcript_sha256": canonical["publisher_caption_sha256"], "source_timeline": canonical["source_timeline"],
                "turns": _canonical_turns(canonical), "audio_alignment_verified": False}
    return parse(source, artifact)


def _spans(text: str) -> set[str]:
    words = _normal(text).split()
    return {sha_text(" ".join(words[index:index + SPAN_WORDS])) for index in range(len(words) - SPAN_WORDS + 1)}


def _exposure_index(directories: list[Path]) -> dict:
    if not directories:
        raise ValueError("At least one actual train/development comparison corpus is required")
    families, sources, urls, exchanges, answers, spans, artifacts = set(), set(), set(), set(), set(), set(), []
    for directory in directories:
        corpus = Corpus.load(directory)
        hashes = {name: _hash(_inside(directory, name)) for name in ("sources.json", "evidence.jsonl", "profile.json")}
        for name in ("corpus-manifest.json", "split-manifest.json"):
            if (directory / name).exists():
                hashes[name] = _hash(_inside(directory, name))
        artifacts.append({"files": hashes, "identity_sha256": digest(hashes), "source_count": len(corpus.sources), "record_count": len(corpus.evidence)})
        # All material already exposed to the developer is compared, including
        # historical pilot test records. An old test is not a fresh holdout.
        for source in corpus.sources.values():
            sources.add(source.id)
            families.add((source.model_extra or {}).get("event_family_id", source_event_id(source)))
            families.add(source_event_id(source))
            urls.add(canonical_source_url(str(source.url)))
        for record in corpus.evidence.values():
            qa = record.question_answer
            text = qa.answer if qa else record.text
            if len(_normal(text).split()) >= 10:
                answers.add(sha_text(_normal(text)))
            spans.update(_spans(text))
            if qa:
                exchanges.add(sha_text(_normal(qa.question + "\n" + qa.answer)))
                spans.update(_spans(qa.question))
                spans.update(_spans(qa.context_before))
    return {"schema_version": "persona-heldout-exposure-index.v1", "span_words": SPAN_WORDS,
            "scope": "all records in explicitly supplied comparison corpora, including exposed historical tests",
            "comparison_corpora": artifacts, "source_ids": sorted(sources), "event_family_ids": sorted(families),
            "source_urls": sorted(urls), "exchange_hashes": sorted(exchanges), "long_answer_hashes": sorted(answers),
            "span_hashes": sorted(spans), "semantic_duplicate_review": "not_performed"}


def _check_leakage(case: dict, exposure: dict) -> None:
    if (case["source_id"] in exposure["source_ids"] or case["event_family_id"] in exposure["event_family_ids"]
            or case["source_event_id"] in exposure["event_family_ids"]
            or canonical_source_url(case["source_url"]) in exposure["source_urls"]):
        raise ValueError(f"Held-out event/source was already exposed: {case['source_id']}")
    if sha_text(_normal(case["question"] + "\n" + case["reference_answer"])) in exposure["exchange_hashes"]:
        raise ValueError(f"Held-out exchange duplicates comparison material: {case['id']}")
    answer = _normal(case["reference_answer"])
    if len(answer.split()) >= 10 and sha_text(answer) in exposure["long_answer_hashes"]:
        raise ValueError(f"Held-out answer duplicates comparison material: {case['id']}")
    if (_spans(case["reference_answer"]) | _spans(case["context_before"])) & set(exposure["span_hashes"]):
        raise ValueError(f"Held-out reference/context shares a {SPAN_WORDS}-word span with comparison material: {case['id']}")


def _case(source: dict, item: dict, transcript: dict) -> dict:
    qa = item["question_answer"]
    context = []
    if qa["context_before"]:
        # Extraction preserves exactly one immediately preceding original turn.
        question_cues = set(qa["transcript_locator"]["question_cue_ids"])
        position = next(i for i, turn in enumerate(transcript["turns"]) if question_cues.intersection(turn["cue_ids"]))
        previous = transcript["turns"][position - 1] if position else None
        if previous is None or previous["text"] != qa["context_before"]:
            raise ValueError("Original held-out context cannot be traced to its publisher turn")
        context.append({"role": "assistant" if previous["speaker"] == target_speaker(source) else "user", "content": qa["context_before"]})
    return {"schema_version": CASE_SCHEMA, "purpose": PURPOSE, "id": "heldout-" + item["id"],
            "source_id": source["id"], "source_event_id": source["source_event_id"], "event_family_id": source["event_family_id"],
            "split": "test", "source_url": source["url"], "published_at": source.get("published_at"),
            "event_at": source.get("recorded_at"), "ingested_at": item["ingested_at"],
            "question": qa["question"], "context_before": qa["context_before"], "context": context,
            "reference_answer": qa["answer"], "questioner": qa["questioner"], "respondent": qa["respondent"],
            "original_language": qa["original_language"], "completeness": qa["completeness"],
            "question_sha256": sha_text(qa["question"]), "answer_sha256": sha_text(qa["answer"]),
            "context_sha256": digest(context), "source_artifact_sha256": item["source_artifact_sha256"],
            "transcript_locator": qa["transcript_locator"], "review": qa["review"],
            "reference_review_status": "publisher_structure_only", "projection_eligibility": dict(PROJECTIONS)}


def build_heldout(registry_dir: Path, acquisition_dir: Path, output: Path, *, comparison_corpora: list[Path],
                  report_dir: Path | None = None) -> dict:
    registry, registration = load_registry(registry_dir)
    _new_directory(output)
    if report_dir is not None:
        _new_directory(report_dir)
    acquisition_path = _inside(acquisition_dir, "heldout-acquisition.json")
    acquisition = read_json(acquisition_path)
    if (acquisition.get("schema_version") != ACQUISITION_SCHEMA or acquisition.get("purpose") != PURPOSE
            or acquisition.get("registry_files") != _registry_hashes(registry_dir)
            or acquisition.get("registry_files") != _registry_hashes(acquisition_dir)
            or acquisition.get("source_registry_sha256") != digest(registry)
            or acquisition.get("split_registration_sha256") != digest(registration)):
        raise ValueError("Held-out acquisition/frozen registry hashes do not match")
    if acquisition.get("documents"):
        document_path = _inside(acquisition_dir, "heldout-documents.json")
        if (_hash(document_path) != acquisition.get("document_registration_sha256")
                or _documents(document_path, registry, registration) != acquisition["documents"]):
            raise ValueError("Held-out publisher document registration changed")
    exposure = _exposure_index(comparison_corpora)
    sources = {row["id"]: row for row in registry["sources"]}
    attempts = acquisition.get("attempts", [])
    seen, cases, reports, rejected, duplicates = set(), [], [], [], set()
    for attempt in attempts:
        identifier = _identifier(attempt.get("source_id"))
        if identifier in seen or identifier not in sources or sources[identifier]["split"] != "test":
            raise ValueError("Held-out acquisition contains duplicate, unregistered or non-test sources")
        seen.add(identifier)
        source = sources[identifier]
        if attempt.get("status") not in {"acquired", "failed"}:
            raise ValueError("Unknown held-out acquisition status")
        if attempt["status"] != "acquired":
            reports.append({"source_id": identifier, "status": "acquisition_failed", "error": attempt.get("error")})
            continue
        record = attempt["record"]
        filename = record["artifact"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Acquired artifact must be a filename within its source directory")
        artifact = _inside(acquisition_dir, f"raw/{identifier}/{filename}")
        document = acquisition.get("documents", {}).get(identifier)
        requested_url = (document or {}).get("acquisition_url", source.get("acquisition_url", source["url"]))
        if (record.get("source_id") != identifier or record.get("requested_url") != requested_url
                or _hash(artifact) != record.get("sha256") or artifact.stat().st_size != record.get("bytes")):
            raise ValueError("Held-out acquired source identity, bytes or hash changed")
        _utc(record.get("retrieved_at"))
        try:
            transcript = _parse_heldout(source, artifact, document)
            extracted, failures = extract_exchanges(source, transcript, acquired_at=record["retrieved_at"])
        except (ValueError, OSError) as exc:
            reports.append({"source_id": identifier, "status": "not_original_qa", "reason": str(exc), "artifact_sha256": record["sha256"]})
            continue
        rejected.extend(failures)
        accepted = 0
        for item in extracted:
            if item["question_answer"]["completeness"] != "complete_exchange":
                rejected.append({"id": item["id"], "reason": "incomplete_published_reference"})
                continue
            case = _case(source, item, transcript)
            _check_leakage(case, exposure)
            if _spans(case["reference_answer"]) & (_spans(case["question"]) | _spans(case["context_before"])):
                rejected.append({"id": case["id"], "reason": "reference_overlaps_supplied_question_or_context"})
                continue
            key = sha_text(_normal(case["question"] + "\n" + case["reference_answer"]))
            if key in duplicates:
                rejected.append({"id": case["id"], "reason": "duplicate_heldout_exchange"})
                continue
            duplicates.add(key)
            cases.append(case)
            accepted += 1
        reports.append({"source_id": identifier, "event_family_id": source["event_family_id"], "source_url": source["url"],
                        "status": "extracted", "artifact_sha256": record["sha256"], "transcript_sha256": transcript["transcript_sha256"],
                        "acquired_at": record["retrieved_at"], "reference_cases": accepted})
    if not cases:
        raise ValueError("No complete held-out references passed extraction and isolation checks; publisher landing pages are not transcripts: "
                         + json.dumps({"sources": reports, "rejections": rejected}))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".heldout-build-", dir=output.parent))
    try:
        for name in REGISTRY_FILES:
            shutil.copyfile(_inside(registry_dir, name), temporary / name)
        write_jsonl(temporary / "cases.jsonl", cases)
        write_json(temporary / "exposure-index.json", exposure)
        recipe = {"schema_version": "persona-heldout-recipe.v1", "purpose": PURPOSE, "parser_version": PARSER_VERSION,
                  "acquisition_manifest_sha256": _hash(acquisition_path), "documents": acquisition.get("documents", {}),
                  "sources": reports, "selections": [{key: item[key] for key in ("id", "source_id", "question_sha256", "answer_sha256", "context_sha256", "transcript_locator")} for item in cases],
                  "rejections": rejected}
        write_json(temporary / "recipe.json", recipe)
        manifest = {"schema_version": DATASET_SCHEMA, "purpose": PURPOSE, "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_registry_sha256": digest(registry), "split_registration_sha256": digest(registration),
                    "files": {name: _hash(temporary / name) for name in FILES}, "cases": len(cases),
                    "source_events": len({item["event_family_id"] for item in cases}), "case_split": "test",
                    "event_counts": dict(Counter(item["event_family_id"] for item in cases)),
                    "projection_eligibility": dict(PROJECTIONS), "reference_review_status": "publisher_structure_only",
                    "human_review_completed": False, "original_wording_local_only": True,
                    "leakage_policy": {"span_words": SPAN_WORDS, "checked_corpora": exposure["comparison_corpora"],
                                       "semantic_duplicate_review": "not_performed"},
                    "limitations": ["Only explicitly supplied comparison corpora were checked; externally exposed wording is unknown.",
                                    "Exact normalized and shared-word-span checks do not establish semantic independence.",
                                    "Publisher structure is not independent human reference review, acoustic verification or efficacy.",
                                    "Null dates remain unknown; acquiring an old interview does not establish current positions."]}
        write_json(temporary / "manifest.json", manifest)
        load_heldout(temporary)
        temporary.rename(output)
        if report_dir is not None:
            write_json(report_dir / "manifest.json", manifest)
            write_json(report_dir / "recipe.json", recipe)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_heldout(directory: Path) -> tuple[dict, list[dict]]:
    """Verify an isolated held-out dataset and return its private references.

    Consumers must pin ``manifest.json`` itself in their experiment registration
    and must never include ``reference_answer`` in inference prompts.
    """
    manifest = read_json(_inside(directory, "manifest.json"))
    if (manifest.get("schema_version") != DATASET_SCHEMA or manifest.get("purpose") != PURPOSE
            or manifest.get("case_split") != "test" or not _projections_disabled(manifest.get("projection_eligibility"))
            or manifest.get("human_review_completed") is not False
            or manifest.get("reference_review_status") != "publisher_structure_only"):
        raise ValueError("Expected an evaluation-only held-out artifact")
    if set(manifest.get("files", {})) != set(FILES):
        raise ValueError("Held-out file inventory changed")
    for name, expected in manifest["files"].items():
        if _hash(_inside(directory, name)) != expected:
            raise ValueError(f"Held-out artifact hash mismatch: {name}")
    if any((directory / name).exists() for name in ("profile.json", "evidence.jsonl", "train.jsonl", "train_raw.jsonl", "projections", "corpus")):
        raise ValueError("Evaluation-only directory must not contain runtime or training projections")
    registry, registration = load_registry(directory)
    if digest(registry) != manifest["source_registry_sha256"] or digest(registration) != manifest["split_registration_sha256"]:
        raise ValueError("Held-out frozen source/split registration changed")
    sources = {row["id"]: row for row in registry["sources"]}
    exposure = read_json(directory / "exposure-index.json")
    if exposure.get("schema_version") != "persona-heldout-exposure-index.v1" or exposure.get("span_words") != SPAN_WORDS:
        raise ValueError("Unknown held-out leakage policy")
    if (not exposure.get("comparison_corpora")
            or manifest.get("leakage_policy", {}).get("checked_corpora") != exposure["comparison_corpora"]
            or manifest.get("leakage_policy", {}).get("span_words") != SPAN_WORDS):
        raise ValueError("Held-out comparison coverage differs from its frozen exposure index")
    recipe = read_json(directory / "recipe.json")
    if recipe.get("schema_version") != "persona-heldout-recipe.v1" or recipe.get("purpose") != PURPOSE:
        raise ValueError("Unknown held-out selection recipe")
    source_reports = {row["source_id"]: row for row in recipe["sources"] if row["status"] == "extracted"}
    selections = {row["id"]: row for row in recipe["selections"]}
    cases = [json.loads(line) for line in (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [_identifier(item.get("id")) for item in cases]
    if (not cases or type(manifest.get("cases")) is not int or len(cases) != manifest["cases"]
            or len(set(ids)) != len(ids) or set(ids) != set(selections) or len(selections) != len(recipe["selections"])):
        raise ValueError("Held-out case count/selection identity changed")
    for item in cases:
        source = sources.get(item.get("source_id"))
        if (not source or source["split"] != "test" or item.get("schema_version") != CASE_SCHEMA
                or item.get("purpose") != PURPOSE or item.get("split") != "test"
                or item.get("event_family_id") != source["event_family_id"] or item.get("source_event_id") != source["source_event_id"]
                or item.get("source_url") != source["url"] or item.get("published_at") != source.get("published_at")
                or item.get("event_at") != source.get("recorded_at") or not _projections_disabled(item.get("projection_eligibility"))
                or item.get("reference_review_status") != "publisher_structure_only" or item.get("completeness") != "complete_exchange"):
            raise ValueError("Held-out case crossed its frozen evaluation-only source contract")
        if (not isinstance(item.get("question"), str) or not item["question"].strip()
                or not isinstance(item.get("reference_answer"), str) or not item["reference_answer"].strip()
                or item.get("question_sha256") != sha_text(item["question"])
                or item.get("answer_sha256") != sha_text(item["reference_answer"])):
            raise ValueError("Held-out original question/reference hash mismatch")
        report = source_reports.get(item["source_id"], {})
        if (item.get("source_artifact_sha256") != report.get("artifact_sha256")
                or item.get("transcript_locator", {}).get("transcript_sha256") != report.get("transcript_sha256")
                or item.get("ingested_at") != report.get("acquired_at") or item.get("respondent") != target_speaker(source)
                or item.get("original_language") not in {"en", "ko"}
                or item.get("review", {}).get("reviewer_type") != "agent"
                or item.get("review", {}).get("speaker_status") != "publisher_labels_checked"
                or item.get("review", {}).get("timing_status") != "not_audio_verified"):
            raise ValueError("Held-out reference source lineage or structural review changed")
        context = item.get("context")
        if (not isinstance(context, list) or len(context) > 1 or any(set(row) != {"role", "content"}
                or row["role"] not in {"user", "assistant"} or not isinstance(row["content"], str) for row in context)
                or item.get("context_sha256") != digest(context)
                or item.get("context_before") != (context[0]["content"] if context else "")):
            raise ValueError("Held-out original dialogue context changed")
        if selections[item["id"]] != {key: item[key] for key in ("id", "source_id", "question_sha256", "answer_sha256", "context_sha256", "transcript_locator")}:
            raise ValueError("Held-out selection differs from its frozen recipe")
        _utc(item.get("ingested_at"))
        _check_leakage(item, exposure)
        if _spans(item["reference_answer"]) & (_spans(item["question"]) | _spans(item["context_before"])):
            raise ValueError("Held-out reference text overlaps its supplied question/context")
    if (manifest.get("event_counts") != dict(Counter(item["event_family_id"] for item in cases))
            or manifest.get("source_events") != len({item["event_family_id"] for item in cases})):
        raise ValueError("Held-out event counts changed")
    return manifest, cases
