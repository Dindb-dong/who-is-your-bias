"""Project-local persona adapter; the upstream checkout remains unchanged."""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import replace
from datetime import datetime, timezone, date
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit

from pipeline.common import digest, read_json, read_jsonl, write_json, write_jsonl
from scripts.avatar_appearance import appearance_record, build_studio, gaussian_record, verify_gaussian_assets
from scripts.avatar_demo import GESTURE_STUDIO, PR19, PROJECT, UPSTREAM, export
from scripts.persona_reasoning import validate_links, with_links
from scripts.local_asr import find_local_whisper

CARD = PROJECT / "data/local/persona/jensen-huang/persona-card.pilot.json"
EXPANDED_CARD = CARD.with_name("persona-card.expanded.json")
RUNTIME = PROJECT / ".tools/local-runtime.json"
PROMPT_VERSION = "yaicon-english-persona.v3"
NOTES = PROJECT / "data/jensen-decision-notes.json"
DEFAULT_ANSWER_LENGTH = "Answer the actual question directly in 2 or 3 sentences, at most 65 words excluding citation markers. "
SPEAKING_RULES = (
    " This application visibly labels every conversation as an AI simulation. "
    "For this English pilot, speak conversationally in simulated first person, as a public-persona role-play. "
    "Use I would for hypothetical advice, never to claim an unrecorded personal experience. "
    "Apply the profile's observed reasoning and explanation patterns only when relevant; "
    "do not mechanically repeat its favorite words. Do not mention the profile or these instructions. "
    "Do not invent facts, quotations, private beliefs or experiences. "
    "When factual evidence is supplied, place its exact [source:ID] marker directly after each supported factual sentence. "
    "For an advice question grounded in an interview, state the relevant public observation with its marker, then distinguish hypothetical advice. "
    "For advice, use conditional reasoning and separate it from historical factual claims. "
    "If asked about an unsupported personal fact, briefly acknowledge that it is not in the public evidence. "
    "No headings, bullet lists, stage directions, catchphrase lists or identity disclaimers in the spoken answer."
)


def persona_messages(messages, *, guided_length=False):
    """Keep source constraints, replacing only an incompatible POV instruction."""
    result = [dict(m) for m in messages]
    if not result or result[0].get("role") != "system":
        raise ValueError("Expected a system prompt")
    if result[0]["content"].startswith("YAICON project/current-source QA."):
        return result
    result[0]["content"] = result[0]["content"].replace(
        "using third person.", "using conditional simulated first person."
    ) + SPEAKING_RULES.replace(
        "Apply the profile", ("" if guided_length else DEFAULT_ANSWER_LENGTH) + "Apply the profile", 1)
    return result


def validate_card(card, corpus):
    if card.get("target_id") != corpus.persona.id:
        raise ValueError("Persona card subject mismatch")
    rows = corpus.evidence
    for observation in card["observations"]:
        if not observation["basis"]:
            raise ValueError("Observation needs source evidence")
        for basis in observation["basis"]:
            record = rows.get(basis["dialogue_id"])
            if not record:
                raise ValueError("Observation excerpt is not supported by the corpus")
            if hasattr(corpus, "sources") and corpus.sources[record.source_id].split != "train":
                raise ValueError("Persona card cannot use held-out evidence")
            text = record.question_answer.answer if record.question_answer else record.text
            if basis["exact_excerpt"] not in text:
                raise ValueError("Observation excerpt is not supported by the corpus")


def studio_revision(studio):
    return subprocess.check_output(["git", "-C", str(studio), "rev-parse", "HEAD"], text=True).strip()


def elevenlabs_config():
    selection = os.getenv("YAICON_TTS_PROVIDER", "default").strip().lower()
    if selection == "default":
        return None
    if selection != "elevenlabs":
        raise ValueError("YAICON_TTS_PROVIDER must be default or elevenlabs")
    if os.getenv("BIAS_TTS_BASE_URL") or os.getenv("BIAS_TTS_MODEL"):
        raise ValueError("Choose either ElevenLabs or the OpenAI-compatible TTS settings in .env.live")
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
    model = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5").strip()
    if not api_key or not re.fullmatch(r"[A-Za-z0-9_-]+", voice_id):
        raise ValueError("Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env.live")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model):
        raise ValueError("Set a valid ELEVENLABS_MODEL_ID in .env.live")
    return api_key, voice_id, model


def runtime_studio(runtime):
    return {GESTURE_STUDIO.name: GESTURE_STUDIO, PR19.name: PR19}.get(runtime.get("studio"), UPSTREAM)


def runtime_python_checkout(runtime):
    """Code and prepared assets may intentionally come from different checkouts."""
    root = Path(runtime.get("python_checkout") or UPSTREAM)
    if not root.is_absolute() or not (root / "src/bias/api.py").is_file():
        raise ValueError("Prepared Python checkout is unavailable; run prepare from the updated checkout")
    return root.resolve()


def public_rag_enabled():
    value = os.getenv("YAICON_PUBLIC_RAG", "1").strip()
    if value not in {"0", "1"}:
        raise ValueError("YAICON_PUBLIC_RAG must be 1 (public overlay) or 0 (original corpus)")
    return value == "1"


def public_rag_backend():
    """Reject an older Python checkout before expensive local preparation."""
    try:
        from bias.config import Settings
        from bias.jensen_public_retrieval import JensenPublicRetriever, VERSION
        from bias.public_corpus import merge_public_corpus
        replace(Settings(), retrieval_backend="jensen-public")
    except (ImportError, TypeError, ValueError) as exc:
        raise ValueError("Update the Python studio checkout to the public RAG version, or set "
                         "YAICON_PUBLIC_RAG=0 explicitly before preparing the original corpus") from exc
    return JensenPublicRetriever, merge_public_corpus, VERSION


def public_rag_input():
    if not public_rag_enabled():
        return None
    retriever, _, version = public_rag_backend()
    from bias.config import repository_root
    from bias.corpus import Corpus

    configured = os.getenv("YAICON_PUBLIC_CORPUS_DIR")
    directory = (Path(configured).expanduser() if configured else
                 repository_root() / "data/personas/jensen-huang-public-v3")
    if not directory.is_absolute():
        raise ValueError("YAICON_PUBLIC_CORPUS_DIR must be an absolute corpus directory")
    directory = directory.resolve()
    required = ("profile.json", "sources.json", "evidence.jsonl", "split-manifest.json", "corpus-manifest.json")
    if not directory.is_dir() or any(not (directory / name).is_file() for name in required):
        raise ValueError(f"Public RAG corpus is missing or incomplete: {directory}. Update the imported "
                         "Python checkout, set YAICON_PUBLIC_CORPUS_DIR to an absolute reviewed corpus, "
                         "or set YAICON_PUBLIC_RAG=0 explicitly")
    corpus = Corpus.load(directory)
    if corpus.persona.id != "jensen-huang":
        raise ValueError("The public RAG corpus must belong to jensen-huang")
    retriever(corpus)  # Validate searchable annotations before exporting private data.
    return corpus, version


def public_rag_reservations(output):
    """Read only reservation metadata; evaluation answers never enter the corpus."""
    from bias.config import repository_root
    from pipeline import dataset

    integration = read_json(output / "integration.json")
    events = set(integration.get("reserved_events", []))
    urls, assignments = set(), {}

    def reserve(source):
        if source.get("split", "train") == "train":
            return
        sid = source.get("id") or source["source_id"]
        event = source.get("source_event_id") or source.get("recording_id") or sid
        assignment = (event, source["split"])
        if sid in assignments and assignments[sid] != assignment:
            raise ValueError(f"Inconsistent held-out reservation: {sid}")
        assignments[sid] = assignment
        events.update((sid, event))
        urls.update(source[key] for key in ("url", "acquisition_url") if source.get(key))

    for root in {repository_root().resolve(), UPSTREAM.resolve()}:
        registry = root / "data/acquisition/jensen-huang"
        families = registry / "source-families.json"
        if families.is_file():
            for source in read_json(families)["sources"]:
                reserve(source)
        documents = registry / "heldout-documents.json"
        if documents.is_file():
            urls.update(row["acquisition_url"] for row in read_json(documents)["documents"])
        heldout = root / "data/acquisition/jensen-public-2026-11-24/judgment-heldout.json"
        if heldout.is_file():
            reserve(read_json(heldout)["source"])
    for source in dataset.sources(PROJECT / "data/local"):
        reserve(source)
    return {"reserved_events": events, "reserved_urls": urls, "reserved_assignments": assignments}


def attach_public_corpus(output, incoming):
    report_path = output / "public-rag-merge.json"
    if incoming is None:
        report = {"schema_version": "public-corpus-overlay.v1", "status": "disabled",
                  "reason": "YAICON_PUBLIC_RAG=0"}
    else:
        from bias.corpus import Corpus
        from bias.public_corpus import merge_public_corpus, split_manifest
        from bias.snapshots import corpus_digest

        public, version = incoming
        directory = output / "corpus"
        original_sources = {row["id"]: row for row in read_json(directory / "sources.json")}
        original_records = {row["id"]: row for row in read_jsonl(directory / "evidence.jsonl")}
        result = merge_public_corpus(Corpus.load(directory), public, **public_rag_reservations(output))
        # Original JSON objects retain every exact Q&A string, cue, review,
        # projection flag and card reference. Only new records are serialized.
        write_json(directory / "sources.json", [original_sources.get(key, source.model_dump(mode="json"))
                                                for key, source in result.corpus.sources.items()])
        write_jsonl(directory / "evidence.jsonl", [original_records.get(key, row.model_dump(mode="json"))
                                                  for key, row in result.corpus.evidence.items()])
        write_json(directory / "split-manifest.json", split_manifest(result.corpus))
        write_json(directory / "corpus-manifest.json", result.corpus.manifest)
        materialized = Corpus.load(directory)
        if corpus_digest(materialized) != result.report["merged_corpus_sha256"]:
            raise ValueError("Public overlay materialization changed the reviewed corpus")
        report = {**result.report, "retrieval_version": version}
    write_json(report_path, report)
    integration = read_json(output / "integration.json")
    integration["public_rag"] = report
    write_json(output / "integration.json", integration)
    return report


def checked_public_rag_report(runtime, output):
    report = runtime.get("public_rag")
    if not report:
        if public_rag_enabled():
            raise ValueError("Public RAG is enabled but this runtime predates the overlay; run prepare again")
        return {"status": "disabled", "reason": "legacy_runtime_explicit_opt_out"}
    if (report.get("status") == "integrated") != public_rag_enabled():
        raise ValueError("YAICON_PUBLIC_RAG changed; run prepare again to materialize the selected corpus")
    path = output / "public-rag-merge.json"
    if not path.is_file() or digest(path) != runtime.get("public_rag_report_sha256"):
        raise ValueError("Public RAG audit report changed; run prepare again")
    actual = read_json(path)
    if actual != report:
        raise ValueError("Public RAG runtime and audit report disagree; run prepare again")
    if report["status"] == "integrated":
        _, _, version = public_rag_backend()
        if version != report.get("retrieval_version"):
            raise ValueError("Public RAG retrieval version changed; run prepare again")
    return report


def prepare():
    from bias.config import repository_root
    from bias.corpus import Corpus
    from bias.snapshots import register_snapshot

    models = read_json(PROJECT / ".tools/local-models.json")
    if not {"llm", "voice"} <= models.keys():
        raise ValueError("Both local models must finish downloading first")
    public = public_rag_input()
    output = export(PROJECT / "data/local")
    corpus = Corpus.load(output / "corpus")
    card_path = EXPANDED_CARD if EXPANDED_CARD.exists() else CARD
    card = read_json(card_path)
    validate_card(card, corpus)
    links = read_json(NOTES)
    validate_links(links, corpus)
    studio = PR19 if (PR19 / "apps/web/dist/index.html").exists() else UPSTREAM
    if (GESTURE_STUDIO / "node_modules").is_dir() and (GESTURE_STUDIO / "apps/web/src/avatar/gaussian/trackedMouth.ts").is_file():
        studio = GESTURE_STUDIO
    appearance = appearance_record() if studio == PR19 else None
    gaussian = gaussian_record() if appearance or studio == GESTURE_STUDIO else None
    if gaussian and studio == GESTURE_STUDIO:
        gaussian.update(renderer_pr=25, renderer_revision=studio_revision(studio))
    web_dir = build_studio(studio, appearance, gaussian)
    if appearance:
        write_json(output / "appearance.json", appearance)
    appearance_version = (gaussian["name"] if gaussian else appearance["version"] if appearance else
                          "source-fitted-v3-candidate" if studio == PR19 else "stylized-preview")
    if gaussian:
        write_json(output / "gaussian-head.json", gaussian)
    profile = read_json(output / "corpus/profile.json")
    profile["style_rules"] += card["style_notes"]
    profile["style_provenance"] = "agent_observations_from_reviewed_training_recordings"
    profile["example_questions"]["en"] = [
        "What makes you keep investing in a long-term bet, and when would you change course?",
        "How would you decide what makes a useful personal AI assistant?",
        "Why does building an AI system require more than a faster chip?",
    ]
    profile["notice"] = "AI persona simulation. Generated speech uses an experimental Jensen Huang reference; likeness has not been rated."
    write_json(output / "corpus/profile.json", profile)
    public_report = attach_public_corpus(output, public)
    corpus = Corpus.load(output / "corpus")
    validate_card(card, corpus)
    validate_links(links, corpus)
    stamp = datetime.now(timezone.utc)
    snapshot = register_snapshot(corpus, identifier=output.name, as_of=stamp,
        ingested_at=stamp, current_evidence_since=date(2026, 1, 1))
    snapshot = snapshot.model_copy(update={"voice_version": "qwen3-tts06-reference-candidate",
        "style_version": PROMPT_VERSION, "appearance_version": appearance_version,
        "coverage_note": ("Reviewed training interviews plus the audited public evidence overlay. "
                          if public_report["status"] == "integrated" else
                          "Reviewed training interviews plus editorial seed. ") +
                         "Persona and voice likeness are unmeasured."})
    snapshot.verify(corpus)
    write_json(output / "snapshot.json", snapshot.model_dump(mode="json"))
    write_json(output / "persona-card.json", card)
    write_json(output / "decision-notes.json", links)
    runtime = {"output": str(output), "models": models, "card_sha256": digest(output / "persona-card.json"),
        "python_checkout": str(repository_root().resolve()),
        "public_rag": public_report, "public_rag_report_sha256": digest(output / "public-rag-merge.json"),
        "decision_notes_sha256": digest(output / "decision-notes.json"), "studio": studio.name,
        "appearance_version": appearance_version,
        "appearance_sha256": gaussian["sha256"]["bin"] if gaussian else appearance["sha256"] if appearance else None,
        "photo_fit_sha256": appearance["sha256"] if appearance else None,
        "gaussian_head": gaussian,
        "web_dir": str(web_dir),
        "decision_notes_enabled": os.environ.get("YAICON_DECISION_NOTES", "1") != "0",
        "prompt_version": PROMPT_VERSION, "language": "en", "llm_port": 8103, "voice_port": 8104,
        "app_port": 8105, "external_inference": (
            bool(elevenlabs_config()) or os.getenv("BIAS_TRANSLATION_PROTOCOL") in {"google-web", "google-cloud"}
            or any(os.getenv(f"BIAS_{kind}_BASE_URL") for kind in ("LLM", "TTS", "ASR", "TRANSLATION")))}
    write_json(RUNTIME, runtime)
    write_json(output / "local-runtime.json", runtime)
    return output


def create_app():
    from bias.api import create_app as upstream_app
    from inspect import signature
    from bias.config import Settings, ProviderSettings, repository_root
    from bias.corpus import Corpus
    from bias.providers import OpenAICompatibleLLM

    runtime = read_json(RUNTIME)
    output = Path(runtime["output"])
    if runtime.get("python_checkout") and runtime_python_checkout(runtime) != repository_root().resolve():
        raise ValueError("This API imported a different Python checkout than prepare; restart with the local launcher")
    public_report = checked_public_rag_report(runtime, output)
    card_path = output / "persona-card.json"
    if not card_path.exists():
        card_path = CARD
    if digest(card_path) != runtime["card_sha256"]:
        raise ValueError("Persona card changed; run prepare again")
    notes_path = output / "decision-notes.json"
    if digest(notes_path) != runtime["decision_notes_sha256"]:
        raise ValueError("Persona decision notes changed; run prepare again")
    links = read_json(notes_path) if runtime["decision_notes_enabled"] else []
    studio = runtime_studio(runtime)
    web_dir = Path(runtime.get("web_dir") or runtime.get("appearance_web_dir") or studio / "apps/web/dist")
    if not (web_dir / "index.html").is_file():
        raise ValueError("Local studio build is missing; run prepare again")
    photo_hash = runtime.get("photo_fit_sha256", runtime.get("appearance_sha256"))
    if photo_hash:
        appearance = read_json(output / "appearance.json")
        asset = web_dir / "models" / f'{appearance["version"]}.glb'
        license_file = asset.with_suffix(".LICENSE.md")
        if (digest(asset) != photo_hash
                or digest(license_file) != appearance["license_sha256"]):
            raise ValueError("Photo-fit studio changed; run prepare again")
    if runtime.get("gaussian_head"):
        verify_gaussian_assets(web_dir / "models/gaussian-head")
    def provider(kind, address=None, model=None):
        base = os.getenv(f"BIAS_{kind}_BASE_URL")
        name = os.getenv(f"BIAS_{kind}_MODEL")
        if base or name:
            return ProviderSettings(base, name, os.getenv(f"BIAS_{kind}_API_KEY"))
        return ProviderSettings(address, model)

    llm = provider("LLM", "http://127.0.0.1:8103/v1", "default_model")
    elevenlabs = elevenlabs_config()
    speech_protocol = {}
    if elevenlabs:
        from bias.providers import ELEVENLABS_BASE_URL
        key, voice_id, model = elevenlabs
        tts = ProviderSettings(ELEVENLABS_BASE_URL, model, key)
        speech_protocol = {"tts_protocol": "elevenlabs"}
    else:
        tts = provider("TTS", "http://127.0.0.1:8104/v1", "qwen3-tts")
        voice_id = os.getenv("BIAS_TTS_VOICE", "default" if os.getenv("BIAS_TTS_BASE_URL") else "reference")
    asr = provider("ASR")
    translation_protocol = os.getenv("BIAS_TRANSLATION_PROTOCOL", "openai-compatible")
    translation = (provider("TRANSLATION") if translation_protocol == "openai-compatible"
                   else ProviderSettings(api_key=os.getenv("BIAS_TRANSLATION_API_KEY")))
    local_asr = find_local_whisper(PROJECT) if not asr.configured and "transcription" in signature(upstream_app).parameters else None
    custom_tts = bool(elevenlabs or os.getenv("BIAS_TTS_BASE_URL"))
    adaptive = os.getenv("BIAS_ADAPTIVE_RESPONSES", "false").lower() in {"true", "1"}
    concise = os.getenv("BIAS_LIVE_CONCISE", "false").lower() in {"true", "1"}
    external = translation_protocol in {"google-web", "google-cloud"} or any(
        urlsplit(item.base_url).hostname not in {"127.0.0.1", "localhost", "::1"}
        for item in (llm, tts, asr, translation) if item.base_url)
    baseline = Settings()
    retrieval_backend = os.getenv("BIAS_RETRIEVAL_BACKEND") or (
        "jensen-public" if public_report["status"] == "integrated" else baseline.retrieval_backend)
    qa_delivery = {}
    if hasattr(baseline, "qa_knowledge_dir"):
        qa_delivery = {"qa_knowledge_dir": Path(os.environ["BIAS_QA_KNOWLEDGE_DIR"]).expanduser() if os.getenv("BIAS_QA_KNOWLEDGE_DIR") else None,
                       "translation": translation,
                       "translation_timeout": float(os.getenv("BIAS_TRANSLATION_TIMEOUT",
                                                            "12" if translation_protocol == "openai-compatible" else "2.5"))}
    elif os.getenv("BIAS_QA_KNOWLEDGE_DIR") or translation.configured:
        raise ValueError("Update the Python studio checkout before enabling QA knowledge or translated subtitles")
    if hasattr(baseline, "translation_protocol"):
        qa_delivery["translation_protocol"] = translation_protocol
    elif translation_protocol != "openai-compatible":
        raise ValueError("Update the Python studio checkout before enabling Google captions")
    policy = {"adaptive_responses": adaptive, "live_concise": concise} if hasattr(baseline, "adaptive_responses") else {}
    config = replace(baseline, corpus_dir=output / "corpus", persona_snapshot=output / "snapshot.json",
        retrieval_backend=retrieval_backend,
        web_dir=web_dir, llm=llm, tts=tts, asr=asr,
        tts_voice=voice_id,
        tts_persona_id="generic" if elevenlabs else os.getenv("BIAS_TTS_PERSONA_ID", "generic" if custom_tts else "jensen-huang"),
        tts_language_hint=os.getenv("BIAS_TTS_LANGUAGE_HINT", "false" if custom_tts else "true").lower() in {"true", "1"},
        llm_max_tokens=280,
        motion_catalog=studio / "data/motion/jensen-multievent.v1.json" if studio in (GESTURE_STUDIO, PR19) else None,
        motion_asset_root=studio / "apps/web/public",
        llm_context_window=15000, max_response_chars=3000, request_timeout=180, retrieval_limit=3, **policy, **speech_protocol, **qa_delivery)
    corpus = Corpus.load(config.corpus_dir)
    if public_report["status"] == "integrated":
        from bias.snapshots import corpus_digest
        if corpus_digest(corpus) != public_report["merged_corpus_sha256"]:
            raise ValueError("Public RAG corpus changed; run prepare again")
    validate_card(read_json(card_path), corpus)
    validate_links(links, corpus)

    class LocalPersonaLLM(OpenAICompatibleLLM):
        async def stream(self, messages):
            system = messages[0]["content"] if messages else ""
            guided_length = bool(policy) and ("adaptive-answer.v1" in system or "Live spoken-answer mode:" in system or "현장 음성 답변은" in system)
            transformed = persona_messages(messages, guided_length=guided_length)
            # Current/project facts must not gain old biographical decision
            # notes or an instruction to speak as a current NVIDIA executive.
            transformed, selected = (transformed, []) if system.startswith("YAICON project/current-source QA.") else with_links(transformed, links)
            prompt_dir = output / "effective-prompts"
            prompt_dir.mkdir(exist_ok=True)
            write_json(prompt_dir / f"{uuid4().hex}.json", {"prompt_version": PROMPT_VERSION,
                "decision_note_ids": selected, "messages": transformed})
            async for chunk in super().stream(transformed):
                yield chunk

    llm_options = {"token_limit_field": os.getenv("BIAS_LLM_TOKEN_LIMIT_FIELD", "max_tokens")} if policy else {}
    app = upstream_app(config, corpus=corpus, llm=LocalPersonaLLM(config.llm,
        timeout=180, max_chars=3000, max_tokens=280, **llm_options),
        **({"transcription": local_asr} if local_asr else {}))

    def runtime_info():
        return {"llm": config.llm.model if os.getenv("BIAS_LLM_BASE_URL") else runtime["models"]["llm"]["repo"],
            "local_only": not external, "server_transcription": asr.model or (local_asr.model_name if local_asr else None),
            "prompt_version": PROMPT_VERSION, "voice": config.tts.model if custom_tts else "Qwen3-TTS Base / experimental reference",
            "retrieval_backend": config.retrieval_backend,
            "retrieval_version": public_report.get("retrieval_version") if config.retrieval_backend == "jensen-public" else None,
            "knowledge_version": (corpus.manifest or {}).get("id"),
            "public_rag": {"status": public_report["status"],
                           "public_corpus_id": public_report.get("public_corpus_id"),
                           "added_records": len(public_report.get("added_evidence_ids", [])),
                           "excluded_events": len(public_report.get("excluded_events", [])),
                           "corpus_sha256": public_report.get("merged_corpus_sha256")},
            "corpus_counts": {"sources": len(corpus.sources), "records": len(corpus.evidence),
                              "training_records": len(corpus.training_evidence),
                              "question_answers": sum(row.question_answer is not None for row in corpus.evidence.values())},
            "voice_provider": "elevenlabs" if elevenlabs else "openai-compatible" if custom_tts else "local-qwen",
            "voice_playback": "whole_answer_default_with_optional_segment_playback",
            "studio": runtime["studio"], "decision_notes_enabled": runtime["decision_notes_enabled"],
            "appearance_version": runtime.get("appearance_version"), "appearance_sha256": runtime.get("appearance_sha256"),
            "gaussian_head": runtime.get("gaussian_head"),
            "persona_quality": "not_evaluated", "voice_likeness": "not_evaluated"}

    # The upstream web app's root mount must follow local API routes.
    studio_mount = next((route for route in app.router.routes if route.name == "studio"), None)
    if studio_mount is not None:
        app.router.routes.remove(studio_mount)
    app.add_api_route("/api/local-runtime", runtime_info, methods=["GET"])
    if studio_mount is not None:
        app.router.routes.append(studio_mount)
    return app


if __name__ == "__main__":
    print(prepare())
