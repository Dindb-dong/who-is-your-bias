"""Synthetic fixtures exercise private dialogue preservation and event isolation."""
from datetime import date, datetime, timezone
import os
from pathlib import Path

import pytest

from bias.corpus import Corpus, CorpusError
from bias.models import Evidence, Persona, Source
from bias.public_corpus import merge_public_corpus, split_manifest
from bias.snapshots import corpus_digest, register_snapshot
from pipeline.common import digest, read_json, read_jsonl, write_json, write_jsonl
from scripts import local_persona


def source(sid, *, event=None, url=None, split="train", **extra):
    return Source(id=sid, title=f"Synthetic {sid}", url=url or f"https://example.org/{sid}",
                  publisher="Synthetic publisher", published_at=date(2024, 1, 1),
                  source_type="synthetic_test_fixture", split=split,
                  source_event_id=event or sid, **extra)


def evidence(rid, sid, *, text=None):
    return Evidence(id=rid, source_id=sid, text=text or f"Synthetic evidence {rid}.", topics=["test"],
                    locator={"label": "Synthetic turn"}, kind="stated_view",
                    confidence="source_grounded", review_status="reviewed")


def original_qa():
    # Not an actual Jensen quotation. The exact strings exist only in this test.
    return Evidence.model_validate({"id": "private-qa", "source_id": "private-source",
        "topics": ["test"], "situations": ["reflection"], "locator": {"label": "Synthetic exchange"},
        "kind": "question_answer", "confidence": "source_grounded", "review_status": "reviewed",
        "projection_eligibility": {"factual": True, "text_style": True,
                                   "aligned_speech": False, "visible_behavior": False},
        "question_answer": {"question": "Synthetic question?", "answer": "An exact synthetic answer.",
            "context_before": "Synthetic context.", "context_kind": "original_text",
            "original_language": "en", "questioner": "Synthetic interviewer", "respondent": "Jensen Huang",
            "source_event_id": "private-event", "completeness": "complete_exchange",
            "timing_basis": "unavailable", "transcript_locator": {
                "question_cue_ids": ["cue-1"], "answer_cue_ids": ["cue-2"], "transcript_sha256": "a" * 64},
            "review": {"reviewer_type": "human", "reviewer": "Synthetic reviewer", "reviewed_at": "2024-01-02",
                       "basis": "Synthetic fixture only", "speaker_status": "publisher_labels_checked"},
            "annotation_basis": "Synthetic fixture only"}})


def corpus(sources, records, *, manifest=None, assignments=None):
    return Corpus(Persona(id="jensen-huang", name="Synthetic Jensen fixture", tagline="Test",
                          description="Synthetic corpus, not a persona evaluation.",
                          voice_binding="keep-original", style_rules=["Keep the reviewed original style."]),
                  {row.id: row for row in sources}, {row.id: row for row in records},
                  manifest or {"id": "synthetic-corpus"}, assignments)


def private_corpus():
    return corpus([source("private-source", event="private-event", url="https://example.org/interview")],
                  [original_qa()], manifest={"id": "private-v3", "original_binding": "untouched",
                      "source_content_sha256": {"private-source": "a" * 64}})


def write_corpus(path, value, *, omit_original_text=False):
    write_json(path / "profile.json", value.persona.model_dump(mode="json"))
    write_json(path / "sources.json", [row.model_dump(mode="json") for row in value.sources.values()])
    rows = [row.model_dump(mode="json", exclude_none=True) for row in value.evidence.values()]
    if omit_original_text:
        for row in rows:
            if row.get("question_answer"):
                row.pop("text")
    write_jsonl(path / "evidence.jsonl", rows)
    write_json(path / "split-manifest.json", split_manifest(value))
    write_json(path / "corpus-manifest.json", value.manifest)


def test_original_dialogue_metadata_and_profile_survive_public_alias_and_duplicate():
    original = private_corpus()
    public = corpus([source("public-alias", event="new-event-name", url="https://example.org/interview/?utm_source=x")],
                    [evidence("private-qa", "public-alias", text="This must never replace the original answer."),
                     evidence("new-note", "public-alias")], manifest={"id": "public-v3"})
    result = merge_public_corpus(original, public)
    merged = result.corpus
    assert merged.evidence["private-qa"] is original.evidence["private-qa"]
    assert merged.sources["private-source"] is original.sources["private-source"]
    assert merged.persona is original.persona
    assert merged.evidence["new-note"].source_id == "private-source"
    assert set(merged.sources) == {"private-source"}
    assert merged.manifest["original_binding"] == "untouched"
    assert merged.manifest["source_content_sha256"] == {"private-source": "a" * 64}
    assert result.report["preserved_original_evidence_ids"] == ["private-qa"]
    assert result.report["reused_source_ids"] == {"public-alias": "private-source"}
    assert result.report["merged_corpus_sha256"] == corpus_digest(merged)
    assert original.manifest["id"] == "private-v3"
    assert set(original.evidence) == {"private-qa"}


@pytest.mark.parametrize("boundary", ["reserved_event", "reserved_url", "frozen_source", "existing_holdout"])
def test_conflict_excludes_whole_event_and_preserves_unrelated_additions(boundary):
    original = private_corpus()
    kwargs = {}
    conflicting = source("incoming-source", event="incoming-event", url="https://example.org/heldout?utm_medium=x")
    if boundary == "reserved_event":
        kwargs["reserved_events"] = {"incoming-event"}
    elif boundary == "reserved_url":
        kwargs["reserved_urls"] = {"https://example.org/heldout#turn2"}
    elif boundary == "frozen_source":
        kwargs["reserved_assignments"] = {"incoming-source": ("unseen-event", "test")}
    else:
        heldout = source("heldout-source", event="heldout-event", url="https://example.org/heldout", split="test")
        original = corpus([*original.sources.values(), heldout],
                          [*original.evidence.values(), evidence("heldout-note", heldout.id)])
    incoming = corpus([conflicting, source("same-event-other-page", event="incoming-event"), source("safe-source")],
                      [evidence("conflict-note", conflicting.id), evidence("other-page-note", "same-event-other-page"),
                       evidence("safe-note", "safe-source")])
    result = merge_public_corpus(original, incoming, **kwargs)
    assert result.report["added_evidence_ids"] == ["safe-note"]
    assert "conflict-note" not in result.corpus.evidence
    assert "other-page-note" not in result.corpus.evidence
    assert result.corpus.evidence["private-qa"] is original.evidence["private-qa"]
    assert result.report["excluded_events"][0]["source_ids"] == ["incoming-source", "same-event-other-page"]
    if boundary == "existing_holdout":
        assert result.corpus.sources["heldout-source"].split == "test"
        assert "heldout-note" not in {row.id for row in result.corpus.training_evidence}


def test_renamed_event_cannot_bypass_reserved_mirror_and_absent_reservations_persist():
    original = private_corpus()
    incoming = corpus([source("a", event="reserved", url="https://example.org/mirror"),
                       source("b", event="renamed", url="https://example.org/mirror#later"),
                       source("c", event="renamed")],
                      [evidence("note-a", "a"), evidence("note-b", "b"), evidence("note-c", "c")])
    result = merge_public_corpus(original, incoming, reserved_assignments={"future-source": ("reserved", "test")})
    assert result.report["added_evidence_ids"] == []
    assert len(result.report["excluded_events"]) == 2
    assert {"source_event_id": "reserved", "split": "test", "source_ids": ["future-source"]} in split_manifest(result.corpus)["events"]


def test_source_id_collision_cannot_rebind_original_qa_to_another_interview():
    incoming = corpus([source("private-source", event="another-event")], [evidence("new-note", "private-source")])
    result = merge_public_corpus(private_corpus(), incoming)
    assert result.report["added_evidence_ids"] == []
    assert result.report["excluded_events"][0]["reasons"] == ["source_identity_conflict"]


def test_persona_mismatch_is_rejected():
    incoming = private_corpus()
    incoming = Corpus(incoming.persona.model_copy(update={"id": "someone-else"}), incoming.sources, incoming.evidence)
    with pytest.raises(CorpusError, match="persona differs"):
        merge_public_corpus(private_corpus(), incoming)


def test_materialized_overlay_preserves_raw_private_qa_and_snapshot_binding(tmp_path, monkeypatch):
    original = private_corpus()
    public = corpus([source("new-source")], [evidence("new-note", "new-source")], manifest={"id": "public-v3"})
    write_corpus(tmp_path / "corpus", original, omit_original_text=True)
    write_json(tmp_path / "integration.json", {"private_binding": "keep"})
    before = read_jsonl(tmp_path / "corpus/evidence.jsonl")[0]
    profile_digest = digest(tmp_path / "corpus/profile.json")
    monkeypatch.setattr(local_persona, "public_rag_reservations", lambda output: {
        "reserved_assignments": {"absent-holdout": ("unseen", "test")}})
    report = local_persona.attach_public_corpus(tmp_path, (public, "test-retrieval-v1"))
    assert read_jsonl(tmp_path / "corpus/evidence.jsonl")[0] == before
    assert digest(tmp_path / "corpus/profile.json") == profile_digest
    assert read_json(tmp_path / "integration.json")["private_binding"] == "keep"
    loaded = Corpus.load(tmp_path / "corpus")
    assert loaded.source_assignments["absent-holdout"] == ("unseen", "test")
    assert report == read_json(tmp_path / "public-rag-merge.json")
    stamp = datetime(2026, 11, 24, tzinfo=timezone.utc)
    snapshot = register_snapshot(loaded, identifier="synthetic-overlay", as_of=stamp,
                                 ingested_at=stamp, current_evidence_since=date(2026, 1, 1))
    snapshot.verify(loaded)
    assert snapshot.corpus_sha256 == report["merged_corpus_sha256"]
    local_persona.validate_card({"target_id": "jensen-huang", "observations": [{"basis": [
        {"dialogue_id": "private-qa", "exact_excerpt": "exact synthetic answer"}]}]}, loaded)


def test_public_overlay_is_default_missing_data_fails_and_optout_is_explicit(tmp_path, monkeypatch):
    import bias.config
    monkeypatch.delenv("YAICON_PUBLIC_RAG", raising=False)
    monkeypatch.delenv("YAICON_PUBLIC_CORPUS_DIR", raising=False)
    monkeypatch.setattr(bias.config, "repository_root", lambda: tmp_path)
    assert local_persona.public_rag_enabled()
    with pytest.raises(ValueError, match="missing or incomplete"):
        local_persona.public_rag_input()
    monkeypatch.setenv("YAICON_PUBLIC_CORPUS_DIR", "relative/corpus")
    with pytest.raises(ValueError, match="absolute corpus directory"):
        local_persona.public_rag_input()
    monkeypatch.setenv("YAICON_PUBLIC_RAG", "0")
    assert local_persona.public_rag_input() is None


def test_stale_or_tampered_runtime_cannot_silently_use_original_corpus(tmp_path, monkeypatch):
    monkeypatch.delenv("YAICON_PUBLIC_RAG", raising=False)
    with pytest.raises(ValueError, match="predates the overlay"):
        local_persona.checked_public_rag_report({}, tmp_path)
    monkeypatch.setenv("YAICON_PUBLIC_RAG", "0")
    assert local_persona.checked_public_rag_report({}, tmp_path)["status"] == "disabled"
    report = {"status": "disabled"}
    write_json(tmp_path / "public-rag-merge.json", report)
    runtime = {"public_rag": report, "public_rag_report_sha256": digest(tmp_path / "public-rag-merge.json")}
    write_json(tmp_path / "public-rag-merge.json", {"status": "integrated"})
    with pytest.raises(ValueError, match="audit report changed"):
        local_persona.checked_public_rag_report(runtime, tmp_path)


def test_old_frontend_selection_does_not_change_prepared_python_checkout(tmp_path, monkeypatch):
    from scripts.run_local_persona import process_environment
    core = tmp_path / "updated-python"
    api = core / "src/bias/api.py"
    api.parent.mkdir(parents=True)
    api.write_text("# synthetic code marker\n")
    old_assets = tmp_path / "pr19-renderer"
    old_assets.mkdir()
    monkeypatch.setattr(local_persona, "PR19", old_assets)
    runtime = {"python_checkout": str(core), "studio": old_assets.name}
    assert local_persona.runtime_studio(runtime) == old_assets
    paths = [Path(value) for value in process_environment(runtime)["PYTHONPATH"].split(os.pathsep)]
    assert core / "src" in paths
    assert core in paths
    assert old_assets not in paths
    assert old_assets / "src" not in paths


def test_missing_prepared_code_root_is_not_replaced_by_asset_checkout(tmp_path):
    with pytest.raises(ValueError, match="Python checkout is unavailable"):
        local_persona.runtime_python_checkout({"python_checkout": str(tmp_path / "missing"), "studio": "pr19"})


def test_local_app_uses_materialized_overlay_and_explicit_backend_override(tmp_path, monkeypatch):
    import bias.api
    from bias.jensen_public_retrieval import VERSION
    from fastapi import FastAPI

    # Only app construction is exercised. No model, microphone, voice or avatar
    # service runs; the real upstream API factory is replaced at its boundary.
    write_corpus(tmp_path / "corpus", private_corpus())
    write_json(tmp_path / "integration.json", {})
    monkeypatch.setattr(local_persona, "public_rag_reservations", lambda output: {})
    public = corpus([source("new-source")], [evidence("new-note", "new-source")])
    report = local_persona.attach_public_corpus(tmp_path, (public, VERSION))
    write_json(tmp_path / "persona-card.json", {"target_id": "jensen-huang", "observations": []})
    write_json(tmp_path / "decision-notes.json", [])
    (tmp_path / "web").mkdir()
    (tmp_path / "web/index.html").write_text("Synthetic app fixture")
    runtime_path = tmp_path / "runtime.json"
    write_json(runtime_path, {"output": str(tmp_path), "models": {"llm": {"repo": "synthetic-model"}},
        "public_rag": report, "public_rag_report_sha256": digest(tmp_path / "public-rag-merge.json"),
        "card_sha256": digest(tmp_path / "persona-card.json"),
        "decision_notes_sha256": digest(tmp_path / "decision-notes.json"),
        "decision_notes_enabled": False, "studio": "synthetic", "web_dir": str(tmp_path / "web")})
    monkeypatch.setattr(local_persona, "RUNTIME", runtime_path)
    monkeypatch.setenv("YAICON_PUBLIC_RAG", "1")
    monkeypatch.delenv("BIAS_RETRIEVAL_BACKEND", raising=False)
    for key, value in {"BIAS_EMBEDDING_BACKEND": "http", "BIAS_EMBEDDING_MODEL": "synthetic-embedding",
                       "BIAS_EMBEDDING_REVISION": "fixture-revision", "BIAS_EMBEDDING_BASE_URL": "https://embeddings.example.org/v1",
                       "BIAS_EMBEDDING_API_KEY": "synthetic-key", "BIAS_EMBEDDING_QUERY_PREFIX": "question: ",
                       "BIAS_EMBEDDING_PASSAGE_PREFIX": "document: ", "BIAS_DENSE_MIN_SIMILARITY": "0.73",
                       "BIAS_PERSONA_GRAPH": "/synthetic/graph-must-stay-disabled.json"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(local_persona, "elevenlabs_config", lambda: None)
    captured = []

    def upstream(config, **kwargs):
        captured.append((config, kwargs["corpus"]))
        return FastAPI()

    monkeypatch.setattr(bias.api, "create_app", upstream)
    app = local_persona.create_app()
    original_config = captured[-1][0]
    assert original_config.retrieval_backend == "jensen-public"
    assert original_config.embedding_model is None
    assert original_config.persona_graph is None
    assert captured[-1][1].evidence["private-qa"].question_answer == original_qa().question_answer
    info = next(route.endpoint for route in app.router.routes if route.path == "/api/local-runtime")()
    assert info["retrieval_version"] == VERSION
    assert info["corpus_counts"]["question_answers"] == 1
    assert info["public_rag"]["added_records"] == 1
    monkeypatch.setenv("BIAS_RETRIEVAL_BACKEND", "lexical")
    local_persona.create_app()
    assert captured[-1][0].retrieval_backend == "lexical"
    assert captured[-1][0].embedding_model is None
    monkeypatch.setenv("BIAS_RETRIEVAL_BACKEND", "hybrid")
    hybrid_app = local_persona.create_app()
    configured = captured[-1][0]
    assert configured.retrieval_backend == "hybrid"
    assert configured.embedding_backend == "http"
    assert configured.embedding_model == "synthetic-embedding"
    assert configured.embedding_revision == "fixture-revision"
    assert configured.embedding_base_url == "https://embeddings.example.org/v1"
    assert configured.embedding_api_key == "synthetic-key"
    assert configured.embedding_query_prefix == "question: "
    assert configured.embedding_passage_prefix == "document: "
    assert configured.dense_min_similarity == 0.73
    assert configured.persona_graph is None
    assert configured.tts == original_config.tts
    assert configured.tts_voice == original_config.tts_voice
    assert configured.tts_persona_id == original_config.tts_persona_id
    assert configured.motion_catalog == original_config.motion_catalog
    hybrid_info = next(route.endpoint for route in hybrid_app.router.routes if route.path == "/api/local-runtime")()
    assert hybrid_info["local_only"] is False
    assert "synthetic-key" not in str(hybrid_info)
