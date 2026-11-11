from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import pytest

from bias.corpus import Corpus, CorpusError
from tools.persona_data import heldout
from tools.persona_data.parsers import digest
from tools.persona_data.pipeline import acquire_sources, build, write_json


ROOT = Path(__file__).resolve().parents[2]
COMPARISON = ROOT / "data/personas/elon-musk"
ANSWER = "A distinct synthetic response describes amber instruments and violet observatories while keeping every qualification intact."
CONTEXT = "This preceding synthetic statement belongs to the original speaker."


def lex_html(answer=ANSWER, context=CONTEXT):
    rows = [("Elon Musk", "00:00:01", context), ("Lex Fridman", "00:00:05", "How would the independent amber observatory work?"),
            ("Elon Musk", "00:00:10", answer)]
    return ("<html>" + "".join(f'<div class="ts-segment"><span class="ts-name">{speaker}</span><span class="ts-timestamp">({stamp})</span><span class="ts-text">{html.escape(text)}</span></div>' for speaker, stamp, text in rows) + "</html>").encode()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    value = {"id": "lex-018-2019", "source_event_id": "lex-018-2019", "event_family_id": "lex-018-2019", "split": "test",
             "url": "https://lexfridman.com/elon-musk/", "title": "Synthetic publisher fixture", "published_at": None,
             "publisher": "Lex Fridman", "source_type": "interview", "prior_exposure": {"original_qa_split": None}}
    write_json(registry / "source-families.json", {"schema_version": "persona-source-families.v1", "sources": [value]})
    write_json(registry / "split-registration.json", {"schema_version": "persona-split-registration.v1", "events": [
        {"source_event_id": value["source_event_id"], "event_family_id": value["event_family_id"], "split": "test", "source_ids": [value["id"]]}]})
    payload = {"raw": lex_html()}

    def collect(source, destination):
        artifact = destination / source["id"] / "publisher.html"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(payload["raw"])
        return {"source_id": source["id"], "artifact": artifact.name, "sha256": hashlib.sha256(payload["raw"]).hexdigest(),
                "bytes": len(payload["raw"]), "retrieved_at": "2026-11-11T10:00:00+00:00",
                "requested_url": source.get("acquisition_url", source["url"]), "final_url": source.get("acquisition_url", source["url"])}

    monkeypatch.setattr(heldout, "collect_source", collect)
    return registry, tmp_path / "acquisition", tmp_path / "evaluation", payload


def acquire(setup):
    registry, acquisition, output, _ = setup
    heldout.acquire_heldout(registry, acquisition, ["lex-018-2019"])
    return registry, acquisition, output


def materialize(setup):
    registry, acquisition, output = acquire(setup)
    manifest = heldout.build_heldout(registry, acquisition, output, comparison_corpora=[COMPARISON])
    return output, manifest


def test_materialization_is_private_evaluation_only_and_preserves_original_context(setup):
    output, manifest = materialize(setup)
    loaded, cases = heldout.load_heldout(output)
    assert loaded == manifest
    assert len(cases) == 1
    assert cases[0]["reference_answer"] == ANSWER
    assert cases[0]["context"] == [{"role": "assistant", "content": CONTEXT}]
    assert cases[0]["published_at"] is None
    assert cases[0]["split"] == "test"
    assert all(value is False for value in cases[0]["projection_eligibility"].values())
    assert not manifest["human_review_completed"]
    assert not (output / "projections").exists()
    assert not (output / "evidence.jsonl").exists()
    with pytest.raises(CorpusError):
        Corpus.load(output)


def test_existing_training_acquisition_and_build_cannot_accept_heldout(setup, tmp_path):
    registry, acquisition, _ = acquire(setup)
    with pytest.raises(ValueError, match="Reserved test"):
        acquire_sources(registry, tmp_path / "wrong", ["lex-018-2019"])
    # Even manually copying the acquired log into the older shape cannot
    # turn reserved events into a runtime/training corpus.
    manifest = json.loads((acquisition / "heldout-acquisition.json").read_text())
    write_json(acquisition / "acquisition-attempts.json", manifest["attempts"])
    with pytest.raises(ValueError, match="Reserved test"):
        build(registry, acquisition, tmp_path / "wrong-build", report_dir=tmp_path / "wrong-report")


@pytest.mark.parametrize("split", ["train", "validation"])
def test_heldout_acquisition_rejects_non_test_sources(setup, split):
    registry, acquisition, _, _ = setup
    for name, field in (("source-families.json", "sources"), ("split-registration.json", "events")):
        value = json.loads((registry / name).read_text())
        value[field][0]["split"] = split
        write_json(registry / name, value)
    with pytest.raises(ValueError, match="test sources only"):
        heldout.acquire_heldout(registry, acquisition, ["lex-018-2019"])
    assert not acquisition.exists()


def test_comparison_is_required(setup):
    registry, acquisition, output = acquire(setup)
    with pytest.raises(ValueError, match="comparison corpus"):
        heldout.build_heldout(registry, acquisition, output, comparison_corpora=[])
    assert not output.exists()


def test_frozen_registration_change_invalidates_acquisition(setup):
    registry, acquisition, output = acquire(setup)
    path = registry / "source-families.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hashes do not match"):
        heldout.build_heldout(registry, acquisition, output, comparison_corpora=[COMPARISON])


@pytest.mark.parametrize("mutation", ["raw", "artifact_path", "source_id", "bytes"])
def test_acquisition_identity_and_bytes_fail_closed(setup, mutation):
    registry, acquisition, output = acquire(setup)
    path = acquisition / "heldout-acquisition.json"
    manifest = json.loads(path.read_text())
    record = manifest["attempts"][0]["record"]
    if mutation == "raw":
        (acquisition / "raw/lex-018-2019/publisher.html").write_text("changed")
    elif mutation == "artifact_path":
        record["artifact"] = "../publisher.html"
    elif mutation == "source_id":
        record["source_id"] = "other-source"
    else:
        record["bytes"] += 1
    write_json(path, manifest)
    with pytest.raises(ValueError, match="filename|identity, bytes or hash"):
        heldout.build_heldout(registry, acquisition, output, comparison_corpora=[COMPARISON])
    assert not output.exists()


def test_source_family_overlap_is_rejected_even_with_new_source_id(setup):
    registry, acquisition, output, _ = setup
    existing = next(iter(Corpus.load(COMPARISON).sources.values()))
    for name, field in (("source-families.json", "sources"), ("split-registration.json", "events")):
        value = json.loads((registry / name).read_text())
        value[field][0]["event_family_id"] = (existing.model_extra or {}).get("event_family_id", heldout.source_event_id(existing))
        write_json(registry / name, value)
    acquire(setup)
    with pytest.raises(ValueError, match="already exposed"):
        heldout.build_heldout(registry, acquisition, output, comparison_corpora=[COMPARISON])


@pytest.mark.parametrize("kind", ["answer", "embedded_span", "context"])
def test_exact_and_embedded_original_wording_leaks_are_rejected(setup, kind):
    corpus = Corpus.load(COMPARISON)
    original = next(record.text for record in corpus.evidence.values() if len(record.text.split()) >= 14)
    registry, acquisition, output, payload = setup
    if kind == "answer":
        payload["raw"] = lex_html(answer=original)
    elif kind == "embedded_span":
        payload["raw"] = lex_html(answer="A different introduction precedes " + original + " and a different ending follows.")
    else:
        payload["raw"] = lex_html(context=original)
    acquire(setup)
    with pytest.raises(ValueError, match="duplicates comparison|shares a 12-word span"):
        heldout.build_heldout(registry, acquisition, output, comparison_corpora=[COMPARISON])


def test_landing_page_is_not_synthetic_dialogue(setup):
    registry, acquisition, output, payload = setup
    payload["raw"] = b"<html><p>Episode outline: spacecraft, machines, space.</p></html>"
    acquire(setup)
    with pytest.raises(ValueError, match="landing pages are not transcripts"):
        heldout.build_heldout(registry, acquisition, output, comparison_corpora=[COMPARISON])
    assert not output.exists()


def test_case_file_tampering_is_rejected(setup):
    output, _ = materialize(setup)
    path = output / "cases.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        heldout.load_heldout(output)


def test_changed_reference_cannot_hide_behind_updated_file_hash(setup):
    output, manifest = materialize(setup)
    path = output / "cases.jsonl"
    case = json.loads(path.read_text())
    case["reference_answer"] += " Changed after freeze."
    case["answer_sha256"] = heldout.sha_text(case["reference_answer"])
    path.write_text(json.dumps(case) + "\n")
    manifest["files"]["cases.jsonl"] = heldout._hash(path)
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="frozen recipe"):
        heldout.load_heldout(output)


def test_loader_rechecks_leakage_even_when_case_hashes_are_rebound(setup):
    output, manifest = materialize(setup)
    exposure_path = output / "exposure-index.json"
    exposure = json.loads(exposure_path.read_text())
    exposure["event_family_ids"].append("lex-018-2019")
    write_json(exposure_path, exposure)
    manifest["files"][exposure_path.name] = heldout._hash(exposure_path)
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="already exposed"):
        heldout.load_heldout(output)


def test_case_symlink_is_rejected(setup, tmp_path):
    output, _ = materialize(setup)
    path = output / "cases.jsonl"
    outside = tmp_path / "same-cases.jsonl"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        heldout.load_heldout(output)


def test_loader_rejects_training_projection_next_to_references(setup):
    output, _ = materialize(setup)
    (output / "train.jsonl").write_text("")
    with pytest.raises(ValueError, match="runtime or training"):
        heldout.load_heldout(output)


def test_shareable_reports_contain_no_question_or_reference_wording(setup, tmp_path):
    registry, acquisition, output = acquire(setup)
    report = tmp_path / "report"
    heldout.build_heldout(registry, acquisition, output, comparison_corpora=[COMPARISON], report_dir=report)
    combined = "".join(path.read_text() for path in report.iterdir())
    assert ANSWER not in combined and CONTEXT not in combined
    assert "How would the independent amber observatory work?" not in combined


def test_preserve_existing_artifact(setup):
    output, _ = materialize(setup)
    original = (output / "manifest.json").read_bytes()
    registry, acquisition, _, _ = setup
    with pytest.raises(ValueError, match="new held-out"):
        heldout.build_heldout(registry, acquisition, output, comparison_corpora=[COMPARISON])
    assert (output / "manifest.json").read_bytes() == original


def test_unknown_dates_and_unreviewed_reference_cannot_be_reclassified(setup):
    output, manifest = materialize(setup)
    manifest["human_review_completed"] = True
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="evaluation-only"):
        heldout.load_heldout(output)


def test_document_supplement_must_bind_same_registry_and_publisher(setup, tmp_path):
    registry, acquisition, _, _ = setup
    registry_value, registration = heldout.load_registry(registry)
    document = {"schema_version": "persona-heldout-documents.v1", "registered_at": "2026-11-11T10:00:00Z",
                "source_registry_sha256": digest(registry_value), "split_registration_sha256": digest(registration),
                "documents": [{"source_id": "lex-018-2019", "adapter": "lex_publisher_html", "basis": "Synthetic fixture registration.",
                               "discovery_url": "https://lexfridman.com/elon-musk/", "acquisition_url": "https://example.org/repost/"}]}
    path = tmp_path / "documents.json"
    write_json(path, document)
    with pytest.raises(ValueError, match="original publisher"):
        heldout.acquire_heldout(registry, acquisition, ["lex-018-2019"], documents_path=path)
    assert not acquisition.exists()


def test_failed_download_is_recorded_without_substitute(setup, monkeypatch):
    registry, acquisition, _, _ = setup
    def failed(*args, **kwargs):
        raise OSError("Fixture publisher unavailable")
    monkeypatch.setattr(heldout, "collect_source", failed)
    manifest = heldout.acquire_heldout(registry, acquisition, ["lex-018-2019"])
    assert manifest["attempts"] == [{"source_id": "lex-018-2019", "status": "failed", "error": "Fixture publisher unavailable"}]
    assert json.loads((acquisition / "heldout-acquisition.json").read_text()) == manifest


def test_reference_wording_is_excluded_when_already_in_supplied_context(setup):
    registry, acquisition, output, payload = setup
    payload["raw"] = lex_html(context=ANSWER)
    acquire(setup)
    with pytest.raises(ValueError, match="reference_overlaps_supplied_question_or_context"):
        heldout.build_heldout(registry, acquisition, output, comparison_corpora=[COMPARISON])


def test_comparison_coverage_cannot_diverge_from_frozen_index(setup):
    output, manifest = materialize(setup)
    manifest["leakage_policy"]["checked_corpora"] = []
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="comparison coverage"):
        heldout.load_heldout(output)
