"""Production corpus/retrieval boundaries, independently from response style."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bias.config import Settings, repository_root
from bias.corpus import Corpus
from bias.jensen_public_retrieval import JensenPublicRetriever, public_tokens
from bias.snapshots import PersonaSnapshot

ROOT = repository_root()


@pytest.fixture(scope="module")
def public():
    return Corpus.load(ROOT / "data/personas/jensen-huang-public-v3")


@pytest.mark.parametrize("question", [
    "You used to work at Denny's—what jobs did you actually do?",
    "데니스에서 접시만 씻었던 거야, 손님 서빙도 했어?",
])
def test_waiter_record_is_retrieved_in_both_languages(public, question):
    result = JensenPublicRetriever(public).search(question)
    assert "JHB-dennys-roles" in {hit.evidence.id for hit in result.hits}
    record = public.evidence["JHB-dennys-roles"]
    assert all(role in record.text.lower() for role in ("dishwasher", "busboy", "waiter"))
    assert record.model_extra["persona_memory"]["first_person_eligible"]


def test_failed_measurement_and_absent_market_retrieve_different_criteria(public):
    retriever = JensenPublicRetriever(public)
    absent = retriever.search("Our engine really cuts compute time sixfold, but nobody is paying yet. Is that a reason to stop?")
    wrong = retriever.search("The benchmark was faulty; our supposed performance advantage vanished. What changes?")
    assert "JHJ-001" in {hit.evidence.id for hit in absent.hits}
    assert "JHJ-004" in {hit.evidence.id for hit in wrong.hits}
    assert public.evidence["JHJ-004"].model_extra["persona_memory"]["revision_condition"]


def test_heldout_records_cannot_be_reintroduced_by_strong_aliases(public):
    source = next(iter(public.sources.values())).model_copy(update={"id": "private-holdout", "split": "test",
        "source_event_id": "private-holdout-event", "url": "https://example.org/reserved-holdout"})
    record = public.evidence["JHB-dennys-roles"].model_copy(update={"id": "heldout-only", "source_id": source.id,
        "text": "Unique mango zeppelin result.", "retrieval_aliases": ["mango zeppelin " * 100]})
    isolated = Corpus(public.persona, public.sources | {source.id: source}, public.evidence | {record.id: record})
    assert "heldout-only" not in {hit.evidence.id for hit in JensenPublicRetriever(isolated).search("mango zeppelin").hits}


@pytest.mark.parametrize("question", [
    "What is your private phone number?", "Did you approve our Jensen chatbot?",
    "What exact words did you say at the 2035 CUDA keynote?", "지금 엔비디아 주가는 얼마야?",
    "TSMC quasar nebula viscosity topological marmalade xylophone",
])
def test_aliases_do_not_bypass_existing_evidence_limits(public, question):
    assert not JensenPublicRetriever(public).search(question).hits


def test_discovered_instagram_posts_are_not_indexed_facts(public):
    urls = {str(source.url) for source in public.sources.values()}
    assert not any("instagram.com/p/" in url or "instagram.com/reel/" in url for url in urls)
    record = public.evidence["JHB-official-social-channels"]
    assert not record.model_extra["persona_memory"]["first_person_eligible"]
    assert "instagram.com/nvidia" in record.text


def test_production_snapshot_binds_all_exact_sources_and_records(public):
    snapshot = PersonaSnapshot.load(ROOT / "data/snapshots/jensen-public-2026-11-24.json", public)
    assert snapshot.persona_id == "jensen-huang"
    first = next(iter(public.evidence))
    modified = public.evidence[first].model_copy(update={"text": "An unsupported replacement."})
    with pytest.raises(ValueError, match="exact corpus"):
        snapshot.verify(replace(public, evidence=public.evidence | {first: modified}))


def test_new_records_are_not_mislabeled_full_original_dialogue(public):
    for record in public.evidence.values():
        if not record.id.startswith(("JHB-", "JHJ-")):
            continue
        assert record.question_answer is None
        assert "editorial" in record.model_extra["persona_memory"]["evidence_mode"]
    assert not public.style_evidence


def test_custom_corpus_keeps_explicit_backend_and_avoids_default_jensen_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("BIAS_CORPUS_DIR", str(tmp_path))
    monkeypatch.delenv("BIAS_PERSONA_SNAPSHOT", raising=False)
    monkeypatch.delenv("BIAS_RETRIEVAL_BACKEND", raising=False)
    settings = Settings.from_env()
    assert settings.corpus_dir == tmp_path
    assert settings.persona_snapshot is None
    assert settings.retrieval_backend == "lexical"
    monkeypatch.setenv("BIAS_RETRIEVAL_BACKEND", "jensen-public")
    assert Settings.from_env().retrieval_backend == "jensen-public"


def test_technical_language_is_not_changed_into_restaurant_or_medical_terms():
    assert "doctor" not in public_tokens("의사결정 기준")
    assert {"decision", "criterion"} <= set(public_tokens("의사결정 기준"))
    assert "server" in public_tokens("AI server architecture")
    assert "waiter" not in public_tokens("AI server architecture")


def test_custom_build_does_not_overwrite_the_default_snapshot(tmp_path):
    from tools.build_jensen_public_corpus import build
    from datetime import datetime, timezone
    production = ROOT / "data/snapshots/jensen-public-2026-11-24.json"
    before = production.read_bytes()
    build(output=tmp_path / "corpus", as_of=datetime(2026, 11, 25, tzinfo=timezone.utc))
    assert production.read_bytes() == before
    corpus = Corpus.load(tmp_path / "corpus")
    PersonaSnapshot.load(tmp_path / "corpus/snapshot.json", corpus)
    assert "no-priors-2023" not in corpus.sources


def test_failed_build_leaves_existing_corpus_and_snapshot_usable(tmp_path):
    from tools.build_jensen_public_corpus import build
    from datetime import datetime, timezone
    destination = tmp_path / "corpus"
    build(output=destination, as_of=datetime(2026, 11, 25, tzinfo=timezone.utc))
    before = {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()}
    invalid = tmp_path / "invalid-shards"
    invalid.mkdir()
    (invalid / "bad-sources.json").write_text(json.dumps([{
        "id": "unrelated-test-source", "title": "Conflicting event split", "url": "https://example.org/conflict",
        "publisher": "Test fixture", "published_at": "2024-05-21", "source_type": "fixture",
        "split": "test", "source_event_id": "stripe-sessions-2024",
    }]))
    with pytest.raises(ValueError, match="Event spans splits"):
        build(acquisition=invalid, output=destination, as_of=datetime(2026, 11, 25, tzinfo=timezone.utc))
    assert {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()} == before


def test_build_interruption_restores_last_good_corpus(tmp_path, monkeypatch):
    from tools import build_jensen_public_corpus as builder
    from datetime import datetime, timezone
    output = tmp_path / "corpus"
    builder.build(output=output, as_of=datetime(2026, 11, 25, tzinfo=timezone.utc))
    before = (output / "evidence.jsonl").read_bytes()
    real_replace = builder.os.replace

    def interrupt_after_backup(src, dst):
        real_replace(src, dst)
        if Path(src) == output and ".last-good-" in Path(dst).name:
            raise KeyboardInterrupt("fixture interruption")

    monkeypatch.setattr(builder.os, "replace", interrupt_after_backup)
    with pytest.raises(KeyboardInterrupt):
        builder.build(output=output, as_of=datetime(2026, 11, 25, tzinfo=timezone.utc))
    assert (output / "evidence.jsonl").read_bytes() == before
    PersonaSnapshot.load(output / "snapshot.json", Corpus.load(output))


def test_snapshot_copy_failure_preserves_original_bytes(tmp_path, monkeypatch):
    from tools import build_jensen_public_corpus as builder
    from datetime import datetime, timezone
    output, snapshot = tmp_path / "corpus", tmp_path / "external-snapshot.json"
    builder.build(output=output, snapshot_path=snapshot, as_of=datetime(2026, 11, 25, tzinfo=timezone.utc))
    before = snapshot.read_bytes()
    real_copy = builder.shutil.copyfile

    def fail_pending_snapshot(src, dst, *args, **kwargs):
        if Path(dst).name.startswith(".jensen-snapshot-"):
            raise OSError("fixture full disk")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(builder.shutil, "copyfile", fail_pending_snapshot)
    with pytest.raises(OSError, match="full disk"):
        builder.build(output=output, snapshot_path=snapshot, as_of=datetime(2026, 11, 26, tzinfo=timezone.utc))
    assert snapshot.read_bytes() == before
    PersonaSnapshot.load(snapshot, Corpus.load(output))
