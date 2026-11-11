"""Real shipped geometry/catalog at API boundaries; no model or identity inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bias.api import create_app
from bias.config import Settings, repository_root
from bias.motion import MotionLibrary


@pytest.fixture(scope="module")
def library():
    root = repository_root()
    return MotionLibrary.load(root / "data/motion/elon-musk-ted-v1.json", root / "apps/web/public")


@pytest.mark.parametrize("language,question", [
    ("en", "Why compare power plants and electric cars for efficiency?"),
    ("ko", "전기차와 발전소의 에너지 효율은 어떻게 달라?"),
])
def test_evidence_behavior_request_selects_only_explicit_preview_motion(library, actual_corpus, language, question, tmp_path):
    with TestClient(create_app(Settings(web_dir=tmp_path / "absent"), corpus=actual_corpus, motion_library=library)) as client:
        base = {"message": question, "language": language, "mode": "behavior"}
        for mode in (None, "disabled", "reviewed"):
            response = client.post("/api/chat", json=base | ({"motion_mode": mode} if mode else {}))
            assert response.status_code == 200
            assert response.json()["behavior"]["motion"] is None
        response = client.post("/api/chat", json=base | {"motion_mode": "geometry_preview"})
        turn = response.json()
        assert response.status_code == 200
        assert turn["grounding"]["retrieved_ids"][0] == "elon-tesla06-system"
        chosen = turn["behavior"]["motion"]
        assert chosen["source_event_id"] == "ted-2013"
        assert chosen["review_status"] == "geometry_preview"
        assert chosen["context_alignment"] == "asr_word_matches"
        assert chosen["duration_ms"] == pytest.approx((chosen["source_end_seconds"] - chosen["source_start_seconds"]) * 1000)


@pytest.mark.parametrize("mode,message", [("profile", "electric cars power plants"), ("rag", "electric cars power plants"), ("behavior", "What is the weather today?")])
def test_profile_rag_and_evidence_abstention_never_attach_motion(library, actual_corpus, mode, message, tmp_path):
    with TestClient(create_app(Settings(web_dir=tmp_path / "absent"), corpus=actual_corpus, motion_library=library)) as client:
        result = client.post("/api/chat", json={"message": message, "mode": mode, "motion_mode": "geometry_preview"})
        assert result.status_code == 200
        assert result.json()["behavior"]["motion"] is None


def test_matching_motion_context_does_not_replace_missing_answer_evidence(library, actual_corpus, tmp_path):
    question = "Why can power plants charge electric cars more efficiently?"
    assert library.select(question, allow_geometry_preview=True) is not None
    with TestClient(create_app(Settings(web_dir=tmp_path / "absent"), corpus=actual_corpus, motion_library=library)) as client:
        response = client.post("/api/chat", json={
            "message": question, "mode": "behavior", "motion_mode": "geometry_preview",
        })
        assert response.status_code == 200
        turn = response.json()
        assert turn["grounding"]["status"] == "insufficient"
        assert turn["grounding"]["retrieved_ids"] == []
        assert turn["citations"] == []
        assert turn["behavior"]["motion"] is None
        assert "tunnel" not in turn["text"].lower()


def test_clip_gate_and_canonical_bytes_are_the_selection_hash(library, actual_corpus, tmp_path):
    chosen = library.select("power plants electric cars fuel efficiency", allow_geometry_preview=True)
    assert chosen is not None
    with TestClient(create_app(Settings(web_dir=tmp_path / "absent"), corpus=actual_corpus, motion_library=library)) as client:
        summary = client.get("/api/motion/library").json()
        assert summary["available"] is True
        assert summary["preview_runtime_primitives"] == 12
        assert summary["reviewed_runtime_primitives"] == 0
        assert client.get(chosen.clip_url).status_code == 404
        assert client.get(chosen.clip_url + "?preview=false").status_code == 404
        response = client.get(chosen.clip_url + "?preview=true")
        assert response.status_code == 200
        expected = json.dumps(library.clip(chosen.primitive_id, allow_geometry_preview=True), ensure_ascii=False,
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        assert response.content == expected
        assert hashlib.sha256(response.content).hexdigest() == chosen.clip_sha256
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["calibration"]["parent_reference"]["clip_sha256"] == chosen.parent_clip_sha256


def test_holdout_is_not_served_even_with_preview(library, actual_corpus, tmp_path):
    records = {clip.id: clip.source_id for clip in library.catalog.clips}
    heldout = [primitive.id for primitive in library.catalog.primitives if records[primitive.clip_id] == "ted-2017"]
    assert heldout
    with TestClient(create_app(Settings(web_dir=tmp_path / "absent"), corpus=actual_corpus, motion_library=library)) as client:
        assert all(client.get(f"/api/motion/clip/{identifier}?preview=true").status_code == 404 for identifier in heldout)
        assert client.get("/api/motion/clip/missing?preview=true").status_code == 404


def test_optional_catalog_is_loaded_by_configuration_and_failure_keeps_chat_available(actual_corpus, tmp_path):
    root = repository_root()
    configured = Settings(web_dir=tmp_path / "absent", motion_catalog=root / "data/motion/elon-musk-ted-v1.json", motion_asset_root=root / "apps/web/public")
    with TestClient(create_app(configured, corpus=actual_corpus)) as client:
        assert client.get("/api/motion/library").json()["available"] is True
    private = tmp_path / "private-reference-library.json"
    with TestClient(create_app(Settings(web_dir=tmp_path / "absent", motion_catalog=private), corpus=actual_corpus)) as client:
        result = client.get("/api/motion/library")
        assert result.json()["status"] == "invalid_catalog"
        assert str(private) not in result.text
        assert client.post("/api/chat", json={"message": "Electric cars", "motion_mode": "geometry_preview"}).status_code == 200


def test_motion_environment_is_optional_and_request_gate_is_validated(monkeypatch, corpus, tmp_path):
    monkeypatch.delenv("BIAS_MOTION_CATALOG", raising=False)
    assert Settings.from_env().motion_catalog is None
    monkeypatch.setenv("BIAS_MOTION_CATALOG", str(tmp_path / "catalog.json"))
    monkeypatch.setenv("BIAS_MOTION_ASSET_ROOT", str(tmp_path / "assets"))
    assert Settings.from_env().motion_catalog == tmp_path / "catalog.json"
    assert Settings.from_env().motion_asset_root == tmp_path / "assets"
    with TestClient(create_app(Settings(web_dir=tmp_path / "absent"), corpus=corpus)) as client:
        assert client.get("/api/motion/library").json()["status"] == "not_configured"
        assert client.post("/api/chat", json={"message": "Hello", "motion_mode": "personal_clone"}).status_code == 422
