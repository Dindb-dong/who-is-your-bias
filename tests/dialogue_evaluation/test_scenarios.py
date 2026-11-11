"""The registered acceptance cases exercise the HTTP API and exposed state."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import date, datetime, timezone

import httpx
import pytest

from bias.api import create_app
from bias.config import ProviderSettings, Settings
from bias.dialogue import DialogueState
from bias.snapshots import register_snapshot, temporal_intent
from tools.dialogue_evaluation.run import REGISTRY, load_registry, preview_settings, run


def test_registered_multiturn_api_acceptance_suite():
    report = run()
    failures = [(scenario["id"], step["label"], step["failures"])
                for scenario in report["scenarios"] for step in scenario["steps"] if not step["passed"]]
    assert report["passed"], failures
    assert report["scenario_count"] == 29 and report["step_count"] == 100
    assert report["llm_inference_turns"] == 0 and report["tts_inference"] is False
    assert report["acoustic_latency_ms"] is None
    assert all(len(value) == 64 for value in report["implementation_sha256"].values())
    interruption = next(item for item in report["scenarios"] if item["id"] == "stream-interruption")
    assert interruption["steps"][1]["interruption"]["result_delivered"] is False
    assert interruption["steps"][2]["state"]["turn_count"] == 2


@pytest.mark.parametrize("mutation", ["unknown_assertion", "unknown_reference", "duplicate_label", "duplicate_scenario"])
def test_registry_does_not_silently_ignore_invalid_acceptance_contracts(tmp_path, mutation):
    registry = json.loads(REGISTRY.read_text())
    first = registry["scenarios"][0]
    if mutation == "unknown_assertion":
        first["steps"][0]["expect"]["pretend_pass"] = True
    elif mutation == "unknown_reference":
        first["steps"][0]["expect"]["supports"] = ["not-an-earlier-turn"]
    elif mutation == "duplicate_label":
        first["steps"][1]["label"] = first["steps"][0]["label"]
    else:
        registry["scenarios"].append(first)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError):
        load_registry(path)


def test_preview_cannot_silently_run_configured_generation_or_changed_corpus(tmp_path):
    settings = replace(preview_settings(), llm=ProviderSettings("https://model.example/v1", "real-model"))
    with pytest.raises(ValueError, match="configured mode"):
        run(settings)
    registry = json.loads(REGISTRY.read_text())
    registry["preview_corpus_sha256"] = "0" * 64
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="corpus changed"):
        run(registry_path=path)


def test_generated_prompt_keeps_corrected_period_and_user_audience_without_claim_promotion(corpus, tmp_path):
    cutoff = datetime(2026, 11, 11, tzinfo=timezone.utc)
    snapshot = register_snapshot(corpus, identifier="api-dialogue-test", as_of=cutoff, ingested_at=cutoff,
                                 current_evidence_since=date(2026, 1, 1))
    path = tmp_path / "snapshot.json"
    path.write_text(snapshot.model_dump_json())
    calls = []

    class RecordingLLM:
        async def stream(self, messages):
            calls.append(messages)
            context = json.loads(messages[1]["content"].split("\n", 1)[1])
            # Explicitly synthetic response. It also plants an unsupported
            # claim to verify that search/context never adopts assistant text.
            identifier = context["factual_evidence"][0]["id"]
            yield f"Synthetic fixture response. [source:{identifier}] ASSISTANT_SECRET_CLAIM."

    async def scenario():
        settings = Settings(persona_snapshot=path, llm=ProviderSettings("http://fixture.invalid/v1", "fixture"))
        app = create_app(settings, corpus=corpus, llm=RecordingLLM())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                first = (await client.post("/api/chat", json={"message": "Why reuse rockets in 2016? Explain for children.", "language": "en"})).json()
                second = (await client.post("/api/chat", json={"message": "And in 2020?", "language": "en", "session_id": first["session_id"]})).json()
                assert second["context"]["temporal"]["years"] == [2020]
                # Simulate an oversized preceding response. This is a product
                # limit: the original user anchor must outlive dropped pairs.
                app.state.engine.sessions.sessions[first["session_id"]].history[-1]["content"] *= 250
                final = (await client.post("/api/chat", json={"message": "Why?", "language": "en", "session_id": first["session_id"]})).json()
                assert final["provider"] == "openai" and final["context"]["prompt_budget"]["history_turns"] == 0
                prompt_context = json.loads(calls[-1][1]["content"].split("\n", 1)[1])
                assert prompt_context["temporal_years"] == [2020] and prompt_context["temporal_period"] == "exact"
                assert "children" in prompt_context["dialogue_reference"]["user_anchor"]
                assert prompt_context["dialogue_reference"]["role"] == "user_context_not_factual_evidence"
                assert "ASSISTANT_SECRET_CLAIM" not in final["context"]["query_resolution"]["query"]
                assert "ASSISTANT_SECRET_CLAIM" not in json.dumps(prompt_context)
                assert {item["id"] for item in prompt_context["factual_evidence"]} == {"rocket-reuse"}
                assert final["grounding"]["status"] == "unverified"
    asyncio.run(scenario())


@pytest.mark.parametrize(("query", "years"), [
    ("I meant 2013, not 2017.", (2013,)),
    ("I meant 2013 instead of 2017.", (2013,)),
    ("아니, 2017년 말고 2013년 인터뷰 얘기였어.", (2013,)),
    ("2017년이 아니라 2013년을 말한 거야.", (2013,)),
    ("Compare the 2013 and 2017 interviews.", (2013, 2017)),
    ("Not before 2017: what did you say in 2020?", (2017, 2020)),
])
def test_rejected_dates_are_not_requested_periods(query, years):
    assert temporal_intent(query, datetime(2026, 11, 11, tzinfo=timezone.utc)).years == years


def test_correction_that_introduces_unknown_subject_does_not_inherit_old_topic():
    state = DialogueState()
    original = state.resolve("Why are tunnels expensive?")
    state.append(turn_id="initial", resolved=original, assistant="Unverified", cited_ids=[],
                 grounding="unverified", intent=None, snapshot_id=None)
    corrected = state.resolve("I meant pension taxation in 2013.")
    assert corrected.query == corrected.original and not corrected.supporting_turn_ids


@pytest.mark.parametrize(("query", "expected", "period"), [
    ("What was the electric car plan by 2020?", "car-sequence", "through"),
    ("What was the electric car plan before 2020?", "car-sequence", "before"),
    ("What were plans for reusable rockets in 2030?", "rocket-reuse", "forecast"),
])
def test_snapshot_period_is_not_overridden_by_legacy_exact_year_filter(corpus, tmp_path, query, expected, period):
    cutoff = datetime(2026, 11, 11, tzinfo=timezone.utc)
    snapshot = register_snapshot(corpus, identifier="period-test", as_of=cutoff, ingested_at=cutoff,
                                 current_evidence_since=date(2026, 1, 1))
    path = tmp_path / "snapshot.json"
    path.write_text(snapshot.model_dump_json())

    async def scenario():
        app = create_app(Settings(persona_snapshot=path), corpus=corpus)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                response = await client.post("/api/chat", json={"message": query, "language": "en"})
                assert response.status_code == 200
                turn = response.json()
                assert expected in turn["grounding"]["retrieved_ids"]
                assert turn["context"]["temporal"]["period"] == period
                assert turn["context"]["lexical_search"]["temporal_scope_applied_before_search"]
    asyncio.run(scenario())
