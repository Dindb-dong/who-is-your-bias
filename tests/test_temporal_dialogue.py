from datetime import date, datetime, timezone
from dataclasses import replace
import asyncio

import pytest

from bias.config import Settings, ProviderSettings
from bias.context_budget import ByteBudgetCounter, build_snapshot_messages
from bias.dialogue import DialogueState
from bias.engine import Engine
from bias.models import ChatRequest
from bias.retrieval import Retriever
from bias.snapshots import PersonaSnapshot, register_snapshot, select_temporal, temporal_intent


CUTOFF = datetime(2026, 11, 11, 10, 1, 40, tzinfo=timezone.utc)


def snapshot(corpus):
    return register_snapshot(corpus, identifier="test-2026.v1", as_of=CUTOFF, ingested_at=CUTOFF,
                             current_evidence_since=date(2026, 1, 1))


@pytest.mark.parametrize("question,mode", [
    ("What do you believe now?", "current"), ("현재 테슬라는?", "current"),
    ("2026년 생각은?", "current"), ("What did you say in 2013?", "historical"),
    ("2017년 터널 계획은?", "historical"), ("Compare 2013 and 2026", "comparison"),
    ("Back then versus now?", "comparison"), ("당시와 현재를 비교해줘", "comparison"),
    ("What happened in 2027?", "future"), ("2050년 실적은?", "future"),
    ("Why reuse rockets?", "general"), ("로켓을 재사용하는 이유는?", "general"),
    ("What did Musk say in the 2026 Dwarkesh interview?", "historical"),
    ("2026년 인터뷰에서 말했던 계획은?", "historical"),
    ("What were the plans for 2030?", "historical"),
    ("What are your current plans for Mars in 2030?", "current"),
    ("현재 2030년 화성 탐사 계획은 뭐야?", "current"),
])
def test_explicit_temporal_intents_include_korean_year_suffixes(question, mode):
    assert temporal_intent(question, CUTOFF).mode == mode


def test_snapshot_does_not_relabel_old_sources_as_current(corpus):
    snap = snapshot(corpus)
    selected = select_temporal(corpus, snap, "What is the current approach to rockets?")
    assert selected.eligible_ids == frozenset()
    assert set(selected.exclusion_reasons.values()) == {"no_dated_current_position"}
    assert snap.summary()["current_dated_training_sources"] == 0
    assert select_temporal(corpus, snap, "What was the plan in 2016?").eligible_ids == {"car-sequence"}


def test_current_requires_dated_firsthand_event_not_new_publication(corpus):
    source = corpus.sources["rocket-interview"]
    corpus.sources[source.id] = source.model_copy(update={"published_at": date(2026, 2, 1)})
    assert not select_temporal(corpus, snapshot(corpus), "your current rocket approach").eligible_ids
    corpus.sources[source.id] = source.model_copy(update={"published_at": date(2026, 2, 1), "recorded_at": date(2026, 1, 30)})
    assert select_temporal(corpus, snapshot(corpus), "your current rocket approach").eligible_ids == {"rocket-reuse"}


@pytest.mark.parametrize("query", ["What are your current plans for Mars in 2030?", "현재 2030년 화성 탐사 계획은 뭐야?"])
def test_future_plan_target_cannot_bypass_current_position_gate(corpus, query):
    selection = select_temporal(corpus, snapshot(corpus), query)
    assert selection.intent.mode == "current"
    assert not selection.eligible_ids
    assert set(selection.exclusion_reasons.values()) == {"no_dated_current_position"}


@pytest.mark.parametrize("query,expected", [
    ("What did he say in the 2020 interview?", {"rocket-reuse"}),
    ("2020년 인터뷰에서 말했던 것은?", {"rocket-reuse"}),
    ("What did he say in the 2017 interview?", set()),
    ("What had he said by 2020?", {"rocket-reuse", "car-sequence"}),
    ("2020년까지 발언을 정리해줘", {"rocket-reuse", "car-sequence"}),
    ("What did he say before 2020?", {"car-sequence"}),
    ("2020년 이전 발언은?", {"car-sequence"}),
    ("What were the plans for 2030?", {"rocket-reuse", "car-sequence"}),
    ("What did he say in 2020 about plans for 2030?", {"rocket-reuse"}),
])
def test_exact_historical_year_is_distinct_from_by_and_forecast_target(corpus, query, expected):
    assert select_temporal(corpus, snapshot(corpus), query).eligible_ids == expected


def test_same_day_unknown_publication_clock_and_future_material_are_excluded(corpus):
    source = corpus.sources["rocket-interview"]
    for published in [CUTOFF.date(), date(2026, 12, 1)]:
        corpus.sources[source.id] = source.model_copy(update={"published_at": published})
        assert "rocket-reuse" not in select_temporal(corpus, snapshot(corpus), "rocket reuse").eligible_ids


def test_snapshot_cannot_invent_event_date_for_unknown_recording(corpus):
    raw = snapshot(corpus).model_dump(mode="json")
    raw["sources"][0]["event_at"] = "2026-02-01"
    with pytest.raises(ValueError, match="event date differs"):
        PersonaSnapshot.model_validate(raw).verify(corpus)


def test_actual_acquisition_times_cannot_be_backdated(corpus):
    source = corpus.sources["rocket-interview"]
    corpus.sources[source.id] = source.model_copy(update={"ingested_at": "2026-11-11T11:00:00Z"})
    with pytest.raises(ValueError, match="acquisition after"):
        snapshot(corpus)


def test_snapshot_rejects_corpus_changes_unknown_links_and_naive_cutoffs(corpus):
    snap = snapshot(corpus)
    corpus.evidence["rocket-reuse"] = corpus.evidence["rocket-reuse"].model_copy(update={"text": "A changed claim."})
    with pytest.raises(ValueError, match="exact corpus"):
        snap.verify(corpus)
    raw = snap.model_dump(mode="json")
    raw["as_of"] = "2026-11-11T10:01:40"
    with pytest.raises(ValueError, match="offset"):
        PersonaSnapshot.model_validate(raw)
    raw = snap.model_dump(mode="json")
    raw["sources"][0]["ingested_at"] = "2027-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="acquisition"):
        PersonaSnapshot.model_validate(raw)


def add_dialogue(state, query, *, assistant="Unsupported secret claim", turn="t1"):
    resolved = state.resolve(query)
    state.append(turn_id=turn, resolved=resolved, assistant=assistant, cited_ids=[], grounding="unverified",
                 intent=None, snapshot_id="snapshot")


@pytest.mark.parametrize("query", ["How does that reduce costs?", "What about that approach?", "그 방법의 단점은?", "아까 말한 방식은 왜 그래?"])
def test_followups_reference_user_topic_without_promoting_generated_claims(query):
    state = DialogueState()
    add_dialogue(state, "Why reuse rockets?")
    result = state.resolve(query)
    assert result.supporting_turn_ids == ("t1",)
    assert "Why reuse rockets?" in result.query
    assert "Unsupported secret" not in result.query
    assert not state.inspect()["turns"][0]["claims_semantically_verified"]


@pytest.mark.parametrize("query", ["I meant electric cars, not rockets.", "정정할게, 테슬라 전기차 얘기야.", "What about Neuralink?"])
def test_corrections_and_topic_switches_do_not_keep_the_old_topic(query):
    state = DialogueState(); add_dialogue(state, "Why reuse rockets?")
    result = state.resolve(query)
    assert result.query == query
    assert result.supporting_turn_ids == ()


def test_chained_followups_keep_topic_anchor_but_bound_prior_user_text():
    state = DialogueState()
    add_dialogue(state, "What is the current AI approach?", turn="initial")
    for index in range(20):
        add_dialogue(state, "Why that approach?" if index % 2 else "How so?", turn=f"followup-{index}")
    result = state.resolve("How so?")
    assert result.query.startswith("What is the current AI approach?")
    assert len(result.query) < 1600
    assert len(result.supporting_turn_ids) == 12
    assert result.entities == ("AI",)


def test_prompt_budget_drops_whole_evidence_and_excludes_its_citation(corpus):
    corpus.evidence["rocket-reuse"] = corpus.evidence["rocket-reuse"].model_copy(update={"text": "Reusable rockets lower launch costs. " * 250})
    result = Retriever(corpus).search("Why reuse rockets?")
    packed = build_snapshot_messages(corpus, ChatRequest(message="Why reuse rockets?"), result, [], snapshot(corpus), None,
                                    context_window=4096, output_tokens=512, counter=ByteBudgetCounter())
    assert "rocket-reuse" in packed.budget["dropped_evidence_ids"]
    assert packed.supplied_ids == ()
    assert packed.budget["input_units"] + 512 <= 4096
    assert not packed.budget["exact_model_tokens"]
    assert not any("lower launch costs." in m["content"] for m in packed.messages)


def test_engine_temporal_guard_and_bounded_reference_state(corpus, tmp_path):
    path = tmp_path / "snapshot.json"; path.write_text(snapshot(corpus).model_dump_json())
    engine = Engine(Settings(persona_snapshot=path, max_turns=2), corpus)
    async def run():
        first = await engine.chat(ChatRequest(message="Why reuse rockets?"))
        followup = await engine.chat(ChatRequest(message="How does that reduce launch costs?", session_id=first.session_id))
        assert followup.context["query_resolution"]["supporting_turn_ids"] == [first.id]
        current = await engine.chat(ChatRequest(message="What is the current rocket plan?", session_id=first.session_id))
        assert current.grounding.status == "insufficient" and not current.citations
        assert "current position" in current.text
        followup_current = await engine.chat(ChatRequest(message="How about that approach?", session_id=first.session_id))
        assert followup_current.context["temporal"]["mode"] == "current"
        assert not followup_current.citations
        third = await engine.chat(ChatRequest(message="How so?", session_id=first.session_id))
        assert third.context["temporal"]["mode"] == "current"
        assert "current rocket plan" in third.context["query_resolution"]["query"]
        assert not third.citations
        historical = await engine.chat(ChatRequest(message="What did he say in 2020?", session_id=first.session_id))
        for message in ["Why that approach?", "How so?"]:
            later = await engine.chat(ChatRequest(message=message, session_id=first.session_id))
            assert later.context["temporal"]["mode"] == "historical"
            assert later.context["temporal"]["years"] == [2020]
            assert "car-sequence" not in later.grounding.retrieved_ids
        state = engine.sessions.sessions[first.session_id].dialogue
        assert len(state.records) == 2
        other = await engine.chat(ChatRequest(message="What about that?"))
        assert other.context["query_resolution"]["supporting_turn_ids"] == []
    asyncio.run(run())
