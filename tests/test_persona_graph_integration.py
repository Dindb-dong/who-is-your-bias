"""Persona memory is optional, evidence-bound, and shared by chat/stream/live."""
import asyncio
import json
from pathlib import Path

import pytest

from bias.config import ProviderSettings, Settings
from bias.context_budget import ByteBudgetCounter, build_snapshot_messages
from bias.corpus import Corpus
from bias.engine import Engine
from bias.models import ChatRequest
from bias.persona_graph import PersonaGraph, augment_result
from bias.prompts import build_messages
from bias.retrieval import Retriever


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "data/personas/jensen-huang/persona_graph.v1.json"
QUESTION = "How should I decide whether to commit our budget to an untested prototype?"


@pytest.fixture
def jensen():
    return Corpus.load(ROOT / "data/personas/jensen-huang")


def unpack(messages):
    return json.loads(messages[1]["content"].split("\n", 1)[1])


class RecordingLLM:
    def __init__(self):
        self.calls = []

    async def stream(self, messages):
        self.calls.append(messages)
        yield "Hypothetical: You could test a small prototype before committing the entire budget."


def settings(**kwargs):
    return Settings(llm=ProviderSettings("http://localhost:9999/v1", "test"),
                    llm_max_tokens=256, **kwargs)


def test_flag_is_off_by_default_and_explicitly_validated(monkeypatch):
    monkeypatch.delenv("BIAS_PERSONA_GRAPH", raising=False)
    monkeypatch.delenv("BIAS_PERSONA_MEMORY_FORMAT", raising=False)
    assert Settings.from_env().persona_graph is None
    monkeypatch.setenv("BIAS_PERSONA_GRAPH", str(GRAPH))
    monkeypatch.setenv("BIAS_PERSONA_MEMORY_FORMAT", "flat")
    assert Settings.from_env().persona_graph == GRAPH
    assert Settings.from_env().persona_memory_format == "flat"
    monkeypatch.setenv("BIAS_PERSONA_MEMORY_FORMAT", "unknown")
    with pytest.raises(ValueError, match="persona_memory_format"):
        Settings.from_env()


def test_disabled_prompt_is_identical_to_existing_builder(jensen):
    result = Retriever(jensen).search(QUESTION)
    request = ChatRequest(message=QUESTION, language="en")
    kwargs = dict(context_window=8192, output_tokens=256, counter=ByteBudgetCounter())
    before = build_snapshot_messages(jensen, request, result, [], None, None, **kwargs)
    after = build_snapshot_messages(jensen, request, result, [], None, None,
                                    persona_selection=None, **kwargs)
    assert before == after
    assert "persona_memory" not in unpack(after.messages)
    assert "persona_memory" not in after.budget


@pytest.mark.parametrize("format", ["flat", "graph"])
def test_enabled_engine_sends_supported_memory_in_one_generation(jensen, format):
    llm = RecordingLLM()
    engine = Engine(settings(persona_graph=GRAPH, persona_memory_format=format), jensen, llm=llm)
    response = asyncio.run(engine.chat(ChatRequest(message=QUESTION, language="en")))
    assert len(llm.calls) == 1
    context = unpack(llm.calls[0])
    supplied = {item["id"] for item in context["factual_evidence"]}
    memory = context["persona_memory"]
    assert memory["nodes"]
    assert all(set(node["evidence_ids"]) <= supplied for node in memory["nodes"])
    assert response.context["persona_memory"]["value_ids"] == ["validate_before_commitment"]
    assert response.context["prompt_budget"]["persona_memory"]["included"] is True
    assert response.grounding.status == "unverified"  # Graph presence is not semantic verification.
    assert ("relations" in memory) is (format == "graph")


def test_profile_ablation_and_custom_builders_do_not_receive_graph(jensen):
    async def run():
        profile_llm = RecordingLLM()
        profile = Engine(settings(persona_graph=GRAPH), jensen, llm=profile_llm)
        response = await profile.chat(ChatRequest(message=QUESTION, mode="profile", language="en"))
        assert "persona_memory" not in response.context
        assert all("persona_memory" not in message["content"] for message in profile_llm.calls[0])
        custom_llm = RecordingLLM()
        def custom(*args, **kwargs):
            return build_messages(*args, **kwargs)
        frozen = Engine(settings(persona_graph=GRAPH), jensen, llm=custom_llm, prompt_builder=custom)
        response = await frozen.chat(ChatRequest(message="What did you learn about CUDA?", language="en"))
        assert "persona_memory" not in response.context
        assert all("persona_memory" not in message["content"] for message in custom_llm.calls[0])
    asyncio.run(run())


def test_private_claim_guard_is_not_overridden_by_graph(jensen):
    llm = RecordingLLM()
    engine = Engine(settings(persona_graph=GRAPH), jensen, llm=llm)
    response = asyncio.run(engine.chat(ChatRequest(
        message="Did you promise me an internship in our private meeting? How should I proceed?",
        language="en")))
    assert not llm.calls
    assert "persona_memory" not in response.context
    assert response.grounding.status == "insufficient"


def test_context_overflow_drops_interpretation_but_preserves_evidence(jensen):
    graph = PersonaGraph.load(GRAPH, jensen)
    selection = graph.select(QUESTION)
    result = augment_result(selection, jensen, Retriever(jensen).search(QUESTION))
    request = ChatRequest(message=QUESTION, language="en")
    plain = build_snapshot_messages(jensen, request, result, [], None, None,
                                   context_window=8192, output_tokens=256, counter=ByteBudgetCounter())
    packed = build_snapshot_messages(jensen, request, result, [], None, None,
        context_window=plain.budget["input_units"] + 256, output_tokens=256,
        counter=ByteBudgetCounter(), persona_selection=selection)
    assert packed.supplied_ids == plain.supplied_ids
    assert packed.messages == plain.messages
    assert packed.budget["persona_memory"] == {
        "format": "graph", "included": False, "reason": "context_budget"}


def test_flat_and_graph_have_identical_factual_evidence_and_instructions(jensen):
    selection = PersonaGraph.load(GRAPH, jensen).select(QUESTION)
    result = augment_result(selection, jensen, Retriever(jensen).search(QUESTION))
    packages = [build_snapshot_messages(jensen, ChatRequest(message=QUESTION, language="en"),
        result, [], None, None, context_window=16000, output_tokens=256,
        counter=ByteBudgetCounter(), persona_selection=selection, persona_memory_format=format)
        for format in ("flat", "graph")]
    assert packages[0].messages[0] == packages[1].messages[0]
    assert packages[0].supplied_ids == packages[1].supplied_ids
    contexts = [unpack(package.messages) for package in packages]
    memories = [context.pop("persona_memory") for context in contexts]
    assert contexts[0] == contexts[1]
    assert memories[0]["nodes"] == memories[1]["nodes"]
