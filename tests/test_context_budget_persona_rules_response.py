"""V2-only presentation follow-up; prompt transport checks, not output quality."""
import json
from pathlib import Path

import pytest

import bias.context_budget as budget_module
from bias.answer_policy import select_answer_policy
from bias.context_budget import ByteBudgetCounter, build_snapshot_messages
from bias.corpus import Corpus
from bias.models import ChatRequest
from bias.persona_graph import PersonaGraph, augment_result
from bias.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "How should we test a prototype before spending our remaining budget?"


@pytest.fixture
def inputs():
    corpus = Corpus.load(ROOT / "data/personas/jensen-huang-values-v2")
    graph = PersonaGraph.load(ROOT / "data/personas/jensen-huang-values-v2/persona_graph.v2.json", corpus)
    selected = graph.select(QUESTION, limit=1)
    result = augment_result(selected, corpus, Retriever(corpus).search(QUESTION))
    return corpus, selected, result


def pack(inputs, *, question=QUESTION, language="en", format="graph", **kwargs):
    corpus, selected, result = inputs
    return build_snapshot_messages(corpus, ChatRequest(message=question, language=language),
        result, [], None, None, context_window=kwargs.pop("context_window", 16000),
        output_tokens=256, counter=ByteBudgetCounter(), persona_selection=selected,
        persona_memory_format=format, **kwargs)


def context(package):
    return json.loads(package.messages[1]["content"].split("\n", 1)[1])


def test_v2_contract_is_shared_by_flat_and_graph_with_complete_memory(inputs):
    flat, graph = pack(inputs, format="flat"), pack(inputs)
    assert flat.messages[0] == graph.messages[0]
    instruction = graph.messages[0]["content"]
    assert "Use the conditional rules silently" in instruction
    assert "about 60 words or fewer" in instruction
    assert "Unless the user explicitly requests a different length or format" in instruction
    assert "Preserve hypothetical labels" in instruction
    assert "never attach a citation to the extrapolation itself" in instruction
    assert context(flat)["factual_evidence"] == context(graph)["factual_evidence"]
    assert context(flat)["persona_memory"]["nodes"] == context(graph)["persona_memory"]["nodes"]
    path = inputs[1].paths[0]
    rule = next(node for node in context(graph)["persona_memory"]["nodes"] if node["id"] == path.value_id)
    assert rule["conditions"] == list(path.rule.conditions)
    assert rule["exceptions"] == list(path.rule.exceptions)
    assert set(path.evidence_ids) <= set(graph.supplied_ids)


@pytest.mark.parametrize("detail", ["in one sentence", "briefly", "in detail"])
@pytest.mark.parametrize("language", ["en", "ko"])
def test_adaptive_policy_has_precedence_without_new_length_cap(inputs, detail, language):
    question = QUESTION + " " + detail
    policy = select_answer_policy(question, language=language)
    package = pack(inputs, question=question, language=language, answer_policy=policy)
    instruction = package.messages[0]["content"]
    assert policy.instruction(language) in instruction
    assert "about 60 words or fewer" not in instruction
    assert "사용자가 다른 길이·형식을 명시하지 않았다면" not in instruction
    if language == "en":
        assert "Existing turn-specific length/format instructions and stricter limits take priority" in instruction
    else:
        assert "이미 주어진 이번 답변의 길이·형식 지침과 더 엄격한 제한을 우선" in instruction


def test_live_policy_remains_the_only_length_target(inputs):
    package = pack(inputs, live_concise=True)
    instruction = package.messages[0]["content"]
    assert budget_module.live_concise_instruction("en") in instruction
    assert "about 60 words or fewer" not in instruction
    assert "Use the conditional rules silently" in instruction


def test_korean_existing_one_sentence_limit_is_not_relaxed(inputs):
    package = pack(inputs, language="ko")
    instruction = package.messages[0]["content"]
    assert "한국어 한 문장으로 짧게 답하고" in instruction
    assert "짧은 두 문장 이내로" not in instruction
    assert "조건부 원칙은 답변 작성에만 활용" in instruction
    assert "가상 추론 표시와 필요한 불확실성을 유지" in instruction


def test_v1_and_disabled_memory_prompts_are_unchanged(monkeypatch):
    corpus = Corpus.load(ROOT / "data/personas/jensen-huang")
    graph = PersonaGraph.load(ROOT / "data/personas/jensen-huang/persona_graph.v1.json", corpus)
    selection = graph.select(QUESTION)
    result = augment_result(selection, corpus, Retriever(corpus).search(QUESTION))
    def render(memory):
        return build_snapshot_messages(corpus, ChatRequest(message=QUESTION, language="en"),
            result, [], None, None, context_window=16000, output_tokens=256,
            counter=ByteBudgetCounter(), persona_selection=memory)
    before_v1, before_disabled = render(selection), render(None)
    # A changed v2 contract, including a large new string, must not affect any
    # message, evidence choice or budget unit for existing paths.
    monkeypatch.setattr(budget_module, "persona_rules_response_instruction", lambda *args, **kwargs: "NEVER INCLUDE " * 1000)
    assert render(selection) == before_v1
    assert render(None) == before_disabled
    assert "Use the conditional rules silently" not in before_v1.messages[0]["content"]
    assert "persona_memory" not in context(before_disabled)


def test_new_response_contract_is_counted_before_memory_acceptance(inputs, monkeypatch):
    current = pack(inputs)
    assert current.budget["input_units"] == ByteBudgetCounter().count(current.messages)
    with monkeypatch.context() as patch:
        patch.setattr(budget_module, "persona_rules_response_instruction", lambda *args, **kwargs: "")
        previous = pack(inputs)
    assert current.budget["input_units"] > previous.budget["input_units"]
    # The old complete prompt fits this window, but the contract makes memory
    # exceed it. Packing must omit whole memory, not silently exceed the window
    # or discard its exceptions to force a fit.
    window = previous.budget["input_units"] + 256
    narrow = pack(inputs, context_window=window)
    assert "persona_memory" not in context(narrow)
    assert narrow.budget["persona_memory"]["reason"] == "context_budget"
    assert narrow.supplied_ids == previous.supplied_ids
    assert ByteBudgetCounter().count(narrow.messages) <= window - 256
