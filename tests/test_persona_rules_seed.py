"""Production seed routing/budget regression probes; these are development cases.

They validate retrieval and transport only, not generated-answer accuracy or
faithfulness to a real person's responses. The evaluation registry is separate.
"""
import json
from pathlib import Path

import pytest

from bias.context_budget import ByteBudgetCounter, build_snapshot_messages
from bias.corpus import Corpus
from bias.models import ChatRequest
from bias.persona_graph import PersonaGraph, augment_result, select_advice_memory
from bias.retrieval import Retriever
from bias.snapshots import PersonaSnapshot

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/personas/jensen-huang-values-v2"


@pytest.fixture
def seed():
    corpus = Corpus.load(DATA)
    graph = PersonaGraph.load(DATA / "persona_graph.v2.json", corpus)
    snapshot = PersonaSnapshot.load(ROOT / "data/snapshots/jensen-values-2026-11-21.json", corpus)
    return corpus, graph, snapshot


PROBES = [
    ("reassess-before-persisting", "Our premise failed in three replications. Should we continue funding the same idea?"),
    ("test-before-costly-commitment", "We can afford only one trial. How should we test our prototype before spending the remaining budget?"),
    ("choose-need-before-building", "How should we choose whether to build a product that customers already get elsewhere?"),
    ("match-guidance-to-experience", "A brilliant intern is inexperienced. How should I delegate a feature while helping her grow?"),
    ("match-guidance-to-experience", "신입 팀원에게 기능 개발을 위임하려는데 어떻게 지도하면 좋을까요?"),
    ("organize-around-contribution", "I postpone documentation because coding feels more prestigious. How should I view these chores?"),
    ("communicate-under-uncertainty", "We found a mistake affecting our team. How should I communicate the bad news without pretending to know the solution?"),
    ("adapt-process-to-project-phase", "A thousand people depend on our production service. Should I launch another experiment without warning them?"),
    ("allocate-by-opportunity-cost", "Two opportunities need the same people and time. How should we prioritize them?"),
    ("deploy-with-accountability", "We want to deploy a model in a hospital. How should we judge the risk to patients?"),
    ("separate-task-from-career", "Translation tasks are being automated. How should I think about my career?"),
    ("disagree-without-losing-the-problem", "My partner and I disagree strongly about a deadline. How should I keep us working together?"),
    ("judge-leadership-by-fit", "The role has outgrown my skills. How should I decide whether to step down as team leader?"),
]


@pytest.mark.parametrize("expected,question", PROBES)
def test_seed_rule_reaches_prompt_with_conditions_and_source_support(seed, expected, question):
    corpus, graph, snapshot = seed
    selection = select_advice_memory(graph, corpus, question, snapshot=snapshot)
    assert expected in [path.value_id for path in selection.paths]
    result = augment_result(selection, corpus, Retriever(corpus).search(question), limit=4)
    packed = build_snapshot_messages(corpus, ChatRequest(message=question, language="en"), result,
        [], snapshot, None, context_window=16000, output_tokens=256,
        counter=ByteBudgetCounter(), persona_selection=selection)
    context = json.loads(packed.messages[1]["content"].split("\n", 1)[1])
    memory = context["persona_memory"]
    # Some secondary rules can be omitted by the four-evidence budget. Every
    # included rule must be whole, including contrary/qualifying evidence.
    assert expected in memory["rule_ids"]
    for path in selection.paths:
        if path.value_id in memory["rule_ids"]:
            assert set(path.evidence_ids) <= set(packed.supplied_ids)
            node = next(node for node in memory["nodes"] if node["id"] == path.value_id)
            assert node["conditions"] == list(path.rule.conditions)
            assert node["exceptions"] == list(path.rule.exceptions)


def test_multiple_rules_do_not_strip_qualifications_to_fit_evidence_cap(seed):
    corpus, graph, _ = seed
    question = "How should I mentor a beginner while adapting the prototype plan for an upcoming launch?"
    selection = graph.select(question)
    assert len(selection.paths) == 2
    assert len(selection.evidence_ids) > 4
    result = augment_result(selection, corpus, Retriever(corpus).search(question), limit=4)
    packed = build_snapshot_messages(corpus, ChatRequest(message=question, language="en"), result,
        [], None, None, context_window=8192, output_tokens=256,
        counter=ByteBudgetCounter(), persona_selection=selection)
    context = json.loads(packed.messages[1]["content"].split("\n", 1)[1])
    memory = context["persona_memory"]
    assert len(memory["rule_ids"]) == 1
    first = selection.paths[0]
    assert memory["rule_ids"] == [first.value_id]
    assert set(first.evidence_ids) <= set(packed.supplied_ids)
    assert next(n for n in memory["nodes"] if n["id"] == first.value_id)["exceptions"] == list(first.rule.exceptions)


@pytest.mark.parametrize("question", [
    "What is NVIDIA stock worth today?",
    "Did you personally endorse our project?",
    "What models does this avatar project use?",
    "최근 불륜 소문은 사실인가요?",
])
def test_expanded_seed_does_not_create_missing_factual_knowledge(seed, question):
    corpus, graph, snapshot = seed
    assert not select_advice_memory(graph, corpus, question, snapshot=snapshot).paths
