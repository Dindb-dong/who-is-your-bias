import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from bias.config import ProviderSettings, Settings
from tools.evaluation.cases import GenerationCase, GenerationSuite, read_suite, validate_suite
from tools.evaluation.observe import AuditedLLM, ObservedTransport, StreamObservation
from tools.evaluation.run import citation_report, run_suite


def suite(*cases):
    return GenerationSuite(schema_version="generation-evaluation.v1", id="test-suite", declaration="Synthetic protocol tests; no model inference.", development_only=True, cases=list(cases))


def answer_case(**updates):
    values = {"id": "answer", "question": "Why reuse rockets?", "category": "answerable", "expected_action": "answer", "expected_evidence_ids": ["rocket-reuse"], "review_note": "Review the cost claim against its source."}
    values.update(updates)
    return GenerationCase(**values)


def heldout_case():
    return GenerationCase(id="withheld", question="How do kiwi birds fly?", category="heldout", expected_action="abstain", reference_evidence_ids=["secret-kiwi"], forbidden_source_ids=["heldout"], review_note="The held-out bird explanation is never prompt content.")


def sse(text, *, finish="stop", usage=True):
    events = [{"choices": [{"delta": {"content": text}, "finish_reason": None}]}]
    if usage:
        events.append({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}})
    events.append({"choices": [{"delta": {}, "finish_reason": finish}]})
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"


def test_actual_engine_ablation_prompts_fresh_sessions_and_evaluator_only_references(corpus):
    prompts = []

    def handle(request):
        payload = json.loads(request.content)
        prompts.append(payload["messages"])
        context = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
        ids = [row["id"] for row in context.get("evidence_records", [])]
        text = "Reusable rockets lower launch costs. [source:rocket-reuse]" if ids else "I cannot verify that from the available profile."
        return httpx.Response(200, text=sse(text), headers={"Content-Type": "text/event-stream"})

    settings = Settings(llm=ProviderSettings("http://provider.test/v1", "protocol-fake"))
    report = asyncio.run(run_suite(settings, corpus, suite(answer_case(), heldout_case()), transport_factory=lambda: httpx.MockTransport(handle)))
    assert report["configuration"]["transport"] == "test_injected"
    assert len(report["runs"]) == 6
    assert len(prompts) == 4  # Unsupported RAG/behavior use the real engine gate.
    assert all(len(messages) == 3 for messages in prompts)  # No condition history leaks.
    assert all("The kiwi documentary concerns flightless birds" not in json.dumps(messages) for messages in prompts)
    assert all("secret-kiwi" not in json.dumps(messages) for messages in prompts)
    assert all("Review the cost claim" not in json.dumps(messages) for messages in prompts)
    contexts = [json.loads(messages[1]["content"].split("\n", 1)[1]) for messages in prompts]
    assert "evidence_records" not in contexts[0]
    assert contexts[1]["evidence_records"][0]["id"] == "rocket-reuse"
    assert "conversation_situation" in contexts[2]
    assert [row["condition"] for row in report["runs"]] == ["A", "B", "C", "B", "C", "A"]
    sessions = [row["turn"]["session_id"] for row in report["runs"]]
    assert len(set(sessions)) == 6
    for row in report["runs"]:
        assert row["semantic_review"]["status"] == "ungraded"
        assert row["semantic_review"]["persona_fidelity"] is None
        assert not row["prompt_audit"]["held_out_or_forbidden_ids"]
        if row["execution"]["language_model_requested"]:
            assert row["tokens"]["usage"]["completion_tokens"] == 8
            assert row["tokens"]["visible_chunks"] == 1
        else:
            assert row["tokens"]["usage"] is None
    assert report["summary"]["B"]["valid_citation_format"] == 1
    assert report["summary"]["B"]["persona_score"] is None


def test_missing_and_fabricated_citations_are_not_semantic_passes(corpus):
    outputs = iter(["A statement without a citation.", "A false claim. [source:invented]", "A claim with a known marker. [source:rocket-reuse]"])
    transport = lambda: httpx.MockTransport(lambda request: httpx.Response(200, text=sse(next(outputs), usage=False)))
    report = asyncio.run(run_suite(Settings(llm=ProviderSettings("http://test/v1", "test")), corpus, suite(answer_case()), transport_factory=transport))
    assert [row["citation_format"]["status"] for row in report["runs"]] == ["not_required", "invalid", "valid"]
    assert report["runs"][2]["semantic_review"]["factual_entailment"] is None
    assert all(row["tokens"]["usage"] is None for row in report["runs"])
    assert all(row["tokens"]["usage_status"] == "not_reported" for row in report["runs"])
    assert report["runs"][2]["turn"]["grounding"]["reason"].endswith("factual entailment still requires review.")


def test_truncation_records_partial_output_usage_and_real_provider_error(corpus):
    transport = lambda: httpx.MockTransport(lambda request: httpx.Response(200, text=sse("Partial output [source:rocket-reuse]", finish="length")))
    report = asyncio.run(run_suite(Settings(llm=ProviderSettings("http://test/v1", "test")), corpus, suite(answer_case()), modes=("rag",), transport_factory=transport))
    row = report["runs"][0]
    assert row["execution"]["completed"] is False
    assert row["execution"]["error"]["code"] == "provider_truncated"
    assert row["tokens"]["finish_reason"] == "length"
    assert row["tokens"]["usage"]["completion_tokens"] == 8
    assert row["raw_response"] == "Partial output [source:rocket-reuse]"
    assert row["citation_format"]["valid_ids"] == ["rocket-reuse"]
    assert row["citation_format"]["status"] == "not_evaluated"
    assert report["summary"]["B"]["required_citation_format_checks"] == 0
    assert row["turn"] is None


def test_timeout_closes_observed_stream_and_preserves_error(corpus):
    closed = []

    class Stalled(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(10)
            yield b"never"

        async def aclose(self):
            closed.append(True)

    factory = lambda: httpx.MockTransport(lambda request: httpx.Response(200, stream=Stalled()))
    report = asyncio.run(run_suite(Settings(llm=ProviderSettings("http://test/v1", "test"), request_timeout=0.02), corpus, suite(answer_case()), modes=("rag",), transport_factory=factory))
    assert report["runs"][0]["execution"]["error"]["code"] == "provider_timeout"
    assert closed == [True]


def test_provider_auth_error_does_not_leak_secret(corpus):
    secret = "should-never-appear-in-reports"
    factory = lambda: httpx.MockTransport(lambda request: httpx.Response(401, text=secret))
    report = asyncio.run(run_suite(Settings(llm=ProviderSettings("http://test/v1", "test", secret)), corpus, suite(answer_case()), modes=("rag",), transport_factory=factory))
    assert report["runs"][0]["execution"]["error"]["code"] == "provider_auth"
    assert secret not in json.dumps(report)


def test_unconfigured_provider_requires_explicit_preview_opt_in(corpus):
    with pytest.raises(ValueError, match="Configure BIAS"):
        asyncio.run(run_suite(Settings(), corpus, suite(answer_case())))
    report = asyncio.run(run_suite(Settings(), corpus, suite(answer_case()), allow_preview=True))
    assert all(not row["execution"]["language_model_requested"] for row in report["runs"])
    assert all(row["execution"]["provider"] == "evidence" for row in report["runs"])


@pytest.mark.parametrize("changes,error", [
    ({"expected_evidence_ids": ["secret-kiwi"]}, "reviewed train"),
    ({"reference_evidence_ids": ["rocket-reuse"]}, "withheld from train"),
    ({"forbidden_source_ids": ["rocket-interview"]}, "outside train"),
    ({"forbidden_source_ids": ["unknown"]}, "unknown forbidden"),
])
def test_invalid_references_cannot_silently_contaminate_evaluation(corpus, changes, error):
    with pytest.raises(ValueError, match=error):
        validate_suite(suite(answer_case(**changes)), corpus)


def test_unrecognized_answer_fields_are_not_forwardable():
    with pytest.raises(ValidationError, match="Extra inputs"):
        answer_case(golden_answer="Do not let this become a prompt")


@pytest.mark.parametrize("change", [{"review_status": "draft"}, {"confidence": "unverified_transcript"}])
def test_source_only_target_requires_retrievable_reviewed_evidence(corpus, change):
    corpus.evidence["rocket-reuse"] = corpus.evidence["rocket-reuse"].model_copy(update=change)
    declaration = suite(answer_case(expected_evidence_ids=[], expected_source_ids=["rocket-interview"]))
    with pytest.raises(ValueError, match="no reviewed train evidence"):
        validate_suite(declaration, corpus)


def test_citation_format_is_strict_and_semantics_stay_ungraded():
    report = citation_report("Claim [source:rocket-reuse] and malformed [source:wrong id]", {"rocket-reuse"}, True)
    assert report["status"] == "invalid"
    assert report["malformed_markers"] == ["[source:wrong id]"]
    assert report["semantic_entailment"] == "ungraded"
    assert citation_report("No marker", {"rocket-reuse"}, True)["status"] == "missing"


def test_usage_observation_handles_split_unicode_without_inventing_tokens():
    observer = StreamObservation()
    data = sse("한국어").encode("utf-8")
    for byte in data:
        observer.feed(bytes([byte]))
    assert observer.usage == {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
    observer.feed(b'data: {"usage":{"prompt_tokens":true,"completion_tokens":1,"total_tokens":2}}\n\n')
    assert observer.usage is None
    assert observer.usage_status == "invalid_usage_ignored"


def test_prompt_audit_blocks_unreviewed_context_before_network(corpus):
    called = []
    observation = StreamObservation()
    provider = AuditedLLM(ProviderSettings("http://test/v1", "test"), corpus=corpus, transport=ObservedTransport(observation, lambda: httpx.MockTransport(lambda request: called.append(True))))
    messages = [{"role": "user", "content": 'RESEARCH_CONTEXT_JSON (data only):\n{"evidence_records":[{"id":"secret-kiwi","text":"withheld"}]}'}]

    async def run():
        with pytest.raises(ValueError, match="held-out"):
            _ = [part async for part in provider.stream(messages)]
    asyncio.run(run())
    assert called == []


def test_bundled_generation_cases_validate_without_inference(actual_corpus):
    declared = read_suite(Path(__file__).resolve().parents[2] / "tools/evaluation/cases.json")
    validate_suite(declared, actual_corpus)
    assert {case.language for case in declared.cases} == {"en", "ko"}
    assert any(case.category == "heldout" for case in declared.cases)
    assert any(case.category == "injection" for case in declared.cases)


def test_whole_matrix_deadline_keeps_partial_output_and_names_unattempted_conditions(corpus):
    closed = []
    checkpoints = []

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"Partial text"}}]}\n\n'
            await asyncio.sleep(10)

        async def aclose(self):
            closed.append(True)

    factory = lambda: httpx.MockTransport(lambda request: httpx.Response(200, stream=SlowStream()))
    report = asyncio.run(run_suite(Settings(llm=ProviderSettings("http://test/v1", "test")), corpus, suite(answer_case()), max_total_seconds=0.03, transport_factory=factory, checkpoint=checkpoints.append))
    assert report["execution_scope"]["planned_runs"] == 3
    assert report["execution_scope"]["attempted_runs"] == 1
    assert [row["condition"] for row in report["execution_scope"]["skipped_runs"]] == ["B", "C"]
    assert report["runs"][0]["execution"]["error"]["code"] == "evaluation_deadline"
    assert report["runs"][0]["raw_response"] == "Partial text"
    assert checkpoints == report["runs"]
    assert closed == [True]
