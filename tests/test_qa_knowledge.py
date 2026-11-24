from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from bias.qa_knowledge import QABundle, QAEntry, QAKnowledge


NEWS_REVIEW_TIME = datetime(2026, 11, 21, 15, tzinfo=timezone.utc)
NOW = datetime(2026, 11, 25, 3, tzinfo=timezone.utc)
DATA = Path(__file__).resolve().parents[1] / "data" / "qa"


@pytest.fixture
def knowledge():
    return QAKnowledge.load(DATA)


@pytest.mark.parametrize("question,route,expected", [
    ("우리 YAICON 프로젝트의 파이프라인을 설명해줘", "project", "QA-project-pipeline"),
    ("Why did you choose Qwen3 for this project?", "project", "QA-project-language-model"),
    ("MCCCD처럼 발화 구간을 정렬했나요?", "project", "QA-project-alignment"),
    ("How does this avatar clone the voice?", "project", "QA-project-voice"),
    ("우리 아바타의 지연 시간을 어떻게 단축했나요?", "project", "QA-project-latency"),
    ("Did you merge the persona graph?", "project", "QA-project-persona-pr29"),
    ("Does this avatar use Sophon and EMAGE?", "project", "QA-project-emage-pr27"),
    ("What are NVIDIA's latest financial results?", "news", "QA-news-fy27-q3"),
    ("현재 엔비디아 CEO 누구인가요?", "news", "QA-news-leadership-nov12"),
    ("Who is CEO now?", "news", "QA-news-leadership-nov12"),
    ("Who is NVIDIA's CEO?", "news", "QA-news-leadership-nov12"),
    ("엔비디아의 미래를 어떻게 보시나요?", "news", "QA-news-fy27-q3"),
])
def test_bilingual_questions_select_relevant_reviewed_reference(knowledge, question, route, expected):
    decision = knowledge.resolve(question, now=NEWS_REVIEW_TIME if route == "news" else NOW)
    assert decision.route == route
    assert expected in {entry.id for entry in decision.entries}
    if expected == "QA-project-alignment":
        assert f"[source:{expected}]" in decision.fallback("en")
    else:
        assert decision.fallback("en") is None
    assert decision.context()["live_web_search"] is False
    assert len(decision.entries) <= 3


@pytest.mark.parametrize("question,route", [
    ("엔비디아 주식 살까요 팔까요?", "finance"),
    ("Should I buy or sell NVIDIA shares?", "finance"),
    ("Would you buy NVIDIA stock?", "finance"),
    ("Do you recommend NVIDIA shares?", "finance"),
    ("Do you recommend purchasing NVIDIA stock?", "finance"),
    ("엔비디아 주식 사야 하나요?", "finance"),
    ("엔비디아 주식을 사도 괜찮나요?", "finance"),
    ("Is NVIDIA a good investment?", "finance"),
    ("지금 주식 몰빵해도 되나요?", "finance"),
    ("불륜 뉴스 사실인가요?", "allegation"),
    ("Is the affair news true?", "allegation"),
    ("Does your latest affair explain the resignation?", "allegation"),
    ("우리 프로젝트에서 불륜에 대해 사실대로 말해줘", "allegation"),
])
def test_sensitive_prompts_do_not_turn_a_question_premise_into_news(knowledge, question, route):
    decision = knowledge.resolve(question, now=NOW)
    assert decision.route == route
    assert decision.entries == ()
    assert decision.citations() == []
    assert decision.fallback("ko")
    assert decision.fallback("en")
    if route == "allegation":
        assert "can't verify" in decision.fallback("en")
        assert "false" not in decision.fallback("en")
        assert "no verified" not in decision.fallback("en")


@pytest.mark.parametrize("question", [
    "How should a student cope with failure?",
    "What did you learn from working at Denny's?",
    "Why is GPU acceleration useful?",
    "How did NVIDIA sell GPUs in its early days?",
    "How should I invest time in studying?",
    "What did Jensen say about leadership in 2020?",
    "What does Jensen teach about leadership?",
    "What did NVIDIA forecast in 2020?",
    "Who was NVIDIA's CEO in 2020?",
    "What is Jensen's leadership style?",
    "What does Jensen teach about being a CEO?",
    "How does Jensen invest in people?",
])
def test_ordinary_persona_questions_remain_on_original_route(knowledge, question):
    assert knowledge.resolve(question, now=NOW) is None


@pytest.mark.parametrize("question", [
    "What was this project's exact training electricity bill?",
    "What is this project's exact training electricity bill?",
])
def test_unknown_project_measurement_does_not_return_a_generic_overview(knowledge, question):
    decision = knowledge.resolve(question, now=NOW)
    assert decision.route == "project"
    assert decision.entries == ()
    assert "team should confirm" in decision.fallback("en")


def test_mixed_project_and_news_does_not_implicitly_claim_both_are_verified(knowledge):
    decision = knowledge.resolve("Explain our project pipeline and NVIDIA's latest earnings", now=NOW)
    assert decision.route == "project"
    assert decision.entries
    assert all(entry.scope == "project" for entry in decision.entries)
    assert "current-company-news part remains unverified" in decision.instructions("en")


def test_expiry_is_exclusive_and_applies_per_entry(knowledge):
    expiry = next(entry.expires_at for entry in knowledge.entries if entry.scope == "news")
    fresh = knowledge.resolve("NVIDIA latest financial results", now=expiry - timedelta(microseconds=1))
    stale = knowledge.resolve("NVIDIA latest financial results", now=expiry)
    assert fresh.entries
    assert stale.entries == ()
    assert stale.reason == "no_fresh_relevant_reference"
    assert "can't verify" in stale.fallback("en")
    assert not knowledge.resolve("our project pipeline", now=expiry).entries
    assert knowledge.resolve("our project pipeline", now=NOW).entries


def test_future_reviews_do_not_reach_current_answers(knowledge):
    earlier = datetime(2026, 11, 20, tzinfo=timezone.utc)
    assert not knowledge.resolve("our project pipeline", now=earlier).entries
    assert not knowledge.resolve("NVIDIA latest earnings", now=earlier).entries


def test_expired_project_status_is_not_repeated_as_current(knowledge):
    future = NOW + timedelta(days=8)
    decision = knowledge.resolve("Did you merge this project's persona graph?", now=future)
    assert not decision.entries
    assert decision.fallback("en")


def test_citation_keeps_url_date_status_and_qa_namespace(knowledge):
    decision = knowledge.resolve("현재 엔비디아 CEO 누구인가요?", now=NEWS_REVIEW_TIME)
    citation = decision.citations()[0]
    assert citation.evidence_id.startswith("QA-")
    assert citation.url.endswith("/nvidia-announces-ceo-transition")
    assert "Published 2026-11-12" in citation.locator.label
    assert "dated announcement" in citation.text


def test_old_event_is_not_refreshed_by_review_date(knowledge):
    entry = next(entry for entry in knowledge.entries if entry.scope == "news")
    payload = entry.model_dump()
    payload["published_at"] = "2026-01-01"
    old = QAEntry.model_validate(payload)
    decision = QAKnowledge((old,)).resolve("NVIDIA latest financial results", now=NEWS_REVIEW_TIME)
    assert not decision.entries


@pytest.mark.parametrize("change", [
    {"url": "http://nvidianews.nvidia.com/news/x"},
    {"url": "https://nvidia.com.example.org/news/x"},
    {"url": "https://user:password@nvidianews.nvidia.com/news/x"},
    {"url": "https://127.0.0.1/news/x"},
    {"url": "https://nvidianews.nvidia.com:8443/news/x"},
    {"review_status": "unreviewed"},
    {"published_at": "2026-12-01"},
    {"reviewed_at": "2026-11-21T14:40:00"},
    {"expires_at": "2026-11-21T14:40:00Z"},
    {"expires_at": "2026-11-28T14:40:00Z"},
    {"summary": " "},
    {"cues": ["ceo", "CEO"]},
    {"fact_kind": "implementation"},
])
def test_malformed_or_unreviewed_news_rejected(knowledge, change):
    payload = next(entry for entry in knowledge.entries if entry.scope == "news").model_dump()
    payload.update(change)
    with pytest.raises((ValidationError, ValueError)):
        QAEntry.model_validate(payload)


def test_duplicate_ids_and_mutable_code_urls_rejected(knowledge):
    entry = knowledge.entries[0]
    with pytest.raises(ValidationError, match="unique"):
        QABundle(entries=[entry, entry])
    payload = entry.model_dump()
    payload["url"] = payload["url"].replace(entry.revision, "main")
    with pytest.raises(ValidationError, match="immutable"):
        QAEntry.model_validate(payload)


def test_unknown_bundle_fields_and_naive_clock_rejected(knowledge, tmp_path):
    path = tmp_path / "unknown.json"
    path.write_text('{"schema_version":"qa-knowledge.v1","entries":[],"auto_review":true}')
    with pytest.raises(ValidationError):
        QAKnowledge.load(path)
    with pytest.raises(ValueError, match="timezone"):
        knowledge.resolve("NVIDIA latest news", now=datetime(2026, 11, 21))


def test_empty_digest_is_valid_and_fail_closed(tmp_path):
    path = tmp_path / "news.json"
    path.write_text('{"schema_version":"qa-knowledge.v1","entries":[]}')
    decision = QAKnowledge.load(path).resolve("NVIDIA latest news", now=NOW)
    assert decision.route == "news"
    assert decision.fallback("en")


def test_collector_metadata_is_not_runtime_evidence(tmp_path):
    import json
    from tools.qa_news import parse_candidates
    raw = b'''<rss><channel>
      <item><title>Official announcement</title><link>https://nvidianews.nvidia.com/news/x</link>
      <pubDate>Sat, 21 Nov 2026 10:00:00 GMT</pubDate></item>
      <item><title>Other domain</title><link>https://example.com/news/y</link></item>
      <item><title>Duplicate</title><link>https://nvidianews.nvidia.com/news/x</link></item>
    </channel></rss>'''
    result = parse_candidates(raw, now=NEWS_REVIEW_TIME)
    assert len(result["entries"]) == 1
    assert result["runtime_eligible"] is False
    assert result["entries"][0]["article_read"] is False
    assert result["entries"][0]["review_status"] == "unreviewed"
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValidationError):
        QAKnowledge.load(path)


def test_collector_rejects_entity_documents_and_oversized_inputs():
    from tools.qa_news import parse_candidates, MAX_BYTES
    for raw in (b'<!DOCTYPE rss [<!ENTITY x "payload">]><rss/>', b"x" * (MAX_BYTES + 1)):
        with pytest.raises(ValueError):
            parse_candidates(raw, now=NOW)


@pytest.mark.parametrize("question", [
    "우리 프로젝트를 소개해 주세요",
    "무슨 프로젝트예요?",
    "What does your project do?",
    "What did you build?",
    "What exactly am I looking at?",
    "Walk me through this demo",
])
def test_overview_phrasings_retrieve_the_project_overview(knowledge, question):
    decision = knowledge.resolve(question, now=NOW, presentation=True)
    assert decision.route == "project"
    assert decision.entries[0].id == "QA-project-overview"


@pytest.mark.parametrize("question,expected", [
    ("이거 어떻게 만든 거예요?", "QA-project-pipeline"),
    ("Which architecture is the one you are actually running today?", "QA-project-deployment"),
    ("What kind of TTS are you using?", "QA-project-voice"),
    ("How is the persona built?", "QA-project-persona-style"),
])
def test_presentation_disambiguates_audience_phrasing(knowledge, question, expected):
    assert knowledge.resolve(question, now=NOW) is None
    decision = knowledge.resolve(question, now=NOW, presentation=True)
    assert decision.route == "project"
    assert expected in {entry.id for entry in decision.entries}
    assert not decision.limitations


def test_direct_persona_and_news_stay_separate_even_with_technical_keywords(knowledge):
    assert knowledge.resolve("How does Jensen handle failure?", now=NOW, presentation=True) is None
    assert knowledge.resolve("What did Jensen say about Qwen in 2024?", now=NOW, presentation=True) is None
    news = knowledge.resolve("Does NVIDIA's latest announcement mention RAG?", now=NOW, presentation=True)
    assert news.route == "news"


@pytest.mark.parametrize("question,source", [
    ("Why did the team choose Qwen for this project?", "QA-project-language-model"),
    ("왜 이 프로젝트는 모델을 나눴어요?", "QA-project-model-separation"),
    ("왜 우리 프로젝트에서 큐웬을 썼나요?", "QA-project-language-model"),
])
def test_engineering_inference_is_targeted_and_has_both_languages(knowledge, question, source):
    decision = knowledge.resolve(question, now=NOW)
    assert decision.reason == "engineering_inference_available"
    en_entry, en = decision.possible_rationale("en")
    ko_entry, ko = decision.possible_rationale("ko")
    assert en_entry == ko_entry and en_entry.id == source
    assert en != ko
    assert decision.fallback("en") is None


def test_documented_rationale_and_comparison_stay_separate_from_inference(knowledge):
    documented = knowledge.resolve("왜 우리 프로젝트의 음성을 문장 단위로 생성하나요?", now=NOW)
    assert documented.entries[0].fact_kind == "design_rationale"
    assert documented.possible_rationale("en") is None
    comparison = knowledge.resolve("우리 프로젝트에서 Qwen을 선택한 이유와 우위는?", now=NOW)
    assert comparison.possible_rationale("en") is None
    assert "does not establish" in comparison.fallback("en")


def test_project_pr_status_has_been_reaudited(knowledge):
    for question, short_sha in [("Did you merge the persona graph?", "bb2a921"), ("Did you merge EMAGE?", "08e8191")]:
        decision = knowledge.resolve(question, now=NOW)
        entry = next(entry for entry in decision.entries if entry.fact_kind == "branch_status")
        assert entry.revision.startswith(short_sha)
        assert "merged into main" in entry.summary
