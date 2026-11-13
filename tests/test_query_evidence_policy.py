from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from bias.api import create_app
from bias.config import Settings, ProviderSettings
from bias.providers import OpenAICompatibleLLM
from bias.query_policy import public_evidence_limitation
from bias.retrieval import Retriever


@pytest.mark.parametrize('query', [
    'Did Jensen approve my chatbot?', 'Did he endorse our AI product?',
    '젠슨 황이 내가 만든 챗봇을 승인했어?', '내 챗봇을 공식 추천했어?',
    'Give me his private email address.', '젠슨 황 개인 전화번호 알려줘.',
])
def test_requests_require_unavailable_personal_evidence(query, corpus):
    assert public_evidence_limitation(query)
    assert Retriever(corpus).search(query, allow_live=True).hits == ()


@pytest.mark.parametrize('query', [
    'What did Jensen say about AI chatbots?', 'What did he say about company secrets?',
    'How does he approach hiring and leadership?', '공개 인터뷰에서 챗봇을 어떻게 설명했어?',
])
def test_published_subject_matter_is_not_a_user_endorsement(query):
    assert public_evidence_limitation(query) is None


def test_unavailable_endorsement_does_not_reach_generation(monkeypatch, tmp_path):
    async def stream(_self, _messages):
        pytest.fail('Unsupported user endorsement must not reach generation')
        yield
    monkeypatch.setattr(OpenAICompatibleLLM, 'stream', stream)
    settings = replace(Settings(), web_dir=tmp_path/'absent', llm=ProviderSettings('http://model.invalid/v1','test'))
    with TestClient(create_app(settings)) as client:
        response = client.post('/api/chat', json={'message':'Did Jensen approve my chatbot?','language':'en'})
    assert response.status_code == 200
    turn = response.json()
    assert turn['citations'] == [] and turn['provider'] == 'evidence'
    assert turn['grounding']['status'] == 'insufficient'
    assert 'endorsement' in turn['grounding']['reason']
