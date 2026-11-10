from datetime import date
from pathlib import Path

import pytest

from bias.corpus import Corpus
from bias.models import Evidence, Locator, Persona, Source


@pytest.fixture
def corpus():
    sources = {
        "rocket-interview": Source(id="rocket-interview", title="Rocket interview", url="https://example.org/rocket", publisher="Example", published_at=date(2020, 1, 1), source_type="interview"),
        "car-plan": Source(id="car-plan", title="Electric car plan", url="https://example.org/car", publisher="Example", published_at=date(2016, 1, 1), source_type="authored_article"),
        "heldout": Source(id="heldout", title="Unseen conversation", url="https://example.org/heldout", publisher="Example", published_at=date(2024, 1, 1), source_type="interview", split="test"),
    }
    records = [
        Evidence(id="rocket-reuse", source_id="rocket-interview", text="Reusable rockets can lower launch costs through repeated flights.", topics=["rocket", "reuse", "launch", "로켓", "재사용"], locator=Locator(label="Launch discussion"), kind="stated_view", confidence="source_grounded", review_status="reviewed"),
        Evidence(id="car-sequence", source_id="car-plan", text="An expensive sports car can fund development of more affordable electric vehicles.", topics=["tesla", "electric vehicles", "sports car", "테슬라", "전기차"], locator=Locator(label="Plan"), kind="authored_plan", confidence="source_grounded", review_status="reviewed"),
        Evidence(id="secret-kiwi", source_id="heldout", text="The kiwi documentary concerns flightless birds.", topics=["kiwi", "birds"], locator=Locator(label="Birds"), kind="stated_view", confidence="source_grounded", review_status="reviewed"),
    ]
    profile = Persona(id="subject", name="Research Subject", tagline="Test profile", description="Public interview research.", style_rules=["Use concise explanations."])
    return Corpus(profile, sources, {record.id: record for record in records})


@pytest.fixture
def actual_corpus():
    return Corpus.load(Path(__file__).resolve().parents[1] / "data/personas/elon-musk")
