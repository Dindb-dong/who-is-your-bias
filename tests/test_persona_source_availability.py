"""Captured bytes establish availability, without changing historical stances."""
from datetime import date, datetime, timezone

import pytest

from bias.snapshots import PersonaSnapshot, register_snapshot, select_temporal


CUTOFF = datetime(2026, 11, 21, 8, tzinfo=timezone.utc)
CAPTURE = '2026-11-21T07:20:35+00:00'


def captured(corpus):
    source = corpus.sources['rocket-interview']
    corpus.sources[source.id] = source.model_copy(update={
        'published_at': CUTOFF.date(),
        'ingested_at': CAPTURE,
        'availability_observed_at': CAPTURE,
        'publication_date_verified': False,
        'publication_date_basis': 'observed_availability',
        'raw_content_sha256': 'a' * 64,
    })
    return register_snapshot(corpus, identifier='captured.v1', as_of=CUTOFF,
                             ingested_at=CUTOFF, current_evidence_since=date(2026, 10, 21))


def test_verified_capture_allows_same_day_historical_material_but_not_current_stance(corpus):
    snapshot = captured(corpus)
    assert 'rocket-reuse' in select_temporal(corpus, snapshot, 'rocket reuse').eligible_ids
    assert 'rocket-reuse' not in select_temporal(corpus, snapshot, 'current position on rockets').eligible_ids


def test_capture_clock_is_required_and_cannot_be_replaced_with_ingestion(corpus):
    snapshot = captured(corpus)
    raw = snapshot.model_dump(mode='json')
    for row in raw['sources']:
        row['verified_available_at'] = None
    assert 'rocket-reuse' not in select_temporal(corpus, PersonaSnapshot.model_validate(raw), 'rocket reuse').eligible_ids


@pytest.mark.parametrize('change', [
    {'source_content_sha256': 'b' * 64},
    {'verified_available_at': '2026-11-21T07:00:00+00:00'},
    {'verified_available_at': '2026-11-21T09:00:00+00:00'},
    {'verified_available_at': '2026-11-20T07:20:35+00:00'},
    {'source_content_sha256': None},
])
def test_unbound_backdated_or_unavailable_capture_is_rejected(corpus, change):
    snapshot = captured(corpus)
    raw = snapshot.model_dump(mode='json')
    next(row for row in raw['sources'] if row['source_id'] == 'rocket-interview').update(change)
    with pytest.raises(ValueError):
        candidate = PersonaSnapshot.model_validate(raw)
        select_temporal(corpus, candidate, 'rocket reuse')


def test_naive_capture_timestamp_is_rejected(corpus):
    snapshot = captured(corpus)
    raw = snapshot.model_dump(mode='json')
    next(row for row in raw['sources'] if row['source_id'] == 'rocket-interview')['verified_available_at'] = '2026-11-21T07:20:35'
    with pytest.raises(ValueError, match='offset'):
        PersonaSnapshot.model_validate(raw)
