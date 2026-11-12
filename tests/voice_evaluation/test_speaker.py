import math

import pytest

from tools.voice_evaluation.speaker import compare, cosine
from tools.voice_evaluation.metrics import EvaluationError


def test_speaker_cosine_rejects_invalid_embeddings():
    assert cosine([1, 0], [1, 0]) == 1
    assert cosine([1, 0], [-1, 0]) == -1
    assert cosine([1, 0], [0, 1]) == 0
    for bad in ([], [0, 0], [math.nan, 1], [math.inf, 0], [1]):
        with pytest.raises(EvaluationError):
            cosine([1, 0], bad)


def test_reference_events_have_equal_weight_without_identity_threshold():
    refs = [{"embedding": [1, 0], "audio_sha256": "a", "event_family_id": "one"},
            {"embedding": [1, 0], "audio_sha256": "b", "event_family_id": "one"},
            {"embedding": [0, 1], "audio_sha256": "c", "event_family_id": "two"}]
    result = compare([1, 0], refs)
    assert result["event_means"] == {"one": 1, "two": 0}
    assert result["event_balanced_mean"] == 0.5
    assert "same_speaker" not in result
    assert "passed" not in result
