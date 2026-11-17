import pytest

from tools.media.common import MediaError
from tools.motion.preview_export import ARMS, FINGERS, choose_reference


def test_reference_preserves_arm_support_before_finger_count():
    arms = dict.fromkeys(ARMS)
    fingers = dict.fromkeys(FINGERS)
    selected = choose_reference([(0, fingers), (100, arms), (200, {**arms, **fingers})])
    assert selected == {"time_ms": 200, "observable_arm_controls": 6, "observable_finger_controls": 30}
    assert choose_reference([(0, fingers), (100, arms)])["time_ms"] == 100


def test_reference_uses_earliest_tie_and_cannot_fabricate_unobserved_limbs():
    assert choose_reference([(200, {"leftUpperArm": None}), (100, {"leftUpperArm": None})])["time_ms"] == 100
    with pytest.raises(MediaError, match="No selected arm"):
        choose_reference([(0, {"head": None, "chest": None})])
