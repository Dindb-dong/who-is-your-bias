from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.animation.convert import FINGER_JOINTS, export_clip, hand_rotations, local_rotations
from tools.media.common import MediaError
from tools.media.observe import serialize_hand_result
from tests.animation.test_conversion import dataset


def hand_fixture():
    points = [{"x": 0.0, "y": 0.0, "z": 0.0}]
    for finger in range(5):
        for joint in range(4):
            points.append({"x": (2 - finger) * .015, "y": -.02 - joint * .025, "z": joint * .004})
    return {"detection_index": 0, "identity": None, "world_landmarks": points,
            "landmarks": [{"x": .5 + p["x"], "y": .5 + p["y"], "z": p["z"]} for p in points]}


def test_serialization_retains_all_landmarks_and_no_identity():
    point = SimpleNamespace(x=.2, y=.3, z=.1)
    result = SimpleNamespace(hand_landmarks=[[point] * 21], hand_world_landmarks=[[point] * 21],
                             handedness=[[SimpleNamespace(category_name="Left", score=.9)]])
    hand = serialize_hand_result(result)[0]
    assert hand["identity"] is None
    assert len(hand["world_landmarks"]) == 21
    assert hand["handedness"] == [{"label": "Left", "score": .9}]


def test_fingers_have_normalized_local_rotations_and_parent_missing_stays_missing():
    hand = hand_fixture()
    world = hand_rotations(hand, "left")
    assert len(world) == 16
    assert set(world) == {"leftHand", *("left" + joint for joint in FINGER_JOINTS)}
    local = local_rotations(world)
    assert all(sum(value * value for value in q) == pytest.approx(1) for q in local.values())
    incomplete = dict(world); incomplete.pop("leftIndexProximal")
    assert "leftIndexIntermediate" not in local_rotations(incomplete, set(world))
    hand["landmarks"][8]["x"] = 1.1
    assert hand_rotations(hand, "left") == {}
    with pytest.raises(MediaError, match="explicitly"):
        hand_rotations(hand, "auto")


def test_export_selects_hands_explicitly_and_drops_unselected_or_missing_frames(tmp_path):
    def mutate(rows):
        for index, row in enumerate(rows):
            hand = hand_fixture()
            hand["world_landmarks"][8]["z"] += index * .01
            row["hands"] = [hand]
        rows[1]["hands"][0]["detection_index"] = 1
    manifest, review, output = dataset(tmp_path, mutate=mutate)
    data = json.loads(review.read_text()); data["selections"][0]["left_hand_index"] = 0
    review.write_text(json.dumps(data))
    result = export_clip(manifest, review, output, allow_geometry_preview=True)
    assert "leftIndexDistal" in result["frames"][0]["bones"]
    assert not any(name.startswith("left") for name in result["frames"][1].get("bones", {}))
    assert "rightHand" not in result["frames"][2]["bones"]
    assert result["frames"][2]["bones"]["leftIndexDistal"] != [0, 0, 0, 1]
    data["selections"][0]["right_hand_index"] = 0; review.write_text(json.dumps(data))
    with pytest.raises(MediaError, match="both subject-relative"):
        export_clip(manifest, review, tmp_path / "bad.json", allow_geometry_preview=True)


def test_motion_mining_retains_observed_fingers_in_upper_body_descriptors():
    from tools.motion.mining import describe
    import math
    frames=[{"time_ms":i*100,"bones":{"leftIndexProximal":[math.sin(i*.1),0,0,math.cos(i*.1)]}} for i in range(4)]
    features,coverage=describe(frames,"upper_body",250)
    assert features["leftIndexProximal:range_rad"] == pytest.approx(.6)
    assert coverage["leftIndexProximal"] == 1
