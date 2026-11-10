from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from tools.animation.convert import FACE_CHANNELS, export_clip, face_rotation, limit_rotation, local_rotations, pose_rotations
from tools.media.common import MediaError


def matrix_y(angle: float, scale: float = 1) -> list[list[float]]:
    c, s = math.cos(angle) * scale, math.sin(angle) * scale
    return [[c, 0, s, 0], [0, scale, 0, 0], [-s, 0, c, -50], [0, 0, 0, 1]]


def dataset(tmp_path: Path, *, mutate=None, reviewed: bool = False):
    rows = [{"schema_version": "observation.v1", "source_id": "synthetic", "source_timeline": "synthetic_video",
             "timestamp_seconds": index / 10, "source_timestamp_seconds": 10 + index / 10,
             "faces": [{"detection_index": 0, "blendshapes": {"jawOpen": 0.2 + index * 0.2, "eyeBlinkLeft": index / 2, "_neutral": 0.01},
                        "facial_transformation_matrix": matrix_y(index * 0.2)}], "poses": []} for index in range(3)]
    if mutate:
        mutate(rows)
    observations = tmp_path / "observations.jsonl"
    observations.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(observations.read_bytes()).hexdigest()
    manifest = tmp_path / "observations-manifest.json"
    manifest.write_text(json.dumps({"schema_version": "observations.v1", "source_id": "synthetic", "source_timeline": "synthetic_video", "source_start_seconds": 10,
                                    "sample_fps_requested": 10, "video_sha256": "b" * 64,
                                    "observations": {"path": observations.name, "sha256": digest}}))
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema_version": "motion-review.v1", "observations_sha256": digest,
                                 "status": "speaker_reviewed" if reviewed else "geometry_preview", "reviewer": "Synthetic test fixture", "track_id": "synthetic-one",
                                 "track_continuity_reviewed": reviewed, "source_clock_reviewed": reviewed, "speaker_association_reviewed": reviewed,
                                 "attribution_basis": "Synthetic test data only; no real speaker identity asserted.",
                                 "selections": [{"start_ms": 0, "end_ms": 300, "face_index": 0, "pose_index": None}]}))
    return manifest, review, tmp_path / "clip.json"


def test_geometry_preview_requires_explicit_opt_in_and_preserves_unreviewed_status(tmp_path: Path):
    manifest, review, output = dataset(tmp_path)
    with pytest.raises(MediaError, match="allow-geometry-preview"):
        export_clip(manifest, review, output)
    assert not output.exists()
    result = export_clip(manifest, review, output, allow_geometry_preview=True)
    assert result["review"]["status"] == "geometry_preview"
    assert result["review"]["speaker_association_reviewed"] is False
    assert len(result["frames"]) == 3
    assert result["frames"][0]["face"]["jawOpen"] == 0.2
    assert result["calibration"]["face_neutral"] == {}  # Speaking first frames are not silently called neutral.
    assert "_neutral" not in result["frames"][0]["face"]
    assert any("_neutral" in item for item in result["limitations"])
    assert result["frames"][1]["bones"]["head"][1] == pytest.approx(math.sin(0.1))


def test_speaker_replay_requires_all_review_assertions(tmp_path: Path):
    manifest, review, output = dataset(tmp_path, reviewed=True)
    data = json.loads(review.read_text()); data["speaker_association_reviewed"] = False; review.write_text(json.dumps(data))
    with pytest.raises(MediaError, match="reviewed continuity, clock, and speaker"):
        export_clip(manifest, review, output, allow_geometry_preview=True)


def test_review_cannot_be_reused_for_another_observation_file(tmp_path: Path):
    manifest, review, output = dataset(tmp_path)
    data = json.loads(review.read_text()); data["observations_sha256"] = "f" * 64; review.write_text(json.dumps(data))
    with pytest.raises(MediaError, match="exact observations"):
        export_clip(manifest, review, output, allow_geometry_preview=True)


def test_trim_preserves_the_native_source_clock_and_rebases_frames(tmp_path: Path):
    manifest, review, output = dataset(tmp_path)
    result = export_clip(manifest, review, output, start_ms=100, end_ms=300, neutral_ms=100, allow_geometry_preview=True)
    assert result["duration_ms"] == 200
    assert result["source"]["source_start_seconds"] == pytest.approx(10.1)
    assert result["source"]["observation_offset_ms"] == 100
    assert result["frames"][0]["time_ms"] == 0
    assert result["frames"][0]["bones"]["head"] == pytest.approx([0, 0, 0, 1])
    assert len(result["source"]["review_sha256"]) == 64


def test_missing_detections_remain_missing_and_do_not_steal_another_track(tmp_path: Path):
    def mutate(rows):
        rows[1]["faces"][0]["detection_index"] = 1
    manifest, review, output = dataset(tmp_path, mutate=mutate)
    result = export_clip(manifest, review, output, allow_geometry_preview=True)
    assert "face" not in result["frames"][1]
    assert "bones" not in result["frames"][1]
    assert result["frames"][2]["face"]


@pytest.mark.parametrize("mutate, message", [
    (lambda rows: rows[1].update(timestamp_seconds=0), "source timestamp"),
    (lambda rows: rows[1].update(source_timestamp_seconds=999), "source timestamp"),
    (lambda rows: rows[1].update(source_timeline="another_clock"), "source/schema"),
    (lambda rows: rows[1]["faces"].append(rows[1]["faces"][0].copy()), "unique integers"),
    (lambda rows: rows[1]["faces"][0]["blendshapes"].update(jawOpen=float("nan")), "between"),
])
def test_malformed_observations_fail_before_writing_a_clip(tmp_path: Path, mutate, message):
    manifest, review, output = dataset(tmp_path, mutate=mutate)
    with pytest.raises(MediaError, match=message):
        export_clip(manifest, review, output, allow_geometry_preview=True)
    assert not output.exists()


def test_matrix_rotations_remove_scale_handle_half_turns_and_reject_reflections():
    assert face_rotation(matrix_y(math.pi, 1.2)) == pytest.approx([0, 1, 0, 0], abs=1e-7)
    limited = limit_rotation(face_rotation(matrix_y(math.pi)), math.radians(70))
    assert 2 * math.acos(limited[3]) == pytest.approx(math.radians(70))
    reflection = matrix_y(0); reflection[0][0] = -1
    with pytest.raises(MediaError, match="reflection"):
        face_rotation(reflection)
    assert len(FACE_CHANNELS) == 52


def test_newly_observed_parent_does_not_change_child_coordinate_basis():
    head = face_rotation(matrix_y(0.8)); chest = face_rotation(matrix_y(0.3))
    result = local_rotations({"head": head, "chest": chest}, {"head"})
    assert result["head"] == pytest.approx(head)
    assert "head" not in local_rotations({"head": head}, {"head", "chest"})


def test_offscreen_pose_landmarks_are_omitted_even_with_high_model_confidence():
    image = [{"x": 0.5, "y": 0.5, "visibility": 1, "presence": 1} for _ in range(33)]
    world = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(33)]
    for index, xyz in {11: (.2, -.5, 0), 12: (-.2, -.5, 0), 23: (.15, .3, 0), 24: (-.15, .3, 0),
                       13: (.3, -.2, 0), 15: (.4, .1, 0), 19: (.45, .25, 0),
                       14: (-.3, -.2, 0), 16: (-.4, .1, 0), 20: (-.45, .25, 0)}.items():
        world[index] = dict(zip(("x", "y", "z"), xyz))
    image[15]["y"] = 1.3
    result = pose_rotations({"landmarks": image, "world_landmarks": world}, 0.65)
    assert "leftUpperArm" in result
    assert "leftForearm" not in result
    assert "leftHand" not in result
    assert "rightForearm" in result
