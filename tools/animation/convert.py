"""Measured geometry is not a person identity or an emotion label.

Input: tools.media observations plus a hash-bound selection/review document.
Output: avatar-clip.v1, with explicit source clock and source-neutral rotations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tools.media.common import MediaError, finite_number, local_file, sha256, write_json


FACE_CHANNELS = frozenset("""browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight
cheekPuff cheekSquintLeft cheekSquintRight eyeBlinkLeft eyeBlinkRight eyeLookDownLeft eyeLookDownRight
eyeLookInLeft eyeLookInRight eyeLookOutLeft eyeLookOutRight eyeLookUpLeft eyeLookUpRight
eyeSquintLeft eyeSquintRight eyeWideLeft eyeWideRight jawForward jawLeft jawOpen jawRight
mouthClose mouthDimpleLeft mouthDimpleRight mouthFrownLeft mouthFrownRight mouthFunnel mouthLeft
mouthLowerDownLeft mouthLowerDownRight mouthPressLeft mouthPressRight mouthPucker mouthRight
mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper mouthSmileLeft mouthSmileRight
mouthStretchLeft mouthStretchRight mouthUpperUpLeft mouthUpperUpRight noseSneerLeft noseSneerRight tongueOut""".split())

IDENTITY = (0.0, 0.0, 0.0, 1.0)
Vec = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def _number(value: Any, name: str, low: float = 0, high: float = 1e7) -> float:
    return finite_number(value, name, low, high)


def _text(value: Any, name: str, limit: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit or any(ord(char) < 32 for char in value):
        raise MediaError(f"{name} must be a nonempty bounded string")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise MediaError(f"{name} must be a SHA-256 digest")
    return value


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise MediaError(f"{name} must be an object")
    return value


def _dot(a: Vec, b: Vec) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _sub(a: Vec, b: Vec) -> Vec:
    return tuple(x - y for x, y in zip(a, b))


def _unit(value: Vec) -> Vec:
    length = math.sqrt(_dot(value, value))
    if length < 1e-7:
        raise MediaError("A measured orientation has degenerate axes")
    return tuple(item / length for item in value)


def _qnorm(q: Quat) -> Quat:
    length = math.sqrt(sum(item * item for item in q))
    if not math.isfinite(length) or length < 1e-8:
        raise MediaError("A measured orientation is not finite")
    sign = -1 if q[3] < 0 else 1
    return tuple(item / length * sign for item in q)


def qmul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return _qnorm((aw * bx + ax * bw + ay * bz - az * by,
                   aw * by - ax * bz + ay * bw + az * bx,
                   aw * bz + ax * by - ay * bx + az * bw,
                   aw * bw - ax * bx - ay * by - az * bz))


def qinv(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def _qmatrix(columns: tuple[Vec, Vec, Vec]) -> Quat:
    # Orthonormalize source axes to remove uniform scale and numerical drift.
    x = _unit(columns[0])
    y = _unit(_sub(columns[1], tuple(value * _dot(x, columns[1]) for value in x)))
    z = _unit(_cross(x, y))
    if _dot(z, columns[2]) <= 0:
        raise MediaError("A measured rotation contains a reflection")
    m = [[x[row], y[row], z[row]] for row in range(3)]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1) * 2
        return _qnorm(((m[2][1] - m[1][2]) / scale, (m[0][2] - m[2][0]) / scale, (m[1][0] - m[0][1]) / scale, scale / 4))
    largest = max(range(3), key=lambda index: m[index][index])
    j, k = (largest + 1) % 3, (largest + 2) % 3
    scale = math.sqrt(max(0, 1 + m[largest][largest] - m[j][j] - m[k][k])) * 2
    if scale < 1e-8:
        raise MediaError("Cannot normalize a measured rotation")
    values = [0.0, 0.0, 0.0, (m[k][j] - m[j][k]) / scale]
    values[largest] = scale / 4
    values[j] = (m[j][largest] + m[largest][j]) / scale
    values[k] = (m[k][largest] + m[largest][k]) / scale
    return _qnorm(tuple(values))


def face_rotation(matrix: Any) -> Quat | None:
    if matrix is None:
        return None
    if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in matrix):
        raise MediaError("Face transformation must be a row-major 4×4 matrix")
    values = [[_number(value, "face matrix", -1e6, 1e6) for value in row] for row in matrix]
    if any(abs(values[3][index]) > 1e-4 for index in range(3)) or abs(values[3][3] - 1) > 1e-4:
        raise MediaError("Face transformation must be affine")
    return _qmatrix(tuple(tuple(values[row][column] for row in range(3)) for column in range(3)))


def _direction_rotation(direction: Vec) -> Quat:
    target = _unit(direction)
    origin = (0.0, -1.0, 0.0)
    dot = _dot(origin, target)
    if dot < -0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    return _qnorm((*_cross(origin, target), 1 + dot))


def pose_rotations(pose: dict, threshold: float) -> dict[str, Quat]:
    landmarks = pose.get("world_landmarks", [])
    normalized = pose.get("landmarks", [])
    if not isinstance(landmarks, list) or len(landmarks) != 33 or not isinstance(normalized, list) or len(normalized) != 33:
        return {}

    def point(index: int) -> Vec | None:
        item = _object(landmarks[index], "world landmark")
        image = _object(normalized[index], "image landmark")
        visibility = _number(image.get("visibility", 0), "visibility", 0, 1)
        presence = _number(image.get("presence", 1), "presence", 0, 1)
        # Landmarks inferred outside the camera frame are not accepted as measured limbs.
        x = _number(image.get("x"), "image x", -100, 100)
        y = _number(image.get("y"), "image y", -100, 100)
        if visibility < threshold or presence < threshold or not 0 <= x <= 1 or not 0 <= y <= 1:
            return None
        # MediaPipe pose image axes -> right-handed, Y-up, Z toward the viewer.
        return (_number(item.get("x"), "world x", -100, 100), -_number(item.get("y"), "world y", -100, 100), -_number(item.get("z"), "world z", -100, 100))

    result: dict[str, Quat] = {}
    left_shoulder, right_shoulder, left_hip, right_hip = [point(index) for index in (11, 12, 23, 24)]
    if all(item is not None for item in (left_shoulder, right_shoulder, left_hip, right_hip)):
        shoulder_mid = tuple((a + b) / 2 for a, b in zip(left_shoulder, right_shoulder))
        hip_mid = tuple((a + b) / 2 for a, b in zip(left_hip, right_hip))
        x = _sub(left_shoulder, right_shoulder)
        y = _sub(shoulder_mid, hip_mid)
        result["chest"] = _qmatrix((x, y, _cross(x, y)))
    for side, shoulder_index, elbow_index, wrist_index, finger_index in (("left", 11, 13, 15, 19), ("right", 12, 14, 16, 20)):
        shoulder, elbow, wrist, finger = [point(index) for index in (shoulder_index, elbow_index, wrist_index, finger_index)]
        if shoulder is not None and elbow is not None:
            result[side + "UpperArm"] = _direction_rotation(_sub(elbow, shoulder))
        if shoulder is not None and elbow is not None and wrist is not None:
            result[side + "Forearm"] = _direction_rotation(_sub(wrist, elbow))
        if shoulder is not None and elbow is not None and wrist is not None and finger is not None:
            result[side + "Hand"] = _direction_rotation(_sub(finger, wrist))
    return result


PARENTS = {"head": "chest", "leftUpperArm": "chest", "rightUpperArm": "chest", "leftForearm": "leftUpperArm", "rightForearm": "rightUpperArm", "leftHand": "leftForearm", "rightHand": "rightForearm"}


def local_rotations(world: dict[str, Quat], reference_names: set[str] | None = None) -> dict[str, Quat]:
    available = set(world) if reference_names is None else reference_names
    result = {}
    for name, rotation in world.items():
        parent = PARENTS.get(name, "")
        if parent in available and parent not in world:
            continue
        parent_rotation = world[parent] if parent in available else IDENTITY
        result[name] = qmul(qinv(parent_rotation), rotation)
    return result


def limit_rotation(q: Quat, max_radians: float) -> Quat:
    q = _qnorm(q)
    angle = 2 * math.acos(max(-1, min(1, q[3])))
    if angle <= max_radians or angle < 1e-8:
        return q
    sine = math.sqrt(max(0, 1 - q[3] * q[3]))
    axis = tuple(value / sine for value in q[:3])
    return (*[value * math.sin(max_radians / 2) for value in axis], math.cos(max_radians / 2))


def load_review(path: Path, observations_sha256: str, allow_preview: bool) -> dict:
    review = _object(json.loads(local_file(path, max_bytes=1024**2).read_text()), "motion review")
    if review.get("schema_version") != "motion-review.v1" or review.get("observations_sha256") != observations_sha256:
        raise MediaError("Review must use motion-review.v1 and match the exact observations SHA-256")
    status = review.get("status")
    if status not in {"geometry_preview", "speaker_reviewed"}:
        raise MediaError("Review status must be geometry_preview or speaker_reviewed")
    if status == "geometry_preview" and not allow_preview:
        raise MediaError("Unreviewed geometry requires --allow-geometry-preview explicitly")
    for field in ("track_continuity_reviewed", "source_clock_reviewed", "speaker_association_reviewed"):
        if not isinstance(review.get(field), bool):
            raise MediaError(f"{field} must be an explicit boolean")
        if status == "speaker_reviewed" and review[field] is not True:
            raise MediaError("Speaker replay requires reviewed continuity, clock, and speaker association")
    for field in ("reviewer", "track_id", "attribution_basis"):
        _text(review.get(field), field, 1000 if field == "attribution_basis" else 160)
    selections = review.get("selections")
    if not isinstance(selections, list) or not 1 <= len(selections) <= 3601:
        raise MediaError("Review requires bounded explicit selection intervals")
    previous_end = -1.0
    for selected in selections:
        _object(selected, "selection")
        start = _number(selected.get("start_ms"), "selection start", 0, 120000)
        end = _number(selected.get("end_ms"), "selection end", start, 120000)
        if start < previous_end or end <= start:
            raise MediaError("Selections must be positive, ordered, and nonoverlapping")
        previous_end = end
        for field in ("face_index", "pose_index"):
            value = selected.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3):
                raise MediaError(f"{field} must be null or a frame-local detection index 0–3")
    return review


def export_clip(manifest_path: str | Path, review_path: str | Path, output_path: str | Path, *,
                start_ms: float = 0, end_ms: float | None = None, neutral_ms: float | None = None,
                max_gap_ms: float = 250, visibility: float = 0.65, max_rotation_degrees: float = 70,
                allow_geometry_preview: bool = False, overwrite: bool = False) -> dict:
    manifest_file = local_file(manifest_path, max_bytes=1024**2)
    manifest = _object(json.loads(manifest_file.read_text()), "observations manifest")
    if manifest.get("schema_version") != "observations.v1":
        raise MediaError("Use the observations.v1 manifest produced by tools.media observe")
    artifact = _object(manifest.get("observations"), "observations artifact")
    relative = _text(artifact.get("path"), "observations path")
    if Path(relative).is_absolute():
        raise MediaError("Observation path must be relative to its manifest")
    candidate = (manifest_file.parent / relative).resolve()
    if not candidate.is_relative_to(manifest_file.parent.resolve()):
        raise MediaError("Observation path must remain within the manifest directory")
    observations = local_file(candidate, max_bytes=256 * 1024**2)
    digest = sha256(observations)
    if digest != _hash(artifact.get("sha256"), "observations hash"):
        raise MediaError("Observation file does not match its manifest SHA-256")
    review_file = local_file(review_path, max_bytes=1024**2)
    review = load_review(review_file, digest, allow_geometry_preview)
    start_ms = _number(start_ms, "start_ms", 0, 120000)
    max_gap_ms = _number(max_gap_ms, "max_gap_ms", 1, 1000)
    visibility = _number(visibility, "visibility", 0, 1)
    max_radians = math.radians(_number(max_rotation_degrees, "max_rotation_degrees", 1, 180))
    rows = []
    with observations.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            if len(rows) >= 3601:
                raise MediaError("At most 3601 observations are accepted per clip")
            rows.append(_object(json.loads(line), "observation"))
    if not rows:
        raise MediaError("No observations are available")
    last_time = _number(rows[-1].get("timestamp_seconds"), "last timestamp", 0, 120) * 1000
    sample_fps = _number(manifest.get("sample_fps_requested", 10), "sample_fps_requested", 0.5, 30)
    source_start = _number(manifest.get("source_start_seconds"), "source_start_seconds")
    end_ms = _number(end_ms if end_ms is not None else min(120000, last_time + 1000 / max(1, sample_fps)), "end_ms", start_ms, 120000)
    if end_ms <= start_ms:
        raise MediaError("Animation window must have positive duration")
    frames: list[dict] = []
    world_frames: list[dict[str, Quat]] = []
    unknown_shapes: set[str] = set()
    previous_time = -1.0
    for row in rows:
        if row.get("schema_version") != "observation.v1" or row.get("source_id") != manifest.get("source_id") or row.get("source_timeline") != manifest.get("source_timeline"):
            raise MediaError("Observation source/schema differs from its manifest")
        time_ms = _number(row.get("timestamp_seconds"), "timestamp_seconds", 0, 120) * 1000
        declared_source_time = _number(row.get("source_timestamp_seconds"), "source_timestamp_seconds")
        if abs(declared_source_time - source_start - time_ms / 1000) > 0.002:
            raise MediaError("Observation source timestamp does not match the declared source clock")
        if time_ms <= previous_time:
            raise MediaError("Observation timestamps must strictly increase")
        previous_time = time_ms
        if not start_ms <= time_ms < end_ms:
            continue
        selection = next((item for item in review["selections"] if item["start_ms"] <= time_ms < item["end_ms"]), None)
        frame: dict[str, Any] = {"time_ms": round(time_ms - start_ms, 4)}
        world: dict[str, Quat] = {}
        if selection:
            for category, index_key in (("faces", "face_index"), ("poses", "pose_index")):
                candidates = row.get(category, [])
                if not isinstance(candidates, list) or len(candidates) > 4:
                    raise MediaError("A frame contains invalid or excessive detections")
                indexes = [item.get("detection_index") for item in candidates if isinstance(item, dict)]
                if len(indexes) != len(candidates) or len(set(indexes)) != len(indexes) or any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 3 for index in indexes):
                    raise MediaError("Frame-local detection indexes must be unique integers 0–3")
                index = selection.get(index_key)
                chosen = next((item for item in candidates if isinstance(item, dict) and item.get("detection_index") == index), None) if index is not None else None
                if chosen is None:
                    continue
                if category == "faces":
                    shapes = _object(chosen.get("blendshapes", {}), "blendshapes")
                    frame["face"] = {name: _number(value, name, 0, 1) for name, value in shapes.items() if name in FACE_CHANNELS}
                    unknown_shapes.update(set(shapes) - FACE_CHANNELS)
                    rotation = face_rotation(chosen.get("facial_transformation_matrix"))
                    if rotation is not None:
                        world["head"] = rotation
                else:
                    world.update(pose_rotations(chosen, visibility))
        frames.append(frame)
        world_frames.append(world)
    if not frames or not any(frame.get("face") or world for frame, world in zip(frames, world_frames)):
        raise MediaError("The selected window contains no usable selected geometry")
    reference_ms = _number(neutral_ms if neutral_ms is not None else start_ms + frames[0]["time_ms"], "neutral_ms", start_ms, end_ms)
    reference_index = min(range(len(frames)), key=lambda index: abs(frames[index]["time_ms"] + start_ms - reference_ms))
    if abs(frames[reference_index]["time_ms"] + start_ms - reference_ms) > max_gap_ms:
        raise MediaError("No selected observation is close enough to the reference pose timestamp")
    neutral_world = world_frames[reference_index]
    neutral_rotations = local_rotations(neutral_world)
    for frame, world in zip(frames, world_frames):
        rotations = local_rotations(world, set(neutral_world))
        bone_values = {name: list(limit_rotation(qmul(qinv(neutral_rotations[name]), rotation), max_radians)) for name, rotation in rotations.items() if name in neutral_rotations}
        if bone_values:
            frame["bones"] = bone_values
    face_neutral = _object(review.get("face_neutral", {}), "face_neutral")
    if any(name not in FACE_CHANNELS for name in face_neutral):
        raise MediaError("Face neutral calibration contains an unknown channel")
    face_neutral = {name: _number(value, name, 0, 0.99) for name, value in face_neutral.items()}
    result = {
        "schema_version": "avatar-clip.v1", "duration_ms": end_ms - start_ms, "max_gap_ms": max_gap_ms,
        "source": {"source_id": _text(manifest.get("source_id"), "source_id", 160), "source_timeline": _text(manifest.get("source_timeline"), "source_timeline", 160),
                   "source_start_seconds": source_start + start_ms / 1000, "observation_offset_ms": start_ms,
                   "video_sha256": _hash(manifest.get("video_sha256"), "video_sha256"), "observations_sha256": digest, "review_sha256": sha256(review_file)},
        "review": {key: review[key] for key in ("status", "reviewer", "track_id", "track_continuity_reviewed", "source_clock_reviewed", "speaker_association_reviewed", "attribution_basis")},
        "calibration": {"neutral_frame_ms": frames[reference_index]["time_ms"], "face_neutral": face_neutral,
                        "bone_neutral": {name: list(rotation) for name, rotation in neutral_rotations.items()},
                        "coordinate_basis": "right_handed_y_up", "bone_rotation_space": "local_delta"},
        "frames": frames,
        "limitations": [
            "Geometry estimates do not identify a person or establish emotional state.",
            "Detection selection and speaker attribution are reviewer assertions, not biometric verification.",
            "The reference frame establishes a rotation reference, not an automatically detected anatomical neutral face.",
            f"Pose landmarks outside the image or below visibility/presence {visibility} are omitted.",
            "Limb rotations use direction-only swing; axial twist and finger articulation are not measured.",
            f"Rotations are bounded to {max_rotation_degrees:g} degrees relative to the reference pose.",
            "Unobserved channels, gaps, and channels unavailable in the reference frame stay absent.",
            "Ignored source-only blendshape categories: " + (", ".join(sorted(unknown_shapes)) or "none"),
        ],
    }
    if manifest.get("mediapipe_version") and isinstance(manifest.get("models"), dict):
        result["measurement"] = {"engine": "mediapipe", "version": _text(manifest["mediapipe_version"], "mediapipe_version", 160),
                                 "face_model_sha256": _hash(manifest["models"].get("face_sha256"), "face model hash")}
        if manifest["models"].get("pose_sha256"):
            result["measurement"]["pose_model_sha256"] = _hash(manifest["models"]["pose_sha256"], "pose model hash")
    for field in ("input_sha256", "preparation_manifest_sha256"):
        if manifest.get(field):
            result["source"][field] = _hash(manifest[field], field)
    if manifest.get("video_crop"):
        crop = _object(manifest["video_crop"], "video_crop")
        values = {key: _number(crop.get(key), key, 0 if key in {"x", "y"} else 1, 32768) for key in ("x", "y", "width", "height", "input_width", "input_height")}
        if any(not value.is_integer() for value in values.values()) or values["x"] + values["width"] > values["input_width"] or values["y"] + values["height"] > values["input_height"] or crop.get("coordinate_basis") != "input_pixels_before_scale" or crop.get("mirrored") is not False:
            raise MediaError("Invalid fixed video crop provenance")
        result["source"]["video_crop"] = {**{key: int(value) for key, value in values.items()}, "coordinate_basis": "input_pixels_before_scale", "mirrored": False}
    write_json(Path(output_path), result, overwrite=overwrite)
    return result
