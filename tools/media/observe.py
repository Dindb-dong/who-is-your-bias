"""Measure visible geometry with optional MediaPipe Tasks models.

Detections are not identity labels. Blendshape scores are not emotion diagnoses.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .common import MediaError, finite_number, load_manifest, local_file, run_media, sha256, timestamp, verify_artifact, write_json


def landmark_values(points: Any) -> list[dict[str, float]]:
    values = []
    for point in points:
        item = {axis: finite_number(getattr(point, axis), axis, -1e6, 1e6) for axis in ("x", "y", "z")}
        for axis in ("visibility", "presence"):
            value = getattr(point, axis, None)
            if value is not None:
                item[axis] = finite_number(value, axis, 0, 1)
        values.append(item)
    return values


def serialize_face_result(result: Any) -> list[dict[str, Any]]:
    faces = []
    for index, landmarks in enumerate(result.face_landmarks):
        blendshapes = result.face_blendshapes[index] if index < len(result.face_blendshapes) else []
        matrices = result.facial_transformation_matrixes
        matrix = matrices[index].tolist() if index < len(matrices) else None
        faces.append({"detection_index": index, "identity": None, "landmarks": landmark_values(landmarks), "blendshapes": {shape.category_name: finite_number(shape.score, "blendshape score", 0, 1) for shape in blendshapes}, "facial_transformation_matrix": matrix})
    return faces


def serialize_hand_result(result: Any) -> list[dict[str, Any]]:
    """Retain geometry and model labels without assigning a person or arm."""
    hands = []
    for index, landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[index] if index < len(result.handedness) else []
        world = result.hand_world_landmarks[index] if index < len(result.hand_world_landmarks) else []
        hands.append({"detection_index": index, "identity": None,
                      "landmarks": landmark_values(landmarks), "world_landmarks": landmark_values(world),
                      "handedness": [{"label": item.category_name,
                                      "score": finite_number(item.score, "handedness score", 0, 1)} for item in handedness]})
    return hands


def video_timestamps(video: Path) -> list[float]:
    output = run_media(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)], timeout=60)
    try:
        frames = json.loads(output)["frames"]
        times = [finite_number(frame.get("best_effort_timestamp_time"), "video frame PTS") for frame in frames]
    except (KeyError, json.JSONDecodeError) as exc:
        raise MediaError("Cannot read video presentation timestamps") from exc
    if not times or any(right < left for left, right in zip(times, times[1:])):
        raise MediaError("Video must have ordered presentation timestamps")
    return times


def observe_clip(manifest_path: str | Path, *, face_model: str | Path, pose_model: str | Path | None = None, hand_model: str | Path | None = None, sample_fps: float = 10, overwrite: bool = False) -> dict[str, Any]:
    path, manifest = load_manifest(manifest_path)
    video = verify_artifact(path, manifest, "video")
    face_asset = local_file(face_model, max_bytes=100 * 1024**2)
    pose_asset = local_file(pose_model, max_bytes=100 * 1024**2) if pose_model else None
    hand_asset = local_file(hand_model, max_bytes=100 * 1024**2) if hand_model else None
    sample_fps = finite_number(sample_fps, "sample_fps", 0.5, 30)
    output = path.parent / "observations.jsonl"
    summary_path = path.parent / "observations-manifest.json"
    if not overwrite and (output.exists() or summary_path.exists()):
        raise MediaError("Observations already exist; use --overwrite explicitly")
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise MediaError("Install tools/media/requirements-vision.txt in an isolated environment") from exc
    times = video_timestamps(video)
    if times[-1] > manifest["clip"]["duration_seconds"] + 0.2:
        raise MediaError("Video extends beyond the prepared audio clock")
    vision = mp.tasks.vision
    face_options = vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(face_asset)), running_mode=vision.RunningMode.VIDEO, num_faces=4, output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise MediaError("OpenCV cannot decode the prepared video")
    fd, temp_name = tempfile.mkstemp(prefix=".observations-", suffix=".jsonl", dir=path.parent)
    observed_frames = 0
    empty_faces = 0
    frames_with_hands = 0
    next_due = 0.0
    last_ms = -1
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream, ExitStack() as stack:
            face_detector = stack.enter_context(vision.FaceLandmarker.create_from_options(face_options))
            pose_detector = None
            if pose_asset:
                pose_options = vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_asset)), running_mode=vision.RunningMode.VIDEO, num_poses=4, output_segmentation_masks=False)
                pose_detector = stack.enter_context(vision.PoseLandmarker.create_from_options(pose_options))
            hand_detector = None
            if hand_asset:
                hand_options = vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_asset)), running_mode=vision.RunningMode.VIDEO, num_hands=4)
                hand_detector = stack.enter_context(vision.HandLandmarker.create_from_options(hand_options))
            for frame_index, seconds in enumerate(times):
                ok, frame = capture.read()
                if not ok:
                    raise MediaError("Video decoding ended before its declared frame timestamps")
                milliseconds = round(seconds * 1000)
                if seconds + 1e-6 < next_due or milliseconds <= last_ms:
                    continue
                next_due = seconds + 1 / sample_fps
                last_ms = milliseconds
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                face_result = face_detector.detect_for_video(image, milliseconds)
                faces = serialize_face_result(face_result)
                poses = []
                if pose_detector:
                    pose_result = pose_detector.detect_for_video(image, milliseconds)
                    for index, landmarks in enumerate(pose_result.pose_landmarks):
                        poses.append({"detection_index": index, "identity": None, "landmarks": landmark_values(landmarks), "world_landmarks": landmark_values(pose_result.pose_world_landmarks[index])})
                hands = serialize_hand_result(hand_detector.detect_for_video(image, milliseconds)) if hand_detector else []
                row = {"schema_version": "observation.v1", "frame_index": frame_index, "timestamp_seconds": seconds, "source_timestamp_seconds": manifest["source_start_seconds"] + seconds, "source_id": manifest["source_id"], "source_timeline": manifest["source_timeline"], "faces": faces, "poses": poses, "hands": hands, "face_observable": bool(faces), "review_status": "unreviewed"}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                observed_frames += 1
                empty_faces += not bool(faces)
                frames_with_hands += bool(hands)
        if not observed_frames:
            raise MediaError("No frames were sampled")
        os.replace(temp_name, output)
    finally:
        capture.release()
        Path(temp_name).unlink(missing_ok=True)
    summary = {"schema_version": "observations.v1", "created_at": timestamp(), "source_id": manifest["source_id"], "source_timeline": manifest["source_timeline"], "source_start_seconds": manifest["source_start_seconds"], "video_sha256": manifest["artifacts"]["video"]["sha256"], "observations": {"path": output.name, "sha256": sha256(output)}, "observed_frames": observed_frames, "frames_without_detected_face": empty_faces, "sample_fps_requested": sample_fps, "mediapipe_version": importlib.metadata.version("mediapipe"), "models": {"face_sha256": sha256(face_asset), "pose_sha256": sha256(pose_asset) if pose_asset else None, "hand_sha256": sha256(hand_asset) if hand_asset else None}, "identity_attribution": "none", "review_status": "unreviewed", "limitations": ["Detection indexes are frame-local, not stable person identities.", "Blendshape scores and geometry do not establish emotional state.", "Missing detection means unobserved, not absence of a behavior.", "No linkage between a detected face and the audio speaker is inferred."]}
    summary["frames_with_detected_hands"] = frames_with_hands
    summary["limitations"].append("Handedness is a model label, not a reviewed subject-side or speaker association; review selects each hand explicitly.")
    summary["preparation_manifest_sha256"] = sha256(path)
    summary["input_sha256"] = manifest["input"]["sha256"]
    if manifest.get("preparation", {}).get("video_crop"):
        summary["video_crop"] = manifest["preparation"]["video_crop"]
    write_json(summary_path, summary, overwrite=overwrite)
    return summary
