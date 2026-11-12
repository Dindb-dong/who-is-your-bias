"""Measure hash-bound visual references for a bounded, approximate mesh fit.

This is not photogrammetry. MediaPipe's inferred depth and weak-perspective pose
normalization support coarse ratios; texture, topology and similarity need review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

# Stable anatomical feature anchors in MediaPipe's published face topology.
RATIO_PAIRS = {
    "face_aspect": ((234, 454), (10, 152)),
    "jaw_to_cheek": ((172, 397), (234, 454)),
    "nose_to_face": ((48, 278), (234, 454)),
    "mouth_to_face": ((61, 291), (234, 454)),
    "eyes_to_face": ((33, 263), (234, 454)),
    "lower_face_to_face": ((2, 152), (10, 152)),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratios(points) -> dict[str, float]:
    result = {}
    for name, (pair, reference) in RATIO_PAIRS.items():
        distance = lambda ij: math.dist(points[ij[0]], points[ij[1]])
        scale = distance(reference)
        if not math.isfinite(scale) or scale <= 1e-7:
            raise ValueError("Degenerate reference facial geometry")
        value = distance(pair) / scale
        if not math.isfinite(value) or not 0 < value < 3:
            raise ValueError("Invalid measured facial ratio")
        result[name] = value
    return result


def measure(manifest_path: Path, model: Path, output: Path) -> dict:
    import cv2
    import mediapipe as mp
    import numpy as np

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("persona_id") != "jensen-huang":
        raise ValueError("This fit requires the explicit Jensen reference manifest")
    if output.exists():
        raise ValueError("Choose a new output; measured reference sets are immutable")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=4,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    accepted, excluded = [], []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for source in manifest["sources"]:
            path = (manifest_path.parent / source["filename"]).resolve()
            if not path.is_relative_to(manifest_path.parent.resolve()) or digest(path) != source["sha256"]:
                raise ValueError("Reference bytes differ or escape their manifest")
            if source.get("split") != "train":
                raise ValueError("Held-out visual references cannot enter identity fitting")
            if source.get("exclude_reason"):
                excluded.append({"id": source["id"], "reason": source["exclude_reason"]}); continue
            frame = cv2.imread(str(path))
            if frame is None:
                excluded.append({"id": source["id"], "reason": "image_decode_failed"}); continue
            height, width = frame.shape[:2]
            resized = cv2.resize(frame, (round(width * min(1, 1600 / max(width, height))), round(height * min(1, 1600 / max(width, height)))))
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
            if len(result.face_landmarks) != 1:
                excluded.append({"id": source["id"], "reason": "requires_exactly_one_reviewed_visible_face", "detections": len(result.face_landmarks)}); continue
            points = np.array([[p.x * width, -p.y * height, -p.z * width] for p in result.face_landmarks[0]], dtype=float)
            points -= points[1]
            rotation = np.asarray(result.facial_transformation_matrixes[0], dtype=float)[:3, :3]
            u, _, vt = np.linalg.svd(rotation); rotation = u @ vt
            normalized = points @ rotation
            accepted.append({"id": source["id"], "event_family": source["event_family"], "sha256": source["sha256"],
                             "size": [width, height], "ratios": ratios(normalized),
                             "landmarks": [[p.x, p.y, p.z] for p in result.face_landmarks[0]],
                             "pose_normalized_points": normalized.tolist(),
                             "facial_transformation_matrix": result.facial_transformation_matrixes[0].tolist(),
                             "blendshapes": {x.category_name: float(x.score) for x in result.face_blendshapes[0]}})
    if len(accepted) < 2:
        raise ValueError("At least two accepted photographs are required for this pilot")
    families = sorted({row["event_family"] for row in accepted})
    # Multiple assets from one portrait session do not outvote another session.
    family_ratios = {family: {name: float(np.median([row["ratios"][name] for row in accepted if row["event_family"] == family]))
                              for name in RATIO_PAIRS} for family in families}
    target = {name: float(np.median([row[name] for row in family_ratios.values()])) for name in RATIO_PAIRS}
    report = {"schema_version": "avatar-identity-measurements.v1", "persona_id": "jensen-huang",
              "reference_manifest_sha256": digest(manifest_path), "model_sha256": digest(model),
              "method": "event-balanced weak-perspective landmark ratios", "accepted": accepted,
              "excluded": excluded, "event_families": families, "target_ratios": target,
              "limitations": ["Inferred depth and pose-normalized image ratios are not a metric scan.",
                              "Expressions, lens effects and occlusions can bias proportions.",
                              "Human likeness quality remains ungraded."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n"); temp.replace(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path); parser.add_argument("model", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = measure(args.manifest, args.model, args.output)
    print(json.dumps({"accepted": len(result["accepted"]), "excluded": len(result["excluded"]), "target_ratios": result["target_ratios"]}))


if __name__ == "__main__":
    main()
