"""Export a frozen, unreviewed source window with a coverage-based reference.

This selects a rotation reference from observed geometry. It cannot establish
anatomical neutral, person identity, gesture semantics or reviewed performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.animation.convert import FINGER_JOINTS, export_clip, hand_rotations, pose_rotations
from tools.media.common import MediaError, local_file, sha256, write_json
from tools.motion.preview_selection import make_preview


ARMS = {side + joint for side in ("left", "right") for joint in ("UpperArm", "Forearm", "Hand")}
FINGERS = {side + joint for side in ("left", "right") for joint in FINGER_JOINTS}


def choose_reference(samples: list[tuple[float, dict]]) -> dict:
    """Maximize observed arms, then fingers, then choose the earliest time."""
    candidates = [(len(ARMS.intersection(bones)), len(FINGERS.intersection(bones)), -time_ms)
                  for time_ms, bones in samples]
    if not candidates or not any(arms or fingers for arms, fingers, _ in candidates):
        raise MediaError("No selected arm or finger geometry can define a reference")
    arms, fingers, negative_time = max(candidates)
    return {"time_ms": -negative_time, "observable_arm_controls": arms, "observable_finger_controls": fingers}


def export_window(recipe_path: Path, measurements: Path, window_id: str, output: Path) -> dict:
    recipe_file = local_file(recipe_path, max_bytes=128 * 1024)
    recipe = json.loads(recipe_file.read_text())
    windows = [row for row in recipe.get("windows", []) if row.get("id") == window_id]
    if len(windows) != 1 or Path(window_id).name != window_id or window_id in {".", ".."}:
        raise MediaError("Choose exactly one frozen, bounded window ID")
    window, config = windows[0], recipe["frozen_export"]
    if config.get("review_status") != "geometry_preview" or any(config.get(field) is not False for field in
            ("track_continuity_reviewed", "source_clock_reviewed", "speaker_association_reviewed")):
        raise MediaError("This exporter only supports explicitly unreviewed geometry previews")
    root = measurements.resolve()
    directory = (root / window_id).resolve()
    if not directory.is_relative_to(root):
        raise MediaError("The observation bundle must remain inside its supplied root")
    manifest_path = local_file(directory / "observations-manifest.json", max_bytes=1024**2)
    manifest = json.loads(manifest_path.read_text())
    acquisition = next(row for row in recipe["acquisitions"] if row["source_id"] == window["source_id"])
    expected_start = window["start_seconds"] + acquisition["source_offset_seconds"]
    if (manifest.get("source_id") != window["source_id"] or manifest.get("source_start_seconds") != expected_start
            or manifest.get("input_sha256") != acquisition["input_sha256"]
            or manifest.get("source_timeline") != acquisition["source_timeline"]):
        raise MediaError("Observation provenance differs from the frozen source window")
    if (manifest.get("mediapipe_version") != recipe["runtime_version"] or manifest.get("sample_fps_requested") != recipe["sample_fps"] or any(
            manifest.get("models", {}).get(name + "_sha256") != recipe[name + "_model_sha256"]
            for name in ("face", "pose", "hand"))):
        raise MediaError("Measurement runtime or model hashes differ from the recipe")
    preparation_file = local_file(directory / "manifest.json", max_bytes=1024**2)
    if sha256(preparation_file) != manifest.get("preparation_manifest_sha256"):
        raise MediaError("Prepared clip provenance differs from its observation manifest")
    prepared = json.loads(preparation_file.read_text())
    crop = prepared.get("preparation", {}).get("video_crop")
    actual_crop = [crop[key] for key in ("x", "y", "width", "height")] if crop else None
    if (actual_crop != window.get("crop") or crop != manifest.get("video_crop")
            or abs(prepared.get("source_end_seconds", 0) - expected_start - window["duration_seconds"]) > .002):
        raise MediaError("Prepared crop or duration differs from the frozen window")
    artifact = manifest["observations"]
    observations = (directory / artifact["path"]).resolve()
    if not observations.is_relative_to(directory) or sha256(observations) != artifact["sha256"]:
        raise MediaError("Observation bytes differ from their manifest")
    rows = [json.loads(line) for line in local_file(observations, max_bytes=256 * 1024**2).read_text().splitlines() if line.strip()]
    if not 1 <= len(rows) <= 3601:
        raise MediaError("Use one bounded observation window")
    review_path = output / "reviews" / (window_id + ".json")
    review = make_preview(manifest_path, review_path, reviewer="geometry-preview-export",
                          track_id=window_id + "-selected-track", face_index=config["face_index"], pose_index=config["pose_index"])
    samples = []
    for row, selected in zip(rows, review["selections"]):
        bones = {}
        pose = next((item for item in row.get("poses", []) if item["detection_index"] == selected.get("pose_index")), None)
        if pose is not None:
            bones.update(pose_rotations(pose, config["visibility_threshold"]))
        for side in ("left", "right"):
            hand = next((item for item in row.get("hands", []) if item["detection_index"] == selected.get(side + "_hand_index")), None)
            if hand is not None:
                bones.update(hand_rotations(hand, side))
        samples.append((row["timestamp_seconds"] * 1000, bones))
    reference = choose_reference(samples)
    clip_path = output / "motion/jensen" / (window_id + ".json")
    clip = export_clip(manifest_path, review_path, clip_path, end_ms=window["duration_seconds"] * 1000,
                       neutral_ms=reference["time_ms"], max_gap_ms=config["max_gap_ms"],
                       visibility=config["visibility_threshold"], max_rotation_degrees=config["max_rotation_degrees"],
                       allow_geometry_preview=True)
    observed = [set(frame.get("bones", {})) for frame in clip["frames"]]
    coverage = {"complete_" + side + "_arm_frames": sum(all(side + joint in names for joint in ("UpperArm", "Forearm", "Hand")) for names in observed)
                for side in ("left", "right")}
    coverage["per_channel_frames"] = {name: sum(name in names for names in observed) for name in sorted(set().union(*observed))}
    receipt = {"schema_version": "gesture-preview-export.v1", "window_id": window_id, "recipe_sha256": sha256(recipe_file),
               "observed_frames": len(rows), "reference": reference, "coverage": coverage,
               "clip": {"path": str(clip_path.relative_to(output)), "bytes": clip_path.stat().st_size, "sha256": sha256(clip_path)},
               "source": clip["source"], "review": clip["review"]}
    write_json(output / "receipts" / (window_id + ".json"), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("window_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_window(args.recipe, args.measurements, args.window_id, args.output), indent=2))


if __name__ == "__main__":
    main()
