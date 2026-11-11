"""Deterministic kinematic mining with train-only normalization and medoids.

Descriptors summarize measured geometry, not personality or emotions. Missing
features stay missing; faces and upper bodies are fitted as separate families.
The validation event never influences descriptor selection, scales, prototypes,
or distance thresholds.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bias.motion import MotionCatalog, MotionClip, MotionError, MotionPrimitive, MotionSource, validate_clip

VERSION = "1.0.0"
BODY_CHANNELS = {"chest", "leftUpperArm", "rightUpperArm", "leftForearm", "rightForearm", "leftHand", "rightHand"}


def geodesic(left: list[float], right: list[float]) -> float:
    """Shortest rotation angle; q and -q encode the same orientation."""
    # Clip validation permits small normalization/serialization error. Angular
    # distance must divide it out; otherwise a stationary q with norm < 1
    # acquires artificial range and velocity on every adjacent frame.
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not math.isfinite(left_norm) or not math.isfinite(right_norm) or left_norm == 0 or right_norm == 0:
        raise MotionError("Angular distance requires finite nonzero quaternions")
    dot = abs(sum(a * b for a, b in zip(left, right))) / (left_norm * right_norm)
    return 2 * math.acos(max(-1.0, min(1.0, dot)))


def describe(frames: list[dict], family: str, max_gap_ms: float, minimum_coverage: float = 0.8) -> tuple[dict[str, float], dict[str, float]]:
    category = "face" if family == "face" else "bones"
    names = sorted({name for frame in frames for name in frame.get(category, {}) if category == "face" or name in BODY_CHANNELS})
    features: dict[str, float] = {}
    coverage: dict[str, float] = {}
    for name in names:
        observed = [(frame["time_ms"], frame[category][name]) for frame in frames if name in frame.get(category, {})]
        coverage[name] = len(observed) / len(frames)
        pairs = [(left, right) for left, right in zip(observed, observed[1:]) if 0 < right[0] - left[0] <= max_gap_ms]
        if coverage[name] < minimum_coverage or len(pairs) < 2:
            continue
        if category == "face":
            values = [value for _, value in observed]
            features[name + ":range"] = max(values) - min(values)
            features[name + ":speed"] = sum(abs(b[1] - a[1]) for a, b in pairs) / (sum(b[0] - a[0] for a, b in pairs) / 1000)
        else:
            reference = observed[0][1]
            features[name + ":range_rad"] = max(geodesic(reference, value) for _, value in observed)
            features[name + ":speed_rad_s"] = sum(geodesic(a[1], b[1]) for a, b in pairs) / (sum(b[0] - a[0] for a, b in pairs) / 1000)
    return {key: round(value, 8) for key, value in features.items()}, {key: round(value, 6) for key, value in coverage.items()}


def mine_clip(clip: dict, clip_id: str, *, window_ms: float = 2500, stride_ms: float = 2500, minimum_coverage: float = 0.8) -> list[dict]:
    validate_clip(clip)
    if not 500 <= window_ms <= 10000 or not 500 <= stride_ms <= window_ms:
        raise MotionError("Mining windows/strides must be 0.5–10 seconds, without uncovered strides")
    if not 0.5 <= minimum_coverage <= 1:
        raise MotionError("Minimum feature coverage must be between 0.5 and 1")
    primitives = []
    start = 0.0
    while start + window_ms <= clip["duration_ms"] + 0.001:
        end = start + window_ms
        frames = [frame for frame in clip["frames"] if start <= frame["time_ms"] < end]
        if len(frames) >= 3 and all(b["time_ms"] - a["time_ms"] <= clip["max_gap_ms"] for a, b in zip(frames, frames[1:])):
            # Each window must actually cover most of the requested interval;
            # several clustered frames followed by a long gap are not a clip.
            if frames[0]["time_ms"] - start <= clip["max_gap_ms"] and end - frames[-1]["time_ms"] <= clip["max_gap_ms"]:
                for family in ("face", "upper_body"):
                    descriptor, coverage = describe(frames, family, clip["max_gap_ms"], minimum_coverage)
                    if descriptor:
                        primitives.append({"id": f"{clip_id}-{family.replace('_', '-')}-{round(start):06d}", "clip_id": clip_id, "family": family, "start_ms": start, "end_ms": end, "descriptor": descriptor, "feature_coverage": coverage, "frame_count": len(frames), "geometry_cluster_id": None, "prototype_distance": None})
        start += stride_ms
    return primitives


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def _variation_floor(key: str) -> float:
    # Fixed engineering floors prevent nearly constant model noise from getting
    # enormous standardized weight. They are not calibrated detector accuracy.
    if key.endswith(":range_rad"):
        return math.radians(1)
    if key.endswith(":speed_rad_s"):
        return 0.02
    return 0.0025 if key.endswith(":range") else 0.01


def kmedoids(vectors: list[list[float]], count: int) -> tuple[list[int], list[int]]:
    """Deterministic farthest-first initialization followed by medoid updates."""
    if not vectors or count < 1:
        raise MotionError("Medoid fitting requires training vectors and a positive cluster count")
    distances = [[_distance(left, right) for right in vectors] for left in vectors]
    medoids = [min(range(len(vectors)), key=lambda index: (sum(distances[index]), index))]
    while len(medoids) < min(count, len(vectors)):
        remaining = [i for i in range(len(vectors)) if i not in medoids]
        candidate = max(remaining, key=lambda i: (min(distances[i][m] for m in medoids), -i))
        if min(distances[candidate][m] for m in medoids) < 1e-12:
            break
        medoids.append(candidate)
    assignments: list[int] = []
    for _ in range(100):
        assignments = [min(range(len(medoids)), key=lambda cluster: (distances[index][medoids[cluster]], cluster)) for index in range(len(vectors))]
        updated = []
        for cluster, medoid in enumerate(medoids):
            members = [i for i, assigned in enumerate(assignments) if assigned == cluster]
            updated.append(min(members, key=lambda i: (sum(distances[i][j] for j in members), i)) if members else medoid)
        if updated == medoids:
            break
        medoids = updated
    assignments = [min(range(len(medoids)), key=lambda cluster: (distances[index][medoids[cluster]], cluster)) for index in range(len(vectors))]
    return medoids, assignments


def fit_library(primitives: list[dict], clip_sources: dict[str, MotionSource], *, clusters: int = 3) -> tuple[dict, dict]:
    if not 1 <= clusters <= 20:
        raise MotionError("Use 1–20 geometry clusters per feature family")
    fits: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    for family in ("face", "upper_body"):
        family_rows = [p for p in primitives if p["family"] == family]
        train = [p for p in family_rows if clip_sources[p["clip_id"]].motion_split == "train"]
        held_out = [p for p in family_rows if clip_sources[p["clip_id"]].motion_split != "train"]
        if not train:
            evaluations[family] = {"status": "no_training_geometry", "held_out_windows": len(held_out), "held_out_source_events": sorted({clip_sources[p["clip_id"]].source_event_id for p in held_out})}
            continue
        # A feature must exist on every training window. No feature or scale is
        # selected using validation geometry; no missing observation becomes 0.
        common = set.intersection(*(set(p["descriptor"]) for p in train))
        keys = sorted(key for key in common if statistics.pstdev(p["descriptor"][key] for p in train) > _variation_floor(key))
        if not keys:
            evaluations[family] = {"status": "insufficient_variable_features", "training_windows": len(train), "held_out_windows": len(held_out), "held_out_source_events": sorted({clip_sources[p["clip_id"]].source_event_id for p in held_out})}
            continue
        mean = [statistics.mean(p["descriptor"][key] for p in train) for key in keys]
        scale = [statistics.pstdev(p["descriptor"][key] for p in train) for key in keys]
        def vector(row: dict) -> list[float] | None:
            if any(key not in row["descriptor"] for key in keys):
                return None
            return [(row["descriptor"][key] - center) / spread for key, center, spread in zip(keys, mean, scale)]
        vectors = [vector(row) for row in train]
        medoids, assignments = kmedoids(vectors, clusters)
        prototypes = [vectors[index] for index in medoids]
        distances = [_distance(vectors[i], prototypes[assigned]) for i, assigned in enumerate(assignments)]
        # Training maximum is only an empirical envelope, not a probabilistic
        # confidence interval. A one-window cluster gets an envelope of zero.
        envelopes = [max(distance for distance, assigned in zip(distances, assignments) if assigned == cluster) for cluster in range(len(medoids))]
        for row, assigned, distance in zip(train, assignments, distances):
            row.update(geometry_cluster_id=f"{family.replace('_', '-')}-{assigned + 1:02d}", prototype_distance=round(distance, 8))
        validation_rows = []
        for row in held_out:
            values = vector(row)
            if values is None:
                validation_rows.append({"primitive_id": row["id"], "source_event_id": clip_sources[row["clip_id"]].source_event_id, "status": "missing_required_features", "missing_features": [key for key in keys if key not in row["descriptor"]]})
                continue
            assigned = min(range(len(prototypes)), key=lambda cluster: (_distance(values, prototypes[cluster]), cluster))
            distance = _distance(values, prototypes[assigned])
            row.update(geometry_cluster_id=f"{family.replace('_', '-')}-{assigned + 1:02d}", prototype_distance=round(distance, 8))
            validation_rows.append({"primitive_id": row["id"], "source_event_id": clip_sources[row["clip_id"]].source_event_id, "status": "measured", "prototype_distance": round(distance, 8), "inside_training_envelope": distance <= envelopes[assigned] + 1e-8})
        measured = [row for row in validation_rows if row["status"] == "measured"]
        fits[family] = {"algorithm": "deterministic_k_medoids", "feature_names": keys, "feature_mean": mean, "feature_scale": scale, "variation_floor": {key: _variation_floor(key) for key in sorted(common)}, "excluded_low_variation_features": sorted(common - set(keys)), "training_source_events": sorted({clip_sources[p["clip_id"]].source_event_id for p in train}), "training_windows": len(train), "prototypes": [{"cluster_id": f"{family.replace('_', '-')}-{i + 1:02d}", "primitive_id": train[medoid]["id"], "normalized_descriptor": prototypes[i], "training_radius": envelopes[i], "members": assignments.count(i)} for i, medoid in enumerate(medoids)]}
        evaluations[family] = {"status": "source_held_out_geometry_only" if held_out else "no_held_out_source", "held_out_source_events": sorted({clip_sources[p["clip_id"]].source_event_id for p in held_out}), "held_out_windows": len(held_out), "comparable_windows": len(measured), "missing_feature_windows": len(held_out) - len(measured), "mean_prototype_distance": statistics.mean(row["prototype_distance"] for row in measured) if measured else None, "inside_training_envelope": sum(row["inside_training_envelope"] for row in measured), "windows": validation_rows}
    return fits, evaluations


def build_library(manifest_path: str | Path, asset_root: str | Path, output_path: str | Path) -> dict:
    path = Path(manifest_path)
    if path.stat().st_size > 2 * 1024**2:
        raise MotionError("Motion build manifest is too large")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "motion-input.v1":
        raise MotionError("Expected motion-input.v1 build manifest")
    sources = [MotionSource.model_validate(source) for source in manifest["sources"]]
    records = [MotionClip.model_validate(clip) for clip in manifest["clips"]]
    source_map = {source.id: source for source in sources}
    if len(source_map) != len(sources) or len({clip.id for clip in records}) != len(records):
        raise MotionError("Duplicate motion input identity")
    event_splits: dict[str, tuple[str, str]] = {}
    for source in sources:
        splits = (source.corpus_split, source.motion_split)
        if source.source_event_id in event_splits and event_splits[source.source_event_id] != splits:
            raise MotionError("A source event crosses dataset splits")
        if source.motion_split == "train" and source.corpus_split != "train":
            raise MotionError("A corpus holdout cannot fit motion features")
        event_splits[source.source_event_id] = splits
    root = Path(asset_root).resolve(strict=True)
    primitives = []
    qa = {}
    for record in records:
        relative = Path(record.path)
        asset = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not asset.is_relative_to(root) or not asset.is_file() or asset.stat().st_size > 8 * 1024**2:
            raise MotionError("Invalid bounded motion asset path")
        if record.source_id not in source_map:
            raise MotionError("Unknown clip source")
        raw = asset.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record.sha256:
            raise MotionError("Input motion asset SHA-256 mismatch")
        clip = json.loads(raw)
        if clip.get("source", {}).get("source_id") != record.source_id:
            raise MotionError("Input clip source differs from catalog")
        measured = mine_clip(clip, record.id, window_ms=manifest.get("window_ms", 2500), stride_ms=manifest.get("stride_ms", 2500), minimum_coverage=manifest.get("minimum_feature_coverage", 0.8))
        primitives.extend(measured)
        frames = clip["frames"]
        qa[record.id] = {"source_id": record.source_id, "frames": len(frames), "duration_ms": clip["duration_ms"], "review_status": clip["review"]["status"], "maximum_frame_gap_ms": max(b["time_ms"] - a["time_ms"] for a, b in zip(frames, frames[1:])), "face_channel_coverage": {name: sum(name in frame.get("face", {}) for frame in frames) / len(frames) for name in sorted({name for frame in frames for name in frame.get("face", {})})}, "bone_channel_coverage": {name: sum(name in frame.get("bones", {}) for frame in frames) / len(frames) for name in sorted({name for frame in frames for name in frame.get("bones", {})})}, "mined_windows_by_family": dict(Counter(p["family"] for p in measured))}
    if not primitives:
        raise MotionError("No usable contiguous motion primitives were mined")
    clip_sources = {record.id: source_map[record.source_id] for record in records}
    fitting, evaluation = fit_library(primitives, clip_sources, clusters=manifest.get("clusters_per_family", 3))
    catalog = MotionCatalog(schema_version="motion-library.v1", id=manifest["id"], version=manifest["version"], created_at=datetime.now(timezone.utc).isoformat(), sources=sources, clips=records, primitives=[MotionPrimitive.model_validate(p) for p in primitives], fitting={"implementation_version": VERSION, "input_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "window_ms": manifest.get("window_ms", 2500), "stride_ms": manifest.get("stride_ms", 2500), "minimum_feature_coverage": manifest.get("minimum_feature_coverage", 0.8), "families": fitting}, evaluation={"per_clip_quality": qa, "families": evaluation, "semantic_gesture_accuracy": None, "person_identification_accuracy": None, "conditional_personal_policy_accuracy": None, "reason": "No independent labeled gesture ground truth, speaker review, or learned conditional-person policy is available."}, limitations=["Source-context lexical retrieval selects measured geometry; it is not a learned personal gesture policy.", "Unsupervised geometry cluster IDs are not emotion, intention, or personality labels.", "The motion-validation event is excluded from normalization, feature selection, prototypes, thresholds, and runtime retrieval.", "Frames from one event are correlated; primitive count is not a count of independent people or events.", "Geometry-preview windows have no reviewed association between the image track and the named speaker.", "Direction-only limb measurements omit axial twist and finger articulation; monocular estimates are not ground-truth motion capture.", "A training distance envelope is a descriptive range, not a calibrated confidence interval."])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog.model_dump(), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return catalog.model_dump()
