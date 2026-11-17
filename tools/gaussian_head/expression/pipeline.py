"""Fit named face controls to an existing FLAME head from reviewed observations.

This is supervised ridge calibration, not a new Gaussian identity model. Dense
FLAME coordinates are learned from paired data; no coefficient is called a brow
or mouth blendshape. Artifacts contain licensed derivatives and stay private.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from tools.animation.convert import FACE_CHANNELS, load_review
from tools.gaussian_head.bundle.head_bundle import read_bundle
from tools.media.common import MediaError


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".previous"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ranges(value: object, count: int, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise MediaError(f"{label} needs explicit half-open frame ranges")
    result: list[int] = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2 or any(type(n) is not int for n in pair):
            raise MediaError(f"Invalid {label} frame range")
        start, end = pair
        if not 0 <= start < end <= count or (result and start <= result[-1]):
            raise MediaError(f"{label} frame ranges overlap or leave the bundle")
        result.extend(range(start, end))
    return result


def _path(base: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise MediaError("A local input path is required")
    return (base / value).resolve()


def prepare_dataset(recipe_path: str | Path, output_path: str | Path) -> dict:
    """Pair only reviewed selected faces with FLAME frames on the source clock.

    Recipe splits are explicit and fixed before fitting. Repeated measurements
    of one FLAME frame contribute once; alignment failures stay excluded.
    """
    path = Path(recipe_path).resolve()
    recipe = json.loads(path.read_text())
    if recipe.get("schema") != "gaussian-expression-preparation.v1" or recipe.get("subject_id") != "jensen-huang":
        raise MediaError("Use a Jensen gaussian-expression-preparation.v1 recipe")
    for key in ("source_id", "source_timeline", "event_id", "sequence_id"):
        if not isinstance(recipe.get(key), str) or not recipe[key].strip():
            raise MediaError(f"Declare {key} before preparing expression data")
    stem = _path(path.parent, recipe.get("bundle_stem"))
    header, bundle = read_bundle(str(stem))
    if header.get("schema") != "gaussian-head.v1":
        raise MediaError("Unsupported Gaussian head bundle")
    E, T, fps = header["n_expr"], header["n_frames"], float(header["fps"])
    if type(E) is not int or type(T) is not int or not 1 <= E <= 512 or T < 2 or not np.isfinite(fps) or fps <= 0:
        raise MediaError("Invalid FLAME motion dimensions")
    if bundle["m_expr"].shape != (T, E) or bundle["m_pose"].shape != (T, 15):
        raise MediaError("FLAME tracks disagree with the header")
    gaussian_train = _ranges(recipe.get("gaussian_train_ranges"), T, "Gaussian training")
    train = _ranges(recipe.get("fit_train_ranges"), T, "Calibration training")
    validation = _ranges(recipe.get("validation_ranges"), T, "Calibration validation")
    if not (set(train) | set(validation)) <= set(gaussian_train) or set(train) & set(validation):
        raise MediaError("Calibration train/validation includes held-out Gaussian frames or overlaps")
    start = recipe.get("bundle_source_start_seconds")
    tolerance = recipe.get("max_alignment_error_ms", 35)
    if not isinstance(start, (float, int)) or not np.isfinite(start) or not isinstance(tolerance, (float, int)) or not 0 < tolerance <= 500 / fps:
        raise MediaError("Declare a finite source clock and tolerance no greater than half a FLAME frame")
    runs = recipe.get("observation_runs")
    if not isinstance(runs, list) or not runs:
        raise MediaError("At least one reviewed observations run is required")
    allowed = set(train) | set(validation)
    paired: dict[int, tuple[float, dict]] = {}
    inputs = []
    excluded = {"outside_split": 0, "alignment": 0, "unselected_face": 0, "duplicate_frame": 0}
    for run in runs:
        manifest_path = _path(path.parent, run.get("manifest"))
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != "observations.v1" or manifest.get("source_id") != recipe["source_id"] or manifest.get("source_timeline") != recipe["source_timeline"]:
            raise MediaError("Observation source/schema/clock does not match the recipe")
        artifact = manifest["observations"]
        observations = _path(manifest_path.parent, artifact.get("path"))
        if not observations.is_relative_to(manifest_path.parent) or sha256(observations) != artifact.get("sha256"):
            raise MediaError("Observation file differs from its manifest")
        review_path = _path(path.parent, run.get("review"))
        review = load_review(review_path, artifact["sha256"], False)
        inputs.append({"observations_sha256": artifact["sha256"], "review_sha256": sha256(review_path), "reviewer": review["reviewer"], "track_id": review["track_id"]})
        previous = -float("inf")
        with observations.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") != "observation.v1" or row.get("source_id") != recipe["source_id"] or row.get("source_timeline") != recipe["source_timeline"]:
                    raise MediaError("An observation changed source or schema")
                local_time, source_time = float(row["timestamp_seconds"]), float(row["source_timestamp_seconds"])
                if not np.isfinite([local_time, source_time]).all() or local_time <= previous or abs(source_time - local_time - float(manifest["source_start_seconds"])) > 0.002:
                    raise MediaError("Observation clocks are inconsistent or unordered")
                previous = local_time
                index = round((source_time - start) * fps)
                if index not in allowed:
                    excluded["outside_split"] += 1
                    continue
                error = abs(start + index / fps - source_time)
                if error * 1000 > tolerance:
                    excluded["alignment"] += 1
                    continue
                selection = next((item for item in review["selections"] if item["start_ms"] <= local_time * 1000 < item["end_ms"]), None)
                face = next((item for item in row.get("faces", []) if selection and selection.get("face_index") is not None and item.get("detection_index") == selection["face_index"]), None)
                if not face:
                    excluded["unselected_face"] += 1
                    continue
                values = {name: float(value) for name, value in face.get("blendshapes", {}).items() if name in FACE_CHANNELS}
                if not values or any(not np.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
                    raise MediaError("Face controls must be finite observed values in [0,1]")
                if index in paired:
                    excluded["duplicate_frame"] += 1
                    if paired[index][0] <= error:
                        continue
                paired[index] = (error, values)
    accepted_train, accepted_validation = [n for n in train if n in paired], [n for n in validation if n in paired]
    if len(accepted_train) < 16 or len(accepted_validation) < 4:
        raise MediaError("Need at least 16 aligned train frames and 4 disjoint validation frames")
    common = sorted(set.intersection(*(set(item[1]) for item in paired.values())))
    if "jawOpen" not in common or not any(name.startswith("brow") for name in common):
        raise MediaError("Paired observations need jaw and brow channels")
    def arrays(selected: list[int]):
        return np.array([[paired[n][1][name] for name in common] for n in selected], dtype=np.float64), np.asarray(bundle["m_expr"][selected], dtype=np.float64), np.asarray(bundle["m_pose"][selected, 6:9], dtype=np.float64)
    train_x, train_y, train_jaw = arrays(accepted_train)
    valid_x, valid_y, valid_jaw = arrays(accepted_validation)
    if not all(np.isfinite(a).all() for a in (train_x, train_y, train_jaw, valid_x, valid_y, valid_jaw)):
        raise MediaError("Nonfinite FLAME or observed targets")
    metadata = {
        "schema": "gaussian-expression-dataset.v1", "subject_id": "jensen-huang", "bundle_sha256": sha256(str(stem) + ".bin"),
        "bundle_header_sha256": sha256(str(stem) + ".json"), "n_expr": E, "n_frames": T, "fps": fps,
        "input_names": common, "gaussian_training_frame_indices": gaussian_train,
        "training_frame_indices": accepted_train, "validation_frame_indices": accepted_validation,
        "source_id": recipe["source_id"], "source_timeline": recipe["source_timeline"], "event_id": recipe["event_id"], "sequence_id": recipe["sequence_id"],
        "recipe_sha256": sha256(path), "observation_inputs": inputs, "excluded": excluded,
        "validation_scope": "within_sequence_disjoint_frames", "unseen_event_generalization": False,
    }
    _atomic_npz(Path(output_path), metadata=np.array(json.dumps(metadata, sort_keys=True)), train_x=train_x, train_y=train_y, train_jaw=train_jaw, validation_x=valid_x, validation_y=valid_y, validation_jaw=valid_jaw)
    return {"status": "prepared", "train_frames": len(accepted_train), "validation_frames": len(accepted_validation), "input_channels": len(common), "excluded": excluded, "sha256": sha256(output_path)}


def fit_driver(dataset_path: str | Path, checkpoint_path: str | Path, output_path: str | Path, *, ridge: float = 0.01, batch_size: int = 256, resume: bool = False, max_batches: int | None = None) -> dict:
    """Deterministic, resumable sufficient statistics then a bounded ridge solve.

    No optimizer or RNG is involved. Checkpoints preserve processed row count,
    normalized Gram/cross matrices, exact dataset hash and all fit settings.
    """
    dataset_path, checkpoint_path, output_path = Path(dataset_path), Path(checkpoint_path), Path(output_path)
    if not np.isfinite(ridge) or ridge <= 0 or type(batch_size) is not int or batch_size < 1 or (max_batches is not None and (type(max_batches) is not int or max_batches < 1)):
        raise MediaError("Use positive ridge, batch size, and optional batch limit")
    with np.load(dataset_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        X, Y, jaw, VX, VY = (np.asarray(data[k], dtype=np.float64) for k in ("train_x", "train_y", "train_jaw", "validation_x", "validation_y"))
    names = metadata.get("input_names", [])
    train, validation = metadata.get("training_frame_indices", []), metadata.get("validation_frame_indices", [])
    if metadata.get("schema") != "gaussian-expression-dataset.v1" or metadata.get("subject_id") != "jensen-huang" or not train or not validation or set(train) & set(validation) or not (set(train) | set(validation)) <= set(metadata.get("gaussian_training_frame_indices", [])):
        raise MediaError("Invalid subject or overlapping/held-out training data")
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0] or len(train) != X.shape[0] or X.shape[1] != len(names) or Y.shape[1] != metadata["n_expr"] or jaw.shape != (len(train), 3) or VX.shape != (len(validation), len(names)) or VY.shape != (len(validation), metadata["n_expr"]) or not all(np.isfinite(a).all() for a in (X, Y, jaw, VX, VY)):
        raise MediaError("Invalid paired calibration arrays")
    if len(train) < 16 or len(validation) < 4 or len(set(names)) != len(names) or any(name not in FACE_CHANNELS for name in names) or ((X < 0) | (X > 1)).any() or ((VX < 0) | (VX > 1)).any():
        raise MediaError("Insufficient or invalid named face observations")
    active = np.flatnonzero((X.std(0) > 1e-4) & (np.ptp(X, axis=0) >= 0.05))
    if not any(names[i].startswith("brow") for i in active) or "jawOpen" not in names:
        raise MediaError("No observable brow variation: collect expressive training frames")
    jaw_input = X[:, names.index("jawOpen")]
    if jaw_input.min() > 0.12 or jaw_input.max() < 0.2:
        raise MediaError("Calibration needs both closed and open mouth observations")
    selected_names = [names[i] for i in active]
    mean_x, scale_x, mean_y = X[:, active].mean(0), X[:, active].std(0), Y.mean(0)
    Z = (X[:, active] - mean_x) / scale_x
    C, E, N = len(active), Y.shape[1], len(X)
    signature = {"schema": "gaussian-expression-fit-checkpoint.v1", "dataset_sha256": sha256(dataset_path), "ridge": ridge, "batch_size": batch_size, "input_names": selected_names, "examples": N, "n_expr": E, "algorithm_revision": 1}
    xtx, xty, next_row = np.zeros((C, C)), np.zeros((C, E)), 0
    if checkpoint_path.exists():
        if not resume:
            raise MediaError("Checkpoint exists; use --resume for the exact same dataset and settings")
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            state = json.loads(str(saved["metadata"]))
            if state.get("signature") != signature:
                raise MediaError("Checkpoint dataset or fit settings changed")
            xtx, xty, next_row = np.array(saved["xtx"]), np.array(saved["xty"]), state["next_row"]
            if xtx.shape != (C, C) or xty.shape != (C, E) or type(next_row) is not int or not 0 <= next_row <= N or not np.isfinite(xtx).all() or not np.isfinite(xty).all():
                raise MediaError("Invalid calibration checkpoint statistics")
    elif resume:
        raise MediaError("No checkpoint exists to resume")
    batches = 0
    while next_row < N:
        end = min(N, next_row + batch_size)
        z, y = Z[next_row:end], Y[next_row:end] - mean_y
        xtx += z.T @ z
        xty += z.T @ y
        next_row = end
        _atomic_npz(checkpoint_path, metadata=np.array(json.dumps({"signature": signature, "next_row": next_row, "phase": "sufficient_statistics"}, sort_keys=True)), xtx=xtx, xty=xty, mean_x=mean_x, scale_x=scale_x, mean_y=mean_y)
        batches += 1
        if max_batches is not None and batches >= max_batches and next_row < N:
            return {"status": "checkpointed", "processed_examples": next_row, "total_examples": N, "checkpoint_sha256": sha256(checkpoint_path)}
    normalized = np.linalg.solve(xtx + np.eye(C) * ridge * N, xty)
    weights = (normalized / scale_x[:, None]).T
    bias = mean_y - weights @ mean_x
    lower, upper = np.quantile(Y, [0.005, 0.995], axis=0)
    input_upper = X[:, active].max(0)
    prediction = np.clip(np.clip(VX[:, active], 0, input_upper) @ weights.T + bias, lower, upper)
    validation_rmse = float(np.sqrt(np.mean((prediction - VY) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((mean_y - VY) ** 2)))
    eligible = validation_rmse < baseline_rmse
    gain = float(np.dot(jaw_input, jaw[:, 0]) / max(1e-9, np.dot(jaw_input, jaw_input)))
    gain = float(np.clip(gain, 0, 0.8))
    report = {
        "status": "fitted" if eligible else "rejected_validation", "method": "ridge_regression", "processed_examples": N, "optimizer_updates": 0,
        "validation_frames": len(validation), "validation_scope": metadata["validation_scope"], "unseen_event_generalization": False,
        "coefficient_rmse": validation_rmse, "constant_baseline_rmse": baseline_rmse, "eligible_for_runtime": eligible,
        "active_channels": selected_names, "unsupported_or_constant_channels": [name for name in names if name not in selected_names],
        "dataset_sha256": signature["dataset_sha256"], "checkpoint_sha256": sha256(checkpoint_path),
    }
    _atomic_json(output_path.with_suffix(".report.json"), report)
    if not eligible:
        return report
    driver = {
        "schema": "gaussian-expression-driver.v1", "subject_id": "jensen-huang", "bundle_sha256": metadata["bundle_sha256"], "n_expr": E,
        "input_names": selected_names, "input_bounds": [[0.0, float(value)] for value in input_upper], "weights": weights.tolist(), "bias": bias.tolist(),
        "output_bounds": np.stack([lower, upper], axis=1).tolist(), "neutral_expr": Y[jaw_input <= 0.12].mean(0).tolist(), "jaw_open_radians": gain,
        "training": {"method": "ridge_regression", "dataset_sha256": signature["dataset_sha256"], "examples": N, "frame_indices": train, "validation_frame_indices": validation},
        "validation": report,
    }
    _atomic_json(output_path, driver)
    return {**report, "driver_sha256": sha256(output_path)}
