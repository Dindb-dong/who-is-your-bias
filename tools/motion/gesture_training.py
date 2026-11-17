"""Fit a small, source-bound temporal gesture prior from observed joint rotations.

This is a masked low-rank kinematic fit, not an audio/text-conditioned personal
gesture model. It neither invents observations for occluded joints nor promotes
source review. Generated rotations are explicitly experimental geometry previews.
No downloads are performed. Source recordings and person-specific weights stay
in ignored/private paths; the plan and measured receipt can be versioned.

Example (from the repository root)::

  python -m tools.motion.gesture_training prepare \
    data/motion/jensen-gesture-training.v1.json data/local/jensen-gesture
  python -m tools.motion.gesture_training train \
    data/local/jensen-gesture --output apps/web/public/models/gesture/jensen-gesture-basis.json

Resume uses the same command with --resume. Checkpoints are written atomically
per group; the immediately preceding checkpoint is retained as .previous.json.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bias.motion import FINGER_CHANNELS, MotionError, validate_clip

VERSION = "1.0.0"
GROUPS = {
    "left_arm": ["leftUpperArm", "leftForearm", "leftHand"],
    "right_arm": ["rightUpperArm", "rightForearm", "rightHand"],
    "left_fingers": sorted(name for name in FINGER_CHANNELS if name.startswith("left")),
    "right_fingers": sorted(name for name in FINGER_CHANNELS if name.startswith("right")),
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path, maximum: int = 16 * 1024**2) -> dict:
    if not path.is_file() or path.stat().st_size > maximum:
        raise MotionError(f"Missing or oversized input: {path.name}")
    result = json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(MotionError("Nonfinite JSON")))
    if not isinstance(result, dict):
        raise MotionError("Expected a JSON object")
    return result


def atomic_json(path: Path, value: dict, *, previous: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if previous and path.exists():
        os.replace(path, path.with_suffix(".previous.json"))
    os.replace(temporary, path)


def bounded_path(root: Path, relative: str) -> Path:
    component = Path(relative)
    result = (root / component).resolve()
    if component.is_absolute() or ".." in component.parts or not result.is_relative_to(root):
        raise MotionError("Input paths must remain inside the declared root")
    return result


def quaternion_log(value: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(value, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-12:
        raise MotionError("Expected a finite nonzero XYZW quaternion")
    q = q / norm
    if q[3] < 0:
        q = -q
    length = np.linalg.norm(q[:3])
    return q[:3] * (2 * math.atan2(length, q[3]) / length) if length > 1e-12 else np.zeros(3)


def quaternion_exp(vector: np.ndarray) -> list[float]:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [*(vector * (math.sin(angle / 2) / angle)).tolist(), math.cos(angle / 2)]


def relative_rotation(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    """reference^-1 * value, matching local-delta rig quaternion order."""
    reference = reference / np.linalg.norm(reference)
    value = value / np.linalg.norm(value)
    av, aw = -reference[:3], reference[3]
    bv, bw = value[:3], value[3]
    return np.r_[aw * bv + bw * av + np.cross(av, bv), aw * bw - np.dot(av, bv)]


def sample_rotation(clip: dict, channel: str, at_ms: float) -> np.ndarray | None:
    """Interpolate only two present observations separated by a permitted gap."""
    frames = clip["frames"]
    times = [row["time_ms"] for row in frames]
    pos = bisect.bisect_right(times, at_ms)
    if pos == 0:
        return None
    left = frames[pos - 1]
    lq = left.get("bones", {}).get(channel)
    if lq is None:
        return None
    if abs(left["time_ms"] - at_ms) < 1e-6:
        return np.asarray(lq, dtype=float)
    if pos == len(frames):
        return None
    right = frames[pos]
    rq = right.get("bones", {}).get(channel)
    if rq is None or right["time_ms"] - left["time_ms"] > clip["max_gap_ms"]:
        return None
    alpha = (at_ms - left["time_ms"]) / (right["time_ms"] - left["time_ms"])
    a, b = np.asarray(lq, dtype=float), np.asarray(rq, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + alpha * (b - a)
    else:
        angle = math.acos(min(1.0, dot))
        q = (math.sin((1 - alpha) * angle) * a + math.sin(alpha * angle) * b) / math.sin(angle)
    return q / np.linalg.norm(q)


def prepare_windows(clip: dict, channels: list[str], *, window_ms: int, samples: int, coverage: float) -> tuple[list[dict], list[dict]]:
    accepted, excluded = [], []
    for start in range(0, int(clip["duration_ms"] - window_ms) + 1, window_ms):
        values = np.zeros((samples, len(channels), 3), dtype=float)
        mask = np.zeros_like(values, dtype=bool)
        reasons = []
        for c, channel in enumerate(channels):
            series = [sample_rotation(clip, channel, start + i * window_ms / samples) for i in range(samples)]
            observed = [i for i, q in enumerate(series) if q is not None]
            if len(observed) / samples < coverage:
                reasons.append(f"{channel}:low_coverage")
                continue
            # A late observation cannot define an unobserved gesture onset.
            if observed[0] * window_ms / samples > clip["max_gap_ms"]:
                reasons.append(f"{channel}:missing_onset")
                continue
            reference = series[observed[0]]
            for i in observed:
                values[i, c] = quaternion_log(relative_rotation(reference, series[i]))
                mask[i, c] = True
        if reasons:
            excluded.append({"start_ms": start, "reasons": reasons})
        else:
            accepted.append({"start_ms": start, "values": values.reshape(-1).tolist(), "observed": mask.reshape(-1).tolist(), "coverage": float(mask.mean())})
    return accepted, excluded


def prepare(plan_path: Path, run_dir: Path, *, repo_root: Path) -> dict:
    plan = read_json(plan_path, 128 * 1024)
    if plan.get("schema_version") != "gesture-training-plan.v1" or plan.get("persona_id") != "jensen-huang":
        raise MotionError("Expected a Jensen gesture-training-plan.v1")
    partitions = plan.get("event_partitions", {})
    if not partitions or not set(partitions.values()) <= {"train", "validation", "test"} or "train" not in partitions.values():
        raise MotionError("Declare exact event partitions including training events")
    if not (500 <= plan["window_ms"] <= 5000 and 5 <= plan["sample_count"] <= 100 and 0.5 <= plan["minimum_coverage"] <= 1):
        raise MotionError("Unsupported sampling plan")
    manifest_path = bounded_path(repo_root, plan["source_manifest"])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != plan["source_manifest_sha256"]:
        raise MotionError("Source manifest hash changed; freeze a new plan")
    manifest = read_json(manifest_path)
    if manifest.get("persona_id") != plan["persona_id"] or manifest.get("schema_version") != "motion-input.v1":
        raise MotionError("Source manifest persona/schema mismatch")
    sources = {source["id"]: source for source in manifest["sources"]}
    if len(sources) != len(manifest["sources"]):
        raise MotionError("Duplicate source IDs")
    if set(partitions) != {source["source_event_id"] for source in sources.values()}:
        raise MotionError("Partition every source event exactly once")
    for source in sources.values():
        if partitions[source["source_event_id"]] == "train" and (source["corpus_split"] != "train" or source["motion_split"] != "train"):
            raise MotionError("An existing held-out event cannot become training data")
    dataset = {"schema_version": "gesture-dataset.v1", "plan": plan, "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(), "groups": {}, "clips": []}
    groups = {name: {"channels": channels, "windows": [], "excluded": []} for name, channels in GROUPS.items()}
    clip_ids = set()
    for record in manifest["clips"]:
        if record["id"] in clip_ids:
            raise MotionError("Duplicate clip ID")
        clip_ids.add(record["id"])
        path = bounded_path((repo_root / "apps/web/public").resolve(), record["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise MotionError("Source clip hash changed")
        clip = read_json(path, 8 * 1024**2)
        validate_clip(clip)
        if clip["source"]["source_id"] != record["source_id"]:
            raise MotionError("Source clip identity mismatch")
        source = sources[record["source_id"]]
        event = source["source_event_id"]
        identity = {"clip_id": record["id"], "clip_sha256": record["sha256"], "source_event_id": event, "split": partitions[event]}
        dataset["clips"].append({**identity, "observed_frames": len(clip["frames"]), "duration_ms": clip["duration_ms"], "source": clip["source"], "source_url": source["url"], "review": clip["review"]})
        for name, group in groups.items():
            windows, excluded = prepare_windows(clip, group["channels"], window_ms=plan["window_ms"], samples=plan["sample_count"], coverage=plan["minimum_coverage"])
            group["windows"].extend({**identity, **window} for window in windows)
            group["excluded"].extend({**identity, **row} for row in excluded)
    dataset["groups"] = groups
    dataset["dataset_sha256"] = digest(dataset)
    destination = run_dir / "dataset.json"
    if destination.exists():
        existing = read_json(destination)
        if existing != dataset:
            raise MotionError("Prepared run already exists with different inputs; use a new directory")
    else:
        atomic_json(destination, dataset)
    summary = {"dataset_sha256": dataset["dataset_sha256"], "source_events": partitions, "source_clips": len(dataset["clips"]), "observed_source_frames": sum(c["observed_frames"] for c in dataset["clips"]), "groups": {name: {"accepted_train_windows": sum(w["split"] == "train" for w in group["windows"]), "accepted_validation_windows": sum(w["split"] == "validation" for w in group["windows"]), "excluded_windows": len(group["excluded"])} for name, group in groups.items()}}
    atomic_json(run_dir / "preparation.json", summary)
    return summary


def _arrays(windows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([window["values"] for window in windows], dtype=float), np.asarray([window["observed"] for window in windows], dtype=bool)


def solve_latent(values: np.ndarray, observed: np.ndarray, mean: np.ndarray, basis: np.ndarray, ridge: float) -> np.ndarray:
    latent = np.zeros((len(values), len(basis)))
    for i, (row, mask) in enumerate(zip(values, observed)):
        if not mask.any():
            continue
        b = basis[:, mask]
        latent[i] = np.linalg.solve(b @ b.T + np.eye(len(basis)) * ridge, b @ (row[mask] - mean[mask]))
    return latent


def fit_basis(values: np.ndarray, observed: np.ndarray, *, rank: int, iterations: int, ridge: float, seed: int, checkpoint: Path | None = None, binding: str = "", checkpoint_every: int = 10, resume: bool = False) -> dict:
    """Alternating ridge least squares. Missing entries never enter the loss."""
    if values.ndim != 2 or observed.shape != values.shape or len(values) < 3 or not np.isfinite(values).all():
        raise MotionError("At least three finite masked training windows are required")
    if not (1 <= rank <= min(8, len(values) - 1) and 1 <= iterations <= 500 and 0 < ridge <= 1 and 1 <= checkpoint_every <= 100):
        raise MotionError("Invalid bounded fitting parameters")
    support = observed.sum(axis=0)
    usable = support >= 2
    observed = observed & usable[None, :]
    if not observed.any():
        raise MotionError("Training windows share no supported observations")
    configuration = {"rank": rank, "ridge": ridge, "seed": seed, "shape": list(values.shape), "binding": binding, "implementation_version": VERSION, "data_sha256": digest({"values": values.tolist(), "observed": observed.tolist()})}
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise MotionError("Resume checkpoint is missing")
        saved = read_json(checkpoint)
        if saved.get("configuration") != configuration:
            raise MotionError("Resume dataset/configuration mismatch")
        mean, basis, latent = (np.asarray(saved[key], dtype=float) for key in ("mean", "basis", "latent"))
        completed, history = saved["iterations_completed"], saved["loss_history"]
        if completed > iterations:
            raise MotionError("Requested iteration precedes saved checkpoint")
    else:
        if checkpoint is not None and checkpoint.exists():
            raise MotionError("Checkpoint exists; use --resume or a new run")
        mean = np.divide(np.where(observed, values, 0).sum(axis=0), observed.sum(axis=0), out=np.zeros(values.shape[1]), where=observed.sum(axis=0) > 0)
        centered = np.where(observed, values - mean, 0.0)
        # Mean filling initializes factors only; its entries are excluded from every update/loss.
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:rank].copy()
        latent = np.zeros((len(values), rank))
        completed, history = 0, []
    for step in range(completed, iterations):
        latent = solve_latent(values, observed, mean, basis, ridge)
        design = np.column_stack([np.ones(len(values)), latent])
        penalty = np.diag([1e-10, *([ridge] * rank)])
        for j in range(values.shape[1]):
            mask = observed[:, j]
            if mask.sum() < 2:
                mean[j], basis[:, j] = 0.0, 0.0
                continue
            a = design[mask]
            weights = np.linalg.solve(a.T @ a + penalty, a.T @ values[mask, j])
            mean[j], basis[:, j] = weights[0], weights[1:]
        residual = (mean + latent @ basis - values)[observed]
        loss = float(np.mean(residual**2))
        history.append({"iteration": step + 1, "observed_mse": loss})
        if checkpoint is not None and ((step + 1) % checkpoint_every == 0 or step + 1 == iterations):
            atomic_json(checkpoint, {"schema_version": "gesture-checkpoint.v1", "configuration": configuration, "iterations_completed": step + 1, "mean": mean.tolist(), "basis": basis.tolist(), "latent": latent.tolist(), "loss_history": history, "optimizer": "deterministic_alternating_ridge_least_squares", "random_state": "No stochastic updates; SVD initialization is deterministic for this runtime."}, previous=True)
    return {"mean": mean, "basis": basis, "latent": latent, "support": support, "usable": usable, "iterations_completed": iterations, "loss_history": history}


def evaluate_fit(fit: dict, windows: list[dict], ridge: float) -> dict:
    if not windows:
        return {"windows": 0, "status": "no_compatible_windows"}
    values, observed = _arrays(windows)
    observed &= fit["usable"][None, :]
    if not observed.any():
        return {"windows": len(windows), "status": "no_supported_coordinates"}
    latent = solve_latent(values, observed, fit["mean"], fit["basis"], ridge)
    reconstruction = fit["mean"] + latent @ fit["basis"]
    error = lambda prediction: float(np.sqrt(np.mean((prediction - values)[observed] ** 2)))
    return {"windows": len(windows), "observed_coordinates": int(observed.sum()), "neutral_rmse_rad": error(np.zeros_like(values)), "mean_rmse_rad": error(np.broadcast_to(fit["mean"], values.shape)), "reconstruction_rmse_rad": error(reconstruction), "source_events": sorted({w["source_event_id"] for w in windows}), "status": "masked_reconstruction_only", "scope": "Latent codes are inferred from each evaluation window. This measures representation reconstruction, not causal gesture prediction or perceptual quality."}


def train(run_dir: Path, output: Path, *, resume: bool = False, iterations: int | None = None) -> dict:
    dataset = read_json(run_dir / "dataset.json", 32 * 1024**2)
    recorded_digest = dataset.pop("dataset_sha256", None)
    if recorded_digest != digest(dataset):
        raise MotionError("Prepared dataset digest mismatch")
    dataset["dataset_sha256"] = recorded_digest
    plan = dataset["plan"]
    steps = plan["iterations"] if iterations is None else iterations
    model = {"schema_version": "gesture-basis.v1", "implementation_version": VERSION, "id": plan["id"], "persona_id": plan["persona_id"], "review_status": "geometry_preview", "coordinate_basis": "right_handed_y_up", "rotation_space": "local_delta", "sample_count": plan["sample_count"], "duration_ms": plan["window_ms"], "flattening": "time_channel_xyz", "dataset_sha256": recorded_digest, "groups": {}, "limitations": ["Experimental learned geometry; source speaker association is unreviewed.", "No audio/text-conditioned policy, gesture semantics or learned torso motion.", "Validation events are separate for this fit but were previously exposed in repository development.", "Local joint deltas are rebased at each observed window onset; monocular geometry omits axial twist."]}
    report = {"schema_version": "gesture-training-result.v1", "implementation_version": VERSION, "run_id": plan["id"], "persona_id": plan["persona_id"], "dataset_sha256": recorded_digest, "source_manifest_sha256": plan["source_manifest_sha256"], "event_partitions": plan["event_partitions"], "source_clips": dataset["clips"], "algorithm": "masked_alternating_ridge_low_rank_temporal_basis", "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": sys.platform, "device": "cpu"}, "groups": {}, "limitations": model["limitations"], "quality_acceptance": {"human_motion_review": False, "audio_conditioned_generation": False, "semantic_gesture_prediction": False, "unseen_person_generalization": False}}
    for name, group in dataset["groups"].items():
        training = [w for w in group["windows"] if w["split"] == "train"]
        validation = [w for w in group["windows"] if w["split"] != "train"]
        if len(training) < 3:
            report["groups"][name] = {"status": "insufficient_training_windows", "train_windows": len(training), "excluded_windows": group["excluded"]}
            continue
        values, observed = _arrays(training)
        rank = min(plan["rank"], len(training) - 1)
        checkpoint = run_dir / (name + ".checkpoint.json")
        fit = fit_basis(values, observed, rank=rank, iterations=steps, ridge=plan["ridge"], seed=plan["seed"], checkpoint=checkpoint, binding=recorded_digest + ":" + name, checkpoint_every=plan["checkpoint_every"], resume=resume and checkpoint.exists())
        # Runtime limits are engineering caps informed only by training values.
        vector_lengths = np.linalg.norm(values.reshape(len(values), plan["sample_count"], len(group["channels"]), 3), axis=-1)
        joint_observed = observed.reshape(len(values), plan["sample_count"], len(group["channels"]), 3).all(axis=-1)
        cap = 0.65 if name.endswith("arm") else 0.8
        observed_lengths = vector_lengths[joint_observed]
        angle_limit = min(cap, max(0.05, float(np.quantile(observed_lengths, 0.95))))
        model["groups"][name] = {"channels": group["channels"], "rank": rank, "mean": fit["mean"].tolist(), "basis": fit["basis"].tolist(), "latent_examples": fit["latent"].tolist(), "latent_ids": [w["clip_id"] + ":" + str(w["start_ms"]) for w in training], "training_source_events": sorted({w["source_event_id"] for w in training}), "support_counts": fit["support"].tolist(), "max_angle_rad": angle_limit, "iterations_completed": steps}
        report["groups"][name] = {"status": "fitted_geometry_preview", "rank": rank, "iterations_completed": steps, "training": evaluate_fit(fit, training, plan["ridge"]), "validation": evaluate_fit(fit, validation, plan["ridge"]), "loss_history": fit["loss_history"], "excluded_windows": group["excluded"], "max_angle_rad": angle_limit, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "checkpoint_bytes": checkpoint.stat().st_size}
    if not model["groups"]:
        raise MotionError("No group had enough observations for fitting")
    atomic_json(output, model)
    report["model_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    report["model_bytes"] = output.stat().st_size
    atomic_json(run_dir / "training-result.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("plan", type=Path)
    prep.add_argument("run_dir", type=Path)
    prep.add_argument("--repo-root", type=Path, default=Path.cwd())
    fit = commands.add_parser("train")
    fit.add_argument("run_dir", type=Path)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--resume", action="store_true")
    fit.add_argument("--iterations", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.plan.resolve(), args.run_dir, repo_root=args.repo_root.resolve())
    else:
        report = train(args.run_dir, args.output, resume=args.resume, iterations=args.iterations)
        result = {"run_id": report["run_id"], "model_sha256": report["model_sha256"], "model_bytes": report["model_bytes"], "groups": {name: {"status": group["status"], "iterations_completed": group.get("iterations_completed"), "training": group.get("training"), "validation": group.get("validation")} for name, group in report["groups"].items()}}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
