from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bias.motion import MotionError
from tools.motion.gesture_training import (
    evaluate_fit, fit_basis, prepare, prepare_windows, quaternion_exp,
    quaternion_log, sample_rotation, train,
)


def source_clip() -> dict:
    return {
        "duration_ms": 2000,
        "max_gap_ms": 250,
        "frames": [
            {"time_ms": i * 100, "bones": {"leftForearm": quaternion_exp(np.array([0.1 * math.sin(i / 3), 0.02 * i, 0.0]))}}
            for i in range(20)
        ],
    }


def matrix() -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0, 2 * np.pi, 36)
    values = np.array([a * np.sin(times) + b * np.cos(times) for a, b in [(1, 0), (0.5, 0.4), (0.2, -0.7), (0.7, -0.2), (-0.2, 0.3)]])
    mask = np.ones_like(values, dtype=bool)
    mask[1, 7:10] = False
    mask[3, 14:17] = False
    return values, mask


def test_quaternion_sign_and_roundtrip_do_not_change_motion():
    vector = np.array([0.2, -0.3, 0.1])
    q = quaternion_exp(vector)
    np.testing.assert_allclose(quaternion_log(q), vector, atol=1e-12)
    np.testing.assert_allclose(quaternion_log([-x for x in q]), vector, atol=1e-12)
    with pytest.raises(MotionError):
        quaternion_log([0, 0, 0, 0])


def test_sampling_does_not_fill_missing_joints_or_bridge_cuts():
    clip = source_clip()
    assert sample_rotation(clip, "leftForearm", 150) is not None
    del clip["frames"][2]["bones"]["leftForearm"]
    assert sample_rotation(clip, "leftForearm", 150) is None
    assert sample_rotation(clip, "leftForearm", 200) is None
    clip["frames"] = [f for f in clip["frames"] if not 300 <= f["time_ms"] <= 600]
    assert sample_rotation(clip, "leftForearm", 450) is None
    assert sample_rotation(clip, "leftForearm", 2001) is None


def test_preparation_keeps_occlusions_masked_and_rebases_observed_onset():
    clip = source_clip()
    for i in [7, 8]:
        clip["frames"][i]["bones"] = {}
    windows, excluded = prepare_windows(clip, ["leftForearm"], window_ms=2000, samples=20, coverage=0.6)
    assert not excluded and len(windows) == 1
    observed = np.asarray(windows[0]["observed"]).reshape(20, 3)
    assert not observed[7:9].any()
    assert observed[0].all()
    np.testing.assert_allclose(windows[0]["values"][:3], [0, 0, 0], atol=1e-12)
    for i in range(4):
        clip["frames"][i]["bones"] = {}
    windows, excluded = prepare_windows(clip, ["leftForearm"], window_ms=2000, samples=20, coverage=0.6)
    assert not windows and "leftForearm:missing_onset" in excluded[0]["reasons"]


def test_missing_values_never_contribute_to_the_fit():
    values, mask = matrix()
    contaminated = values.copy()
    contaminated[~mask] = 10000
    args = dict(rank=2, iterations=4, ridge=0.01, seed=1729)
    first = fit_basis(values, mask, **args)
    second = fit_basis(contaminated, mask, **args)
    for name in ("mean", "basis", "latent"):
        np.testing.assert_allclose(first[name], second[name], atol=1e-12)
    assert first["loss_history"][-1]["observed_mse"] < 0.01


def test_save_load_resume_matches_uninterrupted_updates(tmp_path):
    values, mask = matrix()
    args = dict(rank=2, ridge=0.01, seed=1729, binding="source-hash")
    checkpoint = tmp_path / "left.checkpoint.json"
    fit_basis(values, mask, iterations=2, checkpoint=checkpoint, checkpoint_every=1, **args)
    resumed = fit_basis(values, mask, iterations=5, checkpoint=checkpoint, resume=True, **args)
    continuous = fit_basis(values, mask, iterations=5, **args)
    for name in ("mean", "basis", "latent"):
        np.testing.assert_allclose(resumed[name], continuous[name], atol=1e-12)
    assert resumed["loss_history"] == continuous["loss_history"]
    assert checkpoint.with_suffix(".previous.json").exists()
    with pytest.raises(MotionError, match="mismatch"):
        fit_basis(values + 0.1, mask, iterations=6, checkpoint=checkpoint, resume=True, **args)
    with pytest.raises(MotionError, match="use --resume"):
        fit_basis(values, mask, iterations=6, checkpoint=checkpoint, **args)


def test_validation_reconstruction_does_not_mutate_fitted_weights():
    values, mask = matrix()
    fit = fit_basis(values, mask, rank=2, iterations=3, ridge=0.01, seed=1729)
    before = {name: fit[name].copy() for name in ("mean", "basis", "latent")}
    windows = [{"values": (values[0] * 100).tolist(), "observed": mask[0].tolist(), "source_event_id": "held-out"}]
    measured = evaluate_fit(fit, windows, 0.01)
    assert measured["status"] == "masked_reconstruction_only"
    assert measured["source_events"] == ["held-out"]
    for name, original in before.items():
        np.testing.assert_array_equal(fit[name], original)


def test_real_plan_verifies_sources_and_preserves_event_boundary(tmp_path):
    root = Path(__file__).resolve().parents[2]
    plan = root / "data/motion/jensen-gesture-training.v1.json"
    summary = prepare(plan, tmp_path, repo_root=root)
    assert summary["source_clips"] == 6 and summary["observed_source_frames"] == 580
    data = json.loads((tmp_path / "dataset.json").read_text())
    assert len([c for c in data["clips"] if c["split"] == "validation"]) == 1
    for group in data["groups"].values():
        assert all(w["source_event_id"] != "wired-big-interview-2024" for w in group["windows"] if w["split"] == "train")
        assert "chest" not in group["channels"]
        assert group["excluded"]
    assert all(c["review"]["status"] == "geometry_preview" for c in data["clips"])
    assert prepare(plan, tmp_path, repo_root=root) == summary


def test_prepare_rejects_a_changed_source_manifest_before_fitting(tmp_path):
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / "data/motion/jensen-gesture-training.v1.json").read_text())
    plan["source_manifest_sha256"] = "0" * 64
    copied = tmp_path / "plan.json"
    copied.write_text(json.dumps(plan))
    with pytest.raises(MotionError, match="manifest hash"):
        prepare(copied, tmp_path / "run", repo_root=root)


def test_training_rejects_modified_prepared_bytes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    prepare(root / "data/motion/jensen-gesture-training.v1.json", tmp_path, repo_root=root)
    path = tmp_path / "dataset.json"
    dataset = json.loads(path.read_text())
    dataset["plan"]["rank"] = 1
    path.write_text(json.dumps(dataset))
    with pytest.raises(MotionError, match="dataset digest"):
        train(tmp_path, tmp_path / "model.json")
