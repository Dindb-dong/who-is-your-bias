"""Synthetic numeric fixtures validate calibration mechanics, not Jensen quality."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.gaussian_head.bundle.head_bundle import write_bundle
from tools.gaussian_head.expression.pipeline import fit_driver, prepare_dataset, sha256
from tools.media.common import MediaError


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    n = 40
    brow = np.array([(i % 8) / 7 for i in range(n)])
    jaw = np.array([((i * 3) % 7) / 10 for i in range(n)])
    expressions = np.stack([0.1 + brow * 0.7, -0.2 + jaw * 0.4], axis=1)
    poses = np.zeros((n, 15)); poses[:, 6] = jaw * 0.35
    write_bundle(str(tmp_path / "head"), {"m_expr": ("f32", expressions), "m_pose": ("f32", poses), "m_trans": ("f32", np.zeros((n, 3)))}, {"n_expr": 2, "n_frames": n, "fps": 10})
    observations = tmp_path / "observations.jsonl"
    rows = [{"schema_version": "observation.v1", "source_id": "synthetic-fixture", "source_timeline": "source-video", "timestamp_seconds": i / 10, "source_timestamp_seconds": 10 + i / 10,
             "faces": [{"detection_index": 0, "blendshapes": {"browInnerUp": float(brow[i]), "jawOpen": float(jaw[i]), "eyeBlinkLeft": 0}}]} for i in range(n)]
    observations.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"schema_version": "observations.v1", "source_id": "synthetic-fixture", "source_timeline": "source-video", "source_start_seconds": 10,
                "observations": {"path": observations.name, "sha256": sha256(observations)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    review = {"schema_version": "motion-review.v1", "observations_sha256": sha256(observations), "status": "speaker_reviewed", "track_continuity_reviewed": True, "source_clock_reviewed": True, "speaker_association_reviewed": True,
              "reviewer": "synthetic-test-only", "track_id": "fixture", "attribution_basis": "Synthetic fixture; no person observation or human review is claimed.",
              "selections": [{"start_ms": 0, "end_ms": 4000, "face_index": 0, "pose_index": None, "left_hand_index": None, "right_hand_index": None}]}
    (tmp_path / "review.json").write_text(json.dumps(review))
    recipe = {"schema": "gaussian-expression-preparation.v1", "subject_id": "jensen-huang", "bundle_stem": "head", "source_id": "synthetic-fixture", "source_timeline": "source-video", "event_id": "synthetic-event", "sequence_id": "synthetic-sequence",
              "bundle_source_start_seconds": 10, "gaussian_train_ranges": [[0, 40]], "fit_train_ranges": [[0, 32]], "validation_ranges": [[32, 40]], "observation_runs": [{"manifest": "manifest.json", "review": "review.json"}]}
    recipe_path = tmp_path / "recipe.json"; recipe_path.write_text(json.dumps(recipe))
    return recipe_path, tmp_path / "dataset.npz"


def test_prepare_alignment_and_disjoint_fit(tmp_path):
    recipe, dataset = fixture(tmp_path)
    prepared = prepare_dataset(recipe, dataset)
    assert prepared["train_frames"] == 32 and prepared["validation_frames"] == 8
    checkpoint, model = tmp_path / "checkpoint.npz", tmp_path / "driver.json"
    result = fit_driver(dataset, checkpoint, model, batch_size=8)
    assert result["status"] == "fitted"
    assert result["coefficient_rmse"] < result["constant_baseline_rmse"] * 0.05
    assert result["optimizer_updates"] == 0  # Closed-form fitted weights, not invented SGD updates.
    driver = json.loads(model.read_text())
    assert driver["bundle_sha256"] == sha256(tmp_path / "head.bin")
    assert driver["training"]["frame_indices"] == list(range(32))
    assert driver["training"]["validation_frame_indices"] == list(range(32, 40))
    assert driver["input_names"] == ["browInnerUp", "jawOpen"]
    assert driver["jaw_open_radians"] == pytest.approx(0.35)
    assert not result["unseen_event_generalization"]


def test_checkpoint_resume_is_exact_and_dataset_bound(tmp_path):
    recipe, dataset = fixture(tmp_path); prepare_dataset(recipe, dataset)
    paused = fit_driver(dataset, tmp_path / "resume.npz", tmp_path / "resumed.json", batch_size=8, max_batches=1)
    assert paused["status"] == "checkpointed" and paused["processed_examples"] == 8
    assert not (tmp_path / "resumed.json").exists()
    with pytest.raises(MediaError, match="settings changed"):
        fit_driver(dataset, tmp_path / "resume.npz", tmp_path / "resumed.json", batch_size=8, ridge=0.1, resume=True)
    resumed = fit_driver(dataset, tmp_path / "resume.npz", tmp_path / "resumed.json", batch_size=8, resume=True)
    full = fit_driver(dataset, tmp_path / "full.npz", tmp_path / "full.json", batch_size=8)
    assert resumed["coefficient_rmse"] == full["coefficient_rmse"]
    a, b = json.loads((tmp_path / "resumed.json").read_text()), json.loads((tmp_path / "full.json").read_text())
    assert a["weights"] == b["weights"] and a["bias"] == b["bias"]
    assert (tmp_path / "resume.npz.previous").exists()


@pytest.mark.parametrize("change", ["train_on_gaussian_test", "validation_on_gaussian_test", "overlap", "clock", "unreviewed", "changed_observations"])
def test_preparation_rejects_leakage_and_unverified_alignment(tmp_path, change):
    recipe_path, dataset = fixture(tmp_path)
    recipe = json.loads(recipe_path.read_text())
    if change == "train_on_gaussian_test":
        recipe["gaussian_train_ranges"] = [[0, 30], [32, 40]]
    elif change == "validation_on_gaussian_test":
        recipe["gaussian_train_ranges"] = [[0, 32]]
    elif change == "overlap":
        recipe["validation_ranges"] = [[30, 40]]
    elif change == "clock":
        recipe["source_timeline"] = "different-clock"
    elif change == "unreviewed":
        path = tmp_path / "review.json"; review = json.loads(path.read_text()); review["status"] = "geometry_preview"; path.write_text(json.dumps(review))
    elif change == "changed_observations":
        with (tmp_path / "observations.jsonl").open("a") as stream:
            stream.write("{}\n")
    recipe_path.write_text(json.dumps(recipe))
    with pytest.raises(MediaError):
        prepare_dataset(recipe_path, dataset)
    assert not dataset.exists()


def test_no_observed_brow_variation_is_not_a_trained_expression_claim(tmp_path):
    recipe, dataset = fixture(tmp_path); prepare_dataset(recipe, dataset)
    with np.load(dataset, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    metadata = json.loads(str(arrays["metadata"]))
    arrays["train_x"][:, metadata["input_names"].index("browInnerUp")] = 0
    np.savez_compressed(dataset, **arrays)
    with pytest.raises(MediaError, match="brow variation"):
        fit_driver(dataset, tmp_path / "fit.npz", tmp_path / "driver.json")


def test_fit_rechecks_calibration_validation_is_inside_gaussian_training(tmp_path):
    recipe, dataset = fixture(tmp_path); prepare_dataset(recipe, dataset)
    with np.load(dataset, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["gaussian_training_frame_indices"] = list(range(32))
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(dataset, **arrays)
    with pytest.raises(MediaError, match="held-out"):
        fit_driver(dataset, tmp_path / "fit.npz", tmp_path / "driver.json")


def test_rejected_validation_never_exports_a_runtime_driver(tmp_path):
    recipe, dataset = fixture(tmp_path); prepare_dataset(recipe, dataset)
    with np.load(dataset, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["validation_y"] = np.repeat(arrays["train_y"].mean(0, keepdims=True), 8, axis=0)
    np.savez_compressed(dataset, **arrays)
    result = fit_driver(dataset, tmp_path / "fit.npz", tmp_path / "driver.json")
    assert result["status"] == "rejected_validation"
    assert not (tmp_path / "driver.json").exists()
    assert (tmp_path / "driver.report.json").exists()
