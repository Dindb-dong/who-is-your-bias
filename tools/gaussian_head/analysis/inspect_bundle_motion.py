"""Inspect a private bundle without exporting coefficients, vertices or images.

Run from the repository root with Python/NumPy and Node >= 22.18:
  python -m tools.gaussian_head.analysis.inspect_bundle_motion \
    --bundle apps/web/public/models/gaussian-head/jensen_a_c1c3_240k_sh \
    --train-end 2324 --expected-binary-sha256 SHA256 --output REPORT.json

Only the explicitly declared training slice is used for motion statistics and
forward checks. The production parser checks the entire buffer layout, while
the production validator sees unchanged static arrays and sliced training motion.
Held-out motion values are not validated, selected, summarized or fitted here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from tools.gaussian_head.bundle.head_bundle import (
    flame_forward,
    gaussian_centers,
    read_bundle,
)


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "apps/web/src/avatar/gaussian/gaussianHeadCore.ts"
REFERENCE = ROOT / "tools/gaussian_head/bundle/head_bundle.py"
KNOWN_C1C3_SHA256 = "32d5bc15901a8673901c6234ae7af946d2bfcf2c0aa5e7ed9c816d1c3aff599c"
FORWARD_TOLERANCE_M = 1e-4

# Import the actual production core. Private NumPy reference coordinates travel
# over stdin only; stdout and the saved report contain aggregate errors only.
NODE_CHECK = r"""
import { readFileSync } from 'node:fs';
const request = JSON.parse(readFileSync(0, 'utf8'));
const { parseBundle, validateGaussianBundle, FlameHead, gaussianCenters } =
  await import(request.core_url);
const header = JSON.parse(readFileSync(request.stem + '.json', 'utf8'));
const raw = readFileSync(request.stem + '.bin');
const B = parseBundle(header, raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const [start, end] = request.train_range;
// Check full motion shapes as metadata, without reading held-out values.
for (const [name, width] of [['m_expr', header.n_expr], ['m_pose', 15], ['m_trans', 3]]) {
  if (header.buffers[name]?.shape.join(',') !== [header.n_frames, width].join(',') ||
      B[name]?.length !== header.n_frames * width) throw new Error('Invalid full motion shape: ' + name);
}
const trainingHeader = {...header, n_frames: end - start, buffers: {...header.buffers}};
const trainingB = {...B};
for (const [name, width] of [['m_expr', header.n_expr], ['m_pose', 15], ['m_trans', 3]]) {
  trainingHeader.buffers[name] = {...header.buffers[name], shape: [end - start, width]};
  trainingB[name] = B[name].subarray(start * width, end * width);
}
validateGaussianBundle(trainingHeader, trainingB);
const head = new FlameHead(header, B);
function compare(actual, expected) {
  if (actual.length !== expected.length) throw new Error('Forward reference shape mismatch');
  let square = 0, maximum = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) throw new Error('Nonfinite forward result');
    const error = Math.abs(actual[i] - expected[i]);
    square += error * error;
    maximum = Math.max(maximum, error);
  }
  return {max_component_error_m: maximum, rms_component_error_m: Math.sqrt(square / actual.length)};
}
const frames = [];
for (const reference of request.references) {
  const frame = reference.frame;
  if (!Number.isSafeInteger(frame) || frame < start || frame >= end) throw new Error('Forward frame outside training range');
  const verts = head.forward(B.m_expr.subarray(frame * header.n_expr, (frame + 1) * header.n_expr),
    B.m_pose.subarray(frame * 15, (frame + 1) * 15), B.m_trans.subarray(frame * 3, (frame + 1) * 3));
  const faceData = head.updateFaceFrames();
  const centers = gaussianCenters(B, faceData);
  for (const values of [faceData, centers]) {
    for (const value of values) if (!Number.isFinite(value)) throw new Error('Nonfinite face frame or Gaussian center');
  }
  const selectedCenters = request.gaussian_indices.flatMap(index =>
    Array.from(centers.subarray(index * 3, index * 3 + 3)));
  frames.push({frame, vertices: compare(verts, reference.vertices),
    gaussian_centers: compare(selectedCenters, reference.centers),
    all_vertices_face_frames_and_gaussian_centers_finite: true});
}
console.log(JSON.stringify({node_version: process.version, layout_valid: true,
  static_and_training_arrays_valid: true, frames}));
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
    }


def inspect(stem: Path, start: int, end: int, expected_sha256: str) -> dict:
    binary = stem.with_suffix(".bin")
    manifest = stem.with_suffix(".json")
    initial_hashes = {"binary_sha256": digest(binary), "manifest_sha256": digest(manifest)}
    if initial_hashes["binary_sha256"] != expected_sha256:
        raise ValueError("Bundle binary SHA-256 does not match the explicit expected input")
    header, arrays = read_bundle(str(stem))
    if not 0 <= start < end <= header["n_frames"] or end - start < 2:
        raise ValueError("Declare at least two available training frames as a half-open range")
    if expected_sha256 == KNOWN_C1C3_SHA256 and (header["n_frames"] != 3319 or end > 2324):
        raise ValueError("Known Jensen c1c3 bundle permits training inspection only within [0,2324)")

    # Copy only the training rows into calculation arrays. Never summarize the
    # held-out tail, including for choosing the uniformly spaced check frames.
    expression = arrays["m_expr"][start:end].astype(np.float64)
    pose = arrays["m_pose"][start:end].astype(np.float64).reshape(-1, 5, 3)
    translation = arrays["m_trans"][start:end].astype(np.float64)
    frames = np.unique(np.linspace(start, end - 1, min(5, end - start), dtype=int)).tolist()
    gaussian_indices = np.unique(np.linspace(0, header["n_gauss"] - 1, min(128, header["n_gauss"]), dtype=int))
    references = []
    for frame in frames:
        vertices = flame_forward(arrays, expression[frame - start], pose[frame - start].ravel(), translation[frame - start])
        centers = gaussian_centers(arrays, vertices)
        if not np.isfinite(vertices).all() or not np.isfinite(centers).all():
            raise ValueError("Nonfinite NumPy forward result")
        references.append({"frame": frame, "vertices": vertices.ravel().tolist(),
                           "centers": centers[gaussian_indices].ravel().tolist()})

    process = subprocess.run(
        ["node", "--input-type=module", "--eval", NODE_CHECK],
        input=json.dumps({"core_url": CORE.as_uri(), "stem": str(stem),
                          "train_range": [start, end], "references": references,
                          "gaussian_indices": gaussian_indices.tolist()}, allow_nan=False),
        text=True, capture_output=True, check=True, timeout=60,
    )
    forward = json.loads(process.stdout)
    forward["selection"] = "up to five uniformly spaced training frames; no motion-dependent selection"
    forward["vertices_compared_per_frame"] = header["n_verts"]
    forward["gaussian_centers_compared_per_frame"] = int(gaussian_indices.size)
    forward["gaussian_centers_checked_finite_per_frame"] = header["n_gauss"]
    forward["tolerance_m"] = FORWARD_TOLERANCE_M
    forward["pass"] = all(
        frame[kind]["max_component_error_m"] <= FORWARD_TOLERANCE_M
        for frame in forward["frames"] for kind in ["vertices", "gaussian_centers"]
    )

    coefficient_std = expression.std(axis=0)
    expression_steps = np.linalg.norm(np.diff(expression, axis=0), axis=1)
    centered = expression - expression.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    basis = arrays["expr_dirs"].astype(np.float64)
    # Exact whole-mesh expression-only variance from covariance and the basis
    # Gram matrix. This has no anatomical region or eyebrow interpretation.
    displacement_rms_m = float(np.sqrt(max(0.0, np.sum(covariance * (basis @ basis.T)) / header["n_verts"])))
    joints = {}
    for joint, name in enumerate(["global", "neck", "jaw", "left_eye", "right_eye"]):
        values = pose[:, joint]
        joints[name] = {
            "axis_angle_norm_degrees": summary(np.rad2deg(np.linalg.norm(values, axis=1))),
            "component_std_degrees": np.rad2deg(values.std(axis=0)).tolist(),
            "component_range_degrees": np.rad2deg(np.ptp(values, axis=0)).tolist(),
            "consecutive_axis_angle_vector_delta_degrees": summary(np.rad2deg(np.linalg.norm(np.diff(values, axis=0), axis=1))),
            "exact_zero_vector_fraction": float(np.mean(np.all(values == 0, axis=1))),
        }
    final_hashes = {"binary_sha256": digest(binary), "manifest_sha256": digest(manifest)}
    if final_hashes != initial_hashes:
        raise ValueError("Input bytes changed during inspection")
    return {
        "schema": "gaussian-head-motion-inspection.v1",
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": {"stem": stem.name, **initial_hashes, "binary_bytes": binary.stat().st_size,
                   "manifest_bytes": manifest.stat().st_size, "input_bytes_unchanged": True},
        "implementation": {
            "production_core": str(CORE.relative_to(ROOT)), "production_core_sha256": digest(CORE),
            "numpy_reference": str(REFERENCE.relative_to(ROOT)), "numpy_reference_sha256": digest(REFERENCE),
        },
        "dimensions": {key: header[key] for key in ["n_verts", "n_faces", "n_expr", "n_gauss", "n_frames", "fps", "parents", "iteration", "sh_baked"] if key in header},
        "scope": {
            "training_frame_range_half_open": [start, end], "analyzed_motion_frames": end - start,
            "excluded_motion_frames": header["n_frames"] - (end - start),
            "full_bundle_layout_checked": True,
            "finite_value_validation": "all static arrays and declared training motion only",
            "held_out_motion_values_inspected": False, "fitting_performed": False,
            "raw_coefficients_vertices_and_images_exported": False,
        },
        "training_expression": {
            "coefficient_std_population": summary(coefficient_std),
            "coefficients_with_std_above_1e_minus_6": int(np.sum(coefficient_std > 1e-6)),
            "consecutive_coefficient_vector_delta_l2": summary(expression_steps),
            "fraction_consecutive_deltas_above_1e_minus_8": float(np.mean(expression_steps > 1e-8)),
            "whole_mesh_expression_deviation_rms_m": displacement_rms_m,
            "old_idle_scalar_0_25_expression_deviation_rms_m": displacement_rms_m * 0.25,
            "displacement_measure": "RMS Euclidean expression-only vertex displacement from training mean before pose correctives/skinning; all vertices; covariance evaluated exactly",
            "interpretation": "Nonconstant latent expression data establishes available motion, not eyebrow amplitude, semantic correspondence, identity fidelity or naturalness.",
        },
        "training_joint_poses": joints,
        "training_translation": {
            "component_std_m": translation.std(axis=0).tolist(),
            "consecutive_vector_delta_m": summary(np.linalg.norm(np.diff(translation, axis=0), axis=1)),
        },
        "production_numpy_forward_comparison": forward,
        "limits": [
            "A head bundle has no articulated torso, arms or fingers; its joint hierarchy is root, neck, jaw and two eyes.",
            "FLAME expression basis indices are latent coordinates, not named brow, blink or mouth labels.",
            "The provided JSON/bin bundle contains no paired semantic face observations or source video.",
            "The exporter's constant fps=15 is not proof of original source frame-0 time, frame sampling, edits or dropped frames.",
            "Semantic calibration still requires the matching jensen_a source sequence or reviewed observations and its verified source-clock mapping; unrelated interview footage cannot supply paired targets.",
            "Numerical agreement checks the production geometry math only; it does not assess rendered face/body seams, likeness, naturalness, or unseen motion.",
            "The old 0.25 scalar comparison is algebraic expression-basis displacement, not a measured fourfold eyebrow or rendered-motion improvement.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, required=True)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # A mistyped output path must never overwrite the source bundle.
    stem = args.bundle.resolve()
    output = args.output.resolve()
    if output in [stem.with_suffix(".bin"), stem.with_suffix(".json")]:
        parser.error("The report must not overwrite a bundle file")
    report = inspect(stem, args.train_start, args.train_end, args.expected_binary_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output), "forward_pass": report["production_numpy_forward_comparison"]["pass"],
                      "analyzed_motion_frames": report["scope"]["analyzed_motion_frames"]}))
    if not report["production_numpy_forward_comparison"]["pass"]:
        raise SystemExit("Production/reference geometry mismatch; see aggregate report")


if __name__ == "__main__":
    main()
