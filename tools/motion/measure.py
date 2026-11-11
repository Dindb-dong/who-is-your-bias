"""Reproduce a bounded window plan using explicitly supplied local artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
from pathlib import Path

from bias.motion import MotionError
from tools.media.common import local_file, sha256
from tools.media.observe import observe_clip
from tools.media.prepare import prepare_clip


def measure_plan(plan_path: str | Path, media_root: str | Path, output_root: str | Path, *, face_model: str | Path, pose_model: str | Path) -> list[dict]:
    """No downloads, identities, review promotion, or telemetry shims.

    Runtime version is checked before importing MediaPipe. The separately
    audited legacy implementation is required for this particular experiment.
    """
    path = local_file(plan_path, max_bytes=2 * 1024**2)
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != "motion-measurement-plan.v1" or plan.get("runtime_version") != "0.10.21":
        raise MotionError("Use the reviewed motion plan for the separately pinned 0.10.21 runtime")
    try:
        version = importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError as error:
        raise MotionError("Install tools/motion/requirements-vision-offline.txt in a separate environment") from error
    if version != "0.10.21":
        raise MotionError("This experiment requires the separately audited MediaPipe 0.10.21 implementation")
    models = {"face": local_file(face_model, max_bytes=10 * 1024**2), "pose": local_file(pose_model, max_bytes=10 * 1024**2)}
    for name, model in models.items():
        if sha256(model) != plan.get(name + "_model_sha256"):
            raise MotionError(f"{name} model does not match the plan SHA-256")
    media = Path(media_root).resolve(strict=True)
    target = Path(output_root).resolve()
    sources = {source["id"]: source for source in plan["sources"]}
    acquisitions = {}
    for acquisition in plan["acquisitions"]:
        relative = Path(acquisition["input_basename"])
        file = (media / relative).resolve()
        if relative.is_absolute() or len(relative.parts) != 1 or not file.is_relative_to(media):
            raise MotionError("Acquired media must be a filename inside the supplied media root")
        file = local_file(file)
        if sha256(file) != acquisition["input_sha256"]:
            raise MotionError("Acquired video differs from the reviewed input hash")
        acquisitions[acquisition["source_id"]] = (acquisition, file)
    windows = plan.get("windows")
    if not isinstance(windows, list) or not 1 <= len(windows) <= 30:
        raise MotionError("A measurement plan must contain 1–30 bounded windows")
    ids = [window.get("id", "") for window in windows]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,119}", name) for name in ids):
        raise MotionError("Measurement window IDs must be unique safe path components")
    if any((target / name).exists() for name in ids):
        raise MotionError("Choose a new measurement directory; existing bundles are never overwritten")
    # Resource containment, not a network workaround. The MediaPipe artifacts
    # and implementation are unmodified and normal execution review applies.
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    results = []
    for window in windows:
        source = sources[window["source_id"]]
        acquisition, file = acquisitions[source["id"]]
        out = target / window["id"]
        prepare_clip(file, out, source_id=source["id"], source_url=source["url"], source_timeline=acquisition["source_timeline"], start=window["start_seconds"], duration=window["duration_seconds"], input_source_offset=acquisition["source_offset_seconds"], crop=window.get("crop"))
        result = observe_clip(out / "manifest.json", face_model=models["face"], pose_model=models["pose"], sample_fps=plan["sample_fps"])
        results.append({"id": window["id"], "observed_frames": result["observed_frames"], "runtime_version": result["mediapipe_version"], "observations_sha256": result["observations"]["sha256"]})
    return results
