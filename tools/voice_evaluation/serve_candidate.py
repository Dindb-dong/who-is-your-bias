"""Serve an explicitly selected, hash-bound speech experiment on loopback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from services.voice.adapters import base_manifest, digest, read_checkpoint
from services.voice.engine import VoiceSettings, read_reference
from services.voice.provenance import verify_binding


def read_pinned(path: Path, expected: str) -> dict:
    if digest(path) != expected:
        raise ValueError(f"Pinned experiment receipt changed: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError("Experiment artifacts must be relative to the selected root")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--condition", choices=["reference", "adapted"], required=True)
    parser.add_argument("--allow-candidates", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    if not args.allow_candidates:
        parser.error("This experiment requires explicit --allow-candidates")
    if not 1024 <= args.port <= 65535:
        parser.error("Use a loopback port between 1024 and 65535")
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    if binding.get("schema_version") != "voice-experiment-binding.v1":
        raise ValueError("Unsupported voice experiment binding")
    protocol = read_pinned(args.binding.parent / binding["protocol"]["path"], binding["protocol"]["sha256"])
    training = read_pinned(args.binding.parent / binding["training"]["path"], binding["training"]["sha256"])
    root = args.artifacts.expanduser().resolve()
    base = artifact(root, binding["base_directory"])
    reference = artifact(root, binding["reference_directory"])
    if base_manifest(base) != training["contract"]["base_files_sha256"]:
        raise ValueError("Base model bytes differ from the frozen training snapshot")
    if (protocol["persona_id"] != training["contract"]["persona_id"]
            or protocol["base_revision"] != training["contract"]["base_revision"]):
        raise ValueError("Frozen evaluation and training identities differ")
    adapted = args.condition == "adapted"
    checkpoint = artifact(root, binding["adapter_directory"]) if adapted else None
    final = training["final_artifact"]
    if adapted:
        receipt = read_checkpoint(checkpoint, persona_id=protocol["persona_id"],
            expected_sha256=final["checkpoint_sha256"])
        if receipt["optimizer_steps"] != training["optimizer_steps"]:
            raise ValueError("Adapter progress differs from the saved completion receipt")
    settings = VoiceSettings(model_id=str(base), revision=protocol["base_revision"],
        local_files_only=True, persona_id=protocol["persona_id"],
        speaker_label=training["contract"]["speaker"], reference_wav=reference / "audio.wav",
        reference_text=(reference / "reference.txt").read_text(encoding="utf-8").strip(),
        reference_manifest=reference / "reference.json", allow_candidate_reference=True,
        adapter_checkpoint=checkpoint,
        adapter_checkpoint_sha256=final["checkpoint_sha256"] if adapted else None,
        api_key=os.getenv("BIAS_VOICE_API_KEY"), **protocol["runtime"])
    identity = verify_binding(manifest_path=settings.reference_manifest,
        persona_id=settings.persona_id, speaker=settings.speaker_label,
        audio_path=settings.reference_wav, text=settings.reference_text,
        embedding_only=settings.x_vector_only_mode, allow_candidate=True)
    if (identity["reference_sha256"] != protocol["reference_audio_sha256"]
            or identity["manifest_sha256"] != protocol["reference_manifest_sha256"]):
        raise ValueError("Reference differs from the frozen evaluation protocol")
    read_reference(settings)
    print(json.dumps({"status": "verified_before_model_load", "condition": binding["conditions"][args.condition],
        "persona_id": settings.persona_id, "reference_identity": identity,
        "adapter_checkpoint_sha256": settings.adapter_checkpoint_sha256,
        "optimizer_steps": training["optimizer_steps"] if adapted else 0,
        "promotion_status": "candidate", "host": "127.0.0.1", "port": args.port}), flush=True)
    if args.verify_only:
        return
    # The explicit settings above own this process. Discard stale service defaults
    # before importing the module's optional global ASGI app; retain any API key
    # in the explicit settings object, without printing it.
    for name in list(os.environ):
        if name.startswith("BIAS_VOICE_"):
            del os.environ[name]
    import uvicorn
    from services.voice.app import create_app
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
