"""Fetch immutable MuseTalk/VAE/Whisper snapshots and create a verified lock."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .common import digest, save

MODELS = {
    "musetalk": ("TMElyralab/MuseTalk", "2bcb936e2fddb4d86db4c62fd45b387d0c061571", {
        "musetalk_config": "musetalkV15/musetalk.json", "musetalk_weights": "musetalkV15/unet.pth"}),
    "vae": ("stabilityai/sd-vae-ft-mse", "31f26fdeee1355a5c34592e401dd41e45d25a493", {
        "vae_config": "config.json", "vae_weights": "diffusion_pytorch_model.safetensors"}),
    "whisper": ("openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af", {
        "whisper_config": "config.json", "whisper_weights": "model.safetensors", "whisper_preprocessor": "preprocessor_config.json"}),
}


def fetch(cache: Path, output: Path, *, offline: bool = False) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download
    lock = {"schema_version": "musetalk-model-lock.v1", "models": {}, "files": {}}
    for name, (repo, revision, files) in MODELS.items():
        location = Path(snapshot_download(repo, revision=revision, cache_dir=str(cache),
            allow_patterns=list(files.values()), max_workers=2, local_files_only=offline))
        lock["models"][name] = {"repo": repo, "revision": revision}
        for key, filename in files.items():
            path = location / filename
            # Keep snapshot filenames: resolving HF symlinks would turn config
            # into a nameless blob and break from_pretrained(directory).
            lock["files"][key] = {"path": str(path.absolute()), "sha256": digest(path), "bytes": path.stat().st_size}
    save(output, lock)
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    fetch(args.cache, args.output, offline=args.offline)


if __name__ == "__main__":
    main()
