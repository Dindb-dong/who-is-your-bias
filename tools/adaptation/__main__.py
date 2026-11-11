from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import export_text, export_voice
from .train import train_text, train_voice


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export accepted training data and run recorded adaptation experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    text_export = commands.add_parser("export-text")
    text_export.add_argument("--projection", type=Path, required=True)
    text_export.add_argument("--output", type=Path, required=True)
    text_export.add_argument("--seed", type=int, default=42)
    text_export.add_argument("--max-per-event", type=int, default=150)
    voice_export = commands.add_parser("export-voice")
    voice_export.add_argument("--projection", type=Path, required=True)
    voice_export.add_argument("--output", type=Path, required=True)
    voice_export.add_argument("--asset-root", type=Path, default=Path("."))
    voice_export.add_argument("--reference-id")
    text_train = commands.add_parser("train-text")
    text_train.add_argument("--dataset", type=Path, required=True)
    text_train.add_argument("--output", type=Path, required=True)
    text_train.add_argument("--model", required=True)
    text_train.add_argument("--revision", required=True)
    text_train.add_argument("--epochs", type=int, default=1)
    text_train.add_argument("--seed", type=int, default=42)
    text_train.add_argument("--max-length", type=int, default=1024)
    text_train.add_argument("--rank", type=int, default=8)
    text_train.add_argument("--learning-rate", type=float, default=0.0002)
    text_train.add_argument("--gradient-accumulation", type=int, default=8)
    text_train.add_argument("--max-steps", type=int)
    text_train.add_argument("--device", default="cpu")
    text_train.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    text_train.add_argument("--gradient-checkpointing", action="store_true")
    voice_train = commands.add_parser("train-voice")
    for key in ("dataset", "output", "upstream", "base-model", "tokenizer-model"):
        voice_train.add_argument("--"+key, type=Path, required=True)
    voice_train.add_argument("--upstream-revision", required=True)
    voice_train.add_argument("--device", default="cuda:0")
    voice_train.add_argument("--batch-size", type=int, default=2)
    voice_train.add_argument("--learning-rate", type=float, default=2e-6)
    voice_train.add_argument("--epochs", type=int, default=3)
    voice_train.add_argument("--speaker", default="musk_research")
    voice_train.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        fn = {"export-text": export_text, "export-voice": export_voice, "train-text": train_text, "train-voice": train_voice}[command]
        result = fn(**args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Adaptation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
