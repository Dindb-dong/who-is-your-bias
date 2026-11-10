"""Run with python -m tools.media from the repository checkout."""

from __future__ import annotations

import argparse
import json
import sys

from .common import MediaError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and inspect local source media with explicit provenance")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create an audio/video clip and review manifest")
    prepare.add_argument("input")
    prepare.add_argument("output_dir")
    prepare.add_argument("--source-id", required=True)
    prepare.add_argument("--source-url", required=True)
    prepare.add_argument("--source-timeline", required=True)
    prepare.add_argument("--start", type=float, default=0)
    prepare.add_argument("--duration", type=float, default=30)
    prepare.add_argument("--input-source-offset", type=float, default=0)
    prepare.add_argument("--sample-rate", type=int, default=16000, choices=[16000, 24000, 48000])
    prepare.add_argument("--audio-only", action="store_true")
    transcribe = commands.add_parser("transcribe", help="Optional ASR; produces a draft transcript for review")
    transcribe.add_argument("manifest")
    transcribe.add_argument("--engine", choices=["faster-whisper", "whisperx"], default="faster-whisper")
    transcribe.add_argument("--model", default="small")
    transcribe.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    transcribe.add_argument("--compute-type", default="int8")
    transcribe.add_argument("--language")
    transcribe.add_argument("--overwrite", action="store_true")
    observe = commands.add_parser("observe", help="Optional MediaPipe face/pose measurements; no emotion or identity inference")
    observe.add_argument("manifest")
    observe.add_argument("--face-model", required=True)
    observe.add_argument("--pose-model")
    observe.add_argument("--sample-fps", type=float, default=10)
    observe.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from .prepare import prepare_clip
            output = prepare_clip(args.input, args.output_dir, source_id=args.source_id, source_url=args.source_url, source_timeline=args.source_timeline, start=args.start, duration=args.duration, input_source_offset=args.input_source_offset, sample_rate=args.sample_rate, include_video=not args.audio_only)
            print(json.dumps({"source_id": output["source_id"], "duration_seconds": output["clip"]["duration_seconds"], "review_status": "unreviewed"}))
        elif args.command == "transcribe":
            from .transcribe import transcribe_clip
            output = transcribe_clip(args.manifest, engine=args.engine, model_name=args.model, device=args.device, compute_type=args.compute_type, language=args.language, overwrite=args.overwrite)
            print(json.dumps({"segments": len(output["segments"]), "review_status": "draft"}))
        else:
            from .observe import observe_clip
            output = observe_clip(args.manifest, face_model=args.face_model, pose_model=args.pose_model, sample_fps=args.sample_fps, overwrite=args.overwrite)
            print(json.dumps({"observed_frames": output["observed_frames"], "review_status": "unreviewed"}))
        return 0
    except (MediaError, OSError) as exc:
        print(f"media: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
