"""Run python -m tools.references from a repository checkout."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

from tools.media.common import MediaError, sha256, write_json
from .workflow import align_transcripts, prepare_reference, read_json, review_reference, verify_reference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare source-grounded voice-reference candidates and explicit reviews")
    commands = parser.add_subparsers(dest="command", required=True)
    asr = commands.add_parser("asr", help="Actual CPU ASR on a prepared 16 kHz context, using a pinned local model")
    asr.add_argument("media_manifest")
    asr.add_argument("--model-directory", required=True)
    asr.add_argument("--model-repo", required=True)
    asr.add_argument("--revision", required=True)
    asr.add_argument("--cpu-threads", type=int, default=4)
    asr.add_argument("--output", required=True)
    align = commands.add_parser("match", help="Match actual ASR word times to publisher-labeled caption context")
    align.add_argument("--publisher", required=True)
    align.add_argument("--asr", required=True)
    align.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare", help="Create an 8–15 second candidate; never approves it automatically")
    prepare.add_argument("--media-manifest", required=True)
    prepare.add_argument("--publisher", required=True)
    prepare.add_argument("--asr", required=True)
    prepare.add_argument("--source-media", required=True)
    prepare.add_argument("--speaker", required=True)
    prepare.add_argument("--cue-id", action="append", required=True)
    prepare.add_argument("--start", type=float, required=True, help="Start in the actual native source clock, not the subtitle clock")
    prepare.add_argument("--duration", type=float, required=True)
    prepare.add_argument("--output", required=True)
    review = commands.add_parser("review", help="Append an explicit, hash-bound review assessment")
    review.add_argument("manifest")
    review.add_argument("--assessment", required=True)
    verify = commands.add_parser("verify", help="Verify artifact hashes and the complete review ledger")
    verify.add_argument("manifest")
    args = parser.parse_args(argv)
    try:
        if args.command == "asr":
            from .asr import transcribe_reference_context
            result = transcribe_reference_context(args.media_manifest, model_directory=args.model_directory,
                model_repo=args.model_repo, revision=args.revision, output=args.output, cpu_threads=args.cpu_threads)
            summary = {"segments": len(result["segments"]), "elapsed_seconds": result["elapsed_seconds"], "review_status": "draft", "speaker_attribution": "unverified"}
        elif args.command == "match":
            result = align_transcripts(read_json(args.publisher), read_json(args.asr))
            result.update(publisher_artifact_sha256=sha256(Path(args.publisher)), asr_artifact_sha256=sha256(Path(args.asr)))
            write_json(Path(args.output), result)
            summary = {key: result[key] for key in ("source_id", "asr_tokens", "matched_tokens", "token_coverage", "offset_summary", "review_status")}
        else:
            if args.command == "prepare":
                result = prepare_reference(args.media_manifest, args.publisher, args.asr, args.source_media, args.output,
                    speaker=args.speaker, cue_ids=args.cue_id, start_seconds=args.start, duration_seconds=args.duration)
            elif args.command == "review":
                result = review_reference(args.manifest, read_json(args.assessment))
            else:
                result = verify_reference(args.manifest)
            summary = {"source_id": result["source_id"], "source_timeline": result["source_timeline"],
                "source_start_seconds": result["source_start_seconds"], "source_end_seconds": result["source_end_seconds"],
                "duration_seconds": result["clip"]["duration_seconds"], "review_status": result["review"]["status"],
                "automatic_checks": result["automatic_checks"], "ledger_entries": result["review_ledger"]["entries"]}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except (MediaError, OSError, ValueError, KeyError, wave.Error) as exc:
        print(f"reference: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
