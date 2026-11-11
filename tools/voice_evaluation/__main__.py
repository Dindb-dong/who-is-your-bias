from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .metrics import EvaluationError
from .run import DEFAULT_CASES, checkpoint, run_synthesis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Separate actual synthesis, independent ASR, and human listening evidence")
    commands = parser.add_subparsers(dest="command", required=True)
    synthesize = commands.add_parser("synthesize")
    synthesize.add_argument("--base-url", default="http://127.0.0.1:8092")
    synthesize.add_argument("--cases", default=DEFAULT_CASES, type=Path)
    synthesize.add_argument("--condition", required=True)
    synthesize.add_argument("--output", required=True)
    synthesize.add_argument("--per-case-timeout", type=float, default=300)
    synthesize.add_argument("--total-timeout", type=float, default=900)
    asr = commands.add_parser("asr")
    asr.add_argument("synthesis")
    asr.add_argument("--model-directory", required=True)
    asr.add_argument("--model-repo", required=True)
    asr.add_argument("--revision", required=True)
    asr.add_argument("--cpu-threads", type=int, default=4)
    asr.add_argument("--output", required=True)
    bundle = commands.add_parser("bundle")
    bundle.add_argument("synthesis", nargs="+")
    bundle.add_argument("--output", required=True)
    bundle.add_argument("--seed", type=int, default=42)
    bundle.add_argument("--reference")
    review = commands.add_parser("review")
    review.add_argument("manifest")
    review.add_argument("review")
    review.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "synthesize":
            result = asyncio.run(run_synthesis(base_url=args.base_url, cases_path=args.cases, condition=args.condition,
                output=args.output, api_key=os.getenv("BIAS_VOICE_API_KEY"),
                per_case_timeout=args.per_case_timeout, total_timeout=args.total_timeout))
            summary = {"status": result["status"], "cases": [{"id": row["id"], "status": row["status"],
                "request_wall_seconds": row.get("request_wall_seconds"), "real_time_factor": row.get("real_time_factor")} for row in result["cases"]]}
        elif args.command == "asr":
            from .asr import transcribe_run
            result = transcribe_run(args.synthesis, model_directory=args.model_directory, model_repo=args.model_repo,
                revision=args.revision, output=args.output, cpu_threads=args.cpu_threads)
            summary = {"status": result["status"], "cases": [{"id": row["id"], "status": row["status"],
                "metrics": row.get("metrics")} for row in result["cases"]]}
        elif args.command == "bundle":
            from .review import create_bundle
            summary = create_bundle(args.synthesis, output=args.output, seed=args.seed, reference=args.reference)
        else:
            from .review import validate_review
            summary = validate_review(args.manifest, args.review)
            if args.output:
                path = Path(args.output)
                if path.exists():
                    raise EvaluationError("Review summary already exists")
                path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint(path, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 2 if summary.get("status") == "incomplete" else 0
    except (EvaluationError, OSError) as exc:
        print(f"voice evaluation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
