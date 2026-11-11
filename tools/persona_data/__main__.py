from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import acquire_sources, build
from .heldout import acquire_heldout, build_heldout, load_heldout
from .combined import combine_acquisitions


def main(argv=None):
    parser = argparse.ArgumentParser(description="Acquire and materialize registered original dialogue; never silently ingest holdouts")
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--registry", type=Path, default=Path("data/acquisition/elon-musk-2026"))
    acquire.add_argument("--source", action="append", required=True)
    acquire.add_argument("--output", type=Path, required=True)
    combine = commands.add_parser("combine-acquisitions", help="Verify and copy separate acquisitions into a new immutable input")
    combine.add_argument("--registry", type=Path, default=Path("data/acquisition/elon-musk-2026"))
    combine.add_argument("--acquisition", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    materialize = commands.add_parser("build")
    materialize.add_argument("--registry", type=Path, default=Path("data/acquisition/elon-musk-2026"))
    materialize.add_argument("--acquisition", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--report", type=Path, required=True)
    materialize.add_argument("--recipe", type=Path)
    materialize.add_argument("--media-reviews", type=Path)
    materialize.add_argument("--asset-root", type=Path, default=Path("."))
    heldout_acquire = commands.add_parser("acquire-heldout", help="Acquire reserved test documents into an evaluation-only directory")
    heldout_acquire.add_argument("--registry", type=Path, default=Path("data/acquisition/elon-musk-2026"))
    heldout_acquire.add_argument("--source", action="append", required=True)
    heldout_acquire.add_argument("--output", type=Path, required=True)
    heldout_acquire.add_argument("--documents", type=Path, help="Optional hash-bound publisher transcript document registration")
    heldout_build = commands.add_parser("build-heldout", help="Materialize private references without runtime/training projections")
    heldout_build.add_argument("--registry", type=Path, default=Path("data/acquisition/elon-musk-2026"))
    heldout_build.add_argument("--acquisition", type=Path, required=True)
    heldout_build.add_argument("--comparison-corpus", type=Path, action="append", required=True)
    heldout_build.add_argument("--output", type=Path, required=True)
    heldout_build.add_argument("--report", type=Path)
    heldout_verify = commands.add_parser("verify-heldout", help="Verify exact source, reference and leakage-check artifacts")
    heldout_verify.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "acquire":
            result = acquire_sources(args.registry, args.output, args.source)
            print(json.dumps({"sources": [{"source_id": row["source_id"], "status": row["status"]} for row in result["attempts"]]}, indent=2))
        elif args.command == "combine-acquisitions":
            print(json.dumps(combine_acquisitions(args.registry, args.acquisition, args.output), indent=2))
        elif args.command == "build":
            result = build(args.registry, args.acquisition, args.output, report_dir=args.report, recipe_path=args.recipe,
                           media_review_path=args.media_reviews, asset_root=args.asset_root)
            print(json.dumps({key: result[key] for key in ("corpus", "source_events", "projections", "rejection_counts", "fresh_test_exchanges_acquired")}, indent=2))
        elif args.command == "acquire-heldout":
            result = acquire_heldout(args.registry, args.output, args.source, documents_path=args.documents)
            print(json.dumps({"purpose": result["purpose"], "sources": [{"source_id": row["source_id"], "status": row["status"]} for row in result["attempts"]]}, indent=2))
        elif args.command == "build-heldout":
            result = build_heldout(args.registry, args.acquisition, args.output, comparison_corpora=args.comparison_corpus, report_dir=args.report)
            print(json.dumps({key: result[key] for key in ("purpose", "cases", "source_events", "human_review_completed")}, indent=2))
        else:
            result, _ = load_heldout(args.directory)
            print(json.dumps({key: result[key] for key in ("purpose", "cases", "source_events", "human_review_completed")}, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Persona data failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
