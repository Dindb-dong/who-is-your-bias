from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import acquire_sources, build


def main(argv=None):
    parser = argparse.ArgumentParser(description="Acquire and materialize registered original dialogue; never silently ingest holdouts")
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--registry", type=Path, default=Path("data/acquisition/elon-musk-2026"))
    acquire.add_argument("--source", action="append", required=True)
    acquire.add_argument("--output", type=Path, required=True)
    materialize = commands.add_parser("build")
    materialize.add_argument("--registry", type=Path, default=Path("data/acquisition/elon-musk-2026"))
    materialize.add_argument("--acquisition", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--report", type=Path, required=True)
    materialize.add_argument("--recipe", type=Path)
    materialize.add_argument("--media-reviews", type=Path)
    materialize.add_argument("--asset-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        if args.command == "acquire":
            result = acquire_sources(args.registry, args.output, args.source)
            print(json.dumps({"sources": [{"source_id": row["source_id"], "status": row["status"]} for row in result["attempts"]]}, indent=2))
        else:
            result = build(args.registry, args.acquisition, args.output, report_dir=args.report, recipe_path=args.recipe,
                           media_review_path=args.media_reviews, asset_root=args.asset_root)
            print(json.dumps({key: result[key] for key in ("corpus", "source_events", "projections", "rejection_counts", "fresh_test_exchanges_acquired")}, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Persona data failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
