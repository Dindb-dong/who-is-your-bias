from __future__ import annotations

import argparse
import json

from .review import import_review, make_review_bundle
from .run import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure source-clock interview features without fabricating review")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--plan", required=True)
    build.add_argument("--corpus", required=True)
    build.add_argument("--cache-root", required=True)
    build.add_argument("--repository", default=".")
    build.add_argument("--output", required=True)
    build.add_argument("--overwrite", action="store_true")
    bundle = sub.add_parser("review-bundle")
    bundle.add_argument("--report", required=True)
    bundle.add_argument("--output-directory", required=True)
    review = sub.add_parser("review")
    review.add_argument("--report", required=True)
    review.add_argument("--annotations", required=True)
    review.add_argument("--output", required=True)
    review.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        result = build_report(args.plan, args.corpus, args.cache_root, args.repository, args.output, overwrite=args.overwrite)
    elif args.command == "review-bundle":
        result = make_review_bundle(args.report, args.output_directory)
    else:
        result = import_review(args.report, args.annotations, args.output, overwrite=args.overwrite)
    print(json.dumps(result.get("summary", result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
