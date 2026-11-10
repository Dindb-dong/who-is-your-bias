from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from bias.config import Settings
from bias.corpus import Corpus
from .collect import collect_source
from .inventory import inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the persona corpus and collect original publisher documents locally")
    parser.add_argument("--corpus", type=Path, default=Settings().corpus_dir)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("inventory")
    audit.add_argument("--output", type=Path)
    collect = commands.add_parser("collect")
    collect.add_argument("--source-id", action="append", required=True, help="Repeat to collect several known source IDs; there is no implicit whole-web crawl")
    collect.add_argument("--destination", type=Path, default=Path("data/raw/sources"))
    args = parser.parse_args(argv)
    try:
        report = inventory(Corpus.load(args.corpus, check_splits=args.command != "inventory"))
        if args.command == "inventory":
            serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized)
            print(serialized)
            return 1 if report["source_event_split_leaks"] or report["url_split_leaks"] else 0
        lookup = {source["id"]: source for source in report["sources"]}
        unknown = set(args.source_id) - lookup.keys()
        if unknown:
            raise ValueError("Unknown source IDs: " + ", ".join(sorted(unknown)))
        results = []
        for identifier in dict.fromkeys(args.source_id):
            source = lookup[identifier]
            for url in source["acquisition_urls"]:
                try:
                    result = collect_source({**source, "acquisition_url": url}, args.destination)
                    results.append({"source_id": identifier, "url": url, "ok": True, "bytes": result["bytes"], "sha256": result["sha256"], "review_status": result["review_status"]})
                except (ValueError, OSError, httpx.HTTPError) as exc:
                    results.append({"source_id": identifier, "url": url, "ok": False, "error": type(exc).__name__, "detail": str(exc)})
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        return 0 if all(result["ok"] for result in results) else 1
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
