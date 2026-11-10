"""Run with python -m tools.transcripts ted --help from the checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .ted import MAX_DOCUMENT_BYTES, TranscriptError, extract_ted, fetch_ted_page, read_document, write_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract public publisher transcripts into local review drafts")
    commands = parser.add_subparsers(dest="command", required=True)
    ted = commands.add_parser("ted", help="Read public TED HTML or saved __NEXT_DATA__ JSON")
    source = ted.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Local cached JSON or public page HTML")
    source.add_argument("--url", help="An ordinary HTTPS TED talk URL")
    ted.add_argument("--source-id", required=True)
    ted.add_argument("--output", type=Path, required=True, help="New local output directory, preferably under ignored data/local")
    args = parser.parse_args(argv)
    try:
        if args.input:
            if not args.input.is_file() or args.input.stat().st_size > MAX_DOCUMENT_BYTES:
                raise TranscriptError("Input must be an existing local file of at most 16 MiB")
            with args.input.open("rb") as stream:
                raw = stream.read(MAX_DOCUMENT_BYTES + 1)
        else:
            raw = fetch_ted_page(args.url)
        document, method = read_document(raw)
        bundle = extract_ted(document, source_id=args.source_id, input_sha256=hashlib.sha256(raw).hexdigest())
        bundle["manifest"]["acquisition_method"] = method if args.input else "public_https"
        bundle["manifest"]["input_name"] = args.input.name if args.input else None
        bundle["manifest"]["retrieved_url"] = args.url
        write_bundle(bundle, args.output)
        print(json.dumps(bundle["summary"], ensure_ascii=False, indent=2))
    except (OSError, TranscriptError) as exc:
        print(f"Transcript acquisition failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
