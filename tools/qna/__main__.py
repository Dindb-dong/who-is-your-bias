"""python -m tools.qna hydrate --output data/local/qna/elon-musk-v1"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .acquire import AcquisitionError
from .hydrate import hydrate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hydrate reviewed original exchanges from pinned public publisher transcripts")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("hydrate")
    command.add_argument("--recipes", type=Path, default=Path("data/qna/elon-musk-v1"))
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--selection", type=Path, help="Explicit versioned selection manifest; default preserves the original v1 selection")
    command.add_argument("--raw", action="append", default=[], metavar="SOURCE_ID=PATH", help="Use a cached publisher HTML/JSON instead of HTTPS")
    command.add_argument("--bounded", action="store_true", help="Rebuild only the tiny committed excerpts")
    args = parser.parse_args(argv)
    inputs = {}
    try:
        for pair in args.raw:
            identifier, separator, path = pair.partition("=")
            if not separator or not identifier or not path or identifier in inputs:
                raise AcquisitionError("--raw must be a unique SOURCE_ID=PATH pair")
            inputs[identifier] = Path(path)
        print(json.dumps(hydrate(args.recipes, args.output, raw_inputs=inputs, bounded=args.bounded, selection_path=args.selection), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Q&A acquisition failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
