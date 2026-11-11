from __future__ import annotations

import argparse
import json
import sys

from tools.media.common import MediaError
from .convert import export_clip


def main() -> int:
    parser = argparse.ArgumentParser(description="Export reviewed or explicitly preview-only MediaPipe geometry as a bounded avatar clip.")
    parser.add_argument("observations_manifest")
    parser.add_argument("review")
    parser.add_argument("output")
    parser.add_argument("--start-ms", type=float, default=0)
    parser.add_argument("--end-ms", type=float)
    parser.add_argument("--neutral-ms", type=float)
    parser.add_argument("--max-gap-ms", type=float, default=250)
    parser.add_argument("--visibility", type=float, default=0.65)
    parser.add_argument("--max-rotation-degrees", type=float, default=70)
    parser.add_argument("--allow-geometry-preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = export_clip(args.observations_manifest, args.review, args.output,
                             start_ms=args.start_ms, end_ms=args.end_ms, neutral_ms=args.neutral_ms,
                             max_gap_ms=args.max_gap_ms, visibility=args.visibility, max_rotation_degrees=args.max_rotation_degrees,
                             allow_geometry_preview=args.allow_geometry_preview, overwrite=args.overwrite)
        print(json.dumps({"schema_version": result["schema_version"], "frames": len(result["frames"]), "duration_ms": result["duration_ms"], "review": result["review"]["status"]}))
        return 0
    except (MediaError, OSError, ValueError) as error:
        print(f"Animation export failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
