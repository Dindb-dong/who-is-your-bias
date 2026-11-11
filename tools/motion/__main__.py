from __future__ import annotations

import argparse
import json
import sys

from bias.motion import MotionError, MotionLibrary
from .mining import build_library


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine and retrieve source-linked observed geometry, with source-event holdouts.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("manifest")
    build.add_argument("asset_root")
    build.add_argument("output")
    measure = commands.add_parser("measure")
    measure.add_argument("plan")
    measure.add_argument("media_root")
    measure.add_argument("output_root")
    measure.add_argument("--face-model", required=True)
    measure.add_argument("--pose-model", required=True)
    context = commands.add_parser("bind-contexts")
    context.add_argument("manifest")
    context.add_argument("asset_root")
    context.add_argument("report")
    context.add_argument("--publisher", action="append", required=True, metavar="SOURCE_ID=PATH")
    context.add_argument("--alignment-dir", action="append", required=True)
    check = commands.add_parser("check")
    check.add_argument("catalog")
    check.add_argument("asset_root")
    check.add_argument("--query")
    check.add_argument("--allow-geometry-preview", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "bind-contexts":
            from .contexts import bind_contexts
            pairs = [item.split("=", 1) for item in args.publisher]
            if any(len(pair) != 2 or not all(pair) for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
                raise MotionError("Publisher arguments must contain unique SOURCE_ID=PATH pairs")
            report = bind_contexts(args.manifest, args.asset_root, args.report, publishers=dict(pairs), alignment_directories=args.alignment_dir)
            print(json.dumps({"contexts": len(report["entries"]), "with_word_support": sum(row["status"] == "asr_word_matches" for row in report["entries"]), "report": args.report}))
        elif args.command == "measure":
            from .measure import measure_plan
            result = measure_plan(args.plan, args.media_root, args.output_root, face_model=args.face_model, pose_model=args.pose_model)
            print(json.dumps({"windows": result, "frames": sum(window["observed_frames"] for window in result)}))
        elif args.command == "build":
            result = build_library(args.manifest, args.asset_root, args.output)
            print(json.dumps({"library_id": result["id"], "clips": len(result["clips"]), "primitives": len(result["primitives"]), "output": args.output}))
        else:
            library = MotionLibrary.load(args.catalog, args.asset_root)
            selected = library.select(args.query, allow_geometry_preview=args.allow_geometry_preview) if args.query else None
            print(json.dumps({"summary": library.summary(), "selection": selected.model_dump() if selected else None}, indent=2))
        return 0
    except (MotionError, OSError, ValueError, KeyError) as error:
        print(f"Motion library failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
