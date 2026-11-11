"""Run frozen multi-turn API scenarios and retain a hash-bound result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias.config import Settings
from .run import REGISTRY, preview_settings, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--configured", action="store_true", help="Use explicit BIAS_* runtime settings; may call configured generation providers")
    parser.add_argument("--scenario", action="append", help="Select registered IDs; omitted runs the complete suite")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path; existing evidence is never overwritten")
    report = run(Settings.from_env() if args.configured else preview_settings(), registry_path=args.registry,
                 configured=args.configured, scenario_ids=set(args.scenario) if args.scenario else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": report["passed"], "scenarios": report["scenario_count"], "steps": report["step_count"],
                      "llm_inference_turns": report["llm_inference_turns"], "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
