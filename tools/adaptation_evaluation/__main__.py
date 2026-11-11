from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path

from bias.config import ProviderSettings, Settings
from bias.corpus import Corpus
from bias.engine import Engine
from tools.persona_data.heldout import load_heldout

from .protocol import freeze_plan, public_summary, validate_plan
from .run import compare_reports, export_review, run_condition


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze and execute controlled facts/style/adapter comparisons")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--corpus", type=Path, required=True)
    inputs = freeze.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--heldout", type=Path)
    inputs.add_argument("--development-cases", type=Path)
    freeze.add_argument("--snapshot", type=Path)
    freeze.add_argument("--selections", type=Path)
    freeze.add_argument("--id", required=True)
    freeze.add_argument("--base-model", required=True)
    freeze.add_argument("--base-revision", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--max-tokens", type=int, default=256)
    freeze.add_argument("--context-window", type=int, default=8192)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--condition", choices=["facts", "style", "adapter", "larger"], required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--adapter-run-sha256")
    run.add_argument("--timeout", type=float, default=180)
    run.add_argument("--total-seconds", type=float, default=3600)
    run.add_argument("--output", type=Path, required=True)
    for name in ["compare", "export-review"]:
        cmd = commands.add_parser(name)
        cmd.add_argument("--plan", type=Path, required=True)
        cmd.add_argument("--run", action="append", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            if args.output.exists():
                raise ValueError("Choose a new frozen plan directory")
            if args.heldout:
                identity, cases = load_heldout(args.heldout)
                exposure = "heldout"
            else:
                identity = json.loads(args.development_cases.read_text())
                if identity.get("exposure") != "development":
                    raise ValueError("Custom cases must explicitly declare development exposure")
                cases, exposure = identity["cases"], "development"
            settings = replace(Settings.from_env(), corpus_dir=args.corpus, persona_snapshot=args.snapshot,
                               llm_max_tokens=args.max_tokens, llm_context_window=args.context_window)
            engine = Engine(settings, Corpus.load(args.corpus))
            selections = json.loads(args.selections.read_text()) if args.selections else None
            plan = freeze_plan(engine, cases, plan_id=args.id, exposure=exposure, selections=selections,
                input_identity=identity, base_identity={"base_model": args.base_model, "base_revision": args.base_revision})
            write(args.output / "plan.json", plan)
            write(args.output / "summary.json", public_summary(plan))
            print(json.dumps({"plan_sha256": plan["plan_sha256"], "cases": len(cases), "exposure": exposure}))
        else:
            plan = json.loads(args.plan.read_text())
            validate_plan(plan)
            if args.command == "run":
                provider = ProviderSettings(args.base_url, args.model, os.getenv("BIAS_EVALUATION_API_KEY"))
                report = asyncio.run(run_condition(plan, args.condition, provider, args.output,
                    adapter_sha256=args.adapter_run_sha256, timeout=args.timeout, total_seconds=args.total_seconds))
                print(json.dumps({"status": report["status"], "attempted": len(report["runs"])}))
            else:
                reports = [json.loads(path.read_text()) for path in args.run]
                if args.command == "compare":
                    result = compare_reports(plan, reports)
                    write(args.output, result)
                else:
                    result = export_review(plan, reports, args.output)
                print(json.dumps(result, ensure_ascii=False))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Adaptation comparison failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
