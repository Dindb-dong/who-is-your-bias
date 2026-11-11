"""Supervise real loopback model servers and a frozen, three-condition run.

The server and HTTP client share one process environment. This also works in
executors whose separate commands have isolated loopback namespaces.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from bias.config import ProviderSettings
from services.language.adapters import validate_adapter_run
from .protocol import validate_plan
from .run import compare_reports, export_review, run_condition


def run_local(plan_path: Path, adapter_path: Path, output: Path, *, dtype="bfloat16", threads=2, timeout=180) -> dict:
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    identity = plan.get("base_identity") or {}
    if output.exists() or dtype not in {"float32", "bfloat16"} or not 1 <= threads <= 16:
        raise ValueError("Choose a new output path and supported CPU runtime settings")
    adapter = validate_adapter_run(adapter_path, base_model=identity["base_model"], base_revision=identity["base_revision"])
    output.mkdir(parents=True)
    reports, launchers = [], []
    for phase, conditions in (("base", ("facts", "style")), ("adapted", ("adapter",))):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {key: value for key, value in os.environ.items() if not key.startswith("BIAS_LANGUAGE_")}
        settings = {"MODEL": identity["base_model"], "REVISION": identity["base_revision"],
            "SERVED_MODEL": "comparison-local", "DTYPE": dtype, "DEVICE": "cpu", "THREADS": str(threads),
            "LOCAL_FILES_ONLY": "true", "GENERATION_TIMEOUT": str(timeout),
            "MAX_INPUT_TOKENS": str(plan["policy"]["context_window"]),
            "MAX_OUTPUT_TOKENS": str(max(32, plan["policy"]["max_tokens"]))}
        if phase == "adapted":
            settings.update(ADAPTER_RUN=str(adapter_path.resolve()), ADAPTER_RUN_SHA256=adapter.run_sha256)
        env.update({"BIAS_LANGUAGE_" + key: value for key, value in settings.items()})
        env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), TOKENIZERS_PARALLELISM="false")
        provider = ProviderSettings(f"http://127.0.0.1:{port}/v1", "comparison-local")
        started = time.monotonic()
        with (output / f"{phase}-server.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "services.language.app:app",
                "--host", "127.0.0.1", "--port", str(port)], env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while time.monotonic() - started < 30:
                    if process.poll() is not None:
                        raise ValueError("Language server exited before listening; inspect its local log")
                    try:
                        response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1)
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.1)
                else:
                    raise TimeoutError("Local language server did not start within 30 seconds")
                # Explicit excluded warmup; no benchmark claims about cold start.
                warm = httpx.post(provider.base_url + "/chat/completions", timeout=timeout,
                    json={"model": provider.model, "messages": [{"role": "user", "content": "Reply with one word: Ready"}],
                          "stream": False, "max_tokens": 12, "temperature": 0, "seed": 42})
                if warm.status_code != 200:
                    raise ValueError("Real language warmup failed; inspect local service log")
                launchers.append({"phase": phase, "warmup_ms": round((time.monotonic()-started)*1000, 3),
                    "warmup_finish_reason": warm.json()["choices"][0]["finish_reason"],
                    "warmup_included_in_condition_metrics": False})
                for condition in conditions:
                    report = asyncio.run(run_condition(plan, condition, provider, output / condition,
                        adapter_sha256=adapter.run_sha256 if condition == "adapter" else None,
                        timeout=timeout, total_seconds=min(14400, timeout * len(plan["cases"]) + 120)))
                    reports.append(report)
                    print(json.dumps({"condition": condition, "status": report["status"], "attempted": len(report["runs"])}), flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    summary = compare_reports(plan, reports)
    summary["local_execution"] = {"phases": launchers, "sequence": ["facts", "style", "adapter"],
        "reason": "One actual model server at a time to bound CPU memory; order is not counterbalanced.",
        "transport": "actual_loopback_http", "hardware_scope": "execution_cpu_not_target_device"}
    (output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    export_review(plan, reports, output / "blind-review")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--adapter-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    result = run_local(args.plan, args.adapter_run, args.output, dtype=args.dtype, threads=args.threads, timeout=args.timeout)
    print(json.dumps({"plan_sha256": result["plan_sha256"], "human_review": result["human_review"]}))


if __name__ == "__main__":
    main()
