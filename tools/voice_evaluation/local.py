"""Supervise one real HTTP speech condition inside a shared loopback namespace."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .run import checkpoint, digest, run_synthesis
from .serve_candidate import read_pinned


def process_memory(pid: int) -> dict:
    def fields(path: Path, names: set[str]) -> dict:
        if not path.exists():
            return {}
        return {key: value.strip() for line in path.read_text().splitlines()
            for key, _, value in [line.partition(":")] if key in names}
    return {"scope": "same invocation as the real speech server",
        "parent_namespace_meminfo": fields(Path("/proc/meminfo"), {"MemTotal", "MemAvailable", "MemFree"}),
        "server_status": fields(Path(f"/proc/{pid}/status"), {"VmRSS", "VmHWM", "Threads", "State"})}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--condition", choices=["reference", "adapted"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-candidates", action="store_true")
    parser.add_argument("--per-case-timeout", type=float, default=300)
    parser.add_argument("--total-timeout", type=float, default=900)
    args = parser.parse_args()
    if not args.allow_candidates:
        parser.error("This experiment requires explicit --allow-candidates")
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    protocol = read_pinned(args.binding.parent / binding["protocol"]["path"], binding["protocol"]["sha256"])
    cases = Path(protocol["cases_path"]).resolve()
    if digest(cases) != protocol["cases_sha256"]:
        raise ValueError("Requested text differs from the frozen evaluation cases")
    output = args.output.expanduser().resolve()
    log_path = output.with_name(output.name + ".service.log")
    launcher_path = output.with_name(output.name + ".launcher.json")
    if output.exists() or log_path.exists() or launcher_path.exists():
        raise ValueError("Choose a new output path; existing experiment files are preserved")
    output.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [sys.executable, "-m", "tools.voice_evaluation.serve_candidate",
        "--binding", str(args.binding.resolve()), "--artifacts", str(args.artifacts.resolve()),
        "--condition", args.condition, "--allow-candidates", "--port", str(port)]
    record = {"schema_version": "voice_local_execution.v1", "status": "starting",
        "implementation_sha256": digest(Path(__file__)),
        "condition": binding["conditions"][args.condition], "binding_sha256": digest(args.binding),
        "transport": "actual_loopback_http", "server_and_client_same_namespace": True,
        "warmup_performed": False, "service_log": log_path.name}
    checkpoint(launcher_path, record)
    started = time.monotonic()
    with log_path.open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(timeout=1, trust_env=False) as client:
                while time.monotonic() - started < 45:
                    if process.poll() is not None:
                        raise RuntimeError("Speech server exited before listening; inspect its saved log")
                    try:
                        response = client.get(base + "/health")
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                else:
                    raise TimeoutError("Speech server did not start within 45 seconds")
            record.update(status="synthesizing", startup_seconds=round(time.monotonic() - started, 6),
                memory_before_model_load=process_memory(process.pid))
            checkpoint(launcher_path, record)
            result = asyncio.run(run_synthesis(base_url=base, output=output, cases_path=cases,
                condition=record["condition"], api_key=os.getenv("BIAS_VOICE_API_KEY"),
                per_case_timeout=args.per_case_timeout, total_timeout=args.total_timeout))
            record["status"] = result["status"]
        except BaseException as exc:
            record.update(status="incomplete", error_type=type(exc).__name__)
            raise
        finally:
            record["memory_before_server_stop"] = process_memory(process.pid)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            log.flush()
            record.update(server_returncode=process.returncode,
                elapsed_seconds=round(time.monotonic() - started, 6),
                service_log_sha256=digest(log_path),
                child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
                rss_scope="Linux child-process high-water RSS, not concurrent total memory")
            checkpoint(launcher_path, record)
    print(json.dumps({"status": record["status"], "condition": record["condition"],
        "cases": [{"id": row["id"], "status": row["status"],
            "request_wall_seconds": row.get("request_wall_seconds"),
            "audio_seconds": row.get("signal", {}).get("duration_seconds")}
            for row in result["cases"]]}, indent=2), flush=True)
    return 0 if record["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
