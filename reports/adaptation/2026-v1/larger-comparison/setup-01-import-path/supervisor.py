"""Recorded one-off larger-base control; run from the repository root."""
from __future__ import annotations
import asyncio
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PLAN = Path("data/local/adaptation-comparison/frozen-v2/plan.json")
OUTPUT = Path("reports/local/adaptation-comparison-larger")
CACHE = Path("data/local/larger-model-cache").resolve()
MAX_SECONDS = 600
SERVER_RSS_LIMIT = 4_000_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def stop(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def main():
    if OUTPUT.exists():
        raise ValueError("Preserve existing run; choose a new output")
    OUTPUT.mkdir(parents=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    report = {"schema_version": "larger-control-execution.v1", "started_at": now(),
              "model": MODEL, "revision": REVISION, "status": "preflight",
              "overall_budget_seconds": MAX_SECONDS, "server_rss_limit_bytes": SERVER_RSS_LIMIT,
              "cache_directory": "data/local/larger-model-cache", "transport": "actual_loopback_http",
              "served_dtype": "bfloat16", "served_device": "cpu", "torch_threads": 2,
              "trained_adapter": None, "human_review": "ungraded", "efficacy_claim": False,
              "supervisor_sha256": sha(__file__), "inference_attempted": False,
              "limits": ["Exploratory larger-base control on four exposed development questions.",
                         "Sequential execution and other host activity do not support a model-speed ranking.",
                         "A completed response is not a factual or persona-quality judgment.",
                         "4 GB means 4,000,000,000 bytes of sampled server RSS, not checkpoint file size or total host RSS."]}
    save(OUTPUT / "execution.json", report)
    downloader = server = None
    monitor = None
    stop_monitor = threading.Event()
    resource_failure = []
    try:
        from tools.adaptation_evaluation.protocol import validate_plan, implementation_hashes
        from tools.adaptation_evaluation.run import run_condition, compare_reports, export_review
        from services.language.adapters import checkpoint_manifest
        from bias.config import ProviderSettings
        plan = json.loads(PLAN.read_text())
        validate_plan(plan)
        report["plan_sha256"] = plan["plan_sha256"]
        report["implementation_sha256"] = implementation_hashes()
        report["policy"] = plan["policy"]
        report["versions"] = {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "huggingface-hub", "httpx", "uvicorn")}
        metadata_url = "https://huggingface.co/api/models/" + MODEL
        response = httpx.get(metadata_url, timeout=20, follow_redirects=False)
        response.raise_for_status()
        metadata = response.json()
        if metadata["sha"] != REVISION or metadata.get("private") or metadata.get("gated"):
            raise ValueError("Official immutable revision or public model status changed")
        (OUTPUT / "official-metadata.json").write_bytes(response.content)
        report["official_metadata"] = {"url": metadata_url, "status": response.status_code,
            "retrieved_at": now(), "sha256": sha(OUTPUT / "official-metadata.json"),
            "safetensors": metadata.get("safetensors")}
        common = {k: v for k, v in os.environ.items() if not k.startswith("BIAS_")}
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            common.pop(k, None)
        common.update(HF_HUB_CACHE=str(CACHE), HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_TELEMETRY="1",
            ORT_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
        download_code = "from huggingface_hub import snapshot_download; print(snapshot_download(" + repr(MODEL) + ",revision=" + repr(REVISION) + ",cache_dir=" + repr(str(CACHE)) + ",allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','*.tiktoken'],max_workers=2),flush=True)"
        report["status"] = "downloading"
        save(OUTPUT / "execution.json", report)
        print(json.dumps({"stage": "downloading", "revision": REVISION}), flush=True)
        with (OUTPUT / "download.log").open("w") as log:
            downloader = subprocess.Popen([sys.executable, "-c", download_code], env=common, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while downloader.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Overall budget exhausted during model acquisition")
                time.sleep(.25)
        report["download_exit_code"] = downloader.returncode
        if downloader.returncode:
            raise RuntimeError("Pinned checkpoint acquisition failed; raw local log retained")
        checkpoint = CACHE / "models--Qwen--Qwen3-1.7B" / "snapshots" / REVISION
        report["checkpoint_sha256"] = checkpoint_manifest(checkpoint)
        report["checkpoint_bytes"] = sum((checkpoint / name).stat().st_size for name in report["checkpoint_sha256"])
        report["download_finished_at"] = now()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        settings = {"MODEL": MODEL, "REVISION": REVISION, "CACHE_DIR": str(CACHE),
            "SERVED_MODEL": "comparison-larger", "DTYPE": "bfloat16", "DEVICE": "cpu", "THREADS": "2",
            "LOCAL_FILES_ONLY": "true", "GENERATION_TIMEOUT": "180", "MAX_INPUT_TOKENS": str(plan["policy"]["context_window"]),
            "MAX_OUTPUT_TOKENS": str(plan["policy"]["max_tokens"])}
        server_env = {**common, **{"BIAS_LANGUAGE_" + k: v for k, v in settings.items()},
                      "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        report["status"] = "starting_server"
        report["settings"] = {k: v for k, v in settings.items() if k != "CACHE_DIR"}
        save(OUTPUT / "execution.json", report)
        with (OUTPUT / "server.log").open("w") as log:
            server = subprocess.Popen([sys.executable, "-m", "uvicorn", "services.language.app:app", "--host", "127.0.0.1", "--port", str(port)],
                                      env=server_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

            def watch():
                peak = 0
                with (OUTPUT / "server-memory.jsonl").open("w") as f:
                    while not stop_monitor.is_set() and server.poll() is None:
                        try:
                            fields = dict(line.split(":", 1) for line in Path(f"/proc/{server.pid}/status").read_text().splitlines() if ":" in line)
                            rss = int(fields.get("VmRSS", "0 kB").split()[0]) * 1024
                            peak = max(peak, rss)
                            report["peak_sampled_server_rss_bytes"] = peak
                            f.write(json.dumps({"elapsed_seconds": round(time.monotonic() - started, 3), "rss_bytes": rss}) + "\n")
                            f.flush()
                            if rss > SERVER_RSS_LIMIT:
                                resource_failure.append({"type": "ServerRssLimitExceeded", "observed_bytes": rss, "limit_bytes": SERVER_RSS_LIMIT})
                            elif time.monotonic() >= deadline:
                                resource_failure.append({"type": "OverallDeadlineExceeded", "limit_seconds": MAX_SECONDS})
                            if resource_failure:
                                os.killpg(server.pid, signal.SIGTERM)
                                return
                        except (OSError, ValueError):
                            pass
                        stop_monitor.wait(.25)

            monitor = threading.Thread(target=watch, daemon=True)
            monitor.start()
            listen_deadline = min(deadline, time.monotonic() + 30)
            while time.monotonic() < listen_deadline:
                if server.poll() is not None:
                    raise RuntimeError("Language service exited before listening")
                try:
                    response = httpx.get(url + "/health", timeout=1, trust_env=False)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                raise TimeoutError("Language service did not listen within 30 seconds")
            report["status"] = "warming_model"
            save(OUTPUT / "execution.json", report)
            warm_started = time.monotonic()
            response = httpx.post(url + "/v1/chat/completions", timeout=min(180, deadline - time.monotonic()), trust_env=False,
                json={"model": "comparison-larger", "messages": [{"role": "user", "content": "Reply with one word: Ready"}],
                      "stream": False, "max_tokens": 12, "temperature": 0, "seed": 42})
            response.raise_for_status()
            report["warmup"] = {"elapsed_seconds": time.monotonic() - warm_started, "finish_reason": response.json()["choices"][0]["finish_reason"], "included_in_case_metrics": False}
            report["loaded_health"] = httpx.get(url + "/health", timeout=10, trust_env=False).json()
            report["inference_attempted"] = True
            report["status"] = "running_condition"
            save(OUTPUT / "execution.json", report)
            print(json.dumps({"stage": "running_condition", "base_parameters": report["loaded_health"]["runtime"]["base_parameters"]}), flush=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Overall budget exhausted after warmup")
            larger = asyncio.run(run_condition(plan, "larger", ProviderSettings(url + "/v1", "comparison-larger"), OUTPUT / "larger",
                                              timeout=min(180, remaining), total_seconds=min(14400, remaining)))
            controls = [json.loads((Path("reports/adaptation/2026-v1/final-comparison") / condition / "run.json").read_text()) for condition in ("facts", "style", "adapter")]
            comparison = compare_reports(plan, [*controls, larger])
            save(OUTPUT / "comparison.json", comparison)
            export_review(plan, [*controls, larger], OUTPUT / "blind-review")
            report.update(status="completed" if larger["status"] == "completed" else "incomplete",
                          attempted_cases=len(larger["runs"]), completed_generations=sum(r["generation"]["completed"] for r in larger["runs"]),
                          larger_run_sha256=larger["run_sha256"], comparison_sha256=sha(OUTPUT / "comparison.json"))
    except BaseException as exc:
        report.update(status="blocked", failure=resource_failure[0] if resource_failure else {"type": type(exc).__name__, "reason": str(exc).split("?")[0][:300]})
    finally:
        stop_monitor.set()
        stop(server)
        stop(downloader)
        if monitor:
            monitor.join(timeout=2)
        report["completed_at"] = now()
        report["elapsed_seconds"] = time.monotonic() - started
        report["server_exit_code_after_cleanup"] = server.returncode if server else None
        if report.get("implementation_sha256"):
            report["implementation_unchanged"] = report["implementation_sha256"] == implementation_hashes()
            if not report["implementation_unchanged"]:
                report.update(status="blocked", failure={"type": "ImplementationChanged"})
        save(OUTPUT / "execution.json", report)
        print(json.dumps({k: report.get(k) for k in ("status", "failure", "attempted_cases", "completed_generations", "elapsed_seconds", "peak_sampled_server_rss_bytes")}), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
