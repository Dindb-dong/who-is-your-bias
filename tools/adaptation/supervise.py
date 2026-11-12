"""Own one training process with a file heartbeat, lock and explicit stop request."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time

from .checkpoints import atomic_json


def _stop(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def run_supervised(command: list[str], directory: Path, name: str, log: Path, *, timeout=14400., interval=1.) -> dict:
    if (not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,99}", name)
            or not command or any(not isinstance(x, str) or not x for x in command)
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                   for x in (timeout, interval)) or not 0 < interval <= timeout <= 86400):
        raise ValueError("Choose a named bounded invocation and a nonempty command")
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / f"{name}.json"
    stop_path = directory / f"stop-{name}.json"
    if state_path.exists() or log.exists():
        raise ValueError("Preserve previous invocation evidence and choose a new name/log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with (directory / "active.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another supervised training invocation still holds the shared lock") from exc
        started = time.monotonic()
        state = {"schema_version": "adaptation-process-supervision.v1", "name": name,
            "invocation_state": "starting", "command": command,
            "started_at": datetime.now(timezone.utc).isoformat(), "supervisor_pid": os.getpid(),
            "stop_request": str(stop_path.absolute()), "log": str(log.absolute()),
            "environment": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM")},
            "timeout_seconds": timeout, "heartbeat_interval_seconds": interval,
            "model_training_completion": "read the trainer run receipt separately"}
        atomic_json(directory / "active-run.json", {"name": name, "state": state_path.name})
        process = None
        try:
            with log.open("x") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(invocation_state="running", child_pid=process.pid)
                while process.poll() is None:
                    state.update(heartbeat_at=datetime.now(timezone.utc).isoformat(),
                                 elapsed_seconds=round(time.monotonic() - started, 3))
                    atomic_json(state_path, state)
                    if stop_path.exists():
                        request = json.loads(stop_path.read_text())
                        if request.get("name") != name or not isinstance(request.get("reason"), str) or not request["reason"].strip():
                            raise ValueError("Stop request must explicitly identify this invocation and its reason")
                        state.update(invocation_state="stopped_by_request", stop_reason=request["reason"])
                        _stop(process)
                        break
                    if time.monotonic() - started >= timeout:
                        state["invocation_state"] = "timed_out"
                        _stop(process)
                        break
                    time.sleep(interval)
                if state["invocation_state"] == "running":
                    state["invocation_state"] = "exited"
        except BaseException as exc:
            state.update(invocation_state="supervisor_interrupted" if isinstance(exc, KeyboardInterrupt) else "supervisor_failed",
                         failure_type=type(exc).__name__)
            raise
        finally:
            _stop(process)
            state.update(heartbeat_at=datetime.now(timezone.utc).isoformat(),
                         finished_at=datetime.now(timezone.utc).isoformat(),
                         elapsed_seconds=round(time.monotonic() - started, 3),
                         child_returncode=process.returncode if process is not None else None,
                         child_stopped=process is None or process.poll() is not None,
                         max_waited_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024)
            atomic_json(state_path, state)
            fcntl.flock(lock, fcntl.LOCK_UN)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--inference-python", type=Path, required=True)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=14400.)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    if not train_args or train_args[0] != "train-text" or not 1 <= args.interop_threads <= 16:
        parser.error("Provide train-text arguments and a bounded interop thread count")
    # Keep the virtualenv executable path; resolving its symlink selects the
    # base interpreter on some hosts. The child records its actual Torch runtime.
    command = [str(args.inference_python.absolute()), "-c",
        "import runpy,torch; torch.set_num_interop_threads(" + str(args.interop_threads) + "); runpy.run_module('tools.adaptation',run_name='__main__')",
        *train_args]
    try:
        result = run_supervised(command, args.state_dir, args.name, args.log, timeout=args.timeout)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Training supervision failed: {exc}\n")
    print(json.dumps(result))
    return 0 if result["child_returncode"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
