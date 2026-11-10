"""Run both development servers and stop their owned process groups on exit."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    npm = shutil.which("npm")
    if not npm or not (root / "node_modules").exists():
        print("Install Node and run npm ci first.", file=sys.stderr)
        return 2
    processes: list[subprocess.Popen] = []
    stopping = False
    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, stop)
    try:
        for command in ([sys.executable, "-m", "uvicorn", "bias.api:app", "--host", "127.0.0.1", "--port", "8000", "--reload"], [npm, "run", "dev", "--workspace", "@bias/web"]):
            processes.append(subprocess.Popen(command, cwd=root, start_new_session=os.name != "nt"))
        print("Studio: http://127.0.0.1:5173 | API: http://127.0.0.1:8000", flush=True)
        while not stopping and all(process.poll() is None for process in processes):
            time.sleep(0.25)
        return next((process.returncode for process in processes if process.returncode), 0)
    finally:
        for process in processes:
            if process.poll() is None:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()


if __name__ == "__main__":
    raise SystemExit(main())

