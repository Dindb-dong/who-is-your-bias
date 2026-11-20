"""Run one command alongside local inference in this exec network namespace."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def main():
    command = sys.argv[1:]
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("usage: run_with_server.py -- COMMAND [ARGS...]")
    with (ROOT / "server.log").open("a") as log:
        server = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "server.py")],
                                  stdout=log, stderr=log)
        try:
            start = time.monotonic()
            while time.monotonic() - start < 90:
                if server.poll() is not None:
                    raise RuntimeError(f"inference server exited {server.returncode}; see {ROOT / 'server.log'}")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8129/health", timeout=1) as response:
                        health = json.load(response)
                    break
                except OSError:
                    time.sleep(0.2)
            else:
                raise RuntimeError("inference server did not become ready within 90 seconds")
            print(json.dumps({"local_runtime_ready": health["model"],
                              "thinking": health["thinking"],
                              "startup_seconds": health["startup_seconds"]}), flush=True)
            env = os.environ.copy()
            env["PERSONA_EVAL_BASE_URL"] = "http://127.0.0.1:8129/v1"
            env["PERSONA_EVAL_MODEL"] = "Qwen3-4B-Q4_K_M"
            result = subprocess.run(command, env=env)
            return result.returncode
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
