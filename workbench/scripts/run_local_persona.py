"""Launch three loopback services with project-local models and logs."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from scripts.local_persona import PROJECT, UPSTREAM, RUNTIME, elevenlabs_config, prepare, runtime_studio
from pipeline.common import read_json, digest, write_json, load_env


def wait_ready(url, child, timeout=180):
    deadline = time.monotonic() + timeout
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"A local service stopped ({child.returncode}); inspect its log.")
        try:
            with client.open(url, timeout=2) as response:
                if response.status == 200:
                    json.load(response)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise TimeoutError(f"Local service did not become ready: {url}")


def main():
    load_env(PROJECT / ".env.live")
    os.environ.setdefault("BIAS_ADAPTIVE_RESPONSES", "true")
    external_ports = {port for kind, port in (("LLM", 8103), ("TTS", 8104)) if os.getenv(f"BIAS_{kind}_BASE_URL")}
    if elevenlabs_config():
        external_ports.add(8104)
    for port in (8103, 8104, 8105):
        if port in external_ports:
            continue
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is already in use. Existing demo: http://127.0.0.1:8105")
    reference = PROJECT / "data/local/persona/jensen-huang/voice-reference"
    if 8104 not in external_ports:
        proof = read_json(reference / "reference.json")
        if digest(reference / "reference.wav") != proof["sha256"] or digest(reference / "reference.txt") != proof["transcript_sha256"]:
            raise ValueError("Voice reference changed; update its review record first")
    output = prepare()
    runtime = read_json(RUNTIME)
    studio = runtime_studio(runtime)
    common = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (PROJECT, studio / "src", studio))),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        HF_HOME=str(PROJECT / ".tools/hf-home"), HF_HUB_CACHE=str(PROJECT / ".tools/hf-cache"),
        TOKENIZERS_PARALLELISM="false")
    voice = dict(common, BIAS_VOICE_MODEL=runtime["models"]["voice"]["repo"],
        BIAS_VOICE_REVISION=runtime["models"]["voice"]["revision"],
        BIAS_VOICE_CACHE_DIR=str(PROJECT / ".tools/hf-cache"), BIAS_VOICE_LOCAL_FILES_ONLY="true",
        BIAS_VOICE_DEVICE="mps", BIAS_VOICE_DTYPE="float32", BIAS_VOICE_ATTENTION="sdpa",
        BIAS_VOICE_REFERENCE_WAV=str(reference / "reference.wav"),
        BIAS_VOICE_REFERENCE_TEXT_FILE=str(reference / "reference.txt"),
        BIAS_VOICE_MAX_NEW_TOKENS="1000", BIAS_VOICE_MAX_INPUT_CHARS="3000",
        BIAS_VOICE_MAX_AUDIO_SECONDS="80", BIAS_VOICE_THREADS="4", PYTORCH_ENABLE_MPS_FALLBACK="1")
    app_env = common.copy()
    rhubarb = PROJECT / ".tools/rhubarb/arm-build/rhubarb/rhubarb"
    if os.access(rhubarb, os.X_OK):
        app_env.setdefault("BIAS_LIPSYNC_BIN", str(rhubarb))
    commands = [
        ("llm", [str(PROJECT / ".tools/venv-llm/bin/mlx_lm.server"), "--model", runtime["models"]["llm"]["path"],
            "--host", "127.0.0.1", "--port", "8103", "--temp", "0.7", "--top-p", "0.8", "--top-k", "20",
            "--max-tokens", "280", "--prefill-step-size", "512", "--prompt-cache-size", "2",
            "--decode-concurrency", "1", "--prompt-concurrency", "1",
            "--allowed-origins", "http://127.0.0.1:8105"], common, "http://127.0.0.1:8103/v1/models"),
        ("voice", [str(PROJECT / ".tools/venv-voice/bin/python"), "-m", "uvicorn", "scripts.local_voice:app",
            "--host", "127.0.0.1", "--port", "8104"], voice, "http://127.0.0.1:8104/health"),
        ("app", [str(UPSTREAM / ".venv/bin/python"), "-m", "uvicorn", "scripts.local_persona:create_app", "--factory",
            "--host", "127.0.0.1", "--port", "8105"], app_env, "http://127.0.0.1:8105/api/health"),
    ]
    commands = [command for command, port in zip(commands, (8103, 8104, 8105)) if port not in external_ports]
    processes, handles = [], []

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for name, command, env, health in commands:
            log = (output / f"{name}.log").open("w")
            handles.append(log)
            child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(child)
            print(f"Starting {name}; log: {output / (name + '.log')}", flush=True)
            wait_ready(health, child)
        write_json(output / "processes.json", {"launcher": os.getpid(), **{c[0]: p.pid for c, p in zip(commands, processes)}})
        print("READY http://127.0.0.1:8105 — AI persona simulation, experimental voice", flush=True)
        if "--no-open" not in sys.argv:
            subprocess.run(["/usr/bin/open", "http://127.0.0.1:8105"], check=False)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        raise RuntimeError("A service stopped; check the run logs.")
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Local demo stopped.")
