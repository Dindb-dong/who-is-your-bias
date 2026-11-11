"""Launch real local LLM, voice, and app servers in one process namespace.

The output intentionally contains no browser first-visible or audible latency.
Those require the application's actual LiveTiming export on a working browser.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .protocol import pcm_info, speech_text, sse_events

QWEN_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"
TOKENIZER_REVISION = "7dd38ad4e9bad454aae9cd937d0cd577604fe229"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def tree_identity(root: Path, paths: list[Path]) -> list[dict]:
    return [{**file_identity(path), "path": str(path.relative_to(root))} for path in sorted(paths) if path.is_file()]


def host_identity() -> dict:
    cpu = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    model = next((line.split(":", 1)[1].strip() for line in cpu.splitlines() if line.startswith("model name")), None)
    memory = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value = line.split(":", 1)
                memory[key] = value.strip()
    return {"scope": "Assistant execution host; not the user's device", "platform": platform.platform(),
        "machine": platform.machine(), "cpu_model": model, "logical_cpus": os.cpu_count(),
        "available_cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "memory": memory, "python": platform.python_version(), "inference_device": "cpu"}


class Processes:
    def __init__(self, output: Path):
        self.output, self.items, self.logs = output, {}, []
        self.stop = threading.Event()
        self.peak_sampled_sum_rss_kib = 0
        self.started = time.perf_counter()
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def start(self, name: str, command: list[str], cwd: Path, env: dict) -> subprocess.Popen:
        log = (self.output / f"{name}.log").open("x")
        self.logs.append(log)
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.items[name] = proc
        return proc

    def _monitor(self) -> None:
        with (self.output / "process-memory.jsonl").open("x") as stream:
            while not self.stop.is_set():
                rows, total = {}, 0
                for name, proc in list(self.items.items()):
                    try:
                        status = Path(f"/proc/{proc.pid}/status").read_text()
                        row = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in status.splitlines()
                            if line.startswith(("VmRSS:", "VmHWM:", "Threads:", "State:"))}
                        total += int(row.get("VmRSS", "0 kB").split()[0])
                        rows[name] = row
                    except (OSError, ValueError):
                        rows[name] = {"returncode": proc.poll()}
                self.peak_sampled_sum_rss_kib = max(total, self.peak_sampled_sum_rss_kib)
                stream.write(json.dumps({"elapsed_seconds": round(time.perf_counter() - self.started, 3), "processes": rows,
                    "sum_rss_kib": total}) + "\n")
                stream.flush()
                self.stop.wait(1)

    def close(self) -> None:
        for proc in reversed(list(self.items.values())):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
        self.stop.set()
        self.monitor.join(timeout=2)
        for log in self.logs:
            log.close()


def ready(client: httpx.Client, url: str, proc: subprocess.Popen, seconds: float = 120) -> dict:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Service exited with code {proc.returncode}; inspect its preserved log")
        try:
            response = client.get(url, timeout=2)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            time.sleep(0.2)
    raise TimeoutError("Service readiness timeout")


def local_json(client: httpx.Client, url: str) -> dict:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def collect_turn(client: httpx.Client, base: str, row: dict, output: Path) -> None:
    start = time.perf_counter()
    elapsed = lambda: round((time.perf_counter() - start) * 1000, 3)
    row["submitted_at"] = now()
    timing = row["transport"] = {"first_token_received_ms": None, "first_content_text_received_ms": None}
    row["browser"] = {"first_visible_text_ms": None, "audio_start_ms": None,
        "first_audio_signal_ms": None, "turn_complete_ms": None, "status": "not_measured"}
    row["acoustic_first_audible_ms"] = None
    row["events"] = []
    row["streamed_text"] = ""
    with (output / f"{row['id']}-events.jsonl").open("x") as event_log:
        with client.stream("POST", base + "/api/chat/stream", json=row["request"]) as response:
            row["chat_http_status"] = response.status_code
            response.raise_for_status()
            for event, data in sse_events(response.iter_lines()):
                at = elapsed()
                row["events"].append(event)
                event_log.write(json.dumps({"event": event, "at_ms": at, "data": data}, ensure_ascii=False) + "\n")
                event_log.flush()
                if event == "token":
                    text = data.get("text")
                    if not isinstance(text, str):
                        raise ValueError("Invalid token payload")
                    row["streamed_text"] += text
                    if len(row["streamed_text"]) > 100000:
                        raise ValueError("The generated text exceeds the benchmark bound")
                    if text and timing["first_token_received_ms"] is None:
                        timing["first_token_received_ms"] = at
                    if text.strip() and timing["first_content_text_received_ms"] is None:
                        timing["first_content_text_received_ms"] = at
                elif event == "result":
                    row["turn"] = data["turn"]
                    timing["answer_received_ms"] = at
                elif event == "error":
                    row["api_error"] = data
                    raise RuntimeError("Application returned an error event")
    timing["stream_complete_ms"] = elapsed()
    turn = row.get("turn", {})
    if row["events"][-1:] != ["done"] or turn.get("provider") != "openai" or not turn.get("text"):
        raise ValueError("Expected a completed actual model answer")
    row["speech_input"] = speech_text(turn["text"])
    row["server_chat_timing"] = turn.get("metrics")
    timing["speech_request_started_ms"] = elapsed()
    response = client.post(base + "/api/speech/prepare", json={"text": row["speech_input"], "language": row["request"]["language"]})
    timing["speech_package_received_ms"] = elapsed()
    row["speech_http_status"] = response.status_code
    response.raise_for_status()
    package = response.json()
    row["speech_package"] = package
    row["server_speech_timing"] = package.get("server_timing")
    audio_url = package.get("audio_url", "")
    if not re.fullmatch(r"/api/speech/clips/[a-f0-9]{32}", audio_url):
        raise ValueError("Speech package contains an unexpected audio URL")
    timing["audio_request_started_ms"] = elapsed()
    data = bytearray()
    with client.stream("GET", base + audio_url) as response:
        row["audio_http_status"] = response.status_code
        response.raise_for_status()
        for chunk in response.iter_bytes():
            if chunk and not data:
                timing["audio_first_byte_received_ms"] = elapsed()
            data.extend(chunk)
            if len(data) > 20_000_000:
                raise ValueError("Generated audio exceeds the benchmark bound")
    timing["audio_available_ms"] = elapsed()
    path = output / f"{row['id']}.wav"
    path.write_bytes(data)
    row["audio"] = {"path": path.name, **pcm_info(bytes(data), package["audio_sha256"])}
    timing["inspection_complete_ms"] = elapsed()
    if row["audio"]["offline_pcm_onset_offset_ms"] is None:
        raise ValueError("The generated WAV contains no signal above the stated threshold")
    if abs(row["audio"]["duration_ms"] - package["audio_duration_ms"]) > 0.1:
        raise ValueError("Speech package and WAV duration disagree")
    if not isinstance(row["server_speech_timing"], dict):
        raise ValueError("This application revision does not expose speech server timing")
    row["status"] = "completed"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("app-repo", "app-python", "voice-python", "llama-server", "llm-metadata", "voice-cache", "reference-wav", "reference-text", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--corpus-dir", type=Path)
    result.add_argument("--question", default="How does tunnel diameter affect cost? One sentence, please.")
    result.add_argument("--repeats", type=int, default=2)
    result.add_argument("--mode", choices=("profile", "rag", "behavior"), default="behavior")
    result.add_argument("--language", choices=("en", "ko"), default="en")
    result.add_argument("--llm-threads", type=int, default=4)
    result.add_argument("--voice-threads", type=int, default=2)
    result.add_argument("--context", type=int, default=4096)
    result.add_argument("--max-output-tokens", type=int, default=96)
    result.add_argument("--max-voice-tokens", type=int, default=384)
    result.add_argument("--timeout", type=float, default=180)
    result.add_argument("--port-base", type=int, default=8120)
    return result


def main() -> int:
    cli = parser()
    args = cli.parse_args()
    if not 1 <= args.repeats <= 10 or not 1 <= args.timeout <= 180:
        cli.error("Use 1–10 repetitions and a timeout of 1–180 seconds")
    if not 1 <= args.llm_threads <= 16 or not 1 <= args.voice_threads <= 16 or not 512 <= args.context <= 8192:
        cli.error("CPU threads/context exceed the bounded benchmark settings")
    if not 1 <= args.max_output_tokens <= 512 or not 1 <= args.max_voice_tokens <= 1536 or not 1024 <= args.port_base <= 65533:
        cli.error("Output budget or port is out of bounds")
    if not args.question.strip() or len(args.question) > 2000:
        cli.error("Use a question of 1–2000 characters")
    for name in ("app_repo", "app_python", "voice_python", "llama_server", "llm_metadata", "voice_cache", "reference_wav", "reference_text"):
        # Resolving a virtualenv's Python symlink selects the base interpreter
        # and silently drops its installed dependencies. Keep its entry path.
        value = getattr(args, name).expanduser().absolute()
        if not value.exists():
            cli.error(f"Missing {name}: {value}")
        setattr(args, name, value)
    corpus = (args.corpus_dir or args.app_repo / "data/qna/elon-musk-v1").resolve()
    if not corpus.is_dir():
        cli.error("The corpus directory does not exist")
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        cli.error("Refusing to overwrite an existing benchmark directory")
    for port in range(args.port_base, args.port_base + 3):
        with socket.socket() as check:
            if check.connect_ex(("127.0.0.1", port)) == 0:
                cli.error(f"Port {port} already has a listener")
    args.output.mkdir(parents=True)
    report = {"schema_version": "demo-http-latency.v1", "started_at": now(), "execution": "actual_localhost_http",
        "scope": "Server transport and complete WAV acquisition, not browser or acoustic latency",
        "host": host_identity(), "runs": [], "completed": False,
        "limitations": ["HTTP first-token receipt does not measure visible browser text.",
            "WAV availability is a transport prerequisite for this full-buffer playback path, not first audible sound.",
            "The PCM onset offset is measured within the saved waveform; it is not a playback timestamp.",
            "No microphone capture, ASR input, acoustic output, human listening, or target-device timing is measured.",
            "Two repetitions are an execution pilot, not a percentile or performance benchmark distribution.",
            "The first measured LLM request follows llama.cpp startup warmup. The first voice request includes lazy load and reference prompt preparation.",
            "Model output is preserved unchanged; semantic support and voice fidelity require independent review."]}
    report_path = args.output / "report.json"
    processes = None
    try:
        metadata = json.loads(args.llm_metadata.read_text())
        model = Path(metadata["path"]).expanduser().resolve()
        actual = file_identity(model)
        if actual["sha256"] != metadata["sha256"] or actual["bytes"] != metadata["bytes"]:
            raise ValueError("The local GGUF differs from its model metadata")
        snapshots = []
        for model_id, revision in ((QWEN_MODEL, QWEN_REVISION), ("Qwen/Qwen3-TTS-Tokenizer-12Hz", TOKENIZER_REVISION)):
            snapshot = args.voice_cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
            files = [p for p in snapshot.rglob("*") if p.is_file()]
            if not files:
                raise ValueError(f"Pinned voice snapshot is absent: {model_id}")
            snapshots.append({"model": model_id, "revision": revision, "files": tree_identity(snapshot, files)})
        report["models"] = {"llm": {**metadata, "verified_local_file": actual}, "voice": snapshots,
            "reference_wav": file_identity(args.reference_wav), "reference_text": file_identity(args.reference_text),
            "reference_review": "Engineering candidate; human listening and speaker fidelity ungraded"}
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.app_repo, capture_output=True, text=True, check=True)
        report["application"] = {"repository": str(args.app_repo), "git_head": git.stdout.strip(),
            "source_files": tree_identity(args.app_repo, list((args.app_repo / "src/bias").glob("*.py")) + list((args.app_repo / "services/voice").glob("*.py"))),
            "corpus_files": tree_identity(corpus, list(corpus.rglob("*.json")) + list(corpus.rglob("*.jsonl")))}
        report["harness_files"] = tree_identity(Path(__file__).parent, list(Path(__file__).parent.glob("*.py")))
        root_env = {key: value for key, value in os.environ.items() if not key.startswith("BIAS_")}
        root_env.update({"PYTHONPATH": str(args.app_repo / "src") + os.pathsep + str(args.app_repo),
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost", "PYTHONUNBUFFERED": "1",
            "ORT_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_CACHE": str(args.voice_cache)})
        llm_url, voice_url, app_url = [f"http://127.0.0.1:{args.port_base + offset}" for offset in range(3)]
        llm_command = [str(args.llama_server), "--model", str(model), "--host", "127.0.0.1", "--port", str(args.port_base),
            "--alias", "qwen3-4b-instruct-q4", "--ctx-size", str(args.context), "--parallel", "1",
            "--threads", str(args.llm_threads), "--threads-batch", str(args.llm_threads),
            "--batch-size", "128", "--ubatch-size", "64", "--n-gpu-layers", "0", "--seed", "42",
            "--temp", "0.7", "--top-k", "20", "--top-p", "0.8", "--predict", str(args.max_output_tokens),
            "--offline", "--no-webui", "--metrics"]
        voice_settings = {"MODEL": QWEN_MODEL, "REVISION": QWEN_REVISION, "CACHE_DIR": str(args.voice_cache),
            "LOCAL_FILES_ONLY": "true", "DEVICE": "cpu", "DTYPE": "bfloat16", "ATTENTION": "sdpa",
            "REFERENCE_WAV": str(args.reference_wav), "REFERENCE_TEXT_FILE": str(args.reference_text),
            "X_VECTOR_ONLY_MODE": "false", "THREADS": str(args.voice_threads), "SEED": "42",
            "MAX_NEW_TOKENS": str(args.max_voice_tokens), "MAX_AUDIO_SECONDS": "60", "DO_SAMPLE": "true",
            "TEMPERATURE": "0.9", "TOP_P": "1.0", "TOP_K": "50", "REPETITION_PENALTY": "1.05"}
        voice_env = {**root_env, **{"BIAS_VOICE_" + key: value for key, value in voice_settings.items()},
            "OMP_NUM_THREADS": str(args.voice_threads), "MKL_NUM_THREADS": str(args.voice_threads)}
        app_env = {**root_env, "BIAS_LLM_BASE_URL": llm_url + "/v1", "BIAS_LLM_MODEL": "qwen3-4b-instruct-q4",
            "BIAS_LLM_MAX_TOKENS": str(args.max_output_tokens), "BIAS_REQUEST_TIMEOUT": str(args.timeout),
            "BIAS_TTS_BASE_URL": voice_url + "/v1", "BIAS_TTS_MODEL": "qwen3-tts", "BIAS_TTS_VOICE": "reference",
            "BIAS_TTS_LANGUAGE_HINT": "true", "BIAS_CORPUS_DIR": str(corpus)}
        voice_command = [str(args.voice_python), "-m", "uvicorn", "services.voice.app:app", "--host", "127.0.0.1", "--port", str(args.port_base + 1)]
        app_command = [str(args.app_python), "-m", "uvicorn", "bias.api:app", "--host", "127.0.0.1", "--port", str(args.port_base + 2)]
        report["runtime"] = {"llama_binary": file_identity(args.llama_server), "commands": {"llm": llm_command, "voice": voice_command, "app": app_command},
            "voice_settings": voice_settings, "request_timeout_seconds": args.timeout, "lipsync": "not_configured",
            "max_output_tokens": args.max_output_tokens, "startup_warmup": "llama.cpp default enabled", "speech_streaming": False}
        save(report_path, report)
        processes = Processes(args.output)
        with httpx.Client(timeout=args.timeout + 5, trust_env=False, follow_redirects=False) as client:
            for name, command, env, url in (("llm", llm_command, root_env, llm_url + "/health"),
                    ("voice", voice_command, voice_env, voice_url + "/health"), ("app", app_command, app_env, app_url + "/api/health")):
                started = time.perf_counter()
                proc = processes.start(name, command, args.app_repo, env)
                report[name + "_initial_health"] = ready(client, url, proc)
                report.setdefault("startup_seconds", {})[name] = round(time.perf_counter() - started, 6)
                save(report_path, report)
            for index in range(args.repeats):
                row = {"id": f"turn-{index + 1:02}", "status": "running", "new_conversation": True,
                    "request": {"message": args.question, "mode": args.mode, "language": args.language, "motion_mode": "disabled"},
                    "voice_before": local_json(client, voice_url + "/health"),
                    "condition": "first_request_after_llm_startup_warmup_voice_cold" if index == 0 else "warm_processes_repeated_prompt_new_conversation"}
                report["runs"].append(row)
                save(report_path, report)
                try:
                    collect_turn(client, app_url, row, args.output)
                except Exception as exc:
                    row["status"] = "failed"
                    row["failure"] = {"type": type(exc).__name__, "message": str(exc)}
                    raise
                finally:
                    save(report_path, report)
                row["voice_after"] = local_json(client, voice_url + "/health")
                save(report_path, report)
                print(json.dumps({"id": row["id"], "status": row["status"], "transport": row["transport"],
                    "server_speech_timing": row["server_speech_timing"], "audio_duration_ms": row["audio"]["duration_ms"]}), flush=True)
            metrics = client.get(llm_url + "/metrics")
            metrics.raise_for_status()
            (args.output / "llama-metrics.txt").write_text(metrics.text)
            report["completed"] = True
    except BaseException as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print(json.dumps({"error": type(exc).__name__, "message": str(exc), "output": str(args.output)}), file=sys.stderr, flush=True)
    finally:
        if processes is not None:
            processes.close()
            report["process_observation"] = {"peak_sampled_sum_rss_kib": processes.peak_sampled_sum_rss_kib,
                "sampling_interval_seconds": 1, "scope": "Sum of three direct server RSS values; shared memory may be counted twice, and unsampled spikes are not captured."}
        report["finished_at"] = now()
        report["artifacts"] = [file_identity(path) for path in sorted(args.output.iterdir()) if path.is_file() and path != report_path and path.suffix != ".tmp"]
        save(report_path, report)
    print(json.dumps({"report": str(report_path), "completed": report["completed"], "browser_measured": False}), flush=True)
    return 0 if report["completed"] else 1
