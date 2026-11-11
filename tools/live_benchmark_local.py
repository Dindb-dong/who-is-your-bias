"""One bounded real local Qwen language/TTS request in one process namespace.

Download pinned models and prepare the reference separately. This harness never
downloads models, trains speech, generates fixture PCM, or approves a reference.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
from pathlib import Path

import httpx

from bias.corpus import Corpus
from bias.retrieval import Retriever
from tools.demo_latency.run import Processes, file_identity, host_identity, now, ready, save, tree_identity
from tools.live_benchmark import run_benchmark


LANGUAGE_MODEL = "Qwen/Qwen3-0.6B"
LANGUAGE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
VOICE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
VOICE_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"
TOKENIZER_MODEL = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
TOKENIZER_REVISION = "7dd38ad4e9bad454aae9cd937d0cd577604fe229"


def snapshot(cache: Path, model: str, revision: str) -> tuple[Path, dict]:
    directory = cache / ("models--" + model.replace("/", "--")) / "snapshots" / revision
    files = [p for p in directory.rglob("*") if p.is_file()]
    if not any(p.suffix == ".safetensors" for p in files) or not (directory / "config.json").is_file():
        raise ValueError("A complete pinned model snapshot is required before execution")
    return directory, {"model": model, "revision": revision, "files": tree_identity(directory, files)}


def verify_embedded_tokenizer(voice_directory: Path, tokenizer_identity: dict) -> None:
    embedded = voice_directory / "speech_tokenizer"
    files = tree_identity(embedded, [p for p in embedded.rglob("*") if p.is_file()])
    observed = {row["path"]: row["sha256"] for row in files}
    expected = {row["path"]: row["sha256"] for row in tokenizer_identity["files"]}
    if not observed or observed != expected:
        raise ValueError("Embedded speech tokenizer differs from the separately pinned tokenizer")


def package_versions(python: Path) -> dict:
    code = """import importlib.metadata as m,json,platform
names=('torch','torchaudio','transformers','huggingface-hub','qwen-tts','peft','accelerate','numpy','librosa','soundfile','sox','fastapi','uvicorn')
values={'python':platform.python_version()}
for name in names:
 try: values[name]=m.version(name)
 except m.PackageNotFoundError: pass
print(json.dumps(values))
"""
    return json.loads(subprocess.run([str(python), "-c", code], check=True, capture_output=True, text=True, timeout=15).stdout)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    for name in ("language-cache", "voice-cache", "reference-wav", "reference-text", "reference-provenance", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("app-python", "language-python", "voice-python", "corpus-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--question", default="How does tunnel diameter affect cost? In 12 words.")
    parser.add_argument("--language", choices=("en", "ko"), default="en")
    parser.add_argument("--mode", choices=("rag", "behavior"), default="behavior")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--max-voice-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--port-base", type=int, default=8140)
    parser.add_argument("--concurrency-note", default="Other host workloads were not independently measured.")
    args = parser.parse_args(argv)
    if not (1 <= args.threads <= 4 and 1 <= args.max_output_tokens <= 256 and 1 <= args.max_voice_tokens <= 768
            and 1 <= args.timeout <= 180 and 1024 <= args.port_base <= 65533):
        parser.error("Threads, token budgets, timeout or port exceed the bounded local harness settings")
    if not args.question.strip() or len(args.question) > 2000:
        parser.error("Use a question of 1–2000 characters")
    args.repo = args.repo.expanduser().absolute()
    defaults = {"app_python": args.repo / ".venv-model/bin/python", "language_python": args.repo / ".venv-model/bin/python",
                "voice_python": args.repo / ".venv-voice/bin/python", "corpus_dir": args.repo / "data/qna/elon-musk-v1"}
    for name in ("app_python", "language_python", "voice_python", "corpus_dir", "language_cache", "voice_cache", "reference_wav", "reference_text", "reference_provenance"):
        value = (getattr(args, name) or defaults[name]).expanduser().absolute()
        if not value.exists():
            parser.error("Missing local prerequisite: " + name)
        # Do not resolve Python symlinks: that discards the virtual environment.
        setattr(args, name, value)
    output = args.output.expanduser().absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        parser.error("Choose a new output directory without symlink ancestors")
    for port in range(args.port_base, args.port_base + 3):
        with socket.socket() as check:
            if check.connect_ex(("127.0.0.1", port)) == 0:
                parser.error("A selected local service port is already occupied")
    output.mkdir(parents=True)
    report = {"schema_version": "live-local-execution.v1", "started_at": now(), "status": "preflight",
              "measured_inference_complete": False, "host": host_identity(), "concurrency_note": args.concurrency_note,
              "limitations": ["One request is an execution measurement, not a latency distribution or isolated hardware benchmark.",
                  "Reference candidate status and prior checks do not establish human listening or speaker similarity.",
                  "No new ASR, speech training, browser playback, avatar rendering or acoustic measurement is performed.",
                  "The first measured request includes lazy model loading; no synthetic warmup is substituted.",
                  "A full returned waveform does not prove the TTS backend terminated at EOS; the backend does not expose that reason."]}
    processes = None
    source_paths = (list((args.repo / "src/bias").glob("*.py")) + list((args.repo / "services/language").glob("*.py"))
                    + list((args.repo / "services/voice").glob("*.py")) + [Path(__file__), args.repo / "tools/live_benchmark.py",
                    args.repo / "tools/demo_latency/run.py", args.repo / "tools/demo_latency/protocol.py"])
    try:
        retrieval = Retriever(Corpus.load(args.corpus_dir)).search(args.question, limit=2, behavior=args.mode == "behavior")
        report["preflight_retrieval"] = {"query": args.question, "evidence_ids": [hit.evidence.id for hit in retrieval.hits],
                                        "reason": retrieval.reason}
        if not retrieval.hits:
            raise ValueError("The selected lexical corpus has no evidence for this measurement question")
        language_directory, language_identity = snapshot(args.language_cache, LANGUAGE_MODEL, LANGUAGE_REVISION)
        voice_directory, voice_identity = snapshot(args.voice_cache, VOICE_MODEL, VOICE_REVISION)
        tokenizer_directory, tokenizer_identity = snapshot(args.voice_cache, TOKENIZER_MODEL, TOKENIZER_REVISION)
        # The service loads the tokenizer embedded in the TTS snapshot. Bind its
        # actual bytes to the separately pinned tokenizer instead of assuming
        # that merely downloading the latter makes it the runtime tokenizer.
        verify_embedded_tokenizer(voice_directory, tokenizer_identity)
        reference = {"wav": file_identity(args.reference_wav), "text": file_identity(args.reference_text),
                     "provenance": file_identity(args.reference_provenance)}
        provenance = json.loads(args.reference_provenance.read_text())
        if (provenance.get("audio_sha256") != reference["wav"]["sha256"]
                or provenance.get("text_sha256") != reference["text"]["sha256"]):
            raise ValueError("Reference audio/text differ from the supplied provenance")
        reference["review_status"] = provenance.get("review_status")
        reference["human_listening_completed"] = provenance.get("human_listening_completed")
        report["models"] = {"language": language_identity, "voice": voice_identity, "speech_tokenizer": tokenizer_identity,
                            "embedded_speech_tokenizer_matches_pinned_files": True}
        report["reference"] = reference
        report["packages"] = {kind: package_versions(python) for kind, python in
                              (("app", args.app_python), ("language", args.language_python), ("voice", args.voice_python))}
        report["application"] = {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repo,
            capture_output=True, text=True, check=True).stdout.strip(), "source_files": tree_identity(args.repo, source_paths),
            "corpus_files": tree_identity(args.corpus_dir, [p for p in args.corpus_dir.rglob("*") if p.is_file()])}
        language_url, voice_url, app_url = [f"http://127.0.0.1:{args.port_base + offset}" for offset in range(3)]
        # Secret environment values are never logged. Service-specific BIAS_
        # settings are replaced so unrelated credentials/adapters cannot leak
        # into this base-model execution. All inference is cache-only.
        common = {key: value for key, value in os.environ.items() if not key.startswith("BIAS_")}
        common.update(PYTHONPATH=str(args.repo / "src") + os.pathsep + str(args.repo), PYTHONUNBUFFERED="1",
            NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost", ORT_DISABLE_TELEMETRY="1",
            HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1")
        language_settings = {"MODEL": LANGUAGE_MODEL, "REVISION": LANGUAGE_REVISION, "CACHE_DIR": str(args.language_cache),
            "SERVED_MODEL": "qwen3-local", "DEVICE": "cpu", "DTYPE": "bfloat16", "LOCAL_FILES_ONLY": "true",
            "THREADS": str(args.threads), "MAX_INPUT_TOKENS": "8192", "MAX_OUTPUT_TOKENS": str(args.max_output_tokens),
            "GENERATION_TIMEOUT": str(args.timeout), "TEMPERATURE": "0", "SEED": "42"}
        voice_settings = {"MODEL": VOICE_MODEL, "REVISION": VOICE_REVISION, "CACHE_DIR": str(args.voice_cache),
            "DEVICE": "cpu", "DTYPE": "bfloat16", "ATTENTION": "sdpa", "LOCAL_FILES_ONLY": "true",
            "REFERENCE_WAV": str(args.reference_wav), "REFERENCE_TEXT_FILE": str(args.reference_text),
            "X_VECTOR_ONLY_MODE": "false", "THREADS": str(args.threads), "SEED": "42", "DO_SAMPLE": "true",
            "MAX_NEW_TOKENS": str(args.max_voice_tokens), "MAX_AUDIO_SECONDS": "60", "TEMPERATURE": "0.9",
            "TOP_K": "50", "TOP_P": "1.0", "REPETITION_PENALTY": "1.05"}
        language_env = {**common, **{"BIAS_LANGUAGE_" + k: v for k, v in language_settings.items()}, "HF_HUB_CACHE": str(args.language_cache)}
        voice_env = {**common, **{"BIAS_VOICE_" + k: v for k, v in voice_settings.items()}, "HF_HUB_CACHE": str(args.voice_cache)}
        app_env = {**common, "BIAS_LLM_BASE_URL": language_url + "/v1", "BIAS_LLM_MODEL": "qwen3-local",
            "BIAS_LLM_MAX_TOKENS": str(args.max_output_tokens), "BIAS_LLM_TOKENIZER_PATH": str(language_directory),
            "BIAS_LLM_CONTEXT_WINDOW": "8192", "BIAS_REQUEST_TIMEOUT": str(args.timeout), "BIAS_RETRIEVAL_LIMIT": "2",
            "BIAS_TTS_BASE_URL": voice_url + "/v1", "BIAS_TTS_MODEL": "qwen3-tts", "BIAS_TTS_VOICE": "reference",
            "BIAS_TTS_LANGUAGE_HINT": "true", "BIAS_CORPUS_DIR": str(args.corpus_dir), "BIAS_PERSONA_SNAPSHOT": "",
            "BIAS_RETRIEVAL_BACKEND": "lexical", "BIAS_STYLE_EXAMPLES": "false", "HF_HUB_CACHE": str(args.language_cache)}
        report["configuration"] = {"language_settings": language_settings, "voice_settings": voice_settings,
            "max_output_tokens": args.max_output_tokens, "request_timeout_seconds": args.timeout,
            "retrieval_backend": "lexical", "retrieval_limit": 2, "style_examples": False, "persona_snapshot": None,
            "question": args.question, "language": args.language, "mode": args.mode, "lipsync": "not_configured"}
        report["status"] = "starting_services"
        save(output / "execution.json", report)
        processes = Processes(output)
        with httpx.Client(timeout=15, trust_env=False, follow_redirects=False) as client:
            for i, (kind, python, module, env, health_url) in enumerate((
                ("language", args.language_python, "services.language.app:app", language_env, language_url + "/health"),
                ("voice", args.voice_python, "services.voice.app:app", voice_env, voice_url + "/health"),
                ("app", args.app_python, "bias.api:app", app_env, app_url + "/api/health"))):
                proc = processes.start(kind, [str(python), "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(args.port_base + i)], args.repo, env)
                ready(client, health_url, proc, seconds=60)
        report["status"] = "measuring"
        save(output / "execution.json", report)
        measured = asyncio.run(run_benchmark(base_url=app_url, language_health_url=language_url + "/health",
            voice_health_url=voice_url + "/health", message=args.question, language=args.language, mode=args.mode,
            timeout=args.timeout, cleanup_timeout=5))
        save(output / "http.json", measured)
        report.update(status=measured["status"], measured_inference_complete=measured["measured_inference_complete"],
                      http_report={"path": "http.json", "sha256": file_identity(output / "http.json")["sha256"]})
        report["application"]["source_files_unchanged"] = report["application"]["source_files"] == tree_identity(args.repo, source_paths)
        if not report["application"]["source_files_unchanged"]:
            report.update(status="failed", measured_inference_complete=False, failure={"type": "ApplicationSourceChanged"})
    except Exception as exc:
        report.update(status="failed", measured_inference_complete=False,
            failure={"type": type(exc).__name__, "reason": "Local prerequisites, startup or execution failed; inspect preserved logs and receipts."})
    finally:
        if processes is not None:
            processes.close()
            report["processes"] = {name: {"returncode_after_cleanup": proc.returncode} for name, proc in processes.items.items()}
            report["peak_sampled_service_sum_rss_bytes"] = processes.peak_sampled_sum_rss_kib * 1024
            report["logs"] = tree_identity(output, list(output.glob("*.log")) + list(output.glob("*.jsonl")))
        report["completed_at"] = now()
        save(output / "execution.json", report)
    print(json.dumps({"status": report["status"], "measured_inference_complete": report["measured_inference_complete"],
                      "output": str(output)}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
