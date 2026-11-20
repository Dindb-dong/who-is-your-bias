"""Local-only, single-instance Qwen3 GGUF inference for a recorded pilot.

This is experimental infrastructure, not a production server. The exact upstream
Jinja template is retained; a Jinja variable disables thinking before rendering.
No response rewriting, model-as-judge, or benchmark scoring occurs here.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "model-manifest.json").read_text())
MODEL_ID = "Qwen3-4B-Q4_K_M"
SETTINGS = dict(n_ctx=8192, n_threads=6, n_threads_batch=6, n_batch=256,
                n_gpu_layers=0, seed=20261120, verbose=False)
started = time.perf_counter()
llm = Llama(model_path=str(ROOT / MANIFEST["file"]), **SETTINGS)
template = llm.metadata["tokenizer.chat_template"]
formatter = Jinja2ChatFormatter(
    template="{% set enable_thinking = false %}" + template,
    eos_token="<|im_end|>", bos_token="<|endoftext|>",
    stop_token_ids=[llm.token_eos()],
)
llm.chat_handler = formatter.to_chat_handler()
(ROOT / "chat-template.jinja").write_text(template)
RUNTIME = {
    "model": MODEL_ID, "model_sha256": MANIFEST["sha256"],
    "settings": SETTINGS, "python": platform.python_version(),
    "packages": {name: importlib.metadata.version(name) for name in (
        "llama-cpp-python", "numpy", "jinja2", "fastapi", "uvicorn")},
    "original_chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
    "chat_template_override": "prefix {% set enable_thinking = false %}",
    "thinking": False,
    "cache_policy": "Llama.reset() before each request; no session history or cross-request cache",
}
STARTUP_SECONDS = round(time.perf_counter() - started, 3)
(ROOT / "runtime-manifest.json").write_text(json.dumps(RUNTIME, indent=2) + "\n")
lock = threading.Lock()
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ready", "startup_seconds": STARTUP_SECONDS, **RUNTIME}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model",
                                          "owned_by": "Qwen"}]}


@app.post("/count_tokens")
def count_tokens(body: dict[str, Any]):
    if not isinstance(body.get("messages"), list):
        raise HTTPException(400, "messages must be a list")
    prompt = formatter(messages=body["messages"]).prompt
    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
    return {"prompt_tokens": len(tokens), "count": len(tokens),
            "context_window": SETTINGS["n_ctx"],
            "formatted_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def parameters(body: dict[str, Any]) -> dict[str, Any]:
    if body.get("model", MODEL_ID) != MODEL_ID:
        raise HTTPException(400, f"Only {MODEL_ID} is loaded")
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise HTTPException(400, "A non-empty messages list is required")
    for message in body["messages"]:
        if message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str):
            raise HTTPException(400, "Only text system/user/assistant messages are supported")
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 256))
    if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 1024:
        raise HTTPException(400, "max_tokens must be between 1 and 1024")
    # Official Qwen3 non-thinking sampling defaults; callers may set explicit
    # controls. Every effective control is logged with the raw model result.
    kwargs = dict(messages=body["messages"], model=MODEL_ID,
                  temperature=body.get("temperature", 0.7),
                  top_p=body.get("top_p", 0.8), top_k=body.get("top_k", 20),
                  min_p=body.get("min_p", 0.0),
                  presence_penalty=body.get("presence_penalty", 1.5),
                  frequency_penalty=body.get("frequency_penalty", 0.0),
                  repeat_penalty=body.get("repeat_penalty", 1.0),
                  seed=body.get("seed", 20261120), max_tokens=max_tokens)
    if "stop" in body:
        kwargs["stop"] = body["stop"]
    return kwargs


def record(kwargs, prompt, response, elapsed, first_token_seconds=None):
    record = {"timestamp_unix": time.time(), "parameters": kwargs,
              "formatted_prompt": prompt,
              "formatted_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "elapsed_seconds": elapsed, "first_token_seconds": first_token_seconds,
              "response": response}
    with (ROOT / "requests.jsonl").open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.post("/v1/chat/completions")
def completions(body: dict[str, Any]):
    kwargs = parameters(body)
    prompt = formatter(messages=kwargs["messages"]).prompt
    token_count = len(llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
    if token_count + kwargs["max_tokens"] > SETTINGS["n_ctx"]:
        raise HTTPException(400, f"Prompt ({token_count}) + max_tokens ({kwargs['max_tokens']}) exceeds context ({SETTINGS['n_ctx']}); no truncation performed")
    if body.get("stream", False):
        def generate():
            with lock:
                llm.reset()
                start = time.perf_counter()
                chunks = []
                first = None
                for chunk in llm.create_chat_completion(**kwargs, stream=True):
                    chunks.append(chunk)
                    if first is None and any(c.get("delta", {}).get("content") for c in chunk.get("choices", [])):
                        first = time.perf_counter() - start
                    yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
                record(kwargs, prompt, chunks, time.perf_counter() - start, first)
                yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    with lock:
        llm.reset()
        start = time.perf_counter()
        response = llm.create_chat_completion(**kwargs, stream=False)
        elapsed = time.perf_counter() - start
        record(kwargs, prompt, response, elapsed)
        return response


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8129, log_level="info")
