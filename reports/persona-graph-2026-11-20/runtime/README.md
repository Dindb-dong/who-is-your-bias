# Recorded local Qwen3 inference runtime

This directory records experimental infrastructure, separate from the production
application. It contains no API credential and does not alter team
models. A public, official Qwen model was fetched through the environment's normal
network route and verified against the LFS SHA-256 at an immutable revision.

- Model: `Qwen/Qwen3-4B-GGUF`, `Qwen3-4B-Q4_K_M.gguf`
- Revision: `bc640142c66e1fdd12af0bd68f40445458f3869b`
- Size: 2,497,280,256 bytes
- SHA-256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`
- Runtime: Python 3.12.14, llama-cpp-python 0.3.36, 6 CPU threads, no GPU
- Context: 8,192 tokens, overflow explicitly rejected without truncation
- Template: original model-embedded Jinja with `enable_thinking = false`
- State: reset before every request; no shared dialogue history or response cache

Exact dependencies, template and settings are in `requirements.lock.txt`,
`chat-template.jinja`, `model-manifest.json` and `runtime-manifest.json`.
The model itself is not a repository artifact. Its durable recovery source is the
public pinned upstream URL recorded in the manifest. This local copy is scratch.

## Startup smoke

`startup-smoke.json` records an actual model response to a neutral RAG question,
including raw response chunks and wall-clock times. It is only a runtime check,
not a persona score, benchmark or team Mac latency measurement.

The first smoke had 0.616 s time to first content and 2.219 s total response time
for a short prompt and a 31-word answer. Longer persona prompts have different
prefill costs. Quantized CPU inference is not the identical backend/precision as
the team's Mac pipeline and cannot establish production quality or latency.

## Launch a client

In the recorded execution environment each command had its own network namespace. Start the model
and client together; starting the server in one command and connecting from a
different command will fail even though they can share files.

```bash
RUNTIME="$PWD/reports/persona-graph-2026-11-20/runtime"
"$RUNTIME/.venv/bin/python" "$RUNTIME/run_with_server.py" -- \
  /path/to/client/python /path/to/client.py
```

The launcher supplies `PERSONA_EVAL_BASE_URL=http://127.0.0.1:8129/v1` and
`PERSONA_EVAL_MODEL=Qwen3-4B-Q4_K_M`. It terminates the server after the client
finishes. No external listener or user login is used.

For reproduction, create `.venv` inside this directory using Python 3.12 and
install `requirements.lock.txt`. Download the exact file from `source_url` in
`model-manifest.json` into this directory and verify its byte count and SHA-256
before starting. `llama-cpp-python` may require a C/C++ compiler when no compatible
wheel is available. The shared team's ordinary `uv sync` does not install this
optional runtime. The original run used a separate scratch directory; copied
runtime sources and model identity are identical.

- `GET /health`: model/runtime identity and readiness.
- `POST /count_tokens` with `{"messages": [...]}`: exact full chat prompt count
  (including template and empty thinking prefix), context window and prompt hash.
- `POST /v1/chat/completions`: text chat completions, streaming or non-streaming.
  Supported sampling controls: `seed`, `temperature`, `top_p`, `top_k`, `min_p`,
  `presence_penalty`, `frequency_penalty`, `repeat_penalty`, `stop`, `max_tokens`.
  Max output tokens must be 1–1024. Any model other than the loaded ID is rejected.

All effective parameters, exact formatted prompt, raw response and timing are
appended to `requests.jsonl`. This is experimental public-source content only.

The default sampling follows the official Qwen3 non-thinking model card:
temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5, repeat_penalty
1.0. The caller must freeze any chosen overrides in its evaluation protocol.

Before actual experiments, checkpoint evaluation inputs, model and runtime
manifests, controls and these launcher sources. Do not infer persona improvement
from the startup smoke or change evaluation questions after examining outputs.
