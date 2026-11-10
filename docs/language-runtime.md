# Local language runtime

The optional `services.language` server connects a real local Transformers model to the existing main application. The browser continues to call the main application's `/api/chat` and `/api/chat/stream` routes. Provider URLs, model configuration, and credentials stay in server environment variables.

This is a small, single-worker research runtime. It has lazy loading, bounded requests, a bounded output queue, streamed text, token usage, and cooperative cancellation. It does not train a persona or repair a model's missing citations. A generated response without valid retrieved citation identifiers remains `unverified` in the main application.

## Install and cache a model

Use a separate environment so the main application does not need PyTorch or Transformers merely to start. From the repository root:

```bash
python3.12 -m venv .venv-language
# Select a matching PyTorch build for the machine first.
.venv-language/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.11.0+cpu'
.venv-language/bin/python -m pip install -e . -r services/language/requirements.txt
```

The language requirements pin Transformers 4.57.3, the version used by the local execution checks. The optional voice environment may have additional packages and does not replace this runtime's declared dependency file. The server sets `ORT_DISABLE_TELEMETRY=1` before optional model imports; set it in the process environment as well when launching other inference tools.

The documented small model is [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), pinned for reproduction to revision `c1899de289a04d12100db370d81485cdf75e47ca`. Its published chat template supports `enable_thinking=False`; this runtime uses that option and the publisher's non-thinking sampling defaults: temperature 0.7, top-p 0.8, and top-k 20. A small model's ability to produce fluent text is separate from instruction following, citation compliance, and persona quality.

Download the public model explicitly to a cache outside the repository:

```bash
ORT_DISABLE_TELEMETRY=1 .venv-language/bin/hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --cache-dir /path/outside/repository/hf-cache
```

The runtime defaults to `local_files_only=true`. It resolves the cached repository/revision to a local snapshot directory before handing it to Transformers, avoiding optional Hub metadata probes under strict offline mode. A missing cache or revision produces an explicit model-unavailable error instead of starting a download during a chat request. `trust_remote_code` is disabled and model weights are loaded through safetensors. No model files belong in Git.

## Start the service and the application

Run the model server in one terminal:

```bash
export ORT_DISABLE_TELEMETRY=1
export BIAS_LANGUAGE_MODEL=Qwen/Qwen3-0.6B
export BIAS_LANGUAGE_SERVED_MODEL=qwen3-local
export BIAS_LANGUAGE_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
export BIAS_LANGUAGE_CACHE_DIR=/path/outside/repository/hf-cache
export BIAS_LANGUAGE_LOCAL_FILES_ONLY=true
export BIAS_LANGUAGE_DEVICE=cpu
export BIAS_LANGUAGE_DTYPE=bfloat16
export BIAS_LANGUAGE_THREADS=4
export BIAS_LANGUAGE_MAX_OUTPUT_TOKENS=160
export BIAS_LANGUAGE_GENERATION_TIMEOUT=120
PYTHONPATH=src:. .venv-language/bin/python -m uvicorn \
  services.language.app:app --host 127.0.0.1 --port 8020
```

The shown bfloat16 configuration matches a CPU with suitable support used in the execution check. The default is float32 for portability; choose a device and precision supported by the actual machine. A previous float32 model-loading attempt in this workspace exited with code 137, as recorded in [inference-validation.md](inference-validation.md). The cause was not independently established, and there is no measured combined LLM/TTS/vision memory guarantee.

Run the main application in another terminal:

```bash
export BIAS_LLM_BASE_URL=http://127.0.0.1:8020/v1
export BIAS_LLM_MODEL=qwen3-local
export BIAS_LLM_MAX_TOKENS=160
export BIAS_REQUEST_TIMEOUT=150
uv run bias serve --host 127.0.0.1 --port 8000
```

Use the existing frontend or built static application with this main API. The LLM token budget must not exceed the language server's configured maximum. The main adapter retains its normal 900-token default unless `BIAS_LLM_MAX_TOKENS` is set. A main-application timeout longer than the language-server deadline lets the server return a useful timeout error first.

For a credential-protected local endpoint, give the language process `BIAS_LANGUAGE_API_KEY` and the main process the matching `BIAS_LLM_API_KEY`. These values are read from the environment and are not returned by either health endpoint. Do not put them in frontend build variables. The example binds both services to loopback; use one uvicorn worker because each additional process would load its own model and have its own admission lock.

The [example environment file](../services/language/.env.example) lists the core settings. Merely creating an environment file does not load it; export its values or use your process manager's environment mechanism.

## Protocol and resource behavior

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Reports whether the model is loaded, whether a worker is busy, its stage, configured limits, load duration when measured, and process peak RSS where supported. It does not load the model. |
| `GET /v1/models` | Returns the configured public model alias; requires the configured bearer credential. |
| `POST /v1/chat/completions` | Text-only system/user/assistant messages; either JSON completion or OpenAI-compatible SSE. |

The request supports `model`, `messages`, `stream`, `max_tokens`, optional temperature, top-p, seed, `n=1`, and `stream_options.include_usage`. Unknown fields, tools, multimodal content, multiple completions, and unserved model names are rejected. The service does not pretend to implement the full OpenAI API.

For SSE, the service sends an assistant role chunk, decoded text chunks, optional measured usage, a finish chunk, and `[DONE]`. Usage is enabled by default and may be disabled with `stream_options.include_usage=false`. It precedes the finish chunk so a client that rejects `finish_reason="length"` can still account for work performed. Empty-choice usage chunks and keepalive comments are tolerated by the application's existing adapter. Usage counts come from the tokenized prompt and the actual generated sequence, including generated EOS; a visible text chunk is not a token.

The decoder uses Transformers' [`TextStreamer` callback](https://huggingface.co/docs/transformers/v4.57.3/en/internal/generation_utils#transformers.TextStreamer) to emit finalized text. That means the first visible chunk may arrive after several generated tokens. Cancellation is checked through [`StoppingCriteria`](https://huggingface.co/docs/transformers/v4.57.3/en/internal/generation_utils#transformers.StoppingCriteria), output callbacks, and the consumer deadline.

Admission is immediate: one request owns the inference worker, and overlapping requests receive HTTP 429 with `Retry-After`. There is no unbounded waiting queue. The worker holds admission through model loading and generation. A client disconnect or deadline sets a cancellation flag; an already running model kernel or file-loading operation cannot be interrupted safely by that flag. Admission remains occupied until the worker actually returns. Health reports `stage="stopping"` during that interval. An unresponsive underlying operation may require restarting the model process.

The prompt has both character and token limits. Requests exceeding them are rejected; source context is never silently truncated to fit. Output tokens, output characters, queued chunks, request bytes, and body-read time are bounded separately. The body-byte guard counts received chunks even without a `Content-Length` header.

| Setting | Default | Purpose |
| --- | ---: | --- |
| `BIAS_LANGUAGE_MAX_INPUT_CHARS` | 32,000 | Sum of message text lengths |
| `BIAS_LANGUAGE_MAX_INPUT_TOKENS` | 8,192 | Tokenized prompt cap, also checked against model context capacity |
| `BIAS_LANGUAGE_MAX_OUTPUT_TOKENS` | 1,024 | Maximum accepted request output budget |
| `BIAS_LANGUAGE_MAX_OUTPUT_CHARS` | 12,000 | Decoded output cap |
| `BIAS_LANGUAGE_MAX_BODY_BYTES` | 131,072 | JSON body byte cap |
| `BIAS_LANGUAGE_QUEUE_SIZE` | 64 | Bounded finalized-text queue |
| `BIAS_LANGUAGE_GENERATION_TIMEOUT` | 120 seconds | Deadline from admitted request, including lazy loading |
| `BIAS_LANGUAGE_THREADS` | 4 | PyTorch CPU thread count |
| `BIAS_LANGUAGE_SEED` | 42 | Default sampling seed; output may vary with runtime/hardware |

Errors before streaming starts use an HTTP error with `{ "error": { "code", "message", "type" } }`. Loading, context, or inference errors discovered after SSE headers are sent become an error event without a success marker. Reaching the requested token limit returns `finish_reason="length"`; the main adapter reports `provider_truncated` and does not store a completed conversation turn. A failed or cancelled response is never replaced with fabricated model output.

## Use an existing model server instead

The main application can point to a separately managed server implementing streamed chat completions. The local Transformers runtime is optional. For example, [vLLM's official serving documentation](https://docs.vllm.ai/en/stable/serving/online_serving/) describes its `/v1/chat/completions` interface. Its [serve CLI](https://docs.vllm.ai/en/stable/cli/serve/) supports a public served-model name and default chat-template options.

On hardware supported by your installed vLLM release, a corresponding configuration is:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8020 \
  --served-model-name qwen3-local \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

Keep the main process's `BIAS_LLM_BASE_URL` and `BIAS_LLM_MODEL` matched to that server. This command is documented as an alternative, not a vLLM execution result from this workspace. Installation, device support, scheduling, and memory behavior belong to that server's version and deployment configuration.

## Observed execution and remaining checks

The initial real HTTP check used the pinned Qwen3-0.6B model, CPU bfloat16, four threads, a 160-token cap, and a 120-second service deadline. The first direct JSON request completed in 40.60 seconds including lazy startup, returned `Hello!`, and reported 19 prompt tokens plus 3 generated tokens with EOS completion.

The main app's ordinary Tesla question then streamed a newly generated response in **13.33 seconds**, with first visible text at **7.52 seconds**. The model returned no citations, so the main app correctly returned `provider="openai"` and `grounding.status="unverified"`. The text matched the earlier direct-model example recorded in the inference notes. No citation was appended by the application or evaluation tool.

A second smoke question appended a long citation-format instruction. The existing lexical retrieval gate found insufficient support and returned an evidence-mode abstention without calling the model. The report therefore correctly marks that two-turn execution assertion as failed. This is an observed query-composition limitation, not a successful second model turn. The revised execution-only smoke uses two ordinary supported questions and leaves citation quality as its own measurement.

The original report remains in ignored `reports/local/language-http-smoke.json`. Peak RSS was not instrumented for that first run and is not claimed.

A subsequent full eight-case A/B/C run on **2026-11-10 UTC** completed all 24 conditions: 20 actual model requests and four deterministic no-evidence gates. It ran with strict offline mode after resolving the pinned cached snapshot directory. The initial warm-up took 12.90 seconds, including 10.28 seconds in lazy loading. The matrix took 221.51 seconds; the complete process experiment, including warm-up and shutdown, took 235.98 seconds. These observed times should not be interpreted as a controlled speed comparison with the earlier smoke.

Final process peak RSS was **2,081,476,608 bytes, approximately 1.94 GiB**. This includes the interpreter, libraries, loaded model, and inference allocations; it is not isolated parameter memory. The LLM process was stopped before other model work. Simultaneous LLM/TTS/ASR/vision residency and continuous multimodal serving remain unmeasured.

Execution succeeded, but grounding did not: all 12 retrieval-conditioned generated answers omitted the required citations. Three profile-only answers emitted the invalid literal marker `RECORD_ID`. No citation repair or semantic/persona score was applied. The [committed measured summary](../tools/evaluation/results/qwen3-06b-cpu.json) and [evaluation report notes](evaluation.md#measured-eight-case-run) retain those results separately from protocol success.

An earlier matrix attempt is also preserved locally: addressing the cached model by repository ID with global offline mode enabled caused an `OfflineModeIsEnabled` error during loading. Its 20 provider-path failures produced no model text; four no-evidence gates completed. That failed attempt is not a model-quality result. Resolving the actual cached snapshot directory fixed the loading path without enabling outbound model requests.

Protocol tests run without optional model downloads:

```bash
uv run pytest -q tests/language tests/evaluation
uv run python -m tools.evaluation.run --validate-only
```

After starting both services, the live smoke is:

```bash
uv run python -m tools.evaluation.http_smoke \
  --app-url http://127.0.0.1:8000 \
  --output reports/local/language-http-smoke-next.json
```

The smoke performs real requests and can take several minutes on CPU. It establishes HTTP/SSE integration, not browser rendering, speech playback, semantic accuracy, or resemblance to a real person.
