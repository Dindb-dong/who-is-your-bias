# Reproducible execution

## Local machine

```bash
make setup
make dev
```

The API and Vite run together; Ctrl-C stops both. For a built application, use `make build` then `make serve` and open port 8000. Use a single API worker because session state is process-local.

The development launcher was smoke-tested by starting both servers, receiving HTTP 200 from the API health endpoint and Vite HTML page, and terminating the launcher cleanly. This local launch check is separate from browser and inference validation.

Configuration is read from the environment. To load `.env`, use `uv run --env-file .env python scripts/dev.py` or `uv run --env-file .env bias serve`. Environment files are excluded from Git and the container build context.

## Container

```bash
docker compose up --build
```

The container combines the built frontend and Python API, runs as an unprivileged user, and exposes only `127.0.0.1:8000` through the supplied Compose configuration. The final image does not include the frontend toolchain, raw research recordings, or model weights. Container files are supplied for reproducibility; verify the build on your Docker installation before treating it as a tested deployment.

The language/speech services run separately. A URL containing `127.0.0.1` inside a container refers to that container, not the host. Configure reachable service URLs for your deployment. Docker Desktop can expose host services through its documented host gateway; Linux setups require an explicit network design. Do not broaden service exposure just to make a sample URL work.

## Voice and media

Heavy model dependencies have separate requirement files so the ordinary studio remains lightweight. Follow [media pipeline](media-pipeline.md) to prepare recordings, extract optional observations, and configure the local reference-voice service. Model downloads and inference can need considerably more memory than the web/API application.

## Local validation

Use Python 3.12 with uv, Node 24, and FFmpeg on PATH.

```bash
make setup
make test
npx playwright install --no-shell chromium
make test-e2e
```

`make test` checks the Python suite, corpus integrity, historical retrieval regressions, TypeScript, frontend unit tests, production build, and avatar fixtures. Run it before `make test-e2e` so Playwright serves the production build. The local target uses full Chromium and two Playwright workers for the software WebGL tests. Free port 8000 and run the browser suite from a shell without model provider configuration. To use an installed browser, set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to its executable instead of installing Playwright Chromium.

The retrieval report is written to `reports/local/retrieval.json`. Browser reports, screenshots and failure traces are under `playwright-report/` and `test-results/`. Record the commands and results on the PR head you tested. Model inference is checked separately.

## Apple Silicon text inference

Use a Metal-enabled [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#metal-build) build and confirm `llama-server --list-devices` reports `MTL0`. The local candidate is [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) in the [pinned ggml-org Q8_0 conversion](https://huggingface.co/ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF/blob/e6f794d44f9395d0184a966c27b5ae99ea356fcb/qwen3-4b-instruct-2507-q8_0.gguf) (4.28 GB).

```bash
mkdir -p models
model=models/qwen3-4b-instruct-2507-q8_0.gguf
curl -fL --max-filesize 4280403520 -o "$model.part" \
  https://huggingface.co/ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF/resolve/e6f794d44f9395d0184a966c27b5ae99ea356fcb/qwen3-4b-instruct-2507-q8_0.gguf &&
printf '%s  %s\n' ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1 "$model.part" |
  shasum -a 256 -c &&
mv "$model.part" "$model"
```

Run the loopback model server in one terminal. Add `--verbose` to inspect Metal layer offload during startup.

```bash
llama-server --model models/qwen3-4b-instruct-2507-q8_0.gguf \
  --alias qwen3-4b-instruct-2507 --host 127.0.0.1 --port 8020 \
  --device MTL0 --n-gpu-layers 99 --ctx-size 8192 --parallel 1 --threads 4
```

Run the studio from a second terminal with the existing OpenAI-compatible provider settings.

```bash
BIAS_LLM_BASE_URL=http://127.0.0.1:8020/v1 \
BIAS_LLM_MODEL=qwen3-4b-instruct-2507 \
BIAS_LLM_CONTEXT_WINDOW=8192 \
BIAS_LLM_MAX_TOKENS=256 make dev
```

For PyTorch LoRA training, the separate [adapter training command](text-adapter-runtime.md#create-a-compatible-training-run) accepts `--device mps` and probes BF16 forward and backward when `--precision bfloat16` is selected. The GGUF above serves base-model inference. Adapter training uses the pinned Transformers checkpoint.

## Remaining production work

The research application intentionally leaves user accounts, usage billing, durable chat storage, distributed queues, and public hosting authentication outside its current local workflow. Add those before running an unrestricted public inference endpoint. High-quality subject likeness also requires reviewed voice references and a finished rigged asset, with separate perceptual evaluation.
