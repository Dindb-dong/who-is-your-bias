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
npx playwright install chromium
make test-e2e
```

`make test` checks the Python suite, corpus integrity, historical retrieval regressions, TypeScript, frontend unit tests, production build, and avatar fixtures. Run it before `make test-e2e` so Playwright serves the production build. Free port 8000 and run the browser suite from a shell without model provider configuration. To use an installed browser, set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to its executable instead of installing Playwright Chromium.

The retrieval report is written to `reports/local/retrieval.json`. Browser reports, screenshots and failure traces are under `playwright-report/` and `test-results/`. Record the commands and results on the PR head you tested. Model inference is checked separately.

## Remaining production work

The research application intentionally leaves user accounts, usage billing, durable chat storage, distributed queues, and public hosting authentication outside its current local workflow. Add those before running an unrestricted public inference endpoint. High-quality subject likeness also requires reviewed voice references and a finished rigged asset, with separate perceptual evaluation.
