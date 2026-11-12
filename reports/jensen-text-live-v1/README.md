# Actual neural-text API functional probe

This separate M7 procedure runs a real local language provider, the actual main
API and an HTTP client within one invocation. It is prepared but **has not yet
run a neural model**. The existing evidence-preview smoke, matched M6 response
experiment and recorded-audio browser replay remain separate measurements.

The [frozen protocol](protocol.json) binds Qwen3-1.7B revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, the independently recorded base-file
and tokenizer hashes, original-dialogue-v4, and distinct
[base](base.snapshot.json) / [candidate adapter](adapter.snapshot.json)
snapshots. Its exact file SHA-256 is
`edac6dfc7fb1545fc164fc5070ae3fdbc52a0a6dd82f7c10082fa88a58aa1210`.

The runtime uses CPU float32, eight threads, a 4,096-token input/output context,
160 output tokens, deterministic decoding, a 240-second per-request budget and
a 1,800-second whole-run budget. Voice, transcription and measured-motion
selection are disabled. It does not change the application's default config.

## What it checks

Four predeclared English turns share one session: a question about CUDA, an
elliptical follow-up, an explicit correction, and a topic switch to AI factories.
Every completed case must use the neural provider and expose actual SSE token,
result and done events. Source bindings and conversation provenance are checked
against unchanged corpus records. Citation marker coverage and semantic review
remain separate: a correct source ID does not establish factual entailment.

A fifth request asks for a longer explanation. The client verifies that the
exact loaded provider is still busy at its first nonempty model token, then
closes the TCP connection. The provider must become idle, the interrupted answer
must not enter session memory, and one recovery response must complete in the
same session. The session is then deleted. Main API and language-child teardown
are recorded on both success and failure.

The implementation is
[tools/runtime_checks/text_live.py](../../tools/runtime_checks/text_live.py).
It reuses the actual TCP server and trace primitives from the existing runtime
harness; it does not reuse its evidence-preview configuration preparation.
The `openai` response field names the existing OpenAI-compatible transport. The
direct provider health receipt identifies the actual local Qwen model.

## Verified preparation and next commands

Python compilation, whitespace checking, thirteen focused harness tests and the base
preflight passed. The tests cover exact observed model/adapter rejection, actual
TCP first-nonempty-token disconnect, generator cancellation, normal SSE, HTTP
error retention and server teardown after a body exception. Their small HTTP
fixtures are explicitly not neural integration evidence. The
preflight validated the exact corpus and snapshot, the local pinned cache,
and the selected virtualenv's dependency availability. **No weights loaded, no
model responses generated, and no functional run success is claimed.**

From the integrated repository, set the following existing paths. The interpreter
path must preserve the virtualenv spelling; resolving its symlink can select the
base Python without the model dependencies.

```bash
ART=/workspace/scratch/6f64e1b53384/artifacts/jensen-2026-11-13
CACHE=/workspace/scratch/6f64e1b53384/model-cache/hub
INFERENCE_PYTHON=/workspace/scratch/6f64e1b53384/who-is-your-bias/.venv-inference/bin/python
PROTOCOL_SHA=edac6dfc7fb1545fc164fc5070ae3fdbc52a0a6dd82f7c10082fa88a58aa1210

PYTHONPATH=src:. "$INFERENCE_PYTHON" -m tools.runtime_checks.text_live \
  --artifacts "$ART" --cache-dir "$CACHE" \
  --inference-python "$INFERENCE_PYTHON" \
  --protocol-sha256 "$PROTOCOL_SHA" --condition base \
  --allow-candidates --verify-only
```

Actual execution waits for the coordinating job to release model memory. The
candidate execution command below additionally requires `ADAPTER_RUN` to name a
freshly restored, completed text training run and `ADAPTER_RUN_SHA` to contain
the exact SHA-256 of its `run.json`. These two values are not known at this
preparation checkpoint and must come from the completed durable training receipt.
An intermediate checkpoint or a manually relabeled receipt is rejected.

```bash
PYTHONPATH=src:. "$INFERENCE_PYTHON" -m tools.runtime_checks.text_live \
  --artifacts "$ART" --cache-dir "$CACHE" \
  --inference-python "$INFERENCE_PYTHON" \
  --protocol-sha256 "$PROTOCOL_SHA" --condition adapter \
  --adapter-run "$ADAPTER_RUN" --adapter-run-sha256 "$ADAPTER_RUN_SHA" \
  --allow-candidates --output "$ART/runtime-checks/neural-text-adapter-v1"
```

For an explicitly selected base run, use `--condition base`, omit both adapter
arguments and select a new output directory. There is no silent adapter fallback.
The protocol also pins the exact Jensen training JSONL hash, 113 examples, each of
the nine event counts and 29 optimizer updates. A same-base adapter trained on
another dataset or a shorter completed run cannot satisfy this separate gate.
The completed local training receipt, every safe adapter/tokenizer file and the
loaded provider's actual identity must agree before any main-API turn runs.

## Outputs and interpretation

Raw HTTP traces, generated answers, session context, source excerpts, provider
health, warmup response and process log go to the new private output directory.
The runner saves `report.json`, `http-trace.json` and `public-summary.json` after
each completed stage. Failures and partial token streams are retained. A failed
or truncated turn is not retried or counted as a completed conversation turn.

The public summary contains response hashes, source IDs/URLs, timings, identity,
cleanup and functional checks. Full source text stays in the private trace.
Archive that directory and verify fresh readback before recording its delivery
as preserved. The runner does not upload artifacts itself.

Success would establish the declared functional route and cancellation behavior
for this bounded run. Human review remains ungraded; response quality, persona
fidelity, simultaneous text/voice memory capacity and production promotion are
not inferred from these checks.
