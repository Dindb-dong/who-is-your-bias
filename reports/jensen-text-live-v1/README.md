# Actual neural-text API functional probe

**The actual base and completed29-adapter runs each passed 46/46 functional
checks.** Both used a real local language provider, the actual main API and a
loopback TCP HTTP client. Each completed the four registered conversation turns,
closed a connection after an active model token, retained no partial turn,
completed one recovery response in the same session, deleted the session and
stopped both servers without a forced kill. See the [base receipt](base-result.json),
[adapter receipt](adapter-result.json), and [comparison](comparison.md).

Each result archive and all 12 payload files were freshly restored and verified.
The original preflight failure is preserved in
[base-first-preflight-failure.json](base-first-preflight-failure.json): the
harness incorrectly compared the service's versioned identity envelope with
the frozen payload alone. The correction requires the exact supported schema
and every unchanged payload field. [Corrected preparation](schema-corrected-preparation.json)
records the single changed source hash and the other 49 unchanged inputs.
Models, prompts, budgets, snapshots and decoding settings were unchanged.

The [separate cancellation diagnostic](cancel-recovery/README.md) was registered
as a fallback, but neither condition is eligible because both original corrected
probes reached and passed cancellation/recovery. It was not executed.

These results establish the bounded functional route. Matched M6 answer quality,
human likeness, the existing evidence-preview smoke and recorded-audio browser
replay remain separate measurements. In particular, the [M6 results](../jensen-evaluation-v1/heldout-review-v1/comparison.md)
do not support adapter promotion. The CPU observations do not establish a
real-time or comparative model-speed claim.

The [frozen protocol](protocol.json) binds Qwen3-1.7B revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, the independently recorded base-file
and tokenizer hashes, original-dialogue-v4, and distinct
[base](base.snapshot.json) / [candidate adapter](adapter.snapshot.json)
snapshots. Its exact file SHA-256 is
`edac6dfc7fb1545fc164fc5070ae3fdbc52a0a6dd82f7c10082fa88a58aa1210`.

The original runtime declares CPU float32, eight threads, a 4,096-token input/output context,
160 output tokens, deterministic decoding, a 240-second per-request budget and
a 1,800-second whole-run budget. Voice, transcription and measured-motion
selection are disabled. It does not change the application's default config.

## Pre-output BF16 runtime amendment

The separate M6 language service was SIGKILLed (`-9`) during its excluded FP32
warmup, before any evaluation question or response. Its supervisor recorded
91.894 seconds and the process owner observed a maximum waited-child RSS of
8,191,066,112 bytes. This is consistent with memory pressure; no kernel OOM
attribution was available. The failed attempt and its fresh-readback receipt are
preserved in the [M6 runtime amendment](../jensen-evaluation-v1/runtime-amendment-bf16-v2.json).
It is not an M7 attempt or a semantic response failure.

The additive [BF16 protocol](protocol-bf16-v2.json), sealed before any M7 output,
has SHA-256
`6c699aaa81a5f90919be7c78579f400c58590db5924abd045e6c76bcf9cd331e`.
It differs from the retained original protocol only in `runtime.dtype`, from
`float32` to `bfloat16`. All questions, snapshots, timeouts, decoding settings,
eight threads and the completed 29-update Jensen adapter gate are unchanged.
The [M7 amendment](runtime-amendment-bf16-v2.json) records the observed failure,
exact artifacts and this prospective selection. The wrapper accepts only these
two exact registered protocol digests and still checks the actual loaded dtype.
There is no automatic precision fallback.

This pre-output registration itself did not establish BF16 runtime suitability.
The subsequent actual observations are the separately preserved results above.

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

## Preparation history and reproduction commands

The [completed-29 preparation](completed29-preparation.json) binds the now-saved
serving export. Both base and exact-adapter configuration-only preflights pass;
no weights or neural responses were produced. Its seven preparation payloads
were saved and freshly restored. The serving run SHA is
`e1102328f6cc85e9c096dc554e6255465d67870706df5d3a4f03bf091c6e3f42`;
the subsequent relocated and corrected preparation is linked above.

The original preparation passed Python compilation, whitespace checking, thirteen
focused harness tests and the base preflight. The tests cover exact observed model/adapter rejection, actual
TCP first-nonempty-token disconnect, generator cancellation, normal SSE, HTTP
error retention and server teardown after a body exception. Their small HTTP
fixtures are explicitly not neural integration evidence. The
preflight validated the exact corpus and snapshot, the local pinned cache,
and the selected virtualenv's dependency availability. Those preparation-only
checks loaded no weights and generated no responses; the subsequent actual
functional results are linked above.

The [BF16 preparation record](preparation-bf16-v2.json) adds 24 passing harness
tests and a successful base configuration-only `--verify-only` check. Both seals
are accepted; altered questions, snapshots, budgets or training bindings are
rejected even when a caller supplies a new matching digest. This check found the
existing corpus/cache/interpreter dependencies and loaded no weights. It does
not establish actual BF16 model execution or memory suitability.

From the integrated repository, set the following existing paths. The interpreter
path must preserve the virtualenv spelling; resolving its symlink can select the
base Python without the model dependencies.

```bash
ART=/workspace/scratch/31a6dc706f63/artifacts/jensen-pr19-completion-v1
CACHE=/workspace/scratch/31a6dc706f63/model-cache/hub
INFERENCE_PYTHON=/workspace/scratch/31a6dc706f63/who-is-your-bias/.venv-inference/bin/python
ADAPTER_RUN=/workspace/scratch/31a6dc706f63/restored/jensen-text-adapter-completed29-v1/text-adapter-v1
ADAPTER_RUN_SHA=e1102328f6cc85e9c096dc554e6255465d67870706df5d3a4f03bf091c6e3f42
PROTOCOL_FILE=reports/jensen-text-live-v1/protocol-bf16-v2.json
PROTOCOL_SHA=6c699aaa81a5f90919be7c78579f400c58590db5924abd045e6c76bcf9cd331e

PYTHONPATH=src:. "$INFERENCE_PYTHON" -m tools.runtime_checks.text_live \
  --artifacts "$ART" --cache-dir "$CACHE" \
  --inference-python "$INFERENCE_PYTHON" \
  --protocol "$PROTOCOL_FILE" --protocol-sha256 "$PROTOCOL_SHA" --condition base \
  --allow-candidates --verify-only
```

The completed runs were serialized against the shared model-memory slot. The
candidate execution command below additionally requires `ADAPTER_RUN` to name a
freshly restored, completed text training run and `ADAPTER_RUN_SHA` to contain
the exact SHA-256 of its `run.json`. The completed-29 preparation above now records both values from the
freshly verified durable training receipt.
An intermediate checkpoint or a manually relabeled receipt is rejected.

```bash
PYTHONPATH=src:. "$INFERENCE_PYTHON" -m tools.runtime_checks.text_live \
  --artifacts "$ART" --cache-dir "$CACHE" \
  --inference-python "$INFERENCE_PYTHON" \
  --protocol "$PROTOCOL_FILE" --protocol-sha256 "$PROTOCOL_SHA" --condition adapter \
  --adapter-run "$ADAPTER_RUN" --adapter-run-sha256 "$ADAPTER_RUN_SHA" \
  --allow-candidates --output "$ART/runtime-checks/neural-text-adapter-bf16-v2"
```

For an explicitly selected base run, use `--condition base`, omit both adapter
arguments and select a new output directory. There is no silent adapter fallback.
To reproduce the original preparation, select `protocol.json` and its original
SHA-256; omitting `--protocol` still selects that original FP32 declaration.
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
