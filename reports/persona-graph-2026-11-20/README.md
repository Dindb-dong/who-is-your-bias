# Jensen persona graph: execution record

Status at this checkpoint: implementation and experiment infrastructure ready;
no experimental answer generation or quality score yet.

- Repository: `Dindb-dong/who-is-your-bias`
- Branch: `feat/jensen-persona-graph-nov20`
- Original main base: `472713e183e0dbdc8da8d35a22e6bd9e80be5f39`
- Earlier verified implementation checkpoint: `f28f11212703f715900899e5f8fbe5f817be0192`
- Production feature is opt-in, using `BIAS_PERSONA_GRAPH`.
- Protocol: [persona-graph-evaluation-protocol.md](../../docs/persona-graph-evaluation-protocol.md)

## Pre-generation checks

310 integration, graph, evaluator and affected runtime regression tests passed
in Python 3.12.14. The test warning concerns an existing Starlette/httpx
deprecation. This is software correctness evidence, not persona answer quality.

The development router selects paths on 9 of 12 authored English advice
probes, and none of six boundary controls. The three missed advice cases remain
in the experiment. Questions were not rewritten to fit the router. Routing was
improved after inspecting coverage, before any model answers: these are
development probes, not held-out events or a representative audience sample.

## Model and planned run

Official `Qwen/Qwen3-4B-GGUF`, revision
`bc640142c66e1fdd12af0bd68f40445458f3869b`, Q4_K_M. The verified file has
2,497,280,256 bytes and SHA-256
`7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
The model is an existing pretrained upstream artifact; no weights are trained
or committed in this change. See [runtime/model-manifest.json](runtime/model-manifest.json).

The runtime uses llama-cpp-python 0.3.36, six CPU threads, 8192 context tokens,
no GPU, no thinking, and resets state for every request. Actual formatted prompts
are token-counted before freezing. All three conditions use greedy decoding,
top-p 1, presence penalty 0, seed 20261120 and **512 maximum output tokens**
(a pre-generation override of the protocol's 256-token default, to reduce
truncation). Raw generation output, not TTS or avatar playback, is evaluated.

Twelve advice cases and six controls × three conditions = 54 scheduled attempts.
Existing deterministic guards are reported separately from LLM completions.
Outputs will be preserved in bounded batches and scored from condition-hidden
packets by independent AI reviewers, explicitly not human raters. Errors and
truncations remain in the denominator. A zero or negative change is a valid
result and will not be edited away.

## Files and resumption

`runtime/` records the exact runtime sources, dependencies, upstream model
identity, and neutral startup check. The large model is recoverable from the
pinned upstream URL, and is not a Git artifact. These runtime sources are
experimental infrastructure, not a production server.

The next checkpoint will add the frozen `manifest.json` before inference. Resume
the generation runner against that same manifest and its existing `results.json`;
the runner rejects mismatched identities and does not repeat saved attempts.

```bash
uv run python -m tools.persona_graph_eval freeze \
  --graph data/personas/jensen-huang/persona_graph.v1.json \
  --model Qwen3-4B-Q4_K_M \
  --revision bc640142c66e1fdd12af0bd68f40445458f3869b:Q4_K_M:7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5 \
  --max-tokens 512 --token-count-url http://127.0.0.1:8129/count_tokens \
  --output reports/persona-graph-2026-11-20/manifest.json

uv run python -m tools.persona_graph_eval run \
  --manifest reports/persona-graph-2026-11-20/manifest.json \
  --base-url http://127.0.0.1:8129/v1 --limit 18 \
  --output reports/persona-graph-2026-11-20/results.json
```

Start the optional local server and client within the same launcher command in
environments with per-command network isolation; see [runtime/README.md](runtime/README.md).
The commands above are the planned invocation until the frozen run is saved.
