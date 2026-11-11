# Controlled text adaptation comparison

`tools.adaptation_evaluation` freezes questions, complete factual passages,
conversation context, decoding settings and model identity before comparing
three text-generation conditions. It connects the
[verified adapter runtime](text-adapter-runtime.md) to the controlled comparison
required by the [2026 roadmap](2026-persona-roadmap.md).

The runner calls the language service's real `/v1/chat/completions` endpoint
with frozen prompts. It uses the application's snapshot selection and prompt
packing code when preparing those prompts. It does not exercise the main
application's session or audio APIs; use the
[dialogue scenarios](dialogue-evaluation.md) and
[live stream benchmark](live-stream-benchmark.md) for those paths.

This document describes the protocol and commands. It does not report a
completed model comparison, human ratings or improved persona quality. A
completed training run proves that the recorded optimizer work produced a
loadable adapter. A completed generation proves that a particular request
finished. Neither establishes that the answer is better.

## The controlled conditions

| Condition | Loaded model | Supplied facts | Wording examples | Purpose |
| --- | --- | --- | --- | --- |
| `facts` | Pinned base checkpoint | Frozen complete passages | None | Factual-context control |
| `style` | The same base checkpoint | Identical passages | Frozen eligible examples | Effect of supplied wording examples |
| `adapter` | The same base plus a verified LoRA run | Identical passages | Identical to `style` | Incremental effect of the trained adapter |
| `larger`, optional | A separately identified larger base | Identical passages | Identical to `style` | Additional model-capacity reference |

The required comparison contains exactly one `facts`, one `style` and one
`adapter` report. `larger` is an optional fourth condition. Its observed base
parameter count must exceed the required baseline's observed count. A
different model name alone does not satisfy that requirement.

The freezer builds the styled prompt first, using whole-evidence prompt
packing. It derives the `facts` prompt by removing only the wording-example
section. This preserves the same factual evidence, history and temporal
instructions even when examples consume part of the context budget. The
`style` and `adapter` message arrays must be byte-identical. The factual
citation allowlist contains only the supplied factual passages.

The plan records whether each case has a wording contrast and whether any
factual candidates fit. If a case has no eligible style example, its `facts`
and `style` conditions do not test a wording contrast. If factual candidates
exist but no complete passage fits, freezing fails instead of silently
truncating a passage. Cases without factual candidates remain identifiable in
the comparison summary.

All conditions request the same output budget, `temperature=0`, `top_p=1`
and seed `42`. The three required conditions must also have the same observed
base-file digest, tokenizer contract, device, precision, backend, context
window, thread count, attention setting, base parameter count, Torch and
Transformers versions, and service generation defaults. These are controlled
execution settings, not a claim of bitwise determinism across machines.

## Keep development and held-out references separate

A development input is an explicitly labeled JSON file:

```json
{
  "exposure": "development",
  "cases": [
    {
      "id": "dev-reusability-en",
      "question": "Why does rocket reusability matter?",
      "language": "en"
    }
  ]
}
```

Each case needs a unique simple identifier and a question. Optional fields
include `language` (`en` or `ko`), `context_before`, `event_family_id`,
`reference_answer` and `reference_review_status`. A private reference answer
is retained for later assessment and is not deliberately inserted into the
generation prompt. Earlier source context is supplied as unverified context,
with the same content in every condition. Development cases may already be
familiar to developers or overlap training material; label them accordingly.
Do not infer a fresh-test result from their scores.

Held-out input must come from `load_heldout` and the separately marked
[evaluation-only data workflow](heldout-evaluation-data.md). The CLI's
`--heldout` option validates that dataset rather than accepting a custom file
with a renamed exposure field. The held-out manifest must bind a leakage
check against the exact retrieval corpus used to freeze the plan. Reference
families are excluded from factual and style projections and, at execution,
must be absent from the adapter's reported training families.

These checks establish the declared local acquisition and training boundary.
They do not establish what a pretrained base model encountered before this
project, nor whether developers have previously viewed a reference. A source
family reserved in a registry is not a usable held-out dataset until actual
original Q&A has been acquired, extracted and verified. Consult the held-out
data report for the current acquisition result before selecting this mode.

## Freeze a new plan

Run commands from the repository root. Keep plans and full outputs in ignored
local directories because they contain source wording and private references.
Use a new output path for each version; commands refuse to overwrite an
existing plan or run.

```bash
uv run python -m tools.adaptation_evaluation freeze \
  --corpus data/local/persona-2026/corpus \
  --snapshot data/local/persona-2026/snapshot.json \
  --development-cases data/local/comparison-cases.json \
  --id text-adaptation-development-v1 \
  --base-model MODEL_REPOSITORY \
  --base-revision IMMUTABLE_40_CHARACTER_REVISION \
  --max-tokens 256 --context-window 8192 \
  --output data/local/comparison-plan-new
```

Replace the paths and model identity with actual local artifacts. Omit
`--snapshot` only when deliberately evaluating a corpus without the snapshot
policy. For validated held-out references, replace `--development-cases FILE`
with `--heldout DATASET_DIRECTORY`; the two inputs are mutually exclusive.
The plan accepts 1–200 cases.

Freezing uses configured retrieval settings from `Settings.from_env`, then
sets the explicit corpus, snapshot and token budgets from the CLI. Configure
the desired retrieval backend and its actual model before freezing. Set
`BIAS_LLM_TOKENIZER_PATH` to the compatible tokenizer when using exact token
counting. The prompt budget records which counter was used. The subsequent
language service still enforces its own token limits.

By default the freezer selects factual and style candidates once. Hybrid
retrieval uses the application's factual and style projections. The lexical
fallback selects style candidates by question-token overlap and identifies
that policy explicitly. This fallback does not establish semantic style
relevance.

For a controlled experiment with preselected eligible training records,
`--selections FILE` accepts this mapping:

```json
{
  "dev-reusability-en": {
    "factual_ids": ["ACTUAL_ELIGIBLE_FACT_ID"],
    "style_ids": ["ACTUAL_ELIGIBLE_STYLE_ID"]
  }
}
```

The mapping must cover exactly the case identifiers. IDs must be unique,
eligible for their projection and requested period, and present in the
declared corpus. Records used as facts are excluded from the style examples.
Explicit selection is reported as
`explicit_fixed_training_ids_not_retrieval_efficacy`; it cannot support a
retrieval-quality claim.

The output contains:

| File | Contents and sharing scope |
| --- | --- |
| `plan.json` | Private questions, source passages, examples, references, complete prompts and their hashes |
| `summary.json` | Source-wording-free identifiers, hashes, policy and implementation identity suitable for a shareable protocol record |

The plan uses `text-adaptation-comparison.v1` and a canonical JSON SHA-256
seal. It binds the corpus, snapshot, input identity, base identity and
implementation files. Validation also reconstructs the factual/style
invariants; merely replacing the outer hash after editing facts does not
make an inconsistent plan valid. If a bound implementation file changes,
freeze a new plan rather than changing the old plan's recorded hashes.

## Execute the local CPU supervisor

Use the optional model environment described in
[text adapter runtime setup](text-adapter-runtime.md#create-a-compatible-training-run),
with the project importable, its inference dependencies installed, a completed
compatible adapter run, and the exact pinned base checkpoint already cached.
The supervisor uses the Python interpreter that launches it and enforces
local-only checkpoint loading.

```bash
python -m tools.adaptation_evaluation.local \
  --plan data/local/comparison-plan-new/plan.json \
  --adapter-run data/local/text-adapter-run-new \
  --dtype bfloat16 --threads 2 --timeout 180 \
  --output data/local/comparison-execution-new
```

`--dtype` accepts `float32` or `bfloat16`; `--threads` accepts 1–16. Choose a
precision the actual execution host supports. The runner starts a loopback
language server, performs an explicit warmup, executes `facts` then `style`,
and stops that server. It starts a second server with the pinned adapter and
executes `adapter`. Only one model server is resident at a time. Both servers
run on CPU, with the same declared settings, in the supervisor's process
environment. This permits real HTTP execution even in an executor where
separate commands cannot share loopback services.

The output includes `base-server.log`, `adapted-server.log`, each condition's
`run.json`, `attempts.jsonl` and `raw-generations.jsonl`, `comparison.json`,
and `blind-review/`.
Subprocesses are terminated in cleanup, including failed phases. Warmup time
is recorded separately and excluded from condition metrics. The order is
fixed, not counterbalanced; timings describe this execution CPU, not the
target deployment device. The supervisor does not run the optional larger
model.

## Execute individual conditions against existing services

An independently supervised service must expose the verified local language
service's `/v1` and companion `/health` contracts. A generic OpenAI-compatible
completion endpoint without the required runtime identity is insufficient.
Before running a condition, load the declared model with an excluded warmup
request; the evaluator requires `ready` and `model_loaded=true`.

```bash
uv run python -m tools.adaptation_evaluation run \
  --plan data/local/comparison-plan-new/plan.json \
  --condition facts \
  --base-url http://127.0.0.1:8021/v1 --model comparison-base \
  --timeout 180 --total-seconds 3600 \
  --output data/local/comparison-facts-new
```

Run `style` against the same base checkpoint, tokenizer and runtime with
`--condition style` and a separate output directory. Run `adapter` against
the verified adapted service with `--condition adapter` and
`--adapter-run-sha256 EXACT_RECEIPT_SHA256`. The expected digest is that of
the completed training `run.json`, not a hash of the directory name. If the
endpoint requires a key, put it in `BIAS_EVALUATION_API_KEY`.

The evaluator checks identity before and after each request. The adapter
condition requires a loaded, verified adapter with the exact requested
receipt hash, package and dataset digests, matching base and tokenizer,
declared training families, and positive completed optimizer work. Other
conditions require no configured or loaded adapter. Generated SSE responses
must report the requested served-model alias and complete the expected
protocol.

`--timeout` bounds an individual request and `--total-seconds` bounds the
whole condition. The raw generation is written before the post-request health
check, and `provider_generation` preserves the provider's original text,
finish reason and error. A subsequent timeout, network failure or changed
identity retains the attempt as a failure rather than turning it into an
unattempted case. Partial responses, failed identity checks, protocol errors
and genuinely unattempted cases remain visible in run artifacts. Inspect `status`,
attempted/completed counts and failures; a CLI process returning normally is
not by itself evidence that every case completed.

For the optional capacity comparison, run `--condition larger` against a
separately identified base-only service. It receives the exact frozen `style`
messages, cases, output budget and decoding policy. Its tokenizer and model
identity may differ, so this is an additional model comparison rather than
the isolated adapter contrast. The aggregate comparison checks its observed
parameter count and records its own runtime settings.

Combine the required reports, and optionally a fourth larger report:

```bash
uv run python -m tools.adaptation_evaluation compare \
  --plan data/local/comparison-plan-new/plan.json \
  --run data/local/comparison-facts-new/run.json \
  --run data/local/comparison-style-new/run.json \
  --run data/local/comparison-adapter-new/run.json \
  --output reports/local/comparison-summary-new.json
```

The summary contains completion counts, failure counts, observed model
identities, output hashes and generation-time medians for complete calls.
Citation syntax and lexical overlap are diagnostics. They do not establish
semantic support, correct temporal reasoning, persona similarity or a human
preference. High overlap can indicate memorization; assess it separately.
All automatically produced summaries retain `human_review: ungraded` and
`efficacy_claim: false`.

## Export a blind human review packet

```bash
uv run python -m tools.adaptation_evaluation export-review \
  --plan data/local/comparison-plan-new/plan.json \
  --run data/local/comparison-facts-new/run.json \
  --run data/local/comparison-style-new/run.json \
  --run data/local/comparison-adapter-new/run.json \
  --output data/local/comparison-review-new
```

The exporter validates the comparison and shuffles complete outputs with a
fixed seed. `reviewer/samples.json` contains opaque sample IDs, questions,
answers, private reference answers, supplied factual passages, and empty
ratings. It omits the model condition. The `private/` directory contains the
answer key, output hashes, comparison summary and failed attempts. Share
only `reviewer/` with the intended reviewers and retain the key separately.
The review packet contains original wording; it is not the source-free
public summary.

The seven rating dimensions are factual support, temporal fidelity, style
similarity, answer quality, unsupported first-person claims, repetitive
mannerisms, and quotation memorization. Reviewer identity/type fields start
empty. Define the scoring rubric and reviewer procedure before opening the
condition key. An automatically created packet is not a completed human
evaluation; this workflow does not fabricate ratings or automatically turn
them into an efficacy claim. Failed and incomplete attempts must remain in
the comparison denominator, even though they are excluded from the set of
complete answers sent for rating.

## Python API and verification

The underlying API lives in `tools.adaptation_evaluation`:

| Function | Responsibility |
| --- | --- |
| `protocol.freeze_plan(engine, cases, ...)` | Freeze eligible evidence, prompts, policy and input identity |
| `protocol.validate_plan(plan)` | Verify seals, implementation identity and experimental invariants |
| `protocol.public_summary(plan)` | Produce the wording-free plan summary |
| `run.run_condition(plan, condition, provider, output, ...)` | Execute one condition asynchronously and retain attempt evidence |
| `run.compare_reports(plan, reports)` | Verify paired inputs and model controls and summarize execution |
| `run.export_review(plan, reports, output, seed=42)` | Create the ungraded blind packet and private key |
| `local.run_local(plan_path, adapter_path, output, ...)` | Supervise sequential real local services and all three conditions |

Run the protocol's regression tests with:

```bash
uv run pytest tests/adaptation_evaluation -q
```

The tests use synthetic model-health and SSE fixtures to check complete-fact
invariance, identical adapter/style prompts, private references, split and
selection rejection, changed-plan detection, runtime identity mismatches,
partial-response handling and blind-export boundaries. Those fixtures test
the protocol. Actual neural execution needs its own preserved model receipts,
completed condition artifacts and declared evaluation scope. Multi-turn
behavior, concurrent audiovisual latency, acoustic output and target-device
performance require their separate evaluations.
