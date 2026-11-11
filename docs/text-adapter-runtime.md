# Verified text adapter training and serving

This connects the text-adaptation exports to the existing local language service.
It implements a missing WP3 runtime path in the [2026 roadmap](2026-persona-roadmap.md).
The API serves one operator-selected base or base-plus-LoRA model per process.
Clients use its public model alias; they cannot select a path, adapter, revision,
or another checkpoint in a completion request.

A completed training receipt establishes which computation and files produced
an adapter. It does not establish improved persona fidelity, grounding, or
general answer quality. The controlled comparison must still run with frozen
facts and consistent prompting. Existing A/B/C retrieval experiments retain
their original protocol and results.

## Create a compatible training run

Use the [original-data export](persona-data.md#text-adaptation) to create a
nonempty training dataset. The exporter rejects evaluation-only held-out cases
before accessing their fields; validation and test projections remain ineligible.

Install the model dependencies in the optional language/training environment:

```bash
python -m pip install -r services/language/requirements-adapter.txt

python -m tools.adaptation train-text \
  --dataset data/local/text-adaptation-new \
  --output data/local/text-adapter-run-new \
  --model MODEL_REPOSITORY --revision IMMUTABLE_40_CHARACTER_REVISION \
  --device cpu --epochs 1 --rank 8 --max-length 1024
```

Replace the model and revision with the actual Hugging Face repository and its
immutable commit. The trainer resolves that pinned checkpoint before loading,
uses safetensors, disables remote code, and hashes the local base config,
weights, and tokenizer files. Hugging Face's normal snapshot-to-blob links are
supported for base checkpoints; the target bytes are verified.

The default precision remains `float32`. An explicit lower-memory experiment
can select these additional options:

```bash
python -m tools.adaptation train-text \
  --dataset data/local/text-adaptation-new \
  --output data/local/text-adapter-bounded-bf16 \
  --model MODEL_REPOSITORY --revision IMMUTABLE_40_CHARACTER_REVISION \
  --device cpu --precision bfloat16 --gradient-checkpointing \
  --max-length 256 --gradient-accumulation 1 --max-steps 4
```

The BF16 path first checks a finite matrix forward/backward operation on the
selected device. That check does not establish native hardware acceleration or
support for every later model operation. Checkpointing uses the declared
non-reentrant implementation and enables the input gradient path for frozen
PEFT embeddings. Unsupported operations fail explicitly. A four-step execution
is an engineering experiment, not a completed quality study.

Whole exchanges that do not fit `--max-length` are excluded and listed in the
receipt. Neither prompts nor supervised answers are silently truncated. Loss is
computed only for the assistant response. Gradient accumulation accounts for
the smaller last group. At most 2,048 individual losses are retained in the
receipt; aggregate example count, sum, minimum, and maximum include every
processed training example. The receipt records whether its detailed history
was truncated.

`run.json` uses `text-adaptation-run.v2` and records the selected precision,
checkpointing, seed, optimizer settings, model/tokenizer revisions, exact base
file hashes, dataset identity, family counts, losses, completed optimizer steps,
package versions, implementation hashes, elapsed time, and output hashes.
Successful output contains `adapter/` and `tokenizer/`. A failed run retains
its failure record. A process killed by the operating system cannot execute its
failure handler; its last receipt may remain `running`, and is deliberately
ineligible for serving. Preserve the external exit/log record in that case.

### One tokenizer and template contract

Training, local inference, and the main API's exact tokenizer budget all set
`enable_thinking=False`. The trainer requires the inference assistant prefix to
match the beginning of the complete supervised sequence, then masks that prefix
from loss. The [Transformers chat-template guidance](https://huggingface.co/docs/transformers/chat_templating)
explains why generation prefixes and training message formatting must agree.

The run binds a token contract covering the vocabulary, special token IDs,
chat template, template options, and actual rendered English/Korean probe token
IDs. After saving, the trainer reloads the saved tokenizer and verifies that
the contract is unchanged. Serving checks both the base tokenizer and saved
tokenizer against that same contract. Matching file names or vocabulary size
alone is insufficient.

Old `text-adaptation-run.v1` receipts lack this compatibility evidence. They
remain valid historical artifacts but cannot be served through this verified
path. Re-run training with the current trainer. Relabeling the old schema or
inventing missing fields does not reconstruct the missing evidence.

## Serve one operator-selected adapter

Configure the exact base repository and revision from the completed run, plus
the directory containing its `run.json`:

```bash
export BIAS_LANGUAGE_MODEL=MODEL_REPOSITORY
export BIAS_LANGUAGE_REVISION=IMMUTABLE_40_CHARACTER_REVISION
export BIAS_LANGUAGE_LOCAL_FILES_ONLY=true
export BIAS_LANGUAGE_SERVED_MODEL=persona-text-adapter
export BIAS_LANGUAGE_ADAPTER_RUN=/absolute/path/to/text-adapter-run-new
# Optional: pin the exact SHA-256 of that run.json receipt as well.
export BIAS_LANGUAGE_ADAPTER_RUN_SHA256=EXACT_64_CHARACTER_RECEIPT_SHA256

python -m uvicorn services.language.app:app --host 127.0.0.1 --port 8021
```

Omit `BIAS_LANGUAGE_ADAPTER_RUN_SHA256` if not supplying a real digest. The
service always calculates and reports the observed receipt hash; supplying the
optional pin also rejects a different receipt before loading. Cache the actual
base checkpoint separately before using the default local-only mode.

Point the main application at the served alias and endpoint:

```bash
export BIAS_LLM_BASE_URL=http://127.0.0.1:8021/v1
export BIAS_LLM_MODEL=persona-text-adapter
uv run bias serve
```

Other API settings, snapshot policy, evidence packing, citation checks and
streaming behavior still apply. A text adapter does not replace source
retrieval or make a historical claim current. Credentials remain server-side.

Leave `BIAS_LANGUAGE_ADAPTER_RUN` unset to serve the original base model.
PEFT is imported only when an adapter is selected. Importing the service or
checking a receipt does not load weights. The first accepted completion owns
lazy loading under the existing single-worker admission lock.

The loader checks the completed status, optimizer work, training-only dataset
identity and family counts, matching base repository/revision, PEFT causal-LM
LoRA configuration, and every file in `adapter/` and `tokenizer/`. Missing,
changed or extra files are rejected. Relative manifest paths must be normalized;
package symlinks, special files, pickle weights and executable Python artifacts
are rejected. The base checkpoint is checked against the training receipt and
both tokenizers must preserve the recorded contract.

It then uses [PEFT's pretrained-adapter loader](https://huggingface.co/docs/peft/package_reference/peft_model#peft.PeftModel.from_pretrained)
with local-only safetensors and `is_trainable=False`, and places the model in
evaluation mode. Adapter/package failures are visible; the service never
silently falls back to the base model. Server responses do not expose local
paths or raw training/source text.

## Runtime identity for controlled comparisons

`GET /health` reports the configured selection before loading. Observed hashes
remain null, and an adapter remains `loaded=false, verified=false`, until the
backend has loaded and passed verification.

| Health field | Meaning |
| --- | --- |
| `identity.base_model` | Repository ID, or null when configured with a local path |
| `identity.base_revision` | Immutable configured revision, or null if not established |
| `identity.base_checkpoint_manifest_sha256` | Digest of the actual base-file hash map after loading |
| `identity.tokenizer_contract_sha256` | Digest of the actual vocabulary/template/rendering contract |
| `identity.chat_template_options` | The common non-reasoning rendering setting |
| `runtime.device`, `dtype`, `context_window` | Actual loaded parameter device/dtype and model context; null before loading |
| `runtime.requested_device`, `requested_dtype` | Operator configuration, separate from observed values |
| `runtime.backend`, `versions`, `threads`, `attention` | Backend stack and execution settings |
| `runtime.base_parameters` | Actual base parameter count before adding PEFT layers; null before loading |
| `runtime.generation_defaults` | Server seed, temperature, top-p and top-k defaults |
| `adapter.configured`, `loaded`, `verified` | Selected adapter and completed loader validation, separately |
| `adapter.run_sha256` | Exact completed training receipt |
| `adapter.artifact_manifest_sha256` | All saved adapter and tokenizer files |
| `adapter.dataset_sha256` | Training JSONL bytes bound by the export manifest |
| `adapter.training_event_families` | Source families represented in the training dataset |
| `adapter.optimizer_steps` | Completed optimizer steps recorded by training |
| `adapter.efficacy_evaluated` | False; artifact validation does not establish efficacy |

Base-only services expose the same base and tokenizer identity fields, allowing
an evaluator to reject mismatched controls. Separate server instances can use
different aliases and ports while proving they share the same frozen base and
token contract. The evaluator must also ensure that held-out reference families
are absent from `training_event_families` and that style/adapter conditions use
identical supplied facts and prompts. Equal model names alone do not establish
those controls.

For offline orchestration, `services.language.adapters` is a standard-library
module. `validate_adapter_run(path, base_model=..., base_revision=...,
expected_run_sha256=...)` returns a `VerifiedAdapterRun` with
`public_identity()`, `verify_base_checkpoint(path)` and
`verify_tokenizer(tokenizer)`. The first call validates the receipt/package;
base and tokenizer verification occur separately when their actual inputs are
available.

## Verification scope

```bash
uv run pytest tests/adaptation/test_adapter_artifacts.py tests/language \
  tests/persona_data tests/test_temporal_dialogue.py -q
```

The artifact and loader unit tests use deliberately synthetic files and fake
model modules. They establish rejection paths, identity matching, exact token
policy, local frozen loading, optional dependency behavior and client selection
boundaries. They do not establish neural inference or trained quality. Any
actual training/inference run must retain its own completed receipt, immutable
model identity, raw comparison outputs and declared evaluation limits.
