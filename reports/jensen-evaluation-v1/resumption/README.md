# Resume the remaining M6 invocations after relocation

The interrupted branch was inspected at
`324b9e8e5af9ff16f471badf2e1478c7c5e2bdd1`. Its durable execution status records
48 attempted matched calls: the six heldout adapter calls and the separately
registered six-case RAG diagnostic are not recorded as executed. A conversation
statement that the adapter process started is not a persisted result. Inspect
any recovered process receipts or partial outputs before launching a new attempt;
retain them separately instead of overwriting or silently counting them twice.

`launch_remaining.py` is new operational code for a relocated workspace. It
does not replace the original launchers or claim their file identity. It accepts
explicit restored paths, validates the original frozen implementation/plan and
contract seals, hashes the actual local base checkpoint, and checks the actual
tokenizer contract before starting a model. The adapter arm additionally requires
the exact completed29 serving receipt and payload hashes. Its default mode is
preflight only. `--execute` runs the unchanged original module through the
existing shared process supervisor.

## Required recovered bytes

Use the recorded object IDs and hashes in the linked receipts. Freshly restore
and verify the archive and every member before use.

| Purpose | Existing receipt |
| --- | --- |
| Both frozen matched plans and snapshot | [Evaluation inputs](../artifacts.json) |
| Completed29 adapter/tokenizer/run | [Completed serving export](../../jensen-recovery/text-v1/completed-serving.json) |
| Original heldout facts/style outputs | [Execution status](../execution-status.json), [central artifact index](../../../data/runs/jensen-2026-11-13/artifacts.json) |
| Separate six-case captured RAG prompts | [RAG artifact](../heldout-rag-v1/artifacts.json) |
| Original base weights | Pinned `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; [base manifest receipt](../base-verification.json) |

The serving adapter archive is not the 4,079,423,234-byte base snapshot. The
language supervisor uses local files only. The base must therefore be restored
or downloaded at its pinned revision before execution. Place it in the normal
`HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/<revision>` cache and require the
recorded checkpoint manifest digest. Preserve the independent acquired source
corpus for M7; frozen M6 plans already carry their original factual context.

The previous observed runtime used Torch `2.11.0+cpu` and Transformers `4.57.3`.
The collector compares the entire reported runtime identity between conditions,
including those versions. Restore the compatible inference environment and
adapter dependencies; do not change the baseline to accommodate a new runtime.

## Proposed invocation

These commands are portable examples, not execution receipts. Set `EVAL_ART`,
`EVAL_CACHE`, and `EVAL_PYTHON` to the actual restored artifact directory,
Hugging Face home, and inference virtualenv executable. Choose an unused output,
log and process name. Use the same `--state-dir` for all heavy M6/M7 model work.

```sh
"$EVAL_PYTHON" reports/jensen-evaluation-v1/resumption/launch_remaining.py adapter \
  --plan "$EVAL_ART/text-evaluation-v1/frozen-v2/heldout/plan.json" \
  --adapter-run "$EVAL_ART/text-adapter-completed29-v1" \
  --hf-home "$EVAL_CACHE" \
  --output "$EVAL_ART/text-evaluation-v1/execution-heldout-bf16-v2/adapter" \
  --state-dir "$EVAL_ART/text-process" --name m6-heldout-adapter-bf16-v2
```

Save the preflight and publish its operational code before adding `--execute`.
Keep CPU BF16, eight threads, the 180-second per-case timeout, seed 42,
4,096-token context and 160-token output budget unchanged. Do not modify any
file included in the frozen plan's implementation manifest while the remaining
matched/RAG invocations are pending.

After saving and freshly verifying the complete adapter condition, run the
separate base-only RAG diagnostic with another unused output:

```sh
"$EVAL_PYTHON" reports/jensen-evaluation-v1/resumption/launch_remaining.py rag \
  --plan "$EVAL_ART/text-evaluation-v1/heldout-rag-v1/plan.json" \
  --hf-home "$EVAL_CACHE" \
  --output "$EVAL_ART/text-evaluation-v1/execution-heldout-rag-bf16-v2" \
  --state-dir "$EVAL_ART/text-process" --name m6-heldout-rag-bf16-v2
```

## Preservation and collection

The original runner writes each attempted generation before its post-generation
identity check, then writes the finalized attempt. The partial-checkpoint helper
can preserve a finalized prefix while the server runs. A partial package is
neither a generation-resume cursor nor a collectible complete condition. If a
previous interrupted attempt is recovered, keep its denominator and provenance
separate from any explicitly registered new invocation.

The completed condition needs `supervisor.json`, both health identities, launcher,
server log, excluded warmup, `condition/run.json`, both raw/finalized attempt
logs, and private review data. Preserve the supervisor's whole file inventory,
upload the archive, freshly retrieve it, and build the durability receipt from
the retrieved bytes. A complete loop may contain six failed answers; completion,
claim support and promotion remain separate results.

Collect the restored facts/style/adapter directories using the unchanged
`tools.adaptation_evaluation.local_condition collect` command and one receipt
covering all three conditions. The existing collector validates exact files,
seals, identities, frozen requests and stopped servers. The operational
[collection helper](../operations/collect_package_m6.py) also creates the
opaque review packet; its old absolute paths must be explicitly relocated in
a new wrapper, without changing the archived original.

`deliver_remaining.py` provides that relocation for the original condition,
partial-prefix, collection and RAG helpers. It verifies each imported helper
against the original source manifest, records its actual new paths and wrapper
hashes, and leaves the helpers' original source files unchanged. It runs no
inference or transfer. All finalizers still require genuine independently
obtained upload/readback receipts. Use a new artifact root with the canonical
relative paths and process names shown above; do not reuse the old workspace.

```sh
"$EVAL_PYTHON" reports/jensen-evaluation-v1/resumption/deliver_remaining.py matched-package \
  --artifact-root "$EVAL_ART"
# Upload the resulting archive, freshly retrieve it under EVAL_ART/readback,
# and register the actual verified record in the central artifact index.
"$EVAL_PYTHON" reports/jensen-evaluation-v1/resumption/deliver_remaining.py matched-finalize \
  --artifact-root "$EVAL_ART"
"$EVAL_PYTHON" reports/jensen-evaluation-v1/resumption/deliver_remaining.py collect \
  --artifact-root "$EVAL_ART"
```

`partial-package`/`partial-finalize --attempts 3` preserve a finalized matched
prefix. `rag-package --mode partial3`, `rag-package --mode final`, and
`rag-verify --mode final --artifact-receipt <actual-record.json>` use the separate
RAG workflow. For RAG validation, supply `--inference-python` when it differs
from the checkout's `.venv-inference/bin/python`. The recorded command must
match the real process owner. The RAG archive also includes both relocation
wrappers and the explicit binding record.

Share only the resulting `agent-review/reviewer/` directory with a fresh agent.
Save categorical factual-support/coverage findings before revealing condition
keys or private reference answers. Retain failed and unattempted cases in the
coordinator's denominator. Human likeness/quality ratings remain null until an
identified human supplies them. Review the RAG diagnostic separately against
its actual captured passages, including both empty-retrieval cases.

## Audit validation

This addition does not change the original M6 plans, contracts, inference,
prompting, comparison, packaging or review rubric. Preparation checks and actual
runtime outcomes are recorded separately by the coordinating session.
