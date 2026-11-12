# Heldout default lexical retrieval-to-generation diagnostic

**Status: six runtime prompts frozen and durably restored; zero model responses
generated.** This is a separate, single-base diagnostic of actual default
retrieval feeding generation. It supplements the [matched fixed-evidence
experiment](../README.md), whose six heldout prompts deliberately use selected
training passages and do not measure retrieval efficacy.

The [protocol summary](protocol.json), [execution contract](execution-contract.json)
and [artifact receipt](artifacts.json) contain public hashes and results of
preparation. Original questions, private reference answers, full retrieval
queries and complete captured prompts remain in the saved private archive.

## Frozen behavior and inputs

The plan SHA-256 is
`3e2c6409b45a67b11798c17f23c6ce869aa9d71309775bf9af8f69afcc7d02d6`.
The source fixed-evidence heldout plan remains
`7fd35de01ea1a364aa7b5a2bf8547bf3a4a6cd3ebee4b3764b8ce7c8957e5c7c`.
All six questions, private references, their original ordering and the prior
answerability labels/rationales are unchanged. Earlier speaker context from the
heldout event remains omitted. The actual 50-case original dataset was loaded
and validated before the same six cases were selected by those frozen IDs.

The preparation tool invokes the current `Engine` with the default `Retriever`
(`qa-bm25-situation.v3`), lexical backend, top four, `style_examples=False`,
the unchanged v4 training corpus and the condition-neutral v2 snapshot. It
forwards the actual search arguments and records the returned ranks. At the
language-model boundary, a capture-only generator saves the actual packed
messages and deliberately produces **no token or invented response**. There
is no hand-selected rescue passage, concept expansion, dense reranker or
reference-derived wording example.

The unchanged prompt packer retains whole factual passages within the declared
4,096-token context and reserves 160 output tokens. The actual pinned Qwen3
tokenizer measured the following inputs:

| Heldout case suffix | Original corpus answerability | Retrieved candidates | Supplied whole passages | Input tokens |
| --- | --- | ---: | ---: | ---: |
| `00072` | Partly answerable | 4 | 3 | 3,783 |
| `00078` | Answerable | 4 | 4 | 3,685 |
| `00186` | Partly answerable | 0 | 0 | 724 |
| `00190` | Answerable | 4 | 3 | 3,792 |
| `00253` | Answerable | 0 | 0 | 720 |
| `00297` | Unanswerable | 4 | 3 | 2,750 |

Three candidate passages were dropped as whole records by the runtime budget;
no answer was shortened. Four prompts exceed the earlier fixed-evidence
experiment's approximate 2,300-token planning target. This diagnostic preserves
default top-four behavior and the original 4,096-token budget, so those longer
inputs are explicitly recorded and will cost more prefill work.

The two empty retrieval results are real outcomes. Jensen's existing runtime
constructs a speculative prompt that says no verified evidence was retrieved;
that instruction is preserved. The original answerability label concerns what
the eligible training corpus can support, not what this retrieval selected.
Conversely, retrieving four candidates for the jacket question does not make
its unsupported requested detail answerable. Human support judgments remain
ungraded, and the candidate counts are not relevance or quality scores.

## Execution

Only the pinned Qwen3-1.7B **base** condition is registered here. Runtime is CPU
float32/eight threads, greedy output, seed 42, 160 output tokens and context
4,096. The exact independent checkpoint/tokenizer hashes are inherited from
the [base verification](../base-verification.json). No adapter is configured.

The existing comparison plan envelope contains `facts`, `style` and `adapter`
keys for runner compatibility, with identical captured messages and no wording
examples in this artifact. This does not register three experimental arms.
Use the diagnostic's `run` command, which invokes only `facts`; do not collect
this artifact as a matched three-condition comparison.

Restore the archive from `artifacts.json`, verify its SHA-256 and every restored
member, then wait for the shared model-memory slot. Run from the project root
using the prepared inference environment. This is a **proposed command**, not
an execution receipt:

```sh
EVAL_RAG="$ART/text-evaluation-v1/heldout-rag-v1"
PYTHONPATH=src:. .venv-inference/bin/python \
  -m tools.adaptation_evaluation.rag_diagnostic run \
  --plan "$EVAL_RAG/plan.json" \
  --contract "$EVAL_RAG/execution-contract.json" \
  --timeout 180 \
  --output "$ART/text-evaluation-v1/execution-heldout-rag-v1"
```

Use a new output directory. The one-condition supervisor verifies the exact
base, tokenizer, runtime and source-bound implementation, saves actual launcher
and health hashes, performs its own excluded short warmup, runs the frozen
cases and terminates its server. Save and freshly verify the complete output
before another expensive condition. Keep failed, truncated and interrupted
attempts in the registered six-case denominator; a loop completing is not a
claim that its answers are complete or supported.

## Scope and review limits

This is an **offline retrieval-to-generation diagnostic**. Retrieval and prompt
packing are captured before model execution; the returned language-service
responses are raw generations. This procedure does not execute the main chat
HTTP endpoint, session continuation, application output guard or display-label
insertion. It does not measure live end-to-end API latency, speech or avatar
behavior. Those integration checks are separate.

Compare each generated claim with the actual supplied passages and allowed
citation IDs. For empty retrieval, assess whether the answer acknowledges the
missing evidence and avoids invented specific facts. For partial support,
identify which detail remains unknown. Retain the original answerability labels
and do not tune retrieval or prompts on these heldout outputs. No retrieval
candidate is promoted by this preparation or by completion alone.

## Reproduction and checks

The additive preparation code was remotely verified at
`1214618d27517d678febf48b437d0aeef778cae6` before the actual freeze. It also binds
the engine, dialogue, query-policy and situation code in addition to the
existing frozen comparison implementation. The wrapper and current code must
match those hashes when validating or executing the saved artifact.

```sh
PYTHONPATH=src:. .venv-inference/bin/python \
  -m tools.adaptation_evaluation.rag_diagnostic freeze \
  --corpus "$ART/original-dialogue-v4/corpus" \
  --heldout "$ART/heldout-original-v1" \
  --prior-plan "$ART/text-evaluation-v1/frozen-v2/heldout/plan.json" \
  --answerability "$ART/text-evaluation-v1/frozen-v2/heldout-answerability.json" \
  --snapshot "$ART/text-evaluation-v1/frozen-v2/snapshot.json" \
  --tokenizer "$EVAL_TOKENIZER" \
  --base-contract reports/jensen-evaluation-v1/execution-contract.json \
  --id jensen-heldout-default-lexical-v1 \
  --output "$ART/text-evaluation-v1/heldout-rag-reproduction"
```

`EVAL_TOKENIZER` must identify the cached immutable
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` tokenizer. Reproduction creates a
new artifact with a new creation timestamp; restore the saved plan to retain
its exact identity. Never overwrite or relabel the original frozen plan.

All 53 adaptation-evaluation tests passed at the code checkpoint, including ten
new tests for actual Engine prompt capture, unchanged ranking and passages,
source/label/snapshot changes, default-retriever identity, and preserving cases
where the runtime does not generate. The actual private archive is 54,495
bytes; a fresh remote materialization matched its SHA-256, all five restored
files matched the original bytes, and both `validate_rag_plan` and
`validate_execution_contract` passed on the restored artifact. These are
preparation and durability checks, not model quality results.
