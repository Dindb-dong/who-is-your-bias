# Jensen matched generation: frozen inputs

The current [execution status](execution-status.json) links every finished
condition's measured report, complete raw-output artifact and fresh-readback
receipt. It retains normal, failed and unattempted calls separately. Partial
prefix reports preserve work while a condition is running and remain excluded
from final collection until the whole condition has been saved.
The final [29-update adapter](../jensen-recovery/text-v1/completed-serving.json)
is now available for the frozen adapter condition. Matched execution and semantic
review do not themselves establish human likeness or model promotion.

**Registration history:** inputs were frozen before any model generations.
The execution-status link above records subsequent actual calls.
There are 12 development cases and six cases from the separately acquired No
Priors 2023 test event. One attempt in each of three conditions is planned: 54
generation calls. This is a controlled generation experiment with explicitly
selected training evidence. It does not measure retrieval effectiveness, live
conversation sessions, speech, avatar quality or persona likeness.

The [protocol](protocol.json), [development summary](development-summary.json),
[heldout summary](heldout-summary.json) and [case summary](case-summary.json)
publish identities, source IDs, classifications and hashes. Original questions,
complete passages, references, prompts and detailed support rationales remain
in the saved private artifact identified by [artifact receipts](artifacts.json).

A separate [default lexical heldout RAG diagnostic](heldout-rag-v1/README.md)
now freezes actual runtime retrieval and prompt packing for the same six
questions. It registers one base condition and keeps the original answerability
labels. Its preparation is separate from the 54 matched-generation calls below.

## What is held constant

| Condition | Weights | Factual passages | Wording examples |
| --- | --- | --- | --- |
| `facts` | Pinned Qwen3-1.7B base | Complete fixed passages | None |
| `style` | Same base | Identical passages | Fixed eligible training examples |
| `adapter` | Same base plus the verified Jensen adapter | Identical passages | Byte-identical prompt to `style` |

The base is `Qwen/Qwen3-1.7B` at immutable revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. All conditions request 160 output
tokens, context window 4096, temperature zero, top-p one and seed 42. The exact
pinned tokenizer measured **703–1,625 input tokens**. No selected factual
passage or wording example was dropped or shortened during packing.

The v4 corpus digest is
`939fc6cbf1b6d195233740f06bb63a631a13264cb25e67e474e3f28236420088`.
Its snapshot cutoff is `2026-11-12T18:36:58.188093Z`; unknown recording dates
remain unknown. Snapshot modality labels are neutral across conditions. Voice
and appearance are explicitly outside this text comparison. The snapshot does
not establish a current, dated Jensen position on the control question.

Sixteen cases have a wording-example contrast. The current-position and
unseen-future controls have neither factual evidence nor an eligible style
example under the temporal policy, so their `facts` and `style` prompts are
identical. Their adapter contrast can still be observed.

The existing Korean factual instruction deliberately remains **one sentence,
third person, no direct quotation**. This limits expressiveness and the meaning
of any style-similarity observation. The source-description fix distinguishes
original Q&A/excerpts, editorial summaries and mixed packed evidence; it does
not loosen those answer constraints. Source disfluencies and interruptions
remain intact. In particular, the selected Sega publisher turn ends at an
interruption; its complete supported claims are the evaluation targets, and no
missing words have been invented to complete it.

## Coverage and reference isolation

Development coverage comprises Korean CUDA, Sega and TSMC questions; values
and conviction; an English accelerated-computing question; fixed-context
follow-up, correction and topic-switch probes; and unsupported endorsement,
unknown current position, private-information and unseen-future controls.
Those topics and the training corpus are development-exposed. They are not
fresh tests. The three dialogue probes supply authored earlier-user context
and the protocol's fixed acknowledgement, not a live previous model answer.

The six heldout case suffixes are `00072`, `00078`, `00186`, `00190`, `00253`
and `00297`. The original 50-case dataset passed its strict loader before this
subset was selected. Its manifest file SHA-256 is
`0c5df9f3a41371570a0fb57426bd1bd78d445ce6eddd6f00b8de941fe403682b`.
The loader binds event isolation and normalized twelve-word-span checks against
both v4 and the previously exposed PR17 Jensen corpus. It does not establish
semantic independence or absence from base-model pretraining.

Agent support classification against eligible **training material**, before
generation, gives three answerable, two partly answerable and one unanswerable
case. The detailed rationales and nearby-but-insufficient source IDs are saved
privately. Original reference questions and answers are unchanged. Earlier
speaker turns from the heldout event are deliberately omitted from inference;
the subset record preserves the original case hashes and that transformation.
No heldout reference answer, heldout factual passage or heldout style example
is in a model prompt. Exact reference wording is not a target when independent
training evidence supports only a broader or partial answer.

## Execution and review procedure

Execution waits for the shared model-memory slot. Adapter execution additionally
requires the exact completed, compatible adapter receipt. The additive
[current execution contract](execution-contract-bf16-v2.json) registers **CPU
BF16 with eight threads** for all conditions. The original FP32 loading process
was killed before its warmup or any frozen question completed; the preserved
[failure and pre-output amendment](runtime-amendment-bf16-v2.json) record that
observation. The original FP32 contract and every plan remain unchanged. This also retains the eight-thread update to the four-thread runtime plan
in the earlier registration without changing either v2 plan, its questions,
messages or decoding. If the excluded loading/warmup preflight cannot fit,
preserve that failure and explicitly register a new common runtime before
starting cases. A matrix-kernel probe does not establish end-to-end generation
speed. Do not mix precision or thread counts across the comparison.

Restore the saved private archive, check its file SHA-256, then validate the
immutable `frozen-v2` plans against the exact bound implementation. The v1
plans in `frozen` are preserved as superseded and unexecuted: their initial
snapshot used historical modality labels. V2 changes only the snapshot ID and
condition-neutral modality labels; cases, facts, selections, references and
cutoff are unchanged. No model output was used for that revision.

Use the [single-condition execution and collection guide](condition-execution.md).
The new supervisor starts one real server, runs exactly `facts`, `style` or
`adapter` with the unchanged frozen runner, saves its report and private review
data, then terminates. Save and verify that complete output before another
condition. The original all-condition `local.py` remains unchanged for protocol
identity; it is not the execution entrypoint for this checkpointed procedure.

Use new output directories. Retain raw generations and failed/partial attempts,
finish reasons, identity checks and latency. Save each completed condition
before proceeding with further expensive work. A transport completion with a
length finish reason is not proof of a complete or correct answer. Each
condition now has its own excluded model load and short warmup. Sequential
order remains unbalanced, and its times describe this CPU rather than a target
device. Interruption between conditions can also change background host load.

Review factual support, temporal fidelity, answer quality, style similarity,
unsupported first-person assertions, repetitive mannerisms and quotation
memorization using the existing blind packet. For supported cases, check each
claim against the supplied passage and permitted citation IDs. For partial
cases, check that unsupported detail remains unknown. For controls, check that
the requested unsupported fact or quotation is not fabricated. Lexical overlap
and citation syntax remain diagnostics rather than semantic or human scores.

All human ratings are **ungraded**. Keep failed and incomplete attempts in the
denominator. The adapter remains opt-in pending actual matched execution and
support/control review. Freezing these inputs does not promote a model or
establish an improvement.

## Preparation checks

Both v2 plan seals passed `validate_plan` with the implementation bound after
checkpoint `6ea11b6df8296e008e866b74fe1ed0e5f3aa0b0c`. Actual-token budgets,
full selected-passage retention, identical factual inputs, identical
style/adapter messages and absence of reference answers from prompts passed.
The heldout loader and snapshot verification passed. V1-to-v2 checks confirm
unchanged questions, references, input-case hashes, selections and factual
hashes. Model execution and human review remain pending.
