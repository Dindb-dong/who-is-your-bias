# PR #19 completion resumption

**The registered remaining execution and agent review are complete.** The
[completion report](../../jensen-pr19-completion.md) records all 54 matched
attempts, six separate RAG attempts, and both actual neural API probes. New
outputs, failures, independent reviews and condition mappings were saved and
freshly restored before subsequent expensive stages. See the
[final targeted validation](final-validation.json) and current PR checks for
the separately measured software gates.

Resumed on 2026-11-13 (Asia/Seoul) from the verified integration head
`324b9e8e5af9ff16f471badf2e1478c7c5e2bdd1`, branch
`feat/jensen-resumable-replication`; base remains
`8515b5f30f3cd412e18db3d54ee28bf2a4c28694`.

At resumption the remote execution record preserved 48 of 54 matched calls.
Its last six heldout adapter calls had no committed result. This session found no local
copy of that interrupted owner, so it makes no claim about its exit or any
unpreserved calls. Existing completed conditions are reused, not rerun.

## Verified recovery

[Restored inputs](restored-inputs.json) records nine actual fresh artifact
retrievals, exact archive byte counts/SHA-256 values, and every extracted file
hash. These include the completed 29-update serving adapter, original v4
corpus, frozen v2 evaluation plans, heldout facts/style outputs, development
collection/review and separate RAG/M7 preparation. All archive hashes match
the existing immutable receipts. No new model generation is claimed here.

The resumed host again has nine logical CPUs and approximately 9.7 GiB RAM,
with no GPU or swap. The inference environment was restored to the recorded
Torch 2.11.0+cpu / Transformers 4.57.3 contract. The initially absent pinned base
cache was retrieved and all nine files independently hashed before generation.
The saved adapter was restored without retraining. The M6 frozen prompts, code
seals, CPU BF16/eight-thread runtime and 160-output-token budget remained
unchanged. The separate M7 harness's exact identity-envelope comparison was
corrected only after its original failure was preserved; its
[lineage](../../jensen-text-live-v1/schema-corrected-preparation.json) records
that one code change without a model, prompt or budget amendment.

## Original remaining order, now completed

1. Validate the exact base cache and final adapter, then execute and preserve
   the six heldout adapter attempts in a new output directory.
2. Collect the three heldout conditions using verified per-file durability
   receipts and perform condition-blind agent support review before unblinding.
3. Execute and preserve the separate six-case heldout retrieval-to-generation
   diagnostic; keep it separate from the 54 matched calls.
4. Execute the prepared real neural-text session, cancellation and recovery
   checks, preserving failed as well as completed attempts.
5. Reconcile measured results, candidate decisions, focused regressions and
   the PR description. Human likeness remains ungraded unless actually rated.

Every listed stage now has a measured result and artifact receipt in the
completion report. The matched adapter failed the quality promotion decision;
both M7 runs passed the bounded functional route. Human likeness remains
ungraded. Do not restart these immutable named attempts when resuming a later
task: restore their existing receipts and choose a new registered experiment
for any changed protocol. No merge is authorized by this completion request.
