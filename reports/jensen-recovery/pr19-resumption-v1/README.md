# PR #19 completion resumption

Resumed on 2026-11-13 (Asia/Seoul) from the verified integration head
`324b9e8e5af9ff16f471badf2e1478c7c5e2bdd1`, branch
`feat/jensen-resumable-replication`; base remains
`8515b5f30f3cd412e18db3d54ee28bf2a4c28694`.

The remote execution record preserves 48 of 54 matched calls. Its last six
heldout adapter calls have no committed result. This session found no local
copy of that interrupted owner, so it makes no claim about its exit or any
unpreserved calls. Existing completed conditions are reused, not rerun.

## Verified recovery

[Restored inputs](restored-inputs.json) records nine actual fresh artifact
retrievals, exact archive byte counts/SHA-256 values, and every extracted file
hash. These include the completed 29-update serving adapter, original v4
corpus, frozen v2 evaluation plans, heldout facts/style outputs, development
collection/review and separate RAG/M7 preparation. All archive hashes match
the existing immutable receipts. No new model generation is claimed here.

The current host again has nine logical CPUs and approximately 9.7 GiB RAM,
with no GPU or swap. The inference environment is being restored to the
recorded Torch 2.11.0+cpu / Transformers 4.57.3 contract. The pinned base model
cache is absent and must be retrieved and independently hashed before running
the final condition. The saved adapter is not a replacement for those base
weights. Frozen prompts, code seals, CPU BF16/eight-thread runtime and the
160-output-token budget remain unchanged.

## Remaining execution order

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

Each coherent output is saved and freshly verified before the next expensive
stage. No merge is authorized by this completion request.
