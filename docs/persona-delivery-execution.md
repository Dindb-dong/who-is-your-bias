# Conditional delivery experiment — 2026-11-21

Target: `Dindb-dong/who-is-your-bias`, PR #29,
`feat/jensen-persona-graph-nov20`. Starting remote head:
`4848dfda9efa6ad0c93a7a8b68cbbc947f9fdae4`. This work does not merge main.

## Implementation checkpoint

- Added opt-in compact compilation of a single v2 graph strategy, retaining its
  claims, epistemic status, all conditions/exceptions and evidence relationships.
- Added separately source-bound conditional reaction/speaking-style guidance;
  at most one observation of each kind, whole-unit prompt budgeting, explicit
  factual/private/news exclusions and invitation-only low-stakes humor.
- Both additions are disabled by default. No voice service, rig or renderer is
  replaced. Text style is not speaker similarity or measured emotion.
- Five delivery observations bind five existing evidence records from three
  interview events. See [source notes](persona-delivery-source-notes.md).
- Existing graph/rule tests: 101 passed. New regressions and generated-answer
  comparisons are pending at this checkpoint; no improvement claim is made.

## Planned comparison

Fresh authored transfer probes, identical source evidence in three conditions:
evidence-only, compact conditional memory, compact memory plus delivery. Same
model/checkpoint, decoding, answer-length instruction and empty history. This
isolates added guidance from extra factual evidence, not the benefit of graph
retrieval itself. It is not held-out original-interview answer accuracy.

Freeze prompts and rubric before generating answers. Preserve each actual
attempt, failure and truncation, and checkpoint bounded batches. Blind raters
must not receive condition labels or candidate prompts. Report AI-only ratings
and small sample limitations, and retain the earlier negative v2 experiment.

The pinned Qwen3-4B Q4_K_M CPU runtime was restored from its immutable upstream
revision and SHA-256 verified. Follow the existing runtime launcher documentation.
Actual persona measurements and remote checkpoints will be appended after runs.

Voice identity/prosody require reference audio and actual TTS comparisons. This
text experiment does not test or improve acoustic identity by itself.
