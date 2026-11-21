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
- Existing graph/rule tests: 101 passed at the first checkpoint. Expanded tests
  now pass 149 cases, including engine wiring, default-off behavior, whole-unit
  budgeting, changed-source rejection, snapshot scope, humor opt-outs and
  sensitive-context exclusions. No generated-answer improvement is claimed yet.

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

Implementation checkpoint `206d8dfab7edab101d2851d5e791022ee14ee575` was pushed
to PR #29 and all nine affected file contents were hash-verified remotely.
Subsequent fixes cover explicit humor opt-outs, financial-loss contexts and
snapshot-only calls; these are included in the 149 passing checks.

## Frozen generation inputs

[`manifest.json`](../reports/persona-delivery-2026-11-21/manifest.json) freezes
12 fresh authored questions, three conditions and 36 planned attempts before
any candidate answer was generated. Eight questions test conditional advice,
two test situational humor boundaries, and two test unsupported facts/private
interactions. The graph selects a path on six of eight advice questions. It
misses both oversight questions because their wording lacks the current lexical
cue conjunction. They remain in the planned denominator; the questions and
selector were not rewritten to force retrieval success.

All conditions receive the same factual evidence and live-concise instruction.
This comparator is matched-evidence retrieval, not the older unaugmented RAG
baseline. The full manifest contains exact prompts, token counts, source/code
hashes, balanced attempt order, generation settings and the rating rubric.

Verified commands from repository root:

```bash
reports/persona-graph-2026-11-20/runtime/.venv/bin/python reports/persona-graph-2026-11-20/runtime/run_with_server.py -- .venv/bin/python -m tools.persona_delivery_eval freeze
reports/persona-graph-2026-11-20/runtime/.venv/bin/python reports/persona-graph-2026-11-20/runtime/run_with_server.py -- .venv/bin/python -m tools.persona_delivery_eval run --limit 6
```

Only the freeze command has run at this checkpoint. `run` resumes by exact
manifest/request hashes and saves every attempt atomically. Each bounded batch
will be pushed and read back before starting the next batch.

Voice identity/prosody require reference audio and actual TTS comparisons. This
text experiment does not test or improve acoustic identity by itself.
