# Jensen replication execution status

Run `jensen-2026-11-13`, started 2026-11-13 (Asia/Seoul).

Repository: `Dindb-dong/who-is-your-bias`. Integration branch:
`feat/jensen-resumable-replication`. Base: PR #18 merge
`8515b5f30f3cd412e18db3d54ee28bf2a4c28694`.
Last verified code checkpoint before this receipt:
`4f8df709f841ff2ef81ada3094a5a242dcd753fd`.

The [historical handoff](jensen-replication-handoff.md) defines M0–M7.
This is a new reproducible run; unverified earlier acquisition/training counts
are not inherited as completed work. No later-session training or source-media
artifacts were found in the accessible earlier workspaces or matching saved-file
search. This access audit does not establish permanent loss.

## Verified starting point

- Read current root instructions and current remote `main`; no nested AGENTS.md
  was present at the base revision. Use isolated worktrees and explicit ownership.
- Actual host: Linux x86_64, 9 logical CPUs, approximately 9.7 GiB RAM,
  57.7 GiB free filesystem, no CUDA/MPS. Processes have no promised lifetime
  beyond this session. Heavy text/voice fits are serialized against memory.
- GitHub connector checkpoint writes use an expected branch head, followed by
  a fresh Git fetch and byte comparison of every affected path. CLI cloning and
  fetching work; authenticated CLI push is unavailable.
- Uploaded an actual 269-byte persistence probe, retrieved it to a separate
  directory, compared every byte and SHA-256, and continued its serialized
  state in a fresh process. See the [artifact manifest](../data/runs/jensen-2026-11-13/artifacts.json).
  This proves the artifact transfer route, not a model training/resume result.
  Model optimizer/RNG save/load/resume remains a separate pre-training gate.
- Locked Python development environment installed. Optional model environments
  and browser dependencies are being prepared.

## Workstream checkpoints

| Workstream | Saved implementation | Current gate |
| --- | --- | --- |
| Corpus | Subject-aware extraction and source metadata; 14-family split registration | Acquire clean original dialogue and actual media, preserve bytes |
| Voice | Speaker-bound exports and hash-verified named references | Align actual source speech, execute resumable candidate adaptation |
| Avatar | 30 finger joints, compatible clip playback, hand-inclusive framing | Fit a new identity from actual references; export and inspect |
| Text | Existing pinned-model trainer inspected | Add optimizer/RNG/cursor persistence and verify resume before fitting |
| Retrieval/evaluation | Historical PR #17 remains baseline | Freeze new development controls and matched comparisons |

First focused checks: corpus extraction 21 passed; voice export/reference
49 passed; avatar finger/rendering 39 passed plus TypeScript typecheck.
These are component checks, not integrated acceptance or persona-quality scores.

## Parallel ownership and resumption

- Corpus: source ledger, acquisition/parser code, source-family splits,
  retrieval improvements and focused data checks.
- Voice: voice export/reference provenance, alignment and voice adaptation.
- Avatar: web rig/fingers/framing, media observation and authored avatar assets.
- Coordinator: artifact manifests, text checkpoints/training, matched response
  experiments, integration and remote readback of every checkpoint.

Each meaningful implementation batch is preserved before the whole workstream
finishes. Raw media/full transcripts/weights stay outside public Git; actual
objects have stable IDs, sizes and digests in the versioned manifest. Acquired,
prepared, trained, uploaded and verified states remain separate.

Resume by fetching the integration branch, reading this file and the artifact
manifest, then the latest workstream report. Do not repeat expensive work until
recoverable artifacts and current checks have been inspected. Training commands
will be recorded only after their exact pinned inputs and checkpoint paths exist.

## Remaining acceptance

M0 storage readback is verified; training-state resume is pending. M1–M7 are
in progress. No Jensen text/voice model quality, source-fitted likeness, human
ratings, or latency target is claimed by these implementation checkpoints.
New follow-up pull requests are published for review; no new merge is authorized.
