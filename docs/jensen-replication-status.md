# Jensen replication execution status

Run `jensen-2026-11-13`, started 2026-11-13 (Asia/Seoul).

Repository: `Dindb-dong/who-is-your-bias`. Integration branch:
`feat/jensen-resumable-replication`. Base: PR #18 merge
`8515b5f30f3cd412e18db3d54ee28bf2a4c28694`.
Implementation review: [PR #19](https://github.com/Dindb-dong/who-is-your-bias/pull/19).
Last verified code checkpoint before this status update:
`cd54c28f61a08c9356e77d44398b5b55891405b6`.

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

## Current preserved results

The user reiterated that full original videos and an actual newly authored
avatar are required, and asked to publish useful intermediate material quickly.
Short clips, audio-only recordings, located URLs and full videos are counted
separately. The prior unverified eight-video claim is not inherited by this run.

- Original dialogue v4: **554 Q&A / 9 events**, including **551 complete
  exchanges**, three excerpts and 410 factual projection records. A corrected
  mixed-speaker-label parser recovers omitted YC/Sega answers; earlier v3 remains
  explicitly superseded. Full v4 bytes were saved and freshly retrieved.
- Text selection: **113 complete answers / all 9 events**, 24-word minimum,
  768-token whole-exchange limit, deterministic cap of 16 per event. Dataset,
  selection and immutable Qwen3-1.7B tokenizer/model hashes are saved.
- Text computation: **four actual saved optimizer updates**, 8,716,288 trainable LoRA
  parameters, BF16 CPU. The actual fresh-process continuation from remote step 1
  to step 2 passed (282.0 seconds for that invocation). Step 2 was then uploaded,
  freshly retrieved and extracted for continuation. Step 4 has now also been saved,
  freshly retrieved and loaded with its optimizer/RNG/cursor intact. Further
  computation uses eight CPU threads with unchanged optimization settings. Its
  29-update plan is still
  in progress; model quality has not yet been evaluated.
- Voice: **19 publisher-matched candidates**, independently screened down to
  **8 clips / 91.84 seconds / four events**. Human listening review is zero.
  Actual acoustic encoding produced **1,151 frames × 16 codebooks**, saved and
  freshly retrieved. **16 actual LoRA updates are complete and durably saved**.
  A fresh process restored remote step 1 and completed step 2; subsequent 6/10/14
  checkpoints and final step 16 were also freshly retrieved. The interrupted final
  block recovered from verified step 14 without inventing an uncompleted update.
  **Eight actual matched waveforms** (four reference-conditioned and four with
  learned weights, including Korean) are now saved and freshly retrieved. The
  loaded adapter/checkpoint hashes were checked at synthesis time. Independent
  ASR and speaker-embedding diagnostics are saved separately. The complete
  input/weight/output bundle was freshly retrieved, all 134 payload hashes matched,
  and the dataset/reference/checkpoint loaders passed. The eight generated WAVs
  and a comparison listening page are committed; human reviews remain zero.
- Avatar: a **2,530,256-byte fitted Jensen GLB** is committed with an explicit
  optional Settings selection and source license. Its measured geometry, licensed
  face texture, dark iris and authored salt-and-pepper material are preserved in
  editable masters. The 20→80-step restored fit matches uninterrupted state.
  Likeness remains imperfect and ungraded; the candidate is not the default.
- Motion: **580 measured face/body/hand frames across six Goldman/Stripe/WIRED
  windows**, six source-bound clips and explicit Motion Lab source choices are
  already in Git. Thirty finger controls preserve confidence/occlusion exclusions.
  All three events and their exact source clocks are preserved. These previews do
  not establish habits.
- Held-out evaluation: 50 original No Priors exchanges passed isolation against
  both the v4 corpus and PR #17 summaries. WEF's former transcript URL is
  unavailable; no replacement transcript is invented.
- Source videos: the [live source ledger](../reports/jensen-recovery/source-progress.json)
  and [media artifact receipts](../reports/jensen-recovery/media-artifacts.json)
  distinguish all located events, complete local files, failed addresses,
  partial uploads and fully reconstructed remote videos. All
  **eight complete recordings are now reconstructed from fresh remote chunks**:
  4,412,488,053 original bytes / 10h35m29s, including four complete keynotes
  totaling 7h34m46s. All full SHA-256 values match. Three full original interview
  audio files (Stanford, Acquired and Lex) are separately saved and verified.
  Transcript counts, recording duration and target-speaker duration are distinct.

## Workstream checkpoints

| Workstream | Saved implementation | Current gate |
| --- | --- | --- |
| Corpus | Authentic v4, held-out references, all eight full video restorations | Source/media acquisition batch complete; preserve event isolation |
| Voice | Source clocks, four-event dataset, 16 completed updates, eight public waveforms, ASR/WavLM diagnostics and verified gateway | Candidate remains optional; human likeness is ungraded |
| Avatar | Actual fitted model, editable master, finger controls, observed clips and actual desktop/mobile browser rendering | Human likeness remains ungraded; candidate stays optional |
| Text | Frozen dataset, four saved updates and successful remote optimizer continuation | Finish bounded run and serve exact adapter |
| Retrieval/evaluation | Frozen lexical/concept/hybrid comparison; both candidates remain opt-in | Matched base/style/adapter and heldout response experiments |

Integrated changed-path Python checks: 186 passed and two optional-Torch modules
skipped in the core environment. Separately, real Torch checkpoint/adapter checks
passed 45 tests in the model environment. Web: TypeScript passed and 236 tests
passed at the then-verified head. Later changes still need final focused checks.
Actual GitHub Actions run [43278388296](https://github.com/Dindb-dong/who-is-your-bias/actions/runs/43278388296)
passed web checks/build, 96 focused Python tests and **50 desktop/mobile browser
cases** at head `05d4062`. That head predates the fitted model; its explicit browser
case and final camera checks are queued separately. Synthetic PCM/detection
fixtures test transport behavior, not neural inference speed or human likeness.

The later run [43283696054](https://github.com/Dindb-dong/who-is-your-bias/actions/runs/43283696054)
at `ff64ea3e49335fcfceea35b5ba2c0f29af4b4784` passed **241 web tests, 96 focused
Python tests, 30 historical browser cases and 20 Jensen browser cases**, including
the exact fitted candidate on desktop and mobile. TypeScript and production build
also passed. The downloaded diagnostic ZIP matches GitHub's SHA-256 and was
separately saved and freshly retrieved; both candidate screenshots were inspected.

The separate [actual TCP preview check](../reports/jensen-runtime-v1/README.md)
passed **91 assertions** against the original v4 corpus, exact avatar bytes,
six explicit motion choices, real streaming and four-turn session provenance.
Its delivery timings describe evidence preview, not neural text generation.
The first failed harness attempt is retained alongside the corrected run.
The actual main-API learned-voice gateway also passed 95 assertions and produced
a 1.52-second WAV in 34.835 seconds on CPU. Its exact loaded step-16 weights and
reference provenance matched. This is not real-time performance. Recorded
generated-audio avatar playback is now queued in browser CI.

On 16 supported + eight unsupported frozen development questions, relevant
top-four hits were 3/16 archived lexical, 11/16 concept lexical and 12/16 pinned
multilingual hybrid. All had zero false-hit controls; both candidates failed
the predeclared 80% recall gate and remain opt-in. Full failures and hashes are
in the [retrieval report](../reports/jensen-retrieval-v2/README.md).

Read the [recovery index](../reports/jensen-recovery/README.md),
[text remote-resume proof](../reports/jensen-recovery/text-v1/resume-step0002.json),
[voice run](jensen-voice-adaptation.md), and
[avatar results](../reports/jensen-avatar/README.md) before repeating work.

## Parallel ownership and resumption

- Corpus: completed source ledger/acquisition and source-family splits; now
  frozen matched generation and heldout answerability preparation.
- Voice: voice export/reference provenance, alignment and voice adaptation.
- Avatar: web rig/fingers/framing, media observation and authored avatar assets.
- Coordinator: artifact manifests, text checkpoints/training, completed retrieval
  comparison, model execution, integration and remote checkpoint verification.

Each meaningful implementation batch is preserved before the whole workstream
finishes. Raw media/full transcripts/weights stay outside public Git; actual
objects have stable IDs, sizes and digests in the versioned manifest. Acquired,
prepared, trained, uploaded and verified states remain separate.

Resume by fetching the integration branch, reading this file and the artifact
manifest, then the latest workstream report. Do not repeat expensive work until
recoverable artifacts and current checks have been inspected. Training commands
will be recorded only after their exact pinned inputs and checkpoint paths exist.

## Remaining acceptance

M0 storage readback and real optimizer/RNG equivalence tests are verified. The
actual text checkpoint survived fresh remote retrieval and completed the next
optimizer update in a fresh process; step 2 was again saved and freshly verified. M1–M7 remain in progress. No Jensen
text/voice model quality, accepted likeness, human ratings, or latency target is
claimed by these implementation checkpoints.
New follow-up pull requests are published for review; no new merge is authorized.


## Frozen response comparison

Twelve development cases and six untouched No Priors references are frozen in
[the response protocol](../reports/jensen-evaluation-v1/README.md). All 54 planned
calls use 160 output tokens and identical whole factual passages, with byte-identical
style/adapter messages. Actual tokenizer counts are 703–1,625 input tokens.
Version 1 plans are retained as superseded; version 2 changes only condition-neutral
snapshot metadata before any model output. The private plans, references and snapshot
have been saved, freshly retrieved, seal-validated and checked against the exact v4
corpus. No generated answer is claimed yet at this checkpoint.
