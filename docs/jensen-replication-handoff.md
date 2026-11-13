# Jensen replication: interrupted-run handoff and remaining work

**Historical handoff: superseded by the [PR #19 completion report](../reports/jensen-pr19-completion.md)
and [current execution status](jensen-replication-status.md).** The pending gates
below describe the original recovery handoff. This document preserves the scope and
reported progress of the interrupted post-PR #17 session. PR #18 is being resolved
as an operating-instructions and recovery-documentation change. Merging that PR
does not deliver a trained Jensen persona, trained voice, or newly fitted avatar.

Prepared on **2026-11-13, Asia/Seoul**. Read [AGENTS.md](../AGENTS.md) before continuing.
The [original execution plan](jensen-replication-run.md) remains the historical
brief; use this handoff for current evidence status and the resumption order.

## 1. Evidence and status rules

Three different kinds of information appear below:

- **Verified committed baseline:** inspectable repository files at PR #17's merge,
  `3038629f3c4aeefff568684a6a7ba10df2b61db8`, and PR #18's original plan commit,
  `414a5a4d09c6072a6b9ca6d531b438a74bfdb6ba`.
- **Reported, unverified:** statements in the interrupted conversation excerpt
  supplied by the user during the recovery audit. The underlying code, logs,
  datasets, weights, media, and asset receipts were not recovered in that audit.
- **Pending:** recovery and verification, or a new reproducible execution when
  recovery is unavailable. Do not mark an item complete solely because the
  earlier conversation said it was running or prepared.

An unavailable workspace is not proof that work never executed, that files were
deleted, or that all bytes are permanently lost. Do not invent missing source
URLs, timestamps, model revisions, hashes, reviewer scores, or completion counts.

The current documentation task does not restart collection, training, rendering,
or response experiments. A future implementation task should resume from the
verified baseline and preserve each resulting milestone separately.

## 2. Repository state before this documentation update

| Item | Verified state |
| --- | --- |
| Main baseline | [PR #17](https://github.com/Dindb-dong/who-is-your-bias/pull/17), merged as [3038629](https://github.com/Dindb-dong/who-is-your-bias/commit/3038629f3c4aeefff568684a6a7ba10df2b61db8) |
| Interrupted PR | [PR #18](https://github.com/Dindb-dong/who-is-your-bias/pull/18), branch `feat/jensen-persona-training` |
| Original PR #18 head | [414a5a4](https://github.com/Dindb-dong/who-is-your-bias/commit/414a5a4d09c6072a6b9ca6d531b438a74bfdb6ba), 2026-11-12 21:35:26 KST |
| Original PR #18 diff | One new file, `docs/jensen-replication-run.md`, 59 added lines |
| Other remote branches | The read-only audit checked all 19 then-existing branches; no later implementation was found on another branch |
| CI and releases | No Actions runs for the PR #18 branch and no repository releases were found at audit time |
| Accessible local material | Earlier PR #16 and older files existed; the later run's adapters, long-form recordings, motion outputs, runtime snapshot, and staging archive were not found |
| Separate saved artifacts | No matching saved backup was located in the audit |
| Unreferenced Git objects | The excerpt mentions binary chunks and blob-SHA preparation. No usable upload receipt or blob SHA was recovered; an upload without a branch update remains an unresolved possibility |

The first new durable checkpoint in this documentation task is
[bd572c4](https://github.com/Dindb-dong/who-is-your-bias/commit/bd572c43bb6970a0e57b8cd65ad50a0d68608c29):
the English root `AGENTS.md`. Its remote branch head and exact file content were
read back and verified before this handoff was written.

These observations are dated audit results, not assertions about every future
branch or artifact location. If an older artifact becomes accessible, inspect its
bytes and provenance before changing the corresponding status below.

## 3. Durable functionality to preserve

### Corpus, retrieval, and conversation

The [Jensen corpus guide](../data/personas/jensen-huang/README.md), [profile](../data/personas/jensen-huang/profile.json),
[source registry](../data/personas/jensen-huang/sources.json), and
[snapshot](../data/snapshots/jensen-huang-2026-11-12.json) describe the actual baseline:

- Jensen Huang is already the active subject.
- Nine publisher sources support 32 editorial evidence records: eight NVIDIA
  summaries and 24 paraphrased interview exchanges. These are not the reported
  617 original Q&As or the 105-answer training dataset.
- The six interview sources include Lex #494, Dwarkesh 2026, Acquired 2023,
  Stripe Sessions 2024, Strange Loop 2025, and CSIS 2025.
- The nine active sources are in the training/retrieval partition; do not reuse
  them as independent unseen-event evaluation material.
- The profile's content cutoff is 2026-04-15, while the snapshot availability
  cutoff is 2026-11-12. Do not turn availability dates into statement dates.
- Existing language serving, optional hybrid retrieval, generation guards,
  persona binding, and evaluation tools are reusable infrastructure.

The [source catalog](../data/acquisition/jensen-huang/source-catalog-v2.md)
already lists nine transcript candidates, 17 discussions queued for review, and
seven keynote rows. Those categories describe located leads, not proven unique
acquisitions. Its local-storage note explicitly records no acquired audio/video
for that earlier collection. Do not conflate catalog rows with the later
conversation's acquisition claims.

### Existing response measurements

The [Jensen RAG report](../reports/jensen-rag/README.md) contains durable PR #17
measurements. They are historical baseline results, not tests rerun for this
documentation change.

| Measurement | Recorded result and limit |
| --- | --- |
| Hybrid threshold 0.84 versus 0.60 | Original benchmark target hits rose from 2/6 to 3/6; held-out target hits remained 2/6 |
| 3B hybrid Korean responses | Median 2.22 s; 7/8 completed within five seconds; target citations in 1/6 substantive answers |
| 4B hybrid Korean responses | Median 3.48 s; 4/8 completed within five seconds; target citations in 2/6 substantive answers |
| Controls | Six held-out controls had no citations in the recorded 4B run; the future-quotation request received a verification limitation |
| Scope | One request per question/configuration through the API SSE path; no repeated statistical estimate, browser/audio playback measurement, or human fidelity score |

The general hybrid default remained 0.84. The lower threshold and the interrupted
session's multilingual candidate must not be promoted solely because they
retrieve more material. Consistent five-second completion and reliable grounded
answers remain acceptance work.

### Avatar, attention, and subject isolation

The [attention guide](jensen-attention.md) documents the existing designed Jensen
portrait, explicit participant selection, and eyes-first/head-follow behavior
with a 220 ms head onset. Camera/screen/pointer inputs and lifecycle controls
already exist. Preserve the implemented behavior and subject isolation.

This prototype is not a qualified source-fitted likeness or learned Jensen
performance. The current [channel vocabulary](../apps/web/src/avatar/channels.ts)
has 52 face controls, 15 visemes, and 16 bone controls through the wrists.
Articulated finger controls and a hand-inclusive upper-body framing contract
remain work. The existing clip schema is `avatar-clip.v1`.

The [avatar generation plan](avatar-generation-pipeline.md) was written against
an older baseline. Its statements about Musk being the running subject and some
missing subject bindings predate PR #16/#17. Use the current Jensen baseline above
for status, while retaining that plan's newly generated identity, complete
arms/hands, and new-performance requirements.

Historical Musk datasets, adapters, measured motion, voice demonstrations, and
latency results remain reference implementations. Do not relabel them as Jensen
deliverables.

## 4. Interrupted-session inventory

The following table reconstructs the user's supplied progress excerpt. Every
right-hand result is **reported, unverified** until its supporting artifacts are
recovered and checked.

| Workstream | Last reported progress | What is still needed |
| --- | --- | --- |
| Original interview dialogue | Extraction grew from 541 to 617 Q&As across 11 events/interviews | Recover or rebuild exact text, speaker attribution, complete-turn segmentation, provenance, and exclusions |
| Broader source coverage | More than 25 distinct events/interviews after deduplication; original dialogue, official automatic captions, and edited news footage separated | Recover the source ledger and verify unique event families and actual acquisition status |
| Values evidence | Nine themes organized; named examples included first principles, changing direction when wrong, perseverance/founding costs, coaching, firsthand information gathering, and responsibility | Recover the actual taxonomy and evidence links; do not invent missing theme labels |
| Full audio acquisition | Complete audio files from eight events | Locate or reacquire bytes; verify duration, hashes, speaker content, and source clocks |
| Text preparation | 105 complete answers from eight interviews selected | Reconstruct the exact accepted/rejected sets, token filtering, splits, and dataset version |
| Text adaptation | A 1.7B LoRA run started; last explicit progress was 11 of 27 planned optimizer updates | Recover checkpoint and configuration, or run a new pinned experiment; final weights and completion are not established |
| Voice preparation | 60 source-balanced segments from five interviews, about 6.4 minutes | Verify speaker intervals, acoustic alignment, transcript accuracy, balance, and saved dataset |
| Voice adaptation and samples | No recoverable completion evidence in the supplied excerpt | Execute and preserve actual weight adaptation if feasible; separately record conditioning-only baselines and English/Korean samples |
| Full video acquisition | Progress reached eight distinct video events; the final four newly acquired keynotes totaled about 6 h 49 min | Recover or reacquire full files and manifests; the duration describes those four keynotes, not all videos |
| Motion extraction | 11 clips from three videos, 88 seconds total; face, upper body, hands/fingers mentioned | Recover exact source intervals, observations, confidence, occlusions, and retargeting outputs |
| New avatar | Face/model creation, motion attachment, rendering, optimization, and validation preparation mentioned | Recover or build the editable asset, export, qualification evidence, and integrated playback |
| Retrieval | Gaps around CUDA, Sega, TSMC, values, and long Korean questions; multilingual improvements on some leadership/supply-chain prompts | Reproduce on frozen cases and repair relevance/grounding failures before promotion |
| Response experiments | Development cases with matched evidence and separate held-out end-to-end cases prepared | Recover frozen inputs or create a newly versioned protocol; run and preserve actual outputs and measurements |
| Persistence tooling | Runtime snapshot helper/CLI, staging snapshot, implementation archive, frontend binary chunks, and blob-SHA preparation mentioned | Recover exact code/receipts or implement and verify these capabilities before relying on them |

**Counting rules:** 617 Q&As, 105 accepted answers, and 25+ events are different
populations. Eight audio events, five voice-dataset interviews, and eight video
events cannot be assumed to be the same set. The three-to-five-to-eight video
updates are successive snapshots, not additive counts. Synthetic QA fixtures
must not be counted as collected interviews or silently treated as authentic
training examples.

The exact 1.7B model ID, model/tokenizer revisions, complete nine-theme taxonomy,
complete acquisition URLs, and final upload receipts are not recoverable from
the excerpt. Model size alone is not a reproducible model identifier.

## 5. User requirements for source coverage

The original request requires broad interviews and video, actual text and speech
adaptation, a newly created avatar, and response experiments. A technical keynote
collection alone does not cover the requested persona.

The following is a **required acquisition checklist**, not a claim that the
earlier nine-theme taxonomy or dataset has been recovered:

| Dimension | Evidence to seek and preserve |
| --- | --- |
| Failure and recovery | Founding setbacks, errors, reversals, uncertainty, and how decisions changed |
| Meaning of work | Motivation, responsibility, sacrifice, persistence, and the costs of founding/leading |
| Hiring and leadership | Hiring criteria, coaching, feedback, delegation, standards, and how reasoning is explained |
| Long-term judgment | Patient investment, difficult tradeoffs, first-principles reasoning, and willingness to change course |
| Responsibility | Accountability to employees, partners, customers, and commitments actually discussed in sources |
| Technical/business context | CUDA, Sega, TSMC, AI factories, supply chains, product strategy, and dated context |
| Reflective formats | Long interviews, oral histories, and commencement speeches alongside keynotes |

Use the existing source catalog as a lead list. Its Stanford GSB, YC Startup
School, How I Built This, and Computer History Museum entries are useful starting
points, subject to their recorded review status. The interrupted excerpt also
mentions a Goldman Sachs interview, an NVIDIA booth tour, and an SK hynix booth
signing video. Resolve exact publishers, URLs, dates, and usable intervals before
treating those descriptions as acquired sources.

For avatar references, deliberately include both close facial views and
upper-body/hand views: laughter/smiles, eyebrow changes, blinking, jaw movement,
head turns, listening reactions, and two-handed gestures. Include seated and
standing material, with useful hand/finger visibility. Preserve exact time ranges,
camera cuts, occlusion notes, and whether observations were measured or manually
annotated.

Maintain an event-family identity across an interview's transcript, podcast,
video, captions, reuploads, and edited excerpts. Expand beyond the reported pilot
where coverage is weak; do not treat its counts as a final quality threshold.

## 6. Known issues to carry forward

1. **Lex clock mismatch.** An approximately six-minute advertisement-related
   difference was reported between video captions and podcast audio. Align to
   actual speech and record the mapping. Do not apply an assumed constant offset
   across an entire recording when insertions/cuts require piecewise alignment.
2. **Incorrect university attribution.** A university page reportedly linked a
   different person's video/transcript and was excluded. The institution was not
   identified in the excerpt. Recover the exclusion record or recheck identity;
   do not guess which university it was.
3. **Website tag contamination.** One answer reportedly included a site's trailing
   tags. The excerpt explicitly says it was not among the 105 answers already in
   training. Evaluation was to use a corrected corpus. Preserve separate training
   and evaluation hashes and the correction; do not silently merge versions.
4. **Unsupported endorsement retrieval.** The multilingual candidate selected
   irrelevant interviews for a question equivalent to "Did Jensen approve my
   chatbot?" Retrieved material must actually support the claim. Add explicit
   relevance/grounding checks and evidence abstention for this control.
5. **No premature promotion.** The interrupted session explicitly withheld the
   multilingual default change. Training completion, valid citation IDs, or
   increased retrieval recall alone do not establish response fidelity.
6. **Evaluation leakage.** Split by event family, not Q&A row or file. Keep
   adaptation data, style examples, retrieval indexes, development tuning, and
   held-out references consistent with that split.
7. **Prepared versus executed.** Frozen inputs, synthetic fixtures, queued jobs,
   generated helpers, and upload chunks are not measured completed runs or
   remotely recoverable outputs.

## 7. Execution order and acceptance gates

Each milestone ends with a verified remote code checkpoint and any required
durable artifact receipt. These milestones are pending; they can become separate
implementation PRs or small commits within an authorized workstream.

### M0 — Restore a trustworthy starting point

- [ ] Read current repository instructions and inspect the current remote head.
- [ ] Check known artifact destinations and any newly available original-session
  files once; verify recoverable bytes before rebuilding.
- [ ] Choose accessible persistent storage for datasets, source media, weights,
  and editable avatar assets. Verify a small save/readback.
- [ ] Establish a run manifest and status record; implement periodic persistence
  and resume support where the existing scripts lack it.
- [ ] Verify a bounded save/load/resume cycle before long jobs.
- [ ] Inspect actual CPU/GPU, RAM, disk, dependency, and process-lifetime limits.
  The old plan's nine-CPU/about-10-GB Linux host is historical, not a current
  hardware guarantee.

**Gate:** code and a real artifact survive a fresh retrieval; the next agent has
an unambiguous checkpoint, artifact locator, and resume procedure.

### M1 — Acquire and version the corpus and media

- [ ] Build the source/event ledger and coverage matrix from Section 5.
- [ ] Acquire permitted actual source files in bounded batches; preserve original
  URLs, dates, bytes, hashes, language, source type, and acquisition outcomes.
- [ ] Separate located URLs, fetched transcripts, downloaded audio/video,
  accepted dialogue, and modality-eligible intervals in all counts.
- [ ] Extract complete questions and answers with verified speaker attribution;
  remove boilerplate/tags and retain explicit exclusion reasons.
- [ ] Verify source/video/podcast clocks and version corrected alignments.
- [ ] Freeze train/development/held-out event families before adaptation.
- [ ] Publish reproducible recipes and bounded summaries while preserving full
  working inputs in the durable artifact destination.

Reuse [persona-data preparation](persona-data.md), [original Q&A](original-qa.md),
[media processing](media-pipeline.md), and `tools/persona_data/`,
`tools/adaptation/`, and `tools/media/` where compatible. Existing Musk example
inputs are not Jensen datasets.

**Gate:** manifest-derived counts, retrievable artifacts, clean accepted examples,
and no held-out event in adaptation or runtime retrieval inputs.

### M2 — Reproduce and improve Jensen retrieval

- [ ] Preserve/reproduce the PR #17 baseline with recorded configuration.
- [ ] Add long Korean technical and values questions, including the reported
  CUDA/Sega/TSMC gaps, as explicitly labeled development cases.
- [ ] Measure lexical and multilingual/hybrid candidates on frozen inputs.
- [ ] Exercise unsupported endorsement, private-information, future-quotation,
  and current-information controls against the dated corpus.
- [ ] Record relevant-evidence recall, false evidence selections, citation
  support, abstention, and latency separately.
- [ ] Keep any candidate opt-in until its acceptance criteria are met.

Use [hybrid retrieval](hybrid-retrieval.md), [grounding quality](grounding-quality.md),
and the existing [Jensen RAG report](../reports/jensen-rag/README.md). A corrected
corpus must have a new version/hash rather than silently replacing the comparison
inputs.

**Gate:** the baseline and candidates are reproducibly compared against
predeclared acceptance criteria. Promote only a qualifying candidate; otherwise
retain the baseline and preserve the negative result, including the endorsement
failure. Do not tune on held-out results.

### M3 — Execute and preserve text adaptation

- [ ] Recover the exact 105-answer selection if possible; otherwise publish a new
  versioned selection and explain differences.
- [ ] Pin model and tokenizer IDs/revisions, prompts, token filters, split hashes,
  learning settings, seed, device, and optimization budget.
- [ ] Use existing text export/training infrastructure where compatible, adding
  the checkpoint/resume behavior required by M0.
- [ ] Execute real optimizer updates and preserve logs, step counters, loss,
  checkpoint hashes, and final export/adapter metadata.
- [ ] Load the adapter in the actual language service and verify runtime identity.
- [ ] Keep it a candidate until matched response experiments justify promotion.

See [verified text-adapter serving](text-adapter-runtime.md),
`tools/adaptation/`, and `services/language/`.

**Gate:** a fresh process loads the exact saved adapter, its actual training
progress is evidenced, and another session can resume or reproduce the run.
Do not report the old 27-step target as completed.

### M4 — Prepare, adapt, and evaluate speech

- [ ] Recover or rebuild the reported 60-interval/five-interview pilot, then extend
  it where quality and coverage require; the old 6.4-minute count is not a pass.
- [ ] Review target speaker, actual acoustic boundaries, text accuracy, silence,
  music, overlapping speech, and event balance.
- [ ] Keep reference conditioning and optimizer-based voice adaptation as
  different conditions with different completion claims.
- [ ] Execute a feasible adaptation run with persistent resumable state, or record
  the concrete compute/dependency blocker without claiming trained weights.
- [ ] Save English/Korean generated samples, source/model/config hashes, timing,
  independent intelligibility checks, and clearly ungraded human-review fields.
- [ ] Verify Jensen/generic subject binding and real API speech playback.

Reuse [voice-reference preparation](research/voice-reference.md),
[voice evaluation](voice-evaluation.md), `tools/adaptation/`, and
`services/voice/`. A configuration declaration is not independent speaker
verification.

**Gate:** actual reusable input/output artifacts and a clear account of which
weights were updated, what was only conditioned, and what quality was measured.

### M5 — Build the new avatar and performance path

- [ ] Follow the [new-avatar generation plan](avatar-generation-pipeline.md) and
  [method research](research/avatar-generation-methods.md), using the current
  baseline corrections in Section 3.
- [ ] Fit a new Jensen identity from multiple photographs and actual video
  references. Previously rejected ready-made celebrity candidates remain outside
  the selected approach.
- [ ] Preserve an editable master and self-contained skinned/morph-enabled GLB:
  head, neck, shoulders, torso through waist, complete arms/hands/fingers, usable
  eyes/eyelids, teeth/mouth interior, hair, glasses, and jacket.
- [ ] Recover or regenerate the reported 11 motion windows, then expand expressive
  and hand-visible coverage. Save source intervals, extraction settings,
  confidence, smoothing, exclusions, and observation hashes.
- [ ] Add the missing articulated finger vocabulary, compatible clip/validation/
  retargeting contracts, and framing that keeps both hands visible.
- [ ] Compose new speech, expressions, gaze, head, arm, hand, and finger movement.
  Preserve the existing eyes-first/head-follow attention behavior and explicit
  participant selection.
- [ ] Inspect exported geometry/deformations and actual desktop/mobile playback,
  including lip sync, blinks, expression composition, gesture transitions,
  cancellation, and recovery.
- [ ] Record the fitting method, likeness limitations, build/export recipe,
  asset hashes, subject binding, and target-device performance.

Likely implementation paths include `apps/web/src/avatar/`,
`apps/web/src/components/AvatarStage.tsx`, `apps/web/src/App.tsx`,
`tools/media/`, `tools/motion/`, `tools/animation/`, and
`src/bias/motion.py`. Confirm the current contracts before editing.

The existing conversion/audition tools can inspect real candidate assets, but
their existence does not mean the new asset has been built. Current limits and
supported encodings are described in [avatar audition](avatar-audition.md);
qualify the output against the current loader.

**Gate:** a usable newly authored avatar, complete hand/finger control, new
performances, reproducible source-to-asset provenance, and real integrated
rendering evidence. A static bust, designed placeholder, or replayed talking-head
video alone does not satisfy this milestone.

### M6 — Run controlled response experiments

Freeze a protocol with the [adaptation comparison](adaptation-comparison.md) tools
before examining results:

| Condition | Controlled input |
| --- | --- |
| `facts` | Base model plus fixed factual evidence |
| `style` | Same base model and facts plus fixed wording/style examples |
| `adapter` | Verified adapter with the same facts and style examples as `style` |

Keep the `style` and `adapter` message arrays byte-identical. Manually selected
development evidence measures generation under fixed context, not retrieval
effectiveness; label that selection explicitly.

The existing [held-out workflow](heldout-evaluation-data.md) reserves entire
families outside training, factual retrieval, and style projections. This is an
unseen-event generalization test. Classify each question's answerability against
the eligible corpus: use other eligible sources when they support an answer,
and expect an explicit evidence limitation when they do not. Failure to reproduce
an inaccessible held-out quotation is not by itself a retrieval regression.
If a later experiment deliberately indexes sources held out only from adaptation,
register that as a separate RAG protocol; do not relax the strict held-out loader
or report the two protocols as the same benchmark.

- [ ] Use the same questions, factual evidence, decoding settings, budgets, and
  output checks; record any unavoidable difference.
- [ ] Compare `style` versus `adapter` to isolate the weight-adaptation effect
  under matched contextual inputs.
- [ ] Run a separate held-out event-family retrieval-to-generation evaluation.
  Keep held-out reference answers out of training, retrieval, and style inputs.
- [ ] Include technical, values/leadership, long Korean, follow-up, correction,
  topic-switch, unsupported endorsement, current/private, and future controls.
- [ ] Preserve frozen input hashes, raw generated responses, cited IDs,
  claim-support review, language quality, errors, exclusions, time-to-first-output,
  completion time, hardware, and model/runtime identity.
- [ ] Save a review bundle and leave human fidelity scores ungraded until real
  review occurs. Report failed cases and the number of runs per case.
- [ ] Make an explicit keep/reject/promote decision for both retrieval and adapter
  candidates; do not infer persona fidelity from a finished training job.

Use `tools/adaptation_evaluation/`. The comparison runner's language endpoint
does not cover main-app sessions, speech, or browser playback.

**Gate:** reproducible comparisons and a durable report with actual outputs,
failure cases, and evidence-based promotion decisions.

### M7 — Integrate, validate, and deliver

- [ ] Connect the accepted corpus/retrieval/model/voice/avatar configuration
  without falling back to historical Musk assets.
- [ ] Test actual API streaming, session follow-ups, live speech/avatar timing,
  cancellation, source inspection, and explicit attention selection.
- [ ] Run focused Python/web/browser regressions for changed runtime paths and
  applicable repository gates. Record commands, revisions, results, and skips.
- [ ] Test desktop/mobile framing and behavior on declared target hardware.
- [ ] Save the final runtime configuration, artifact manifest, demonstration
  evidence, known limitations, and reproduction/resume instructions.
- [ ] Verify remote code and artifacts before declaring an implementation PR
  complete. Merge only within the user's authorization.

**Gate:** another session can retrieve the exact delivery and reproduce its
recorded behavior. Unmeasured appearance, voice similarity, or latency remains
explicitly unmeasured.

## 8. Artifact and status contract for the next run

The following layout is a **proposed contract**, not a claim that these files or
a snapshot CLI already exist:

| Durable item | Required contents |
| --- | --- |
| Repository run entry, e.g. `reports/jensen-replication/<run-id>/README.md` | Scope, status, code revision, actual commands, settings, results, failures, and next action |
| Source/event manifest | Canonical URL, publisher, date, source type, event-family ID, acquisition/review state, clocks, byte count, hashes, and artifact locator |
| Dataset/split manifest | Exact example IDs, accepted/excluded reasons, family splits, modality eligibility, correction history, and content digests |
| Model/checkpoint receipt | Model/tokenizer revision, configuration, data hashes, actual optimizer step, resume state, weights hash, persistent locator, and successful load check |
| Avatar/motion receipt | Master/export hashes, source ranges, rig/channel schema, extraction/retargeting settings, qualification/rendering evidence, and asset locator |
| Evaluation receipt | Frozen protocol hashes, model/runtime identity, raw-output locator, metrics, failed cases, reviewer status, and promotion decision |
| Status record | Completed/in-progress/blocked/unverified work, last verified checkpoint, external artifact receipts, and exact next/resume command |

Keep raw recordings, full transcripts, large datasets, weights, and editable
masters in the authorized durable artifact destination. Git stores recipes,
metadata, bounded generated assets/reports where appropriate, and links/IDs that
can locate the actual bytes. Do not publish full source passages merely to make
a raw experiment report convenient.

Use stable identifiers, not temporary sandbox paths or expiring signed download
URLs, as the only retrieval reference. The same file can be locally prepared
and not yet uploaded; record those states separately.

## 9. Existing command entry points

These are existing repository entry points, not evidence that the interrupted
experiments were rerun. Execute only after prerequisites and pinned inputs are
available.

Basic setup and focused validation:

```bash
uv sync --locked --extra dev
npm ci
uv run bias validate
uv run pytest tests/test_jensen_persona.py
npm run typecheck
npm run test:web
npm run build
npm run test:attention
```

The build may provision the pinned face detector. Training, speech, and avatar
tooling have additional dependencies; consult their guides and the actual host
before installing or launching long jobs. `make evaluate` targets the historical
Musk corpus, not a Jensen acceptance benchmark.

The existing PR #17 evaluation script can reproduce its documented comparison
only with the matching configured model services:

```bash
PYTHONPATH=src python reports/jensen-rag/evaluate_followup.py
PYTHONPATH=src python reports/jensen-rag/evaluate_followup.py --threshold 0.60 qwen2.5:3b-instruct
```

For new Jensen runs, inspect the current command help and fill all required
source, model, split, and artifact inputs. Do not copy historical Musk paths or
claim a placeholder command is executable:

```bash
uv run python -m tools.persona_data --help
uv run python -m tools.adaptation --help
uv run python -m tools.adaptation_evaluation --help
uv run python -m tools.media --help
```

The reusable command families include text/voice export and training, comparison
freeze/run/compare/review export, and media preparation/transcription/observation.
Exact Jensen training commands remain pending until their inputs and checkpoint
behavior have been verified.

## 10. Acceptance checklist for the complete Jensen workstream

- [ ] Broad source coverage is backed by a deduplicated ledger and actual
  acquired/retrievable material, including values and reflective speeches.
- [ ] Prepared dialogue, speech, and motion datasets have explicit provenance,
  exclusions, aligned clocks, and isolated event-family splits.
- [ ] Actual text and voice adaptation results have saved weights and execution
  receipts; conditioning-only results are labeled separately.
- [ ] The new editable avatar meets the face/body/hand requirements and performs
  new synchronized expressions and gestures in the actual runtime.
- [ ] Controlled development comparisons and unseen-event end-to-end experiments
  have preserved raw outputs, failures, timing, and justified promotion decisions.
- [ ] Integrated API, speech, avatar, attention, and browser behavior is verified
  at the delivered revision, with limitations and hardware disclosed.
- [ ] Every expensive output has a verified durable location and a usable resume
  or reproduction procedure.

None of these workstream-level boxes is checked by merging the documentation-only
PR #18. The first implementation action should be M0, followed by incremental
milestones with remote verification rather than another single unsaved run.

