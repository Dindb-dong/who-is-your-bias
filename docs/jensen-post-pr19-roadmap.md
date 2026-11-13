# Jensen quality roadmap after PR #19

Status: planned follow-up work. Prepared on 2026-11-13 (Asia/Seoul).

Repository: `Dindb-dong/who-is-your-bias`.
Evidence baseline: PR #19 head
`03e5769dfe99e2f92b7bec2a0c1ae15d744d1fe3`, based on
`main` at `8515b5f30f3cd412e18db3d54ee28bf2a4c28694`.
The user requested this roadmap be pushed to PR #19 and that PR then be merged.
This document records subsequent implementation work; it does not report that
the work below has already run or that any candidate passed a new quality gate.
Use the [PR record](https://github.com/Dindb-dong/who-is-your-bias/pull/19)
for the actual integration status and current checks.

Read the [completion report](../reports/jensen-pr19-completion.md),
[execution status](jensen-replication-status.md), and [repository instructions](../AGENTS.md)
before resuming. PR #19 completed registered execution, preservation and functional
integration. Candidate promotion remains a separate decision.

## 1. What the next work must improve

| Area | Verified PR #19 observation | Consequence |
| --- | --- | --- |
| Matched response execution | 54 attempts: 22 normal completions and 32 failures; adapter development 2/12 and heldout 0/6 normal | Improve completion and instructions before treating the adapter as useful |
| Failure categories | 26 length limits, one timeout, five HTTP 429 busy responses | Diagnose generation termination separately from provider availability |
| Retrieval development | Relevant top-four hits: lexical 3/16, concept lexical 11/16, multilingual hybrid 12/16; all 0/8 false-hit controls | Both candidates missed the existing 80% gate and remain optional |
| Default lexical RAG | Separate six-call diagnostic: four normal, two length failures; two genuinely empty evidence pools | Improve actual retrieval and packing, with no hand-selected rescue evidence |
| Actual neural API | Base and adapter each passed 46/46 functional checks; all ten completed responses lacked citation records; four adapter answers repeated the same CUDA definition | Add semantic conversation acceptance to functional checks |
| Voice | Eight matched EN/KO waveforms and ASR/embedding diagnostics; no human listening ratings | Assess intelligibility, naturalness and likeness independently |
| Avatar and motion | Optional fitted GLB, editable master, articulated hands and six source-bound clips | Human likeness, contextual motion and physical-device performance remain open |
| Performance | Single CPU runs observed text first tokens around 14–27 seconds; an adapted 9.92-second waveform took 71.09 seconds | Measure and improve the complete path on declared target hardware |

Sources: [matched results](../reports/jensen-evaluation-v1/execution-status.json),
[development review](../reports/jensen-evaluation-v1/development-comparison.md),
[heldout review](../reports/jensen-evaluation-v1/heldout-review-v1/comparison.md),
[retrieval comparison](../reports/jensen-retrieval-v2/README.md),
[default RAG diagnostic](../reports/jensen-evaluation-v1/heldout-rag-v1/README.md),
[actual neural API comparison](../reports/jensen-text-live-v1/comparison.md),
[voice results](jensen-voice-adaptation.md), and
[avatar results](../reports/jensen-avatar/README.md).

The experiments have different scopes. M6 used matched, selected factual passages
to test generation. The default lexical RAG diagnostic captured actual retrieval
and prompts offline, then generated raw language-service responses; it did not
execute the main API, session memory, output guard or UI. M7 used the actual main
API with the explicitly selected `jensen-lexical` candidate. Its functional pass
does not establish acceptance of default lexical retrieval or answer quality.

The runtime defaults remain unchanged: optional providers are unconfigured,
retrieval is `lexical`, wording examples are off, the language adapter requires
explicit configuration, and the UI starts with voice off and the designed Jensen
portrait. Preserve those defaults until the corresponding candidate qualifies.
See [runtime settings](../src/bias/config.py),
[language service](../services/language/engine.py), and [Studio defaults](../apps/web/src/App.tsx).

## 2. Delivery order and ownership

| Order | Work package | Deliverable and dependency |
| --- | --- | --- |
| First | E0 — Evaluation contract and fresh holdout | Frozen questions, event splits, scoring and exposure ledger before new experiments |
| Next implementation PR | R1 — Grounded base responses and retrieval | Reliable bilingual answers, source support, citations and contextual queries; uses E0 |
| Following text PR | T1 — Adapter diagnosis and aligned training | Separately versioned development experiments and matched adapter comparison; uses R1 |
| Parallel after E0 | V1 — Voice listening and targeted improvement | Independent listening results, expanded cases and evidence-based training decisions |
| Parallel after E0 | A1 — Avatar likeness and meaningful motion | Human visual assessment, corrected assets and source-reviewed performance |
| Integration PR | I1 — Complete conversational runtime | Actual neural text, neural voice and browser/avatar path on target hardware |

The labels above identify work packages, not already opened issues or PR numbers.
Keep coherent changes in separate PRs as practical. Save small commits throughout
each package rather than waiting for its final acceptance report. Read-only reviews,
voice listening and visual review can run in parallel; coordinate model jobs against
the actual host memory budget.

## 3. E0 — Freeze the next evaluation before implementation

The six PR #19 heldout cases and their findings are now exposed. Preserve them as
immutable historical evidence. If reused during development, label them regression
or development cases; they cannot establish fresh generalization. Do not assume the
remaining questions from the same event are unexposed without an exposure audit.

1. Reserve new event families before adaptation or tuning. Group transcripts,
   podcasts, video editions, captions and excerpts from one event together.
   Keep confirmatory references out of training, retrieval, style examples and
   runtime prompts. Record both repository exposure and any unknown base-model
   pretraining exposure; event separation cannot prove the latter absent.
2. Freeze bilingual technical, values/leadership, follow-up, correction and
   topic-switch cases, plus unsupported endorsement, private information,
   uncertain current positions and future-event controls.
3. Label corpus answerability separately from the support in the passages
   actually supplied. Define the information each answerable positive case
   should cover using eligible evidence, without requiring inaccessible heldout
   wording or facts absent from the supplied sources.
4. Freeze source/model/tokenizer hashes, retrieval settings, complete messages,
   decoding, output/context budgets, timeouts, trial order, repeat counts and
   scoring rubrics. Publish a new protocol ID before outputs.
5. Separate fixed-evidence generation, retrieval-to-generation and actual API
   conversation experiments. Record their denominators and scope independently.
6. Give reviewers opaque outputs and permitted evidence only. Preserve and
   freshly restore blind findings before opening condition keys or reference
   answers. Record reviewer type; agent judgments are not human likeness scores.

All attempted calls remain in execution denominators, including length failures,
timeouts, busy responses and interrupted calls with unknown outcomes. Semantic
scores use explicitly reported applicable completed subsets. Paired comparisons
must report the number of jointly completed pairs as well as completion for all
attempts. A zero-pair comparison cannot establish an adapter winner.

Citation coverage uses factual claims requiring support as its denominator.
Claim support requires the cited passage to support the actual claim, not merely
a valid citation ID. Report zero-applicability cases as N/A. Track positive answer
coverage so blank answers or universal abstention cannot pass vacuously. Normative
advice and permitted conditional reasoning need their own rubric; they are not
automatically factual fabrication or factual citation failures.

**Delivery:** versioned protocol, split/exposure manifest, scoring definitions,
blind-review procedure and output/artifact contract. No new CLI is implied by
these proposed artifacts; inspect and extend the existing evaluation tooling.

## 4. R1 — Establish reliable grounded base answers

### Implementation

- Reproduce the base behavior with the existing pinned inputs before changing
  retrieval, prompts or model choice. Preserve the original observations.
- Inspect bilingual query construction, abbreviations, subject binding and
  history resolution. Evaluate an elliptical follow-up and an explicit correction
  through actual session handling, not only an isolated search string.
- Improve passage boundaries and ranking while retaining stable source IDs,
  dates, attribution and source-clock mappings. Record passages dropped for
  context budget and their relevance. Avoid silently clipping away qualifications.
- Compare lexical, concept and multilingual candidates on identical eligible
  evidence. Keep unsupported relationship/private-data controls and temporal
  filtering intact. Search similarity is not calibrated claim support.
- Require the base response to fulfill the requested content, language and
  format, cite factual claims, and distinguish absent evidence from partial
  evidence. Evaluate empty-pool fallback using its actual documented policy.
- Check provider cleanup after timeout and the origin of busy responses.
  Preserve transport outcomes separately from semantic quality; M7 streaming
  cancellation does not by itself validate non-streaming timeout cleanup.

The earlier RAG diagnostic supplied whole passages within 4,096 context tokens;
three candidate passages were dropped as whole records. Its two empty pools
must remain recorded. Do not substitute matched M6 passages or heldout answers
to make retrieval appear successful.

### Proposed acceptance gates

The retrieval threshold below already existed. The response percentages are
**proposed future gates**, not measured results or retroactive PR #19 requirements.
Freeze the final protocol and sample counts before trials; publish any later
amendment rather than changing a gate after seeing outputs.

| Measure | Gate for the next registered experiment |
| --- | --- |
| Existing retrieval development set | At least 13/16 relevant top-four hits, 0/8 false-hit controls and no loss of baseline relevant hits |
| Normal response completion | At least 95% of all attempted calls, with per-language counts |
| Requested language | 100% of completed answers comply; failed-call counts remain visible separately |
| Factual citation coverage and claim support | At least 95% for each, with explicit claim denominators and independent source-only review |
| Supported answer coverage | At least 90% of predeclared applicable information requirements; report per-question failures and abstentions |
| Unsupported controls | Zero invented specific private/current/endorsement facts in the registered control set |
| Conversation scenario | Initial answer, requested simplification, explicit correction and topic switch all meet their semantic rubrics |

Existing development thresholds are a minimum regression gate. Default promotion
also requires the newly frozen bilingual confirmation set and source-support
review. Report uncertainty and actual sample sizes; small-set percentages are
not population guarantees.

**Delivery:** code and focused regressions, new frozen comparisons, full failure
accounting, actual API semantic cases, independent findings and an explicit
keep/reject/promote decision.

## 5. T1 — Diagnose and align the text adapter

The [training exporter](../tools/adaptation/datasets.py) learns historical
interviewer questions paired with original answers, and intentionally excludes
factual retrieval from that dataset. [Snapshot serving](../src/bias/context_budget.py)
asks for evidence-based claims and citations; supported Korean Jensen responses
add a short third-person Korean format. This contract difference is observed.
Its contribution to poor completion, copying or instruction adherence is a
hypothesis to test. English heldout failures mean Korean mismatch alone cannot
explain the result. Neither undertraining nor memorization is established by
the existing small experiment.

1. Audit chat-template application, assistant-only loss masking, EOS/stop handling,
   whole-example tokenization and train/serve input formats against exact saved
   model and tokenizer revisions.
2. Diagnose termination and repetition with small development-only runs. Test
   output-budget changes separately from dataset/prompt changes; preserve the
   original 160-token failures. More available tokens are not a proven fix.
3. Preserve original dialogue unchanged. Version any serving-aligned training
   examples separately, including language, length, evidence/citation behavior,
   uncertainty and provenance. Record authored or generated transformations.
4. Run bounded training only after input/version checks and a fresh-process
   save/load/resume proof. Preserve optimizer, scheduler where used, RNG,
   data cursor and completed step counts at the documented interval.
5. Compare base, wording examples and adapter with frozen evidence and decoding.
   Keep style/adapter messages byte-identical when isolating weight adaptation.
   Register any broader model/runtime change as a different experiment.
6. Include semantic conversation checks for simplification, corrections and
   topic switching. Detect repeated generic responses, source copying and
   tautologies independently of transport/session success.

**Acceptance:** satisfy the R1 completion, language, grounding and positive
coverage gates without base regression. Before evaluation, define a paired
improvement criterion for persona wording or conversational appropriateness,
its sample size and reviewer procedure. Report mixed outcomes and survivor-only
pair counts; do not claim a win from training completion or 46/46 API checks.
Keep human persona scores ungraded until actual human review.

**Delivery:** causal-diagnosis report with confirmed findings versus hypotheses,
versioned training contract, resumable candidate artifacts, matched blind
comparison and promotion decision. The completed29 candidate remains recoverable.

## 6. V1 — Evaluate voice before expanding training

Start with the [eight existing generated samples](../reports/voice/jensen-2026-11-13/listening/README.md)
and existing condition-blind bundle. Review the actual training/reference clips
as well as the generated outputs; current automated screens are not human review.

- Use multiple independent listeners with condition order concealed. Score
  content completeness, pronunciation, naturalness/prosody and speaker likeness
  separately, including EN/KO, numbers and longer utterances.
- Record exact audio hashes, listening conditions, rater counts and disagreement.
  Preserve null ratings until someone actually listens.
- Assess unseen-event references and non-target-speaker comparisons where
  permitted. Current WavLM references were training-exposed; raw cosines are
  descriptive diagnostics, not resemblance percentages.
- Keep literal ASR scoring and any prospective numeric normalization distinct.
  The earlier number-case errors partly reflected digit-versus-word rendering;
  do not relabel those scores as proven pronunciation defects.
- Use the findings to select additional clean, speaker-verified, aligned speech
  and train a new version only where justified. Record overlap, music, acoustic
  boundaries, transcript corrections, language and event balance.
- Investigate backend termination information and first-audio delivery. The
  current service returned complete WAVs; waveform validity alone does not
  prove natural phrase ending or streamed speech.

**Acceptance:** predeclare human-rating scales, required rater/sample counts and
decision thresholds before a new comparison. Require demonstrated intelligibility
and source binding, with no material regression against reference conditioning,
then independently assess likeness. Report each language separately. No voice
default change follows solely from the existing ASR or embedding numbers.

**Delivery:** preserved listening results, calibrated evaluation plan, targeted
data/implementation changes, any new resumable voice candidate and a promotion
decision. Existing sixteen-update artifacts remain unchanged.

## 7. A1 — Improve likeness and validate meaningful motion

Start from the [source-fitted v3c and motion evidence](../reports/jensen-avatar/README.md).
Review neutral and animated front, profile and three-quarter views, including
face geometry, hair silhouette, baked lighting, glasses placement, mouth/eyes
and hand framing. Preserve the editable master, export hashes and license.

Evaluate identity likeness separately from facial deformation and render
correctness. Human reviewers should see comparable framing/lighting and have
their method and scores recorded. A lower landmark residual is not human
likeness acceptance.

For motion, review continuous source intervals to confirm the speaker, tracked
person, source clock, hand visibility and speech context. Keep confidence and
occlusion exclusions. Measured clips and repeated gestures across sources do not
by themselves establish a personal habit or an appropriate dialogue-trigger rule.
Keep explicit preview/selection until contextual behavior is independently assessed.

**Acceptance:** predeclare visual and motion review criteria, preserve articulated
face/body/hands, cancellation and neutral reset, and verify actual exported assets
on declared desktop and physical mobile hardware. Any phoneme-aligned mouth
improvement requires a real timing source and validation; amplitude-driven mouth
movement must retain its approximate label.

**Delivery:** review findings, targeted asset/animation changes, source-to-export
receipts, real device captures and separate likeness/performance decisions.
The existing rejected dense-v2 and textured-v3 artifacts remain historical evidence.

## 8. I1 — Qualify the complete conversational path

Exercise an actual user request through main API, the exact neural text provider,
the exact voice provider, browser playback and the chosen avatar. Include source
inspection, follow-up, correction, topic switching and explicit attention selection.

Before trials, declare target hardware, model precision, provider placement,
concurrency, output lengths and resource budgets. Freeze cold/warm latency,
memory and rendering acceptance limits; the CPU observations above are baseline
measurements, not target-device promises. The earlier roughly 10 GB host ran full
text and voice work serially, so concurrent model feasibility must be measured.

Record these separately:

- Request start, first text token, text completion, TTS request and first
  byte/chunk, first non-silent browser signal and playback completion.
- Provider and aggregate process memory, GPU memory where applicable, browser
  frame timing and audio underruns.
- Cancellation-to-idle and reset during text generation, synthesis and playback,
  plus same-session recovery and absence of stale/partial turns or stale audio.
- Repeated cold/warm samples with declared counts, p50/p95 and all failures.
  A browser signal timestamp is not a measured loudspeaker acoustic onset.

Declare whether models stay resident together or use staged loading. Capture
timestamps and resource traces for any claimed overlap; two available services
do not prove simultaneous inference. Add real text/audio streaming only with
observed chunk delivery, ordering, cancellation and recovery evidence.

**Acceptance:** the complete declared path meets its preregistered target-device
latency/resource limits and semantic/speech/render criteria across the registered
cases. Verify desktop and physical-phone behavior. Retain the 750 ms browser
clock regression gate as a functional check; it is not perceptual lip-sync
acceptance. Preserve simulation labels, persona/voice/motion binding and explicit
participant selection.

**Delivery:** reproducible runtime configuration, integrated scenario runner,
raw traces and hashes, real device captures, failure recovery evidence and a
bounded deployment-readiness decision.

## 9. Checkpoint and resumption procedure

1. Read current root/scoped instructions, the remote branch, this roadmap, the
   latest execution record and the [artifact manifest](../data/runs/jensen-2026-11-13/artifacts.json).
   Record the exact new base/head and preserve unrelated work.
2. Inspect existing artifact receipts and recover reusable bytes before
   reacquiring media, retraining or rerunning evaluation. Validate archive
   size/SHA-256 and every payload; a reachable URL alone is insufficient.
3. Create a new run ID and durable status entry with the selected work package,
   owner, scope, protocol version, hardware and next action. Mark planned,
   prepared, running, completed, failed and unverified states distinctly.
4. Save a small coherent code/document commit after each meaningful batch and
   before every long job. Push without rewriting history, then re-read the
   remote head and affected file bytes before continuing.
5. Save expensive outputs in bounded batches to the authorized artifact
   destination, freshly restore them and record stable IDs, hashes and sizes.
   Keep raw recordings, full transcripts, large datasets/weights and editable
   masters outside public Git. Preserve actual optimizer/RNG resume state.
6. After interruption, resume from the last verified receipt in a fresh process
   and output directory. Do not infer success or an exit code for a lost owner.
   Keep prior frozen results, failures, blinded findings and receipts immutable.
7. Use focused checks for changed paths and applicable repository CI. Update the
   handoff with exact commands only after their inputs and arguments are verified;
   clearly label proposed commands. Do not launch model jobs solely for docs.
8. At each work-package close, publish what changed, actual denominators,
   reviewer status, remaining failures and the explicit promotion decision.
   Defaults change only with recorded qualifying evidence and intended scope.

The next implementation entry is E0 followed by R1. Existing tools under
`tools/adaptation_evaluation/`, `tools/adaptation/`, `tools/voice_evaluation/`,
`tools/retrieval_development.py` and the linked guides are starting points;
inspect their current interfaces before composing exact commands. This roadmap
does not invent a new executable command or claim a new experiment has run.
