# Proposal alignment and next implementation milestones

Audit date: **2026-11-11, Asia/Seoul**. Implementation snapshot:
[`f9a9c23`](https://github.com/Dindb-dong/who-is-your-bias/commit/f9a9c23a576469536a3c288b43c07489d709d089),
after PRs #7, #8, and #9 were merged into `main`.

## Purpose and source

The original project proposal is [2026-2 YAICON](https://docs.google.com/document/d/1eHnjPdZbQgCUOTIuixlIQc2oIDqyyHjUvhCFUc30-cw),
including its sections dated November 9 and November 11. This audit read that
document and compared its requirements with the repository at the revision above.
The source document may evolve; the mapping below records the requirements read
on the audit date, not a continuously synchronized copy.

The proposal asks whether providing actual interview answers, and then adding
situation-relevant examples, makes the same language model respond more like the
target person. It also specifies a presentation path using **LLM + RAG →
Qwen3-TTS Base → MuseTalk**, with presentation media generated in advance and
live generation reserved for audience questions.

The user explicitly selected **Elon Musk** as the first subject. That choice is
an authorized adaptation of the proposal's initial artist/idol examples. The
rigged 3D avatar also addresses the user's conversational project request, but
the proposal's MuseTalk presentation path remains a separate deliverable.

The repository has a substantial working application and verified integrations.
Its central proposal-level experiments are still incomplete. This document makes
that distinction explicit and establishes the next acceptance criteria.

## Requirement-to-implementation map

| ID | Proposal requirement | Observed implementation | Status and next step |
| --- | --- | --- | --- |
| R1 | A: the same LLM receives an artist profile and basic style guidance | The `profile` condition provides a profile without retrieved evidence | Basic condition implemented; retain it as the controlled baseline |
| R2 | B: profile plus actual Q&A with a similar topic | The normal retriever selects reviewed summaries of statements, experiences, and plans | Partial; curate original question/answer pairs and preserve conversational context |
| R3 | C: profile plus actual Q&A selected by topic and conversational situation | A situation classifier and score multiplier exist, but no indexed situation label matches the classifier's vocabulary | Ineffective situation reranking on this snapshot; align the data vocabulary and implement a meaningful situation-aware retrieval condition |
| R4 | Ask A/B/C held-out interview questions and compare generated responses with the person's actual answers | Development questions, source-isolation checks, citation diagnostics, and source-support reviews exist | The intended reference-answer/persona comparison has not been completed |
| R5 | Generate speech with Qwen3-TTS Base | The reference-conditioned service produced four preserved WAV samples with ASR and timing measurements | Functional integration exists; naturalness and voice resemblance still need actual listening judgments |
| R6 | Generate lip-sync video with MuseTalk | The application renders and animates a rigged GLB human asset; no MuseTalk implementation is present | Implement and verify the specified video-generation path |
| R7 | Prepare presentation speech/video in advance and support live Q&A | Interactive chat and speech components exist | Assemble a reproducible presentation package and measure the complete live Q&A path |
| R8 | Record speech rate, pre-answer pauses, smile/laughter, head movement, and visible hand movement | Transcript/media tooling and measured motion windows exist | Build reviewed observations linked to actual Q&A, with explicit timestamps, speaker attribution, and unobservable states |

Implementation references: [retrieval](../src/bias/retrieval.py),
[current prompt builder](../src/bias/compact_prompts.py),
[generated-answer evaluation](evaluation.md), [voice evaluation](voice-evaluation.md),
[human avatar](humanoid-avatar.md), and [motion library](motion-library.md).

## Findings that determine the next priority

### 1. The indexed corpus is not yet a real Q&A corpus

There are **163 reviewed evidence records** in total. Runtime retrieval indexes
**159 records from 30 sources**; four records from the held-out source are
excluded. The indexed records contain 98 `stated_view`, 35 `reported_experience`,
and 26 `authored_plan` entries. They are curated, attributed summaries rather
than original question/answer pairs. No indexed record has an original question
and answer field, and none is a transcript record.

This supports factual-source retrieval, but loses features the proposal wants to
compare: how a question was phrased, what preceded it, the person's original
wording, and the way they responded to the conversational situation.

The [publisher transcript work](research/transcript-acquisition.md) already
provides an acquisition/extraction route for complete TED conversations and
preserves adjacent interviewer and subject turns. Those transcripts remain
local draft material outside the runtime index. Recover or reacquire the
originals through that workflow, then curate and review Q&A records before
indexing them. Do not turn a summary into a fabricated quotation.

Sources: [corpus records](../data/personas/elon-musk/evidence.jsonl),
[record schema](../src/bias/models.py), and [corpus status](research/corpus-status.md).

### 2. C's situation multiplier never activates on the current data

The query classifier returns one of:

- `explanation`
- `reflection`
- `praise`
- `challenge`
- `conversation`

The corpus uses free-form descriptions such as `giving advice`,
`responding to criticism`, and `explaining schedule choices`. The classifier's
five possible outputs have **zero exact matches** among the indexed situation
labels.

In [the retriever](../src/bias/retrieval.py), C multiplies a candidate's score by
`1.15` only when the classified situation exactly matches a value in that
record's `situations` list. That condition never becomes true on this snapshot.
For the same query, corpus, limit, and other options, **B and C therefore retrieve
the same records in the same order**.

C still adds a `conversation_situation` hint to the model context and has a
designed animation path. It is not identical to B in every respect. Those
differences do not establish the proposal's retrieval of situation-relevant
actual Q&A.

The fix needs a shared, documented situation vocabulary and reviewed record
labels, followed by retrieval checks where situation relevance matters. A
meaningful C condition need not return different records from B for every
question; it must prioritize the appropriate reaction when topic similarity
alone leaves a relevant distinction unresolved.

### 3. Existing evaluations do not establish the intended persona comparison

The [generation runner](evaluation.md) executes real A/B/C requests and preserves
their outputs, inputs, timings, and failures. The bundled questions are explicitly
development cases. Existing held-out checks primarily verify source isolation
and abstention; they do not complete the proposal's comparison against the
person's actual answers to unseen interview questions.

The [grounding studies](grounding-quality.md) assess citation coverage, source
support, relevance, and related development criteria. Their reviews do not
establish independent persona fidelity. Citation formatting, source-supported
content, similarity to an actual response, and natural speech are separate
outcomes.

Preserve the existing reports as historical results of their original conditions.
After changing the corpus or C retrieval, give the experiment a new version and
retain the previous corpus/retriever/prompt snapshots or reproducible revisions.
Do not relabel old C results as results of the corrected condition.

### 4. Verified rendering and playback are useful foundations

The merged implementation passed its engineering checks, including actual
desktop/mobile rendering and playback scenarios. The [merged-main workflow](https://github.com/Dindb-dong/who-is-your-bias/actions/runs/42947174466)
succeeded. See [engineering verification and recovery](quality-completion.md)
for exact historical measurements and artifacts.

These results support the application's execution and integration claims. They
do not demonstrate the missing Q&A experiment, a verified personal likeness,
independent voice similarity, or a learned personal gesture policy. The present
human asset is an artistic reference, and source-linked motion replay is distinct
from predicting how the person would move in a new situation.

## Recommended implementation order

### Milestone 1 — Actual Q&A, working C retrieval, and reference-answer evaluation

**This is the recommended next PR scope.** Reuse the current application,
retrieval components, and model adapter to complete the central experiment.

#### Data and provenance

Create reviewed conversational records that retain:

| Field group | Required information |
| --- | --- |
| Original exchange | Interviewer question, actual answer, adjacent conversational context, and original language |
| Attribution | Source URL, source event, recording/publication dates, questioner and respondent, speaker-review status |
| Time | Source-clock question/answer boundaries, with alignment evidence and explicit unknown values |
| Retrieval labels | Topic labels and normalized situation labels, plus the annotation basis |
| Review | Review status, correction history, and links to the original material |
| Experiment boundary | Source-event train/development/test assignment, fixed before extraction or retrieval tuning |

Keep source text, translations, summaries, and inferred annotations distinguishable.
Hold out complete interview events, including duplicate publications of the same
event. Existing source-event isolation checks can be reused. The model-facing
context must not contain evaluator-only answers or withheld source records.

#### Controlled A/B/C conditions

- **A:** profile and common response instructions.
- **B:** the same profile/instructions plus topic-relevant actual Q&A.
- **C:** the same profile/instructions plus Q&A selected using topic and situation.

Use the same model revision, question set, language, generation settings, session
policy, and declared context/output budgets. Record any unavoidable differences.
Version the profile, corpus, situation taxonomy, retriever, prompt, and evaluation
cases. Keep source-window motion selection separate from the text experiment so
animation choices cannot be mistaken for improved text retrieval.

#### Evaluation and acceptance

- Freeze a set of original questions and reference answers from held-out
  interviews. Provide only the questions to the model.
- Add meaningful retrieval regressions showing that a reviewed situation can
  affect selection when competing topical examples have different reactions.
- Verify shared situation-label coverage and source-event isolation before
  inference. Preserve failures and missing annotations explicitly.
- Execute the actual A/B/C matrix and retain raw responses, prompt fingerprints,
  selected record IDs, timings, and incomplete/failed requests.
- Prepare a condition-blinded review bundle. Compare content/viewpoint, wording
  and sentence structure, and situation-dependent response. Grade factual source
  support separately.
- Record actual reviewer judgments and unresolved cases. Leave human judgments
  ungraded until a person supplies them; application tests and model-assisted
  reviews must retain their own labels.
- Report whether C improves over B, is indistinguishable, or performs worse.
  Completion means a valid experiment and an honest result, not a predetermined
  positive score.

The exact initial dataset and evaluation sizes should be recorded when the
source-event inventory is reviewed. Do not present a convenient pilot size as
statistical evidence of generalization.

### Milestone 2 — Presentation media and live Q&A

Implement the presentation path specified in the proposal:

1. Generate a prepared presentation's speech through Qwen3-TTS.
2. Generate the corresponding lip-sync video through MuseTalk in a verified,
   compatible execution environment.
3. Package the rendered video, audio, subtitles, source inputs, and reproduction
   metadata so the prepared presentation can be replayed reliably.
4. Connect audience question → grounded answer → speech output as the live path,
   and document its relationship to the existing 3D avatar interface.

Acceptance requires an actual playable presentation artifact, verified audio/video
timing, and an exercised live-question path. Record the chosen model versions,
hardware, generation commands, elapsed times, and failures. A configuration stub
or external API placeholder is not evidence that video generation ran.

Measure the live path on the intended demonstration hardware: question completion
to first visible text, first audible speech, and turn completion. Preserve the
individual LLM, TTS, transport, and rendering timings so optimization targets a
measured bottleneck.

### Milestone 3 — Situation-linked voice and motion quality

Attach observable speech/movement features to the reviewed exchanges:

- speech rate and the pause before an answer, where the source clock permits it;
- visible smile and audible laughter as separate observations;
- head movement with camera cuts/motion accounted for;
- visible arm/hand movement, with occluded or off-screen intervals marked
  unobservable;
- whether a reaction occurs while listening or speaking.

Use multiple independent source events and retain held-out events when evaluating
new selection or modeling methods. Existing geometry extraction, review ledgers,
source-clock handling, and rig controls are reusable components.

Evaluate voice naturalness and similarity through actual listening. Current warm
CPU synthesis took roughly **3.5–3.8 times the audio duration** in the preserved
four-sample evaluation; that measurement does not establish live-demo latency on
another device. Use the [voice report](voice-evaluation.md) as the recorded
baseline, then measure the selected demonstration environment.

Textual A/B/C can be completed with the current LLM and retrieval architecture.
A learned gesture policy, persona graph, or model fine-tuning is a separately
scoped research extension with its own data and evaluation requirements.

## Delivery and reporting

- Keep coherent implementation changes in separate commits and substantial
  milestones in reviewable PRs, following the existing delivery workflow.
- Reference R1–R8 in PR descriptions and update this mapping with implementation
  evidence and remaining gaps.
- Maintain separate statuses for code integration, completed model execution,
  reference-answer evaluation, human review, and presentation readiness.
- Preserve historical measurements and make changed experimental conditions
  explicit rather than overwriting earlier results.

The next implementation priority is **actual Q&A data + effective situation-aware
C retrieval + held-out actual-answer evaluation**. The verified studio provides
the execution foundation for that work.
