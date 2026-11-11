# Delivery plan

## Product objective

Build a real-time research studio that models Elon Musk's publicly documented statements and, as observable media becomes available, his speaking and movement patterns. A response combines evidence selection, dialogue, voice, timed behavior, and a controllable avatar. The interface identifies the simulation and distinguishes generated dialogue, factual source notes, generic speech, and learned assets.

## New avatar generation workstream — 2026-11-12

The user selected **new avatar creation from multiple photographs and video**
for the Jensen Huang visual-avatar workstream and excluded the previously
suggested ready-made celebrity assets. Follow the
[generation pipeline](avatar-generation-pipeline.md) and
[primary-source method evidence](research/avatar-generation-methods.md).
The earlier acquisition records are historical context, not the candidate list
for this workstream.

The output must include a complete upper torso, both arms and hands, articulated
fingers, facial expressions, lip sync and new gestures. The product baseline is
an editable master plus a skinned, morph-enabled GLB. Video-based ExAvatar and
multi-image LHM++ are bounded Gaussian reconstruction experiments with separate
renderer/export requirements; neither is assumed to produce a drop-in GLB.

Implementation order: source coverage and splits; reconstruction pilots; neutral
identity and appearance; face/body/finger rig; runtime finger controls, framing,
speech-expression composition and persona binding; fresh-dialogue auditions;
then qualification and promotion. All stages are **planned**. The document is
grounded in the runtime at `0ef4e8b`; it does not report a generated Jensen model,
training run or completed likeness evaluation. The currently configured Musk
persona also requires an explicit subject-binding boundary before a new visual
can enter a conversational session.

## Review boundaries

| Milestone | Review unit | Acceptance |
| --- | --- | --- |
| Evidence and conversation foundation | PR 1, targeting `main` | Source-disjoint corpus, validated ingestion, retrieval, API, provider integration, grounded abstention, evaluation, tests |
| Interactive avatar studio | PR 2, targeting PR 1 branch | Working browser conversation, evidence inspection, streaming, audio controls, procedural rig, GLB retargeting, responsive layout, browser verification |
| Media pipeline and reproducible delivery | PR 3, targeting PR 2 branch | Local transcription and observation tools, voice-service integration, launch/deployment instructions, contract checks, reproducible evaluation |
| Broader corpus and original transcript acquisition | PR 4, targeting PR 3 branch | Reviewed primary-source expansion, source-event isolation, English/Korean retrieval checks, full publisher transcript extraction, measured coverage |

Foundation PRs 1–4 were merged into `main` in order on 2026-11-10 UTC after the user's explicit merge instruction, preserving their intermediate commits. Follow-up work remains organized into coherent commits and reviewable milestone PRs:

| Follow-up | Review unit | Acceptance |
| --- | --- | --- |
| Local inference and generated-answer evaluation | Local language runtime and A/B/C evaluation PR | Actual HTTP/SSE model connection, bounded lazy runtime, token/timing observations, separate ungraded semantic review |
| Audio-timed embodiment and measured media | Dependent speech/motion PR | Reference-candidate workflow, real ASR and voice HTTP smoke, audio-bound mouth timing, measured clip replay, calibration, browser checks |

Intermediate changes are committed by coherent capability. Completed milestones are published to their branches and opened as pull requests. PRs #5 and #6 were merged into `main` on 2026-11-11 UTC following the next explicit merge instruction. The resumed quality work was divided into PRs #7, #8, and #9: citation/semantic evaluation, multilingual voice quality and review audio, and human-avatar/measured-motion integration. All three were merged in dependency order on 2026-11-11, producing `f9a9c23a576469536a3c288b43c07489d709d089`; its integrated CI passed.

## Proposal alignment and next milestones

The [proposal alignment audit](proposal-alignment.md) maps the original **2026-2 YAICON** plan to the current implementation. Passed engineering checks do not complete the proposal's actual-interview Q&A experiment or its MuseTalk presentation path.

| Priority | Proposed review unit | Acceptance |
| --- | --- | --- |
| 1 | Actual Q&A data, effective situation-aware retrieval, and reference-answer evaluation | Reviewed original exchanges; shared situation vocabulary; meaningful B/C retrieval distinction; fixed source-event holdouts; controlled same-model A/B/C runs; preserved outputs and a blinded review bundle |
| 2 | Prepared presentation media and live Q&A | Actual Qwen3-TTS → MuseTalk presentation artifact, audio/video timing verification, and a measured live question-to-speech path |
| 3 | Situation-linked voice/motion quality | Reviewed observable features tied to Q&A, independent source-event evaluation, real listening judgments, and latency measurements on the demonstration hardware |

The original-Q&A and shared-taxonomy milestone is implemented. Full v2 materialization contains 80 train / four development / eight test exchanges. The first frozen same-model experiment completed 24/24 responses; B/C retrieved no examples on all eight held-out questions, so C superiority is not established. See [the recorded experiment](reference-evaluation.md). Subsequent retrieval changes need a new version and fresh independent evaluation. Preserve existing results as historical runs of their original conditions.

## Executed proposal follow-ups

The three follow-up implementation units are now present: original-Q&A/reference evaluation, prepared presentation/live measurement, and situation-linked observations. The first two are published as [PR #10](https://github.com/Dindb-dong/who-is-your-bias/pull/10) and [PR #11](https://github.com/Dindb-dong/who-is-your-bias/pull/11); observation delivery depends on both. They are review branches, not new merged-main claims.

- The frozen text experiment completed 24 actual calls but produced no retrieval contrast on its eight questions. C efficacy remains inconclusive; redesigning retrieval requires a new version and fresh independent holdouts.
- The prepared Qwen3-TTS → MuseTalk video is 12.24 seconds, 306 frames, and fully decoded with matched audio/video timing. It is English narration over the fixed artistic portrait.
- Two actual CPU language-to-voice requests completed; complete WAV acquisition took 128.284 s and 109.324 s. The longer-answer timeout is preserved. This configuration is not responsive conversational voice; browser/acoustic timings on the intended demonstration device are still unmeasured.
- Observation extraction measured four Q&A links across two events, 19 speech-rate windows, ten linked visual windows and 512 frames. Validated pauses, audible laughter, accepted speaker/movement annotations and human perceptual ratings remain ungraded. The runtime observation index stays empty until review requirements are met.

These remaining quality/evaluation states are not filled with synthetic user ratings. The review bundles, strict importers and target-device export are the concrete handoff for obtaining that evidence. See [current requirement mapping](proposal-alignment.md).

## Fidelity ladder

1. **Runnable without model credentials:** a source-note preview retrieves documented evidence. It does not claim to be an LLM or a personality clone. A procedural, rigged avatar previews synchronized, explicitly designed behaviors.
2. **Configured inference:** connect a local or hosted OpenAI-compatible language model and a speech server. Returned citation IDs are checked against retrieved records; this is not an entailment check. Provider errors are visible.
3. **Measured personal voice and motion:** import local interviews, align utterances, inspect observations, and connect an appropriate voice reference and rigged model. A waveform or facial landmark alone is not evidence of a personality trait.
4. **High likeness:** capture or obtain a high-quality head/body asset and calibrate its blendshapes and bones; evaluate identity, speech, gesture, timing, and conversational behavior separately.

## Evaluation principles

- Split by source interview before extraction; do not place the same interview in both retrieval and held-out evaluation.
- Separate retrieval relevance, evidence validity, temporal source accuracy, and persona fidelity. Automated lexical scores are not human likeness scores.
- Preserve source dates, locator timeline, source IDs, and transform history.
- Unknown, unobservable, and unverified are explicit states.
- Never invent interview quotations, motion captures, audio timestamps, trained checkpoints, or benchmark results.
- Run capability tests independently of paid providers; report separately which external model paths were exercised.


