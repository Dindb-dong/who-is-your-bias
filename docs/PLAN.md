# Delivery plan

## Product objective

Build a real-time research studio that models Elon Musk's publicly documented statements and, as observable media becomes available, his speaking and movement patterns. A response combines evidence selection, dialogue, voice, timed behavior, and a controllable avatar. The interface identifies the simulation and distinguishes generated dialogue, factual source notes, generic speech, and learned assets.

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

Intermediate changes are committed by coherent capability. Completed milestones are published to their branches and opened as pull requests. New follow-up PRs remain available for review; the prior merge instruction is recorded against the four existing foundation PRs.

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

