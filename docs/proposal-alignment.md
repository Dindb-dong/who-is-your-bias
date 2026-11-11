# YAICON proposal implementation status

Source: [2026-2 YAICON](https://docs.google.com/document/d/1eHnjPdZbQgCUOTIuixlIQc2oIDqyyHjUvhCFUc30-cw), read from the shared school Drive on 2026-11-11. The source may contain errors and may evolve. The [original audit preserved from 5559eda](proposal-audit-5559eda.md) records the previous gaps and full acceptance criteria; it is historical, not the current application description.

The central implementation now uses original interview question/answer pairs, a shared situation vocabulary, an event-isolated experiment, and actual same-model A/B/C executions. The first held-out pilot exposed a retrieval coverage failure: none of its eight questions obtained eligible examples. **It does not establish a C advantage.** Keeping that failure visible is part of the experiment's acceptance criteria.

## Requirement map

| ID | Requirement | Current evidence and remaining boundary |
| --- | --- | --- |
| R1 | Profile-only baseline on the same LLM | Identical common instructions, profile, model, generation parameters and budgets across A/B/C; all conditions call the model even without examples in the frozen runner. |
| R2 | Topic-relevant original Q&A | 92 locally materialized exchanges: 80 train, four development, eight test; bounded excerpt default and pinned full-text recipes. Publisher attribution reviewed by an AI agent, not independently by humans. |
| R3 | Topic plus conversational situation | Shared eight-label taxonomy and working 1.35 reranking after a common topical gate; meaningful development ranking regression. No eligible examples on the eight final pilot questions. |
| R4 | Compare against actual withheld answers | 24/24 real Qwen4B responses completed; hashes, requests, raw outputs, timing and failures retained. Condition-blinded review and reconstruction tooling separates source support, model-assisted judgments and ungraded human judgments. |
| R5 | Qwen3-TTS Base speech | Fresh 12.24-second narration synthesized in 79.27 s on CPU, with three generated segments and an independent unprompted ASR report. Human naturalness and likeness remain ungraded. |
| R6 | MuseTalk presentation | Actual MuseTalk 1.5 inference generated 306 frames at 25 fps in 491.07 s. Packaged H.264/AAC timing and full decode verified. Fixed artistic portrait and geometric blend are explicit adaptations; perceptual lip-sync remains unrated. |
| R7 | Prepared presentation plus live audience Q&A | Bundled player, captions, downloads and fresh live-session handoff implemented. Browser timing export and real HTTP language-to-voice benchmark distinguish digital/server measurements from unmeasured target-device/acoustic output; see the actual live report. |
| R8 | Situation-linked observable voice and motion | Measured four original Q&As across two events: 19 matched speech-rate windows and ten visual windows with 512 sampled frames, eight excluded frames and two smile candidates. Source clocks and unknown acoustic boundaries remain explicit. Human review is uncompleted and the reviewed runtime index is empty. |

## Delivered experiment

[Original Q&A](original-qa.md) explains attribution, context, complete versus partial exchanges, source-wide training selection and the exact split manifest. [Reference-answer evaluation](reference-evaluation.md) records the real run and interpretation. [Prepared media](presentation.md) and [live measurement](live-demo-latency.md) record the separate speech/video path.

[Situation observations](situation-observations.md) records the executed R8 pilot and its review workflow. All four whole-answer speech rates and validated pre-answer pauses remain unknown; matched subwindows and visible-smile candidates do not establish audible laughter, semantic gestures or perceptual voice quality. TED2017 is held out for observation validation while remaining a text-training event. The [artifact audit](research/observation-pilot-status.md) verifies that the four exchange digests are unchanged in Q&A v2, but the preserved observation report and its review bundle remain bound to the v1 corpus. The larger text corpus does not expand measured media coverage or supply human judgments.

The five source events have fixed roles: TED2013 and TED2017 train, TED2022 recorded development, and Lex #400/#438 test. Old summary-corpus material is not mixed into this study because its historical split assignments differ. All original v1 selections are retained by the expanded v2 corpus. Full source text remains local, reconstructed from publisher material whose canonical transcript and selection hashes are pinned.

The ordinary application explicitly shows which corpus is loaded. Its no-key preview exposes source excerpts without pretending that they are newly generated persona responses. The interactive research lab remains exploratory: normal application abstention and sessions are not substituted for the frozen experiment's fresh, same-model requests.

The run completed all eight questions under all three conditions. Every paired A/B/C request was identical because retrieval abstained. Six questions nevertheless produced different raw text between requests at temperature zero; the pinned CPU runtime and prompt-cache execution can vary numerically. These differences cannot be credited to situation retrieval. Two independent test events and one repetition support only a descriptive pilot. A future retrieval improvement needs a new version and a fresh independent test set; this exposed pilot must not be reused as an untouched holdout.

## Delivery boundaries

1. Original-Q&A corpus, situation retrieval, controlled reference experiment and review material.
2. Actual Qwen3-TTS → MuseTalk prepared media, presentation UI, live timing and cancellation correctness.
3. Hash-bound situation observations, acoustic-boundary validation, review workflow and measured coverage.

Each milestone is a reviewable dependent PR. The historical grounding reports, 120 archived prompt fingerprints, voice samples and motion catalog retain their original meanings. Text quality, speech naturalness, identity likeness, motion attribution and engineering execution are reported separately.

The source proposal's suggestion of a persona graph, fine-tuning, or learned gesture policy remains a research extension requiring its own data and evaluation. The implemented transparent retrieval baseline and reusable observation contracts make those extensions possible without inventing unobserved behavior.
