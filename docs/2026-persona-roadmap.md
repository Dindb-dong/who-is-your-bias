# Elon Musk 2026: persona, speech, and avatar roadmap

Status: **implementation plan; not a completed capability claim**. Recorded on **2026-11-11, Asia/Seoul**. The user selected Musk as represented in 2026 and requested this documentation update without implementation changes or a new PR.

**Implementation follow-up (2026-11-11):** the user subsequently authorized implementation and merging the existing PRs. PRs #10–#13 are now merged. See [the PR #13 implementation record](2026-implementation.md) and [continued implementation and actual execution evidence](2026-continuation.md). The plan and authoring-time status below remain a dated record.

**New-avatar direction (2026-11-12):** for the Jensen Huang visual-avatar
workstream, the user now requests a newly generated avatar from multiple photos
and video. Its active design is the [avatar generation pipeline](avatar-generation-pipeline.md),
including full upper-body/finger control and subject binding to the existing
application. The acquisition-first priority below records the earlier Musk plan;
it does not select ready-made candidates for the new workstream.

The goal is a convincing, source-grounded conversation with a recognizable Musk avatar: relevant answers, characteristic language and speech, coherent dialogue, and natural audiovisual behavior. The existing engineering pilot provides reusable infrastructure, but its small original-Q&A dataset, short voice reference, and generic face do not establish that experience.

Read the [3D avatar shortlist](research/musk-3d-avatar-shortlist.md) alongside this plan. Existing Musk-specific assets advertised with facial rigs make **3D asset evaluation a priority**. Image/video-based rendering remains a comparison path; it is not a predetermined replacement for 3D.

## 1. Evidence and branch status

At authoring, `main` points to [`5559eda`](https://github.com/Dindb-dong/who-is-your-bias/commit/5559edad3fa460638930303ee212bbbdb706d46b). The implementation described in the latest pilot is in three **open, dependent PRs**, not merged into `main`:

| Work | Immutable head | Scope |
| --- | --- | --- |
| [PR #10](https://github.com/Dindb-dong/who-is-your-bias/pull/10) | `bf2c60ae14d18d4ab814e046c687423177b2e434` | Original Q&A, situation retrieval, reference-answer experiment |
| [PR #11](https://github.com/Dindb-dong/who-is-your-bias/pull/11) | `a8a72cf8be258ef2ee44d8da3de7cf37db446cca` | Actual prepared speech/video, presentation UI, live timing |
| [PR #12](https://github.com/Dindb-dong/who-is-your-bias/pull/12) | `43af562608631f25623c2593c84eeed417153fd2` | Situation-linked observations and review workflow |

These statuses are a dated snapshot. This documentation commit does not merge those PRs. Links to their new files below use immutable GitHub URLs so they also work before the code reaches `main`.

### What already exists

| Observed baseline | Consequence for the next implementation |
| --- | --- |
| The earlier summary corpus contains 163 evidence records from 31 events/documents | Reuse its source inventory, but do not count paraphrases as original speaking-style examples |
| Original-Q&A v2 has 92 exchanges: 80 train, four development, eight test | Only the 80 train exchanges enter retrieval; they come from just TED2013 and TED2017 |
| The bundled original-Q&A preview has ten excerpts, five in training | A successful preview does not imply the full local corpus is active |
| The frozen study completed 24 actual model calls; B/C retrieved no examples for all eight questions | Paired A/B/C inputs were identical; the study cannot establish a situation-retrieval benefit |
| The style profile contains editorial instructions; the prompt limits answers to 1–3 short sentences | No Musk-specific text-weight training or measured style model has been delivered |
| Speech uses a 9.830042-second reference candidate | This is reference conditioning, not training across Musk's situations or a validated voice likeness |
| The 12.24-second prepared clip uses actual Qwen3-TTS and MuseTalk with an artistic proxy portrait | Neural execution is real, but the input face, fixed crop and geometric blend limit realism |
| The browser humanoid is a generic authored asset with 52 face channels and 15 visemes | Its rig is reusable infrastructure; its face is not a verified Musk likeness |
| Two actual CPU live runs received complete WAVs after 128.284 and 109.324 seconds | The path works, but these are not responsive voice latencies or target-device acoustic measurements |
| Four original exchanges have linked observation measurements; accepted human observations remain zero | Source-window measurements are not a learned, validated personal movement policy |

Evidence: [historical corpus inventory](research/corpus-status.md), [original QA](https://github.com/Dindb-dong/who-is-your-bias/blob/43af562608631f25623c2593c84eeed417153fd2/docs/original-qa.md), [frozen experiment](https://github.com/Dindb-dong/who-is-your-bias/blob/43af562608631f25623c2593c84eeed417153fd2/docs/reference-evaluation.md), [style profile](https://github.com/Dindb-dong/who-is-your-bias/blob/43af562608631f25623c2593c84eeed417153fd2/data/qna/elon-musk-v1/profile.json), [prompt](https://github.com/Dindb-dong/who-is-your-bias/blob/43af562608631f25623c2593c84eeed417153fd2/src/bias/qa_prompts.py), [voice reference](research/voice-reference.md), [humanoid](humanoid-avatar.md), [prepared video](https://github.com/Dindb-dong/who-is-your-bias/blob/43af562608631f25623c2593c84eeed417153fd2/docs/presentation.md), [live measurements](https://github.com/Dindb-dong/who-is-your-bias/blob/43af562608631f25623c2593c84eeed417153fd2/docs/live-demo-latency.md), and [observation audit](https://github.com/Dindb-dong/who-is-your-bias/blob/43af562608631f25623c2593c84eeed417153fd2/docs/research/observation-pilot-status.md).

The original [YAICON proposal](https://docs.google.com/document/d/1eHnjPdZbQgCUOTIuixlIQc2oIDqyyHjUvhCFUc30-cw) and [dated proposal audit](proposal-alignment.md) remain useful context. This plan advances the product target and research scope without rewriting historical results.

## 2. Define the 2026 persona explicitly

The initial snapshot is **Musk as of 2026-11-11 in Asia/Seoul**, using only material available and acquired by the recorded build cutoff. It does not include unseen events later in November or December 2026. A build must record an exact UTC cutoff; the date here is the planning boundary, not a claim that all material published that day was collected.

Maintain separate versioned identities for knowledge/positions, language style, voice, and visual appearance. Updating the text corpus does not update an old facial mesh or make a 2013 voice reference representative of 2026.

### Proposed temporal contract

| Field | Meaning |
| --- | --- |
| `event_at` | When the recorded conversation or event occurred; unknown remains null |
| `published_at` | Publisher release time, distinct from the event time |
| `ingested_at` | When this project acquired the artifact |
| `as_of` | Exact cutoff used by this persona build |
| `assertion_period` | Time period explicitly described by the statement, if known |
| `supersedes` / `contradicts` | Reviewed links between statements; do not invent these from recency alone |
| `style_period` / `appearance_reference_period` | Dates represented by the selected behavioral and visual references |

These fields are proposed additions, not current API guarantees.

Current-position answers should prefer recent, relevant firsthand evidence and preserve qualifications. Historical answers should retain their original dates. When an older opinion has no documented update, say what was said then; do not infer a current opinion. General contemporary facts and Musk's expressed positions are separate evidence types.

Older recordings can contribute history, pronunciation, dialogue structure and candidate mannerisms. Include recent recordings in validation, and measure whether older behavior transfers to the 2026 target. Do not assume that one voice sample, expression or attitude is permanent.

A snapshot refresh should acquire new material, review date/attribution changes, rebuild indexes, rerun temporal regressions and publish a new manifest. Keep the prior snapshot reproducible. A live web result must not silently become a reviewed personal belief.

## 3. Expand the corpus across independent situations

### Initial acquisition targets

The following are planning targets, **not collected quantities or guarantees of model quality**:

- 20–30 independent events with useful original dialogue, spanning interviewers, years and settings.
- 1,000–3,000 validated dialogue turns or exchanges, including multi-turn context.
- 2–5 hours of clean, target-speaker speech aligned to accurate transcripts, drawn from several events.
- A smaller shot-continuous visual subset with usable face/eye/head observations and explicitly visible hands.

Reconcile the existing 31-source summary inventory and local transcript recipes first. The target counts events that yield validated original dialogue; it does not mean the repository has fewer than 20 source URLs. Report raw recording hours, target-speaker hours, accepted speech hours, original turns and indexed units separately.

### Acquisition starting points

| Primary source | Useful contribution | Important handling |
| --- | --- | --- |
| [Lex interview 1](https://lexfridman.com/elon-musk/), [interview 2 with transcript PDF](https://lexfridman.com/elon-musk-2/), [#252](https://lexfridman.com/elon-musk-3/) | Long-form dialogue, technical explanations and varied questions | Use original recordings/transcripts; do not ingest advertisements as target speech |
| [TED2022 live interview](https://www.ted.com/talks/elon_musk_elon_musk_talks_twitter_tesla_and_how_his_brain_works_live_at_ted2022) | A distinct event from the recorded Gigafactory interview | Preserve separate event IDs and verify the actual media clock |
| [Tesla Investor Relations archive](https://ir.tesla.com/) | Multi-year technical/business Q&A and follow-ups | Verify Musk's participation and separate other executives and questioners |
| [Everyday Astronaut Starbase interview](https://everydayastronaut.com/starbase-tour-and-interview-with-elon-musk/) | On-site explanation and spontaneous questions | The recording contains direct speech; the accompanying article is a writer's summary |
| [SpaceX content hub](https://www.spacex.com/content) | Presentations and technical updates | Separate rehearsed monologues from spontaneous conversation |
| [2006 Master Plan](https://www.tesla.com/secret-master-plan), [2016 Part Deux](https://www.tesla.com/master-plan-part-deux) | Explicitly attributed written statements | Keep written style and dated positions separate from spoken dialogue |

This list seeds acquisition; it is not a complete 2026 source inventory. Explicitly add and verify recent material before claiming contemporary coverage.

Books about Musk are useful for background but are not automatically his writing. For example, the publisher identifies [*Elon Musk* as Walter Isaacson's book](https://www.simonandschuster.com/books/Elon-Musk/Walter-Isaacson/9781982181284). Separate author narration, reported direct quotations and editorial interpretation. Audiobook narration by another person is not Musk voice data. Byline and actual speaker matter more than the title.

### Acquisition and preparation workflow

1. Register the publisher URL, event family, date, media availability and usage status before extraction.
2. Deduplicate original recordings, excerpts, reposts and transcript mirrors. Assign development/test event families before tuning.
3. Acquire through the existing publisher/media adapters where applicable; retain original hashes and native clock identity.
4. Preserve exact original text separately from normalized text, translations, summaries and generated annotations.
5. Identify speaker turns from explicit labels and reviewed alignment. Detect overlap, music, cuts and uncertain attribution.
6. Align text/audio and, where usable, continuous visual tracks. A caption timestamp alone is not an acoustic boundary.
7. Retain interviewer context, interruptions, corrections and meaningful pauses. Do not force every useful turn into an isolated QA.
8. Produce coverage and rejection reports; sample-check extraction against the original media before index/training export.

Use one shared `event_id`/`segment_id` lineage with four projections: factual/position evidence, text-style examples, aligned speech, and visible behavior. Each projection has its own eligibility and review state. A good transcript does not automatically qualify the associated video or audio.

Required segment metadata includes source URL/hash, recording family, original text, speaker-attribution basis, context, completeness, date, language, source clock, nullable acoustic/visual intervals, transformation history, split and reviewer provenance. Unknown and unobservable remain distinct from a measured zero.

Track coverage across topic, situation, year, interlocutor, format and recording quality. Cap dominance by one long interview. Include explanation, reflection, challenge, humor, advice and ordinary conversational responses; do not manufacture stereotyped behavior for missing categories.

## 4. Replace the narrow retrieval baseline

The current original-Q&A retriever concatenates answer text, repeated questions and tags in a BM25 index, with manual Korean/English aliases. Conservative lexical gates precede the C situation multiplier. A candidate that fails the gate cannot be rescued by situation reranking. See the [pinned implementation](https://github.com/Dindb-dong/who-is-your-bias/blob/43af562608631f25623c2593c84eeed417153fd2/src/bias/retrieval.py).

### Proposed retrieval flow

1. Resolve the current question against conversation entities and prior claims. Preserve the original user text alongside any rewritten search query.
2. Retrieve in parallel with lexical matching and multilingual embeddings.
3. Query both question/context representations and factual answer passages. Retain complete source turns for reconstruction.
4. Fuse candidate lists, then rerank for actual relevance and temporal compatibility.
5. Apply situation suitability only among candidates with adequate topical support.
6. Assemble factual evidence and style examples as separate prompt sections, under one tokenizer-aware context budget.
7. Record retrieved IDs, scores, filter reasons, final supplied passages and corpus/model versions.

[Sentence Transformers' retrieve-and-rerank guide](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) is a reference implementation pattern. [Qwen3 Embedding/Reranker](https://github.com/QwenLM/Qwen3-Embedding) is one multilingual candidate to compare; its upstream benchmark does not establish performance on this corpus.

Do not fix zero coverage by forcing a top result or indiscriminately lowering thresholds. Label whether an answer is genuinely present in the corpus. Measure missed relevant evidence separately from absent evidence. A style example may inform phrasing without supporting the current factual claim or receiving its factual citation.

For long turns, use retrievable subspans linked to their complete parent exchange; preserve qualifications and surrounding context. Never silently truncate a passage into a stronger claim. Deduplicate fused results so repeated excerpts do not consume the context window.

The model context budget must include system/profile tokens, dialogue state, evidence, style examples, user text and reserved output tokens. Character counts alone do not guarantee that the selected model's context limit is respected.

### Controlled experiments

Keep A/B/C on the same model, corpus snapshot, common instructions, output budget and request policy. A receives the common profile; B receives topical evidence; C uses the declared situation-aware selection. Do not add voice, animation, different models or unrelated style instructions to only one condition.

The eight already-exposed pilot questions remain reproducibility/development material. New efficacy claims require fresh independent test events. Preserve the failed pilot unchanged. A valid new result can be negative; however, a run in which every treatment again has identical empty evidence does not test the intended retrieval contrast.

## 5. Learn text style and speech as separate experiments

### Text adaptation

Extract source-attributed examples of answer structure, qualification, self-correction, rhetorical choices, response length and reactions to different questions. Preserve variation rather than imposing universal brevity or adding fillers to every answer.

Compare at least:

| Condition | What changes |
| --- | --- |
| Base + factual retrieval | Strong grounded language baseline |
| Base + identical factual retrieval + style examples | Tests contextual style conditioning |
| Same base + text adapter + identical factual retrieval/style policy | Tests the additional effect of SFT/LoRA |

These are adaptation experiments, separate from the original A/B/C retrieval comparison. Freeze the data splits and supplied factual evidence within each comparison. Compare a larger capable base model on the same cases to determine whether the current 4B model is a bottleneck before spending heavily on adaptation.

Train on real attributed dialogue with the original question and useful preceding context. Keep generated augmentation separately labeled and out of the human-reference test set. Record checkpoint hash, tokenizer, base revision, adapter configuration, precision, seed, optimization settings, hardware, elapsed time and failures.

Keeping updateable facts in retrieval is a design goal, not a guarantee that fine-tuning cannot memorize them. Test stale-claim leakage, unsupported first-person claims, quotation memorization, repetitive mannerisms and general answer quality after adaptation.

### Voice adaptation

First compare several clean references using identical evaluation sentences, then compare the best reference-conditioned baseline with actual speaker adaptation. Preserve accurate transcripts, natural pauses and emphasis while excluding overlapping speakers and unusable noise. Measure timbre and prosody separately.

The [Qwen3-TTS fine-tuning guide](https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/README.md) documents single-speaker SFT for Base models using aligned audio, text and reference audio. Its recommendation for a consistent reference should be retained unless a different design is explicitly tested. The [model feature table](https://github.com/QwenLM/Qwen3-TTS) distinguishes Base cloning from instruction-controlled variants; do not assume an arbitrary prosody instruction works with the currently selected Base model.

Situation-dependent rhythm may require conditional data, reference selection or additional modeling. Treat it as an evaluated capability, not an automatic consequence of voice cloning. Avoid changing the speaker identity or channel sound whenever the conversational situation changes.

Keep English source imitation and Korean output as separate conditions. Korean is a cross-language adaptation target; it does not establish directly observed Korean speaking habits of Musk. ASR intelligibility, voice similarity, expressive timing and listening preference each need their own report.

## 6. Evaluate real Musk 3D assets before choosing the visual route

The [asset research](research/musk-3d-avatar-shortlist.md) identifies five paid facial-rig candidates, three free-download listings and two detailed creator portraits. These are inspected listings, not validated model files. No purchase or model download is part of this documentation update.

Start with the Chatiz and star-game-3d-media candidates because their descriptions explicitly include facial controls and editable formats. Compare the Meta Avatars head package and Lucid Dreams asset if the first candidates fail. The free candidates are useful alternatives but do not yet demonstrate a complete controllable speech face.

Asset qualification must cover 2026 appearance, neutral geometry, skin/eye/hair materials, mouth interior, facial deformation, gaze, blinks, jaw/visemes, rig retargeting, GLB export and actual browser performance. The existing 52 face channels and 15 visemes are a mapping target, not a reason to assume every vendor uses those names. Keep unsupported channels explicit.

For 2026 appearance, prepare dated front, three-quarter and profile references from the chosen period. Record the evidence for any remodeling. An older mesh can be an adaptation starting point; the listing title alone does not verify its age or era.

Use the same speech and lighting/framing conditions for candidate comparisons. Check identity during articulation, not just a neutral beauty render. Isolate jaw-bone and morph contributions so the mouth is not opened twice. Compare short and longer utterances, silence transitions, gaze changes and interruptions.

Keep a real-photo/video MuseTalk path as a controlled visual comparison. Restore appropriate tracking, face parsing and temporally stable blending before attributing all quality limits to the neural model. [MuseTalk](https://github.com/TMElyralab/MuseTalk) documents identity/resolution limits. [LivePortrait](https://github.com/KlingAIResearch/LivePortrait) provides portrait motion transfer, but does not by itself learn appropriate personal reactions to new questions.

Natural behavior requires listening, planning, speaking and interrupted states. Source motion can seed subtle, reviewed movement ranges; a fixed talking loop or random gesture selection does not establish situation-dependent behavior. Preserve the application's simulation label while improving appearance.

## 7. Multi-turn persona and streaming interaction

### Conversation state

Maintain current entities, topic, prior factual claims, open questions, user corrections and the active temporal snapshot. Resolve references such as “that approach” and “what you said earlier” before retrieval. Keep user statements distinguishable from source-supported facts. Store compact state with supporting turn IDs so summaries can be inspected and corrected.

Evaluate follow-up questions, disagreement, changed premises, topic switches, unknown current views and requests to compare historical positions. Preserve session isolation, bounded history and expiration. State persistence beyond the existing session lifecycle is a separately specified feature.

### Proposed streaming contract

| Record | Minimum information |
| --- | --- |
| Turn | Session/turn ID, persona snapshot, state, cancellation owner, monotonic start |
| Committed text segment | Stable sequence, exact text, evidence references, language, completion/failure state |
| Audio chunk | Turn/segment IDs, sequence, sample rate, channels, sample offset, terminal/error state |
| Animation segment | Target rig/version, audio sample interval, timing basis, supported channels |
| Timing report | First visible text, first digital PCM, underruns, interruption response, completion and unmeasured fields |

This contract is proposed. New endpoint names and executable commands must be documented when implemented rather than presented here as existing APIs.

Begin TTS after a linguistically usable text segment is committed, while later language generation continues. Do not speak a partial number, abbreviation or claim that is about to be revised. Use bounded queues and backpressure. Synchronize the avatar to actual audio playback, compensate for buffering and avoid independent timers drifting apart.

Cancellation must invalidate pending text/audio/animation for the same turn, stop playback promptly, and preserve partial/failure status. A completed network response does not establish a completed spoken turn. If microphone interaction is added, specify voice activity, echo handling and false-interruption behavior as well.

[vLLM-Omni's TTS examples](https://docs.vllm.ai/projects/vllm-omni/en/stable/user_guide/examples/online_serving/text_to_speech/) and [speech API](https://docs.vllm.ai/projects/vllm-omni/en/stable/serving/speech_api/) document Qwen3-TTS PCM/WebSocket paths. Buffered input can still wait for input completion. Its documented commitment policy has a Chinese/English scope; Korean segmentation needs separate validation. Server streaming support alone does not finish application-level early playback.

Use a suitable accelerator and benchmark the actual target machine. Keep model/reference features warm where supported. Measure the simultaneous LLM/TTS/avatar workload, not isolated peak throughput. Apple Silicon support, CUDA support and a vendor GPU benchmark are different facts; no specific machine has yet been shown to meet this plan's interaction target.

## 8. Evaluation and acceptance

Register evaluation event families before tuning and retain raw outputs, model/data hashes, supplied evidence and failures. A test reference answer must never enter retrieval, style training or fine-tuning. Deduplicate quoted/reposted material across splits; source URL differences are insufficient.

### Proposed initial test design

- At least 100 retrieval cases with manually checked supporting passages, plus at least 50 genuinely absent-evidence cases; include paraphrases and Korean/English equivalents.
- At least 20 multi-turn scenarios covering follow-ups, corrections, temporal changes, topic shifts and interruptions.
- At least 30 audiovisual samples spanning explanation, brief answers, uncertainty and challenge, with several independent human reviewers and blinded presentation order.
- Separate historical-fidelity and current-2026 tests; include recent independent events in the latter.

These are starting design targets. Record actual event counts and clustering; many questions from one interview are not many independent observations. Choose thresholds on development data, freeze them before testing and report uncertainty, disagreements and exclusions. Do not require C to win in order to publish a valid result.

| Dimension | Measure | Release evidence |
| --- | --- | --- |
| Retrieval | Recall/ranking on answerable cases, false support on absent cases | Compare lexical and hybrid variants, with identical corpus and explicit abstention reasons |
| Grounding | Source support and temporal accuracy per factual claim | Distinguish citation formatting from entailment; report unsupported current/personal claims |
| Text style | Wording, structure, response length and situation response | Blinded comparisons against actual source responses; keep model-assisted and human judgments separate |
| Speech | Intelligibility, timbre, prosody and listening preference | Original references and generated outputs; ASR is only one diagnostic |
| Avatar | 2026 likeness, facial deformation, lip-sync and behavior | Actual imported asset, dynamic examples and perceptual comparison with the previous proxy |
| Interaction | First audible latency, underruns, cancellation and audiovisual drift | Repeated runs on the declared machine with all services active |
| Reproducibility | Data/model/asset versions, commands and artifacts | Another developer can reproduce a run or identify the exact unavailable dependency |

Do not invent universal latency or perceptual-score thresholds from the current CPU pilot. Establish a measured baseline on the intended device, then publish explicit product budgets before optimization comparisons. Report digital playback separately from microphone/loudspeaker measurements.

## 9. Work packages and dependencies

| Package | Deliverables | Exit condition |
| --- | --- | --- |
| WP0: snapshot and evaluation registration | 2026 manifest, source-family inventory, fresh split, evaluation rubrics | Old pilot preserved; new holdouts isolated; known and proposed capabilities distinguished |
| WP1: aligned corpus and hybrid retrieval | Acquisition/review export, four data projections, coverage report, lexical+dense retrieval and reranker | Relevant evidence reaches the model on answerable cases; absent cases remain distinguishable |
| WP2: 3D asset audition, parallel with WP1 | Qualified asset reports, lawful acquisition path, mapping, GLB conversion and same-audio comparisons | At least one candidate passes actual deformation/material/browser checks and improves the assessed experience |
| WP3: text and voice adaptation | Reference baseline, adapter training, checkpoint records and controlled comparisons | Measured improvement without hidden grounding, identity or general-quality regression |
| WP4: temporal dialogue and streaming | Multi-turn state, committed segments, PCM playback, synchronized animation and interruption | Scenario suite and target-machine interaction budgets pass |
| WP5: demonstration and research release | Prepared presentation, live rehearsal, human evaluation and reproducible report | Public claims match actual evidence, with remaining limitations and failures retained |

WP0 starts first. WP1 and WP2 can proceed in parallel. WP3 depends on accepted data; WP4 can start with baseline models and integrate successful adapters later. Evaluation and artifact recording run throughout, rather than waiting for WP5.

The repository already documents runnable setup, ingestion, reference preparation and existing evaluations in [README](../README.md), [collection](research/collection.md), [transcript acquisition](research/transcript-acquisition.md) and the pinned pilot links above. Future packages must add their own actual CLI/API documentation, configuration defaults, failure handling and reproduction examples. Do not claim a proposed training or acquisition command works before it exists.

## 10. Decisions still requiring implementation evidence

- Which 3D candidate preserves likeness under speech and fits the intended distribution model?
- How much recent material is needed to represent 2026 language/voice variation, and what can transfer from older recordings?
- Does contextual style retrieval already suffice, or does a text adapter add useful fidelity?
- Does speaker adaptation improve prosody as well as timbre without reducing intelligibility?
- Is a larger language model necessary after data and retrieval are corrected?
- Which hardware/runtime configuration meets conversational latency while all components run together?

The next implementation priority is **broader aligned data plus hybrid retrieval, in parallel with an actual Musk 3D asset audition**. The completion criterion is a measured improvement in the user's conversation experience, supported by reproducible evidence.

