# Who Is Your Bias?

**A research studio for evidence-grounded conversation, voice, and embodied behavior.**

Active subject: **Jensen Huang**. The project connects documented statements to a conversational model and a synchronized avatar. Generated interactions are explicitly presented as simulations. Historical statements keep their source and date; the system does not claim to reconstruct private thoughts. Earlier Elon Musk corpora, presentations, motion clips and evaluation results remain explicitly historical research assets.

## Run the complete studio

Requires Node 22.12+ (Node 24 recommended), Python 3.11+, and uv.

```bash
uv sync --extra dev
npm ci
npm run build
uv run bias serve
```

Open `http://127.0.0.1:8000`. Jensen's stylized portrait is the default. In **Attention**, try the pointer preview, or explicitly start Camera / Share screen and select a numbered participant. The eyes turn first; the head follows after 220 ms. Selection uses face positions and does not infer gender or identity. See the [Jensen attention guide](docs/jensen-attention.md) for controls, privacy, setup and limitations.

Build and development startup provision a pinned local face detector. Initial setup needs network access to download its verified model; camera frames and inference remain in the browser. The studio also includes source inspection, A/B/C comparison, exportable turn timing, optional voice playback, a generic human rig, local GLB loading and historical presentation/motion labs. See [studio guide](docs/studio.md) for audio and retargeting details.

The Jensen corpus contains 32 editorial evidence records from nine dated sources, including 24 reviewed interview exchanges. When the model is connected and retrieval finds no evidence, the app labels the generated answer as a hypothetical simulation. See the [Jensen corpus](data/personas/jensen-huang/README.md) for provenance and the snapshot cutoff.

The historical [2026 implementation report](docs/2026-implementation.md) records a reproducible 977-QA original-data build, optional multilingual hybrid retrieval, explicit temporal snapshots, follow-up resolution, early PCM speech and same-audio avatar audition for the earlier Musk workstream. Its measurements and remaining limitations do not transfer to Jensen.

## Run the conversation engine

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run bias validate
uv run bias serve
```

The API runs at `http://127.0.0.1:8000`; API documentation is at `/docs`. With no model configured, Jensen responses contain labeled evidence summaries from the dated seed. This mode exercises retrieval, sessions, evidence links, and behavior planning without claiming language-model inference. Optional speech must explicitly bind `BIAS_TTS_PERSONA_ID` to `jensen-huang` or `generic`; foreign or unbound voices and historical Musk motion are unavailable in the Jensen studio.

### Connect an actual model

```bash
cp .env.example .env
# Set BIAS_LLM_BASE_URL, BIAS_LLM_MODEL, and a key if your provider requires one.
uv run --env-file .env bias serve
```

The adapter supports the OpenAI-compatible `/chat/completions` streaming protocol. A local model server can be used without a paid API. For example, an existing Ollama installation exposes this protocol at `http://127.0.0.1:11434/v1`; set the model name to one you have actually installed. Model quality and hardware needs depend on that choice. Credentials stay on the server.

The provider is called only when configured. Errors and missing capabilities remain visible. This repository does not bundle a language-model checkpoint, a trained voice, or a photorealistic head asset.

The optional [local language server](docs/language-runtime.md) runs a server-selected Transformers checkpoint behind that protocol. Its model loads lazily, generation is bounded, and overlapping requests are rejected while the worker is occupied. The studio's A/B/C comparison runs sequentially to support this CPU-oriented configuration. [Generation evaluation](docs/evaluation.md) records actual outputs, timing, provider token usage when available, citation-format checks, and separately ungraded review criteria.

## What is implemented

| Capability | Behavior |
| --- | --- |
| Active Jensen studio | Profile, topics, source library and snapshot from Jensen's own dated seed; explicit speech/motion subject binding |
| Participant attention | Local camera/screen face boxes, explicit selection, eyes-first then head response, pointer/keyboard preview and reduced-motion support |
| Source corpus | Original question/answer pairs, distinct context/annotations, pinned source events, explicit unknown acoustic boundaries and review provenance |
| Retrieval | Preserved lexical baseline; optional pinned multilingual embeddings, fusion and cross-encoder with independent factual/style gates and recorded development tradeoffs |
| Dialogue | OpenAI-compatible streaming adapter; profile, evidence, and behavior modes; explicit corpus-bound historical/current snapshot policy |
| Conversation state | Opaque sessions, bounded user-reference anchors, corrections, TTL, serialized text/audio turns, absolute deadlines and cancellation cleanup |
| Evidence | Per-unit citation coverage, separately labeled semantic review, original source links, frozen prompt comparisons |
| Behavior plans | Timed face/gaze/head/gesture cues with explicit provenance and estimated timing |
| Speech and transcription | Optional compatible providers; committed-sentence synthesis and bounded PCM delivery while later text continues; sample-clock playback and interruption |
| Historical prepared presentation | Earlier Musk-workstream 12.24-second Qwen3-TTS → MuseTalk video, captions, generated WAV, verified media hashes and fresh live-session handoff |
| Live timing | Browser text/PCM events and JSON export; separate actual-service HTTP benchmark; acoustic speaker output remains unmeasured |
| Local reference voice | Optional Qwen3-TTS service with cached prompts, bounded requests, and a fixed server-side reference |
| Speech timing | Optional Rhubarb mouth-cue recognition, SHA-256-bound audio packages, actual playback clock, explicit amplitude fallback |
| Avatar calibration | Bundled compressed human mesh with 163 bones and 67 named face/viseme targets; calibrated controls, portrait/full-body cameras, and local GLB support |
| Historical measured motion | Earlier Musk-workstream 13 source-linked clips and 40 primitives from two events; source-clock replay and explicit review gates; not bound to Jensen |
| Situation observations | Four original Q&As linked to 19 speech-rate windows and ten visual windows; source-clock measurements, explicit unknowns and an empty reviewed runtime index |
| Reference preparation | Publisher-text/ASR comparison, bounded candidate WAVs, clock metadata, artifact hashes, and a review ledger |
| Media processing | FFmpeg clips, source-clock manifests, optional ASR/alignment and facial/body observation adapters |
| Ingestion | Local SRT/VTT/WhisperX imports, speaker selection, hashes, source-clock offsets |
| Evaluation | Reproducible retrieval checks, separate generation cases, actual A/B/C execution, and source-disjoint answerability fixtures |

## Historical corpus and source acquisition

The following counts describe the preserved `data/personas/elon-musk` summary archive, not the active Jensen seed or the original-Q&A v2 experiment.

The initial 7-source development collection has expanded to **31 distinct source events/documents and 163 reviewed evidence records**, spanning **2003–2026**. Retrieval uses **159 records from 30 sources**; the four reference records from Lex #438 remain held out. Evidence is concise, attributed paraphrase, with original URLs and dates.

The collection includes TED interviews, Lex Fridman conversations, Dwarkesh, Y Combinator, Recode, WIRED, TIME, GQ, Stanford, MIT, congressional testimony, and authored Tesla documents. Coverage now includes learning, fear, criticism, uncertainty, humor, motivation, and work habits alongside engineering and business plans. See [integrated corpus status](docs/research/corpus-status.md) for exact contributions, dates, acquisition links, and limitations.

The publisher transcript pipeline also extracted **3,451 timestamped caption cues** from four complete TED interviews: **3:00:45 of declared program duration and 27,411 whitespace-delimited words**. Of those cues, **1,822 carry Musk labels in the publisher text**, 1,125 carry interviewer labels, and 504 remain unassigned. Questions and adjacent speaking turns are preserved. The full local transcript imports remain **draft, unindexed** material. Separately reviewed original-Q&A selections from the pinned transcripts enter the new explicit corpus; a text label is not independently verified voice identity. The repository contains reproducible extraction code and measured coverage, while complete originals remain outside Git.

This is a research corpus and working application, not a validated replica of a person's mind, voice, or appearance. The [initial inference report](docs/inference-validation.md) and [integration results](docs/integration-validation.md) separate real model execution from citation compliance and perceptual quality. The included human avatar is an artistic reference asset with a calibrated face/body rig. A high-quality likeness asset, learned personal mannerisms across many situations, and independent perceptual evaluation remain separate work.

## Historical original interview corpus

The earlier Musk default is a ten-record bounded excerpt pilot, now selected explicitly with `BIAS_CORPUS_DIR=data/qna/elon-musk-v1` and `BIAS_PERSONA_SNAPSHOT=data/snapshots/elon-musk-2026-11-11.json`. The full original-answer study uses 92 locally materialized exchanges (80 train / four development / eight test); source text is pinned and reconstructed locally:

```bash
uv run python -m tools.qna hydrate \
  --selection data/qna/elon-musk-v1/selection-recipes.v2.json \
  --output data/local/qna/elon-musk-v2
BIAS_CORPUS_DIR=data/local/qna/elon-musk-v2 uv run bias serve
```

See [original Q&A](docs/original-qa.md) for source review, source-event isolation, unknown acoustic timestamps and bounded-versus-full materialization. Old summary-corpus results retain their original version and explicit corpus path.

## Research workflow

```bash
uv run bias search "What are AI factories?"
uv run bias search "AI 팩토리를 어떻게 설명했나요?"
uv run bias --corpus data/personas/elon-musk evaluate --fail-on-regression --output reports/local/retrieval.json
uv run python -m tools.research inventory --output reports/local/source-inventory.json
uv run pytest
```

To import a transcript you have obtained:

```bash
uv run bias ingest data/raw/interview.vtt \
  --source-id interview-example \
  --timeline source_video \
  --output data/local/interview-draft.jsonl
```

Imports are **draft, unindexed** records. Review the target speaker, segmentation, dates, source timebase, and attribution before integrating them. Raw audio, video, local transcripts, environment files, and model weights are excluded from Git.

### Interpret results correctly

- Original Q&A and a shared situation taxonomy are implemented. The first held-out pilot completed 24 real A/B/C responses but retrieved no examples for its eight questions, so it does not establish a C advantage. See [reference-answer evaluation](docs/reference-evaluation.md) and [proposal status](docs/proposal-alignment.md). Human judgments remain separate and ungraded.
- A valid citation ID means the response referenced an allowed record; it does not prove that every generated sentence follows from that record.
- Retrieval smoke tests are not an independent persona-fidelity benchmark. Cases derived while authoring the corpus are labeled development fixtures.
- An entire held-out source is excluded from runtime retrieval, including its records and reference answers.
- Conversation behavior cues remain authored demonstration rules. Measured video clips can be replayed in the motion lab; they are not automatically promoted into learned personality or gesture policies. Audio-recognized mouth cues remain predictions, with their timing basis shown explicitly.
- The [situation-observation pilot](docs/situation-observations.md) measured four Q&As from two events: 19 matched speech-rate windows and ten visual windows with 512 sampled frames, eight excluded frames and two smile candidates. Human listening, verified pauses and semantic gesture judgments remain uncompleted; its runtime index is empty. The preserved report remains bound to Q&A v1, although the four linked exchange digests also match v2.
- No-key source previews do not translate English evidence into fabricated Korean quotations. A connected model can answer in Korean with English-source attribution.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Configuration and corpus status; does not assert remote-provider availability |
| `GET /api/persona` | Profile, examples, corpus counts, capabilities |
| `GET /api/sources` | Source inventory and train/test designation |
| `GET /api/motion/library` | Optional measured-motion availability, review counts, and provenance |
| `GET /api/motion/clip/{id}` | Hash-bound eligible source window; unreviewed geometry requires explicit preview |
| `POST /api/chat` | Complete response with citations and behavioral timeline |
| `POST /api/chat/stream` | SSE `meta`, `status`, `token`, `result`, `done` or `error` |
| `POST /api/chat/live` | Committed text segments and validated PCM streaming; requires configured speech |
| `GET /api/sessions/{id}/context` | Inspect bounded user references, citations and temporal provenance |
| `POST /api/speech` | Speech-provider audio, when configured |
| `POST /api/speech/prepare` | Generate audio and optional mouth cues bound to its SHA-256 |
| `GET /api/speech/clips/{id}` | Retrieve an ephemeral prepared clip before expiry |
| `POST /api/transcribe` | Raw-audio transcription, when configured |
| `DELETE /api/sessions/{id}` | Delete an idle conversation session |

## Documentation

- [Agent operating rules](AGENTS.md): explicit instruction loading, durable checkpoints, remote verification, and resumable handoffs.
- [Jensen interrupted-run handoff](docs/jensen-replication-handoff.md): verified PR #17 baseline, unverified post-PR #17 progress, known failures, and remaining persona, voice, avatar, and evaluation milestones.
- [Jensen original execution plan](docs/jensen-replication-run.md): the historical PR #18 brief, now marked interrupted and linked to the recovery plan.
- [Jensen attention guide](docs/jensen-attention.md): explicit participant selection, local capture, eyes-first timing, reduced motion, setup and known limits.
- [PR #16 implementation plan](docs/jensen-attention-plan.md): committed before implementation, with implementation and validation evidence.
- [New avatar generation pipeline](docs/avatar-generation-pipeline.md): the active plan for creating a new Jensen Huang avatar from multiple photographs and video, including complete arms/hands, facial and finger rigs, new performances, runtime work and acceptance gates.
- [Avatar generation method evidence](docs/research/avatar-generation-methods.md): mesh authoring, expressive video and multi-image reconstruction, output-format boundaries, hardware and separate code/model licenses.
- [2026 continuation and execution evidence](docs/2026-continuation.md): merged PR #13, verified adapter serving, real training/comparison boundaries, dialogue regressions, and remaining acceptance work.
- [Congressional source extension](docs/congressional-data.md): six additional official hearing events, source-bound parsing, combined corpus replay, and actual API availability limits.
- [Verified text adapter runtime](docs/text-adapter-runtime.md): hash-bound LoRA training, tokenizer compatibility, operator-selected loading, and actual runtime identity.
- [Controlled adaptation comparison](docs/adaptation-comparison.md): frozen factual/style inputs, base/style/adapter controls, raw failures, and blinded review exports.
- [Dialogue evaluation](docs/dialogue-evaluation.md): English/Korean follow-ups, date corrections, topic switches, bounded context, and source-preview regressions.
- [Held-out evaluation data](docs/heldout-evaluation-data.md): publisher acquisition, source-family isolation, corpus overlap checks, and evaluation-only records.
- [Live neural streaming benchmark](docs/live-stream-benchmark.md): actual HTTP text/PCM measurements, service identity, cancellation, and unmeasured playback/acoustic outcomes.
- [2026 implementation and evidence](docs/2026-implementation.md): package status, exact snapshot, runtime configuration, early PCM protocol and remaining release conditions.
- [Original-data expansion and adaptation](docs/persona-data.md): ten actual acquisitions, nine dialogue events, independent modality eligibility, pinned replay and training exports/launchers.
- [Hybrid retrieval](docs/hybrid-retrieval.md): actual multilingual embedding/reranker runs, false-candidate tradeoffs and reproducible optional configuration.
- [Avatar audition](docs/avatar-audition.md): decoded GLB qualification, Blender conversion, same-audio recording and actual acquisition boundaries.
- [Historical avatar acquisition follow-up](docs/avatar-acquisition-followup.md): the earlier asset-search record and authenticated-download boundary; the new-avatar workstream follows the generation plan above.
- [2026 persona roadmap](docs/2026-persona-roadmap.md): current-pilot limits, the 2026 snapshot, data and learning expansion, retrieval, dialogue, streaming, and acceptance criteria.
- [Historical Musk 3D avatar shortlist](docs/research/musk-3d-avatar-shortlist.md): the dated research record for the earlier Musk asset-acquisition plan.
- [Proposal implementation status](docs/proposal-alignment.md): requirement mapping, completed evidence and remaining review boundaries.
- [Original Q&A corpus](docs/original-qa.md): pinned publisher selections, event splits and working situation retrieval.
- [Actual-answer evaluation](docs/reference-evaluation.md): the complete 24-response run, negative coverage result and reconstructable blinded review.
- [Engineering verification and recovery](docs/quality-completion.md): merged PR status, browser evidence, recovery fixes, measured limits, and verification links.
- [Delivery plan](docs/PLAN.md): review boundaries, acceptance criteria, fidelity ladder.
- [Architecture](docs/architecture.md): evidence, generation, timing, and deployment boundaries.
- [Source provenance](docs/research/sources.md): original URLs, dates, timebases, and corpus limitations.
- [Integrated corpus status](docs/research/corpus-status.md): 31 sources, split isolation, and expanded retrieval evaluation.
- [Publisher acquisition](docs/research/collection.md): explicit document collection, component links, and provenance manifests.
- [Full transcript extraction](docs/research/transcript-acquisition.md): complete TED conversation drafts, speaker-label rules, and measured coverage.
- [Media and voice pipeline](docs/media-pipeline.md): local clips, optional transcription/observations, and the reference-voice server.
- [Execution and deployment](docs/deployment.md): coordinated development servers, container configuration, and local validation.
- [Inference validation](docs/inference-validation.md): real model smoke results, reproducible commands, and measured limitations.
- [Local language server](docs/language-runtime.md): model loading, bounded generation, and main-app connection.
- [Generation evaluation](docs/evaluation.md): A/B/C execution, token usage, citation checks, and human-review fields.
- [Reference preparation](docs/research/voice-reference.md): publisher-linked audio candidates, clock matching, and review evidence.
- [Speech timing](docs/speech-timing.md): real audio transport, recognized mouth cues, and failure behavior.
- [Grounding quality](docs/grounding-quality.md): frozen prompt comparisons, exact generated responses, citation coverage, and separate semantic review.
- [Prepared presentation](docs/presentation.md): actual Qwen3-TTS → MuseTalk artifact, verified timing and reproduction.
- [Live measurements](docs/live-demo-latency.md): browser timing export and real LLM-to-voice transport evidence.
- [Voice evaluation](docs/voice-evaluation.md): four real synthesized cases, independent ASR, runtime measurements, and a generated-output listening bundle.
- [Bundled human avatar](docs/humanoid-avatar.md): asset licenses, compressed GLB build, named controls, and actual deformation checks.
- [Motion library](docs/motion-library.md): two-event measurements, source-context selection, validation results, and reproduction.
- [Situation observations](docs/situation-observations.md): actual Q&A-linked speech and visual measurements, source clocks, review gates and unmeasured outcomes.
- [Observation artifact audit](docs/research/observation-pilot-status.md): preserved pilot hashes, measured coverage and exact v1/v2 exchange compatibility.
- [Avatar calibration](docs/avatar-calibration.md): observed clip export, rig mapping, and channel inspection.
- [Integration validation](docs/integration-validation.md): actual ASR, local HTTP inference, speech packages, and video observations.

Run locally by default. Internet-facing deployment needs application authentication, inference quotas, and a considered persistence policy. The current session store is process-local and intentionally ephemeral; run a single API worker.


