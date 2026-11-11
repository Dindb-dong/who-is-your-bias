# Who Is Your Bias?

**A research studio for evidence-grounded conversation, voice, and embodied behavior.**

Initial subject: **Elon Musk**. The project connects documented statements to a conversational model and a synchronized avatar. Generated interactions are explicitly presented as simulations. Historical statements keep their source and date; the system does not claim to reconstruct private thoughts.

## Run the complete studio

Requires Node 22.12+ (Node 24 recommended), Python 3.11+, and uv.

```bash
uv sync --extra dev
npm ci
npm run build
uv run bias serve
```

Open `http://127.0.0.1:8000`. The studio includes conversation, source inspection, A/B/C comparison, voice playback controls, a bundled human avatar, a lightweight portrait, local GLB loading, and a motion/calibration lab. See [studio guide](docs/studio.md) for development, audio, and retargeting details.

## Run the conversation engine

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run bias validate
uv run bias serve
```

The API runs at `http://127.0.0.1:8000`; API documentation is at `/docs`. With no model configured, responses contain labeled original-interview excerpts from a bounded pilot. This mode exercises retrieval, sessions, evidence links, and behavior planning without claiming language-model inference.

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
| Source corpus | Original question/answer pairs, distinct context/annotations, pinned source events, explicit unknown acoustic boundaries and review provenance |
| Retrieval | Original-Q&A BM25, English/Korean aliases, shared situation labels and working reranking after a common topical gate; held-out coverage remains inadequate |
| Dialogue | OpenAI-compatible streaming adapter; profile, evidence, and behavior experiment modes |
| Conversation state | Opaque sessions, bounded history, TTL, serialized turns, cancellation cleanup |
| Evidence | Per-unit citation coverage, separately labeled semantic review, original source links, frozen prompt comparisons |
| Behavior plans | Timed face/gaze/head/gesture cues with explicit provenance and estimated timing |
| Speech and transcription | Optional compatible provider adapters with real audio bytes and size limits |
| Local reference voice | Optional Qwen3-TTS service with cached prompts, bounded requests, and a fixed server-side reference |
| Speech timing | Optional Rhubarb mouth-cue recognition, SHA-256-bound audio packages, actual playback clock, explicit amplitude fallback |
| Avatar calibration | Bundled compressed human mesh with 163 bones and 67 named face/viseme targets; calibrated controls, portrait/full-body cameras, and local GLB support |
| Measured motion | 13 source-linked clips and 40 primitives from two events; opt-in conversation retrieval, source-clock replay, and explicit review gates |
| Reference preparation | Publisher-text/ASR comparison, bounded candidate WAVs, clock metadata, artifact hashes, and a review ledger |
| Media processing | FFmpeg clips, source-clock manifests, optional ASR/alignment and facial/body observation adapters |
| Ingestion | Local SRT/VTT/WhisperX imports, speaker selection, hashes, source-clock offsets |
| Evaluation | Reproducible retrieval checks, separate generation cases, actual A/B/C execution, and source-disjoint answerability fixtures |

## Historical corpus and source acquisition

The following counts describe the preserved `data/personas/elon-musk` summary archive, not the current original-Q&A default or the v2 experiment.

The initial 7-source development collection has expanded to **31 distinct source events/documents and 163 reviewed evidence records**, spanning **2003–2026**. Retrieval uses **159 records from 30 sources**; the four reference records from Lex #438 remain held out. Evidence is concise, attributed paraphrase, with original URLs and dates.

The collection includes TED interviews, Lex Fridman conversations, Dwarkesh, Y Combinator, Recode, WIRED, TIME, GQ, Stanford, MIT, congressional testimony, and authored Tesla documents. Coverage now includes learning, fear, criticism, uncertainty, humor, motivation, and work habits alongside engineering and business plans. See [integrated corpus status](docs/research/corpus-status.md) for exact contributions, dates, acquisition links, and limitations.

The publisher transcript pipeline also extracted **3,451 timestamped caption cues** from four complete TED interviews: **3:00:45 of declared program duration and 27,411 whitespace-delimited words**. Of those cues, **1,822 carry Musk labels in the publisher text**, 1,125 carry interviewer labels, and 504 remain unassigned. Questions and adjacent speaking turns are preserved. The full local transcript imports remain **draft, unindexed** material. Separately reviewed original-Q&A selections from the pinned transcripts enter the new explicit corpus; a text label is not independently verified voice identity. The repository contains reproducible extraction code and measured coverage, while complete originals remain outside Git.

This is a research corpus and working application, not a validated replica of a person's mind, voice, or appearance. The [initial inference report](docs/inference-validation.md) and [integration results](docs/integration-validation.md) separate real model execution from citation compliance and perceptual quality. The included human avatar is an artistic reference asset with a calibrated face/body rig. A high-quality likeness asset, learned personal mannerisms across many situations, and independent perceptual evaluation remain separate work.

## Original interview corpus

The explicit default is a ten-record bounded excerpt pilot. The source library identifies it. The full original-answer study uses 92 locally materialized exchanges (80 train / four development / eight test); source text is pinned and reconstructed locally:

```bash
uv run python -m tools.qna hydrate \
  --selection data/qna/elon-musk-v1/selection-recipes.v2.json \
  --output data/local/qna/elon-musk-v2
BIAS_CORPUS_DIR=data/local/qna/elon-musk-v2 uv run bias serve
```

See [original Q&A](docs/original-qa.md) for source review, source-event isolation, unknown acoustic timestamps and bounded-versus-full materialization. Old summary-corpus results retain their original version and explicit corpus path.

## Research workflow

```bash
uv run bias search "Why are you boring?"
uv run bias search "땅굴을 왜 파나요?"
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
| `POST /api/speech` | Speech-provider audio, when configured |
| `POST /api/speech/prepare` | Generate audio and optional mouth cues bound to its SHA-256 |
| `GET /api/speech/clips/{id}` | Retrieve an ephemeral prepared clip before expiry |
| `POST /api/transcribe` | Raw-audio transcription, when configured |
| `DELETE /api/sessions/{id}` | Delete an idle conversation session |

## Documentation

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
- [Execution and deployment](docs/deployment.md): coordinated development servers, container configuration, and CI artifacts.
- [Inference validation](docs/inference-validation.md): real model smoke results, reproducible commands, and measured limitations.
- [Local language server](docs/language-runtime.md): model loading, bounded generation, and main-app connection.
- [Generation evaluation](docs/evaluation.md): A/B/C execution, token usage, citation checks, and human-review fields.
- [Reference preparation](docs/research/voice-reference.md): publisher-linked audio candidates, clock matching, and review evidence.
- [Speech timing](docs/speech-timing.md): real audio transport, recognized mouth cues, and failure behavior.
- [Grounding quality](docs/grounding-quality.md): frozen prompt comparisons, exact generated responses, citation coverage, and separate semantic review.
- [Voice evaluation](docs/voice-evaluation.md): four real synthesized cases, independent ASR, runtime measurements, and a generated-output listening bundle.
- [Bundled human avatar](docs/humanoid-avatar.md): asset licenses, compressed GLB build, named controls, and actual deformation checks.
- [Motion library](docs/motion-library.md): two-event measurements, source-context selection, validation results, and reproduction.
- [Avatar calibration](docs/avatar-calibration.md): observed clip export, rig mapping, and channel inspection.
- [Integration validation](docs/integration-validation.md): actual ASR, local HTTP inference, speech packages, and video observations.

Run locally by default. Internet-facing deployment needs application authentication, inference quotas, and a considered persistence policy. The current session store is process-local and intentionally ephemeral; run a single API worker.
