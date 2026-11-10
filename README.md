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

Open `http://127.0.0.1:8000`. The studio includes conversation, source inspection, A/B/C comparison, voice playback controls, a rigged reference avatar, and local GLB loading. See [studio guide](docs/studio.md) for development, audio, and retargeting details.

## Run the conversation engine

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run bias validate
uv run bias serve
```

The API runs at `http://127.0.0.1:8000`; API documentation is at `/docs`. With no model configured, responses contain labeled, curated source notes. This mode exercises retrieval, sessions, evidence links, and behavior planning without claiming language-model inference.

### Connect an actual model

```bash
cp .env.example .env
# Set BIAS_LLM_BASE_URL, BIAS_LLM_MODEL, and a key if your provider requires one.
uv run --env-file .env bias serve
```

The adapter supports the OpenAI-compatible `/chat/completions` streaming protocol. A local model server can be used without a paid API. For example, an existing Ollama installation exposes this protocol at `http://127.0.0.1:11434/v1`; set the model name to one you have actually installed. Model quality and hardware needs depend on that choice. Credentials stay on the server.

The provider is called only when configured. Errors and missing capabilities remain visible. This repository does not bundle a language-model checkpoint, a trained voice, or a photorealistic head asset.

## What is implemented

| Capability | Behavior |
| --- | --- |
| Source corpus | Linked primary-source records, dated paraphrases, explicit timelines, review state |
| Retrieval | BM25, English/Korean query support, topic/situation matching, conservative abstention |
| Dialogue | OpenAI-compatible streaming adapter; profile, evidence, and behavior experiment modes |
| Conversation state | Opaque sessions, bounded history, TTL, serialized turns, cancellation cleanup |
| Evidence | Returned citation IDs checked against retrieved records; original source links |
| Behavior plans | Timed face/gaze/head/gesture cues with explicit provenance and estimated timing |
| Speech and transcription | Optional compatible provider adapters with real audio bytes and size limits |
| Local reference voice | Optional Qwen3-TTS service with cached prompts, bounded requests, and a fixed server-side reference |
| Media processing | FFmpeg clips, source-clock manifests, optional ASR/alignment and facial/body observation adapters |
| Ingestion | Local SRT/VTT/WhisperX imports, speaker selection, hashes, source-clock offsets |
| Evaluation | Reproducible retrieval checks and source-disjoint answerability fixtures |

The initial curated corpus is a development fixture, not a sufficient persona training set. Broader source acquisition, original media, detailed interaction context, and independently evaluated likeness are separate milestones.

## Research workflow

```bash
uv run bias search "Why did Tesla start with an expensive sports car?"
uv run bias search "테슬라는 왜 처음에 비싼 스포츠카를 만들었어?"
uv run bias evaluate --fail-on-regression --output reports/local/retrieval.json
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

- A valid citation ID means the response referenced an allowed record; it does not prove that every generated sentence follows from that record.
- Retrieval smoke tests are not an independent persona-fidelity benchmark. Cases derived while authoring the corpus are labeled development fixtures.
- An entire held-out source is excluded from runtime retrieval, including its records and reference answers.
- Initial motion cues are authored demonstration rules, not measured Musk mannerisms. Lip motion needs actual audio/phoneme alignment to support precise synchronization claims.
- No-key source previews do not translate English evidence into fabricated Korean quotations. A connected model can answer in Korean with English-source attribution.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Configuration and corpus status; does not assert remote-provider availability |
| `GET /api/persona` | Profile, examples, corpus counts, capabilities |
| `GET /api/sources` | Source inventory and train/test designation |
| `POST /api/chat` | Complete response with citations and behavioral timeline |
| `POST /api/chat/stream` | SSE `meta`, `status`, `token`, `result`, `done` or `error` |
| `POST /api/speech` | Speech-provider audio, when configured |
| `POST /api/transcribe` | Raw-audio transcription, when configured |
| `DELETE /api/sessions/{id}` | Delete an idle conversation session |

## Documentation

- [Delivery plan](docs/PLAN.md): review boundaries, acceptance criteria, fidelity ladder.
- [Architecture](docs/architecture.md): evidence, generation, timing, and deployment boundaries.
- [Source provenance](docs/research/sources.md): original URLs, dates, timebases, and corpus limitations.
- [Media and voice pipeline](docs/media-pipeline.md): local clips, optional transcription/observations, and the reference-voice server.
- [Execution and deployment](docs/deployment.md): coordinated development servers, container configuration, and CI artifacts.
- [Inference validation](docs/inference-validation.md): real model smoke results, reproducible commands, and measured limitations.

Run locally by default. Internet-facing deployment needs application authentication, inference quotas, and a considered persistence policy. The current session store is process-local and intentionally ephemeral; run a single API worker.
