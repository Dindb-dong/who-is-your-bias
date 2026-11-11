# 2026 persona implementation and measured boundaries

Implementation record for the [2026 roadmap](2026-persona-roadmap.md), dated
2026-11-11. The original roadmap and failed pilot remain historical records.
This document preserves the implementation and measurements delivered in
[PR #13](https://github.com/Dindb-dong/who-is-your-bias/pull/13), now merged at
`cc23f47f5dfa25cced290e90a8f8c4ef9db22181`. The later
[continuation record](2026-continuation.md) covers actual text training, verified
adapter serving, additional original-source acquisition, controlled generation,
and dialogue/streaming acceptance work. Counts below describe the PR #13 build.
The previously open, dependent PRs [#10](https://github.com/Dindb-dong/who-is-your-bias/pull/10),
[#11](https://github.com/Dindb-dong/who-is-your-bias/pull/11) and
[#12](https://github.com/Dindb-dong/who-is-your-bias/pull/12) were merged, ending at
`0fde5aeb6a494b575099819dbd14ba5baaa78b29` on `main`. Their source, prompt,
evaluation, speech and presentation artifacts were preserved.

## Delivered work and remaining acceptance evidence

| Roadmap package | Executed or implemented in this change | Still required for the package's full exit condition |
| --- | --- | --- |
| WP0: versioned persona and evaluation registration | Exact corpus-bound snapshots; event/publication/acquisition dates; independent modality versions; new source-family split registration; unchanged failed pilot | Acquire and independently review the reserved test events; verify contemporary-position coverage |
| WP1: original corpus and hybrid retrieval | Ten publisher documents acquired, nine usable events, 977 attributed QAs; independent factual/style/audio/visual eligibility; actual pinned multilingual embedding and cross-encoder development runs | Reach the 20–30-event acquisition target; human source review; reduce misses and unsupported candidates on fresh events |
| WP2: real asset audition | Decoded GLB qualification, real Blender conversion, same-audio browser recording and versioned reports; actual bundled-proxy qualification | Acquire a usable Musk-specific candidate through its legitimate download/purchase path; assess 2026 likeness and dynamic quality |
| WP3: text and speech adaptation | 595 complete text examples exported with event caps; hash-bound text LoRA and Qwen3-TTS training launchers; explicit empty speech export | Accepted aligned speech; actual adapter training and controlled base/style/adapter comparisons; human listening review |
| WP4: temporal dialogue and early audio | Bounded user-reference state, exact-period gates, prompt packing, committed-sentence synthesis, streamed PCM, playback-clock animation, interruption and protocol checks | Actual simultaneous LLM/TTS/avatar benchmarks on the intended machine; product latency budgets; acoustic and perceptual measurements |
| WP5: release evidence | Reproducible data recipes, model/input hashes, negative results, browser protocol checks, existing prepared presentation retained | Fresh blinded text/audiovisual evaluation and an assessed demonstration meeting the product target |

This is a working research implementation with measured development results.
It is not a completed or perceptually validated Musk replica. In particular,
the trained text adapter, trained voice adapter and qualified Musk likeness
counts were all zero at the PR #13 snapshot. Consult the continuation record for
subsequent executions; artifact completion and measured persona fidelity remain
separate claims.

## Data, retrieval and the actual experiment

[Original-data preparation](persona-data.md) documents the acquisition and
replay commands. The 977 QAs contain 922 complete published exchanges and 55
qualified excerpts. Independent projection gates admit **745 factual records,
922 style examples, zero aligned speech records and zero visible-behavior
records**. Publisher structure and attribution were reviewed by an AI agent;
this is not a human listening or semantic-entailment assessment. Complete
original transcripts remain in ignored local build directories.

The new split registry has 28 training families, three development families
and three reserved test families. The eight exposed pilot questions remain
reproduction/development material. Fresh test transcripts and reference answers
have not been acquired. Nine acquired dialogue events do not satisfy the
20–30-event roadmap target, and there are no accepted speech hours to count
toward its 2–5-hour target.

The [hybrid report](hybrid-retrieval.md) includes actual CPU inference with
`intfloat/multilingual-e5-small` and the multilingual mMARCO cross-encoder at
pinned revisions. The legacy lexical implementation and frozen pilot path are
preserved. On the expanded original-QA development cases:

| Retrieval policy | Registered supporting record in top four | Absent cases returning a candidate |
| --- | ---: | ---: |
| Preserved lexical baseline | 3/36 | 0/20 |
| E5 cosine 0.78, cross-encoder raw-logit floor −4 | 28/36 | 4/20 |
| One stricter sensitivity, raw-logit floor −3 | 27/36 | 0/20 |

These are 18 underlying answerable exchanges with English/Korean paraphrases,
not 36 independent events. They come from nine training events. Both hybrid
policies find the specified support in only **one of four explicitly dated
2026 cases**. No current-2026 accuracy, situation-retrieval advantage or new
persona-fidelity claim follows. A selected candidate is not proof of an answer's
entailment. The stricter setting is the optional expanded-corpus example because
of this observed development tradeoff; lexical retrieval remains the default.

## Run the expanded application

Materialize a fresh corpus with [the data commands](persona-data.md), then
register a new snapshot using the actual build time. The included
[build snapshot](../reports/data-expansion/2026-v1/persona-snapshot.json) records
the executed build cutoff at `2026-11-11T10:37:37.104004Z` and digest
`ff732e295f6b5488e4c746bca31d4d13852c9b2d480bf083ea3a7dd19978f2a6`.
A later acquisition has its own ingestion times and therefore requires its own
snapshot; do not copy an old manifest onto a changed corpus.

```bash
uv sync --extra dev --extra hybrid
npm ci
npm run build

# First create data/local/persona-2026/corpus using docs/persona-data.md.
PERSONA_BUILD_CUTOFF="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
uv run bias --corpus data/local/persona-2026/corpus snapshot \
  --id elon-musk-original-local.v1 \
  --as-of "$PERSONA_BUILD_CUTOFF" --ingested-at "$PERSONA_BUILD_CUTOFF" \
  --current-since 2026-01-01 --output data/local/persona-2026/snapshot.json

uv run --env-file config/persona-2026.env.example bias serve
```

The example has no credentials or configured generation/speech provider. It
runs attributed source previews after hybrid retrieval. Set actual LLM and TTS
endpoints in a private environment file to generate new answers and speech.
The model revision, query/passage prefixes, dense floor and raw-logit reranker
floor are explicit. `BIAS_STYLE_EXAMPLES=true` adds separately eligible style
examples; these never enter the factual citation allowlist.

The local model packages are optional. Startup prewarms the hybrid corpus index
and configured reranker before serving requests, and a preparation failure
fails startup instead of quietly reverting to lexical search. Health includes
the measured `startup_warmup_ms`. The first query in the CPU development run
took about 117 seconds including cold indexing, model loading and first rerank;
this was not an isolated index measurement or conversational latency. The index
is process-local and rebuilt on restart. Set `BIAS_RETRIEVAL_PREWARM=false` only
when intentionally measuring cold request behavior. Keep a single API worker.
The measured runs used an isolated CPU Torch environment. The ordinary hybrid
lock includes CUDA dependencies on Linux; the [pinned CPU environment recipe](hybrid-retrieval.md#optional-isolated-cpu-environment)
documents the actual package selection and avoids treating a default lockfile
installation as an identical benchmark environment.

The [actual expanded-API smoke](../reports/runtime/2026-v1/preview-api-smoke.json)
then exercised the final startup path with these real models through an
in-process ASGI client: warmup took **65.940 seconds**, and all six source-preview
turns completed in **78.807 seconds including startup**. The first question and
two follow-ups retained the expected original passage; the Korean current-plan
question and its two follow-ups all abstained without citing old evidence.
The three searched turns took 3.234–3.736 seconds each on that CPU run. This is
an integration check without LLM/TTS inference, external network transport or
speaker latency. Reproduce after materializing the example's corpus/snapshot:

```bash
uv run --extra hybrid --env-file config/persona-2026.env.example \
  python -m tools.persona_runtime_smoke --output reports/local/preview-api-smoke-new.json
```

## Temporal evidence and dialogue state

The default bundled preview loads its registered 2026 snapshot. A custom corpus
requires an explicit `BIAS_PERSONA_SNAPSHOT` path. Setting that variable to an
empty value disables the temporal runtime policy for historical reproduction;
it does not activate a different 2026 snapshot. The `search` and `evaluate` CLI
commands remain the historical lexical tools; hybrid evaluations use the
separate documented runner.

A snapshot verifies the exact corpus digest, persona, source families, split,
publication dates, event dates and actual per-source acquisition times. Unknown
event dates stay null. A date-only same-day publication is excluded until its
entire UTC day is within the cutoff. Snapshot acquisition cannot be backdated
to make later material appear available earlier.

| Question intent | Runtime handling |
| --- | --- |
| Current/now/현재/요즘 | Requires a recent verified event date or an explicitly reviewed current assertion period; old or undated recordings do not become current merely because they were republished |
| In a specific historical year | Keeps evidence from that event year, using the publication year only when the event date is unknown; preserves the distinction in the prompt |
| By/as of a historical year | Allows earlier evidence through the requested year |
| Before a historical year | Excludes that year and later years |
| Future outcomes beyond the cutoff | Abstains; a forecast/plan question can retrieve an earlier dated statement but cannot turn its target into an achieved outcome |
| Then versus now / multiple historical years | Preserves source dates and qualifications; does not infer that recency establishes supersession |

Only explicit reviewed `supersedes` links suppress earlier current claims.
`contradicts` links do not silently discard either side. The expanded corpus's
Dwarkesh page establishes publication on February 5, 2026; its recording date
remains unknown. Accordingly, `current_dated_training_sources` is zero in this
build. An attributed question about the published 2026 interview is distinct
from an unqualified claim about the person's current position.

`GET /api/sessions/{id}/context` exposes the bounded dialogue state: original
user statements, search anchors, entities, referenced turn IDs, cited source
IDs, grounding status and temporal intent. Follow-ups retain the user topic
and temporal anchor across multiple turns; explicit corrections and detected
topic switches reset it. User claims and generated assistant prose are never
automatically promoted into retrieved evidence. This resolver uses declared
English/Korean reference and entity rules, not a validated general-purpose
coreference model. Successful text turns are retained; interrupted text
generation is not. A completed text answer may remain if its later audio fails.

### One context budget and citation boundary

Snapshot/hybrid prompts separate factual evidence, wording-only style examples,
bounded dialogue history and the current question. They preserve complete
selected passages and reserve the configured output budget. An evidence passage
that does not fit is dropped whole, and its ID cannot authorize a citation.
If no complete evidence fits, the request fails explicitly before inference.
The bounded user reference and supporting user turn IDs are reserved in the
base context, so dropping an oversized history pair cannot erase the subject
or audience needed to interpret a short follow-up. They remain unverified user
context and never authorize citations.

Set `BIAS_LLM_CONTEXT_WINDOW` to the actual model window and optionally
`BIAS_LLM_TOKENIZER_PATH` to its matching local tokenizer with a chat template.
That path uses actual template tokenization. Without it, the trace explicitly
reports a conservative UTF-8 byte budget plus message overhead, **not exact
model tokens or a universal tokenizer guarantee**. A valid citation identifier
still does not establish semantic entailment; generated claims remain unreviewed.

## Committed text and live PCM

`POST /api/chat/live` accepts the same chat request as `/api/chat/stream` and
requires a configured speech provider. The server synthesizes each committed
sentence while later language tokens continue. It accepts complete 16-bit
mono/stereo PCM WAV segments at 8–48 kHz and transports their PCM frames in
bounded SSE chunks. This is real application-level early audio delivery, not
native incremental acoustic-token generation by the TTS model. Each sentence
still waits for that provider's WAV.

| Event | Meaning |
| --- | --- |
| `meta` | Session, turn, snapshot and `live-segments-pcm.v1` identity |
| `token` | New text for display |
| `segment` | Stable segment ID, sequence, text offsets and allowed citation IDs |
| `audio_segment` | Hash, sample format and expected frame extent of one synthesized segment |
| `audio` | Base64 signed little-endian PCM, global sequence and exact sample offset |
| `audio_segment_end` | Delivered segment boundary |
| `result` | Completed text answer and its grounding/context trace; audio can still be playing |
| `audio_end` | Final delivered chunk/sample totals; explicitly `spoken_complete=false` |
| `done` / `error` | Normal transport termination or explicit partial failure |

The segmenter waits for usable sentence boundaries and handles numbers,
abbreviations and citation brackets. It discards an uncommitted tail on provider
failure/cancellation. Commitment is a transport decision, not factual review or
a guarantee that later model output never revises a previous sentence.

The browser validates turn/session identities, committed segment order, sample
format, offsets and terminal totals before scheduling audio. Web Audio applies
bounded backpressure, queues continuous sample-clock buffers and reports
underruns. Avatar mouth movement follows actual PCM energy and the playback
clock; it remains approximate amplitude-driven lip sync. Completion requires
the scheduled audio to finish, including the exact final accepted sample clock.
Digital first-PCM timing is separate from microphone/loudspeaker latency, which
remains null/unmeasured.

Stop, a new question, navigation and connection loss invalidate the current
turn's pending playback. Server disconnect handling cancels silent LLM/TTS waits
and closes generators even under socket backpressure. The session's turn lock
and lifetime pin remain held through speech-worker cleanup. Queued turns consume
the same absolute deadline as generation/audio; waiting does not reset the
budget. Client-side cancelled/failed states cannot become spoken completion.

## Avatar audition and verification

[Avatar audition](avatar-audition.md) documents the actual GLB decoder, Blender
conversion and Motion Lab flow. The bundled generic proxy has a hash-bound
qualification receipt with seven meshes and 66 distinct nonzero named morphs.
Names alone do not prove deformation; decoded displacement measurements are
recorded. Same-audio reports bind the model, rig mapping, WAV, calibration and
viewport, and invalidate on relevant changes or hidden-tab interruption.

The source listings that could provide a Musk-specific asset require payment
or an authenticated download. No model was purchased, no access control was
bypassed and no listing image was treated as an acquired mesh. This change
qualifies the actual existing proxy and makes candidate comparison runnable;
it does not claim a successful Musk likeness audition.

Verification commands:

```bash
uv run pytest
uv run bias validate
uv run bias --corpus data/personas/elon-musk evaluate --fail-on-regression
npm run check
node --test tools/avatar_audition.test.mjs
npm run test:e2e
```

Focused regressions exercise exact dates, chained follow-ups, whole-passage
budgets, independent retrieval projections, protocol identity, queue deadlines,
ASGI disconnects and cancellation without false completion. Browser protocol
tests use a real local HTTP SSE server and real Web Audio with synthetic sine
PCM; they establish playback/cancellation behavior, not voice similarity or a
neural TTS latency result. The avatar browser test uses the actual bundled GLB.
Local Blender conversion was executed. This workspace blocks Chromium's
process singleton socket before page load, so browser rendering is verified in
the repository's GitHub Actions runner and retained as CI artifacts.
