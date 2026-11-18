# YAICON live Q&A latency experiment

Prepared 2026-11-17 against `main` at
`61385492e18c47a49104ba26cd9cbab46e6068f2` on task branch
`feat/yaicon-live-latency-nov17`.

## Observed baseline and scope

The team reported on November 14 that its local Mac generated approximately
30 seconds of Jensen speech in 59 seconds. The derived real-time factor is
`59 / 30 = 1.97`. This is a team-reported whole-utterance TTS measurement,
not a reproduced benchmark or a measured question-to-first-audible latency.
It must not be compared directly with server first-PCM timing.

The current code already overlaps LLM generation with synthesis of completed
sentences through `/api/chat/live`. It already caches the model's reference
voice prompt after the first synthesis. The Qwen voice service returns a
complete WAV for each sentence. Sending that WAV as PCM chunks does not make
the neural model itself stream audio.

The change here adds an **opt-in shorter spoken-answer prompt** and stage
measurements. It does not change the voice model, retrieve different evidence,
hard-truncate sentences, reduce the LLM token cap, or bypass Korean answer
review. Real Mac latency and answer quality remain unmeasured for this change.

## Activation

Set `BIAS_LIVE_CONCISE=true` in the application's `.env`, then restart the API
using the same corpus/provider configuration:

```bash
uv run --env-file .env bias serve --host 127.0.0.1 --port 8000
```

`GET /api/health` then reports `live_response_policy: live-concise.v1`.
Keep `BIAS_ADAPTIVE_RESPONSES=false` for this fixed-concise A/B comparison.
The separate [adaptive-answer option](yaicon-adaptive-answers.md) takes precedence
when enabled and uses question-dependent targets across chat, stream and live.
The default is `false`, reported as `default`. Ordinary `/api/chat` and
`/api/chat/stream` keep their existing prompt even when the flag is enabled.

For `/api/chat/live`, the prompt asks for a complete short first sentence
(English target: 18 words, excluding citation markers; Korean target:
15 spacing-delimited words) and at most two short sentences overall. English
targets approximately 45 words total. Existing stricter Korean instructions
take precedence. These are model instructions, not guaranteed length limits.
Necessary qualifications, uncertainty and citation markers remain required.
The extra instruction is counted **before** snapshot evidence/history packing.
Check `turn.context.prompt_budget` for any resulting change in supplied evidence.

The hypothesis is that a shorter complete first sentence needs less TTS work
before playback can start. Extra sentence boundaries can also increase TTS call
overhead or change prosody. This requires measurement, not an assumed speedup.

## Measurement contract

The final SSE `audio_end` event adds `server_timings_ms`, measured from the live
request's server start, including any earlier-turn lock wait:

| Field | Event |
| --- | --- |
| `first_text_ms` | Engine emits its first text fragment, after any required review |
| `first_segment_ms` | A complete text segment becomes eligible for TTS |
| `first_synthesis_started_ms` | First call to the speech provider starts |
| `server_first_pcm_ms` | First PCM chunk is yielded to the HTTP response iterator |
| `text_complete_ms` | Engine reports normal text completion |

`server_first_pcm_ms` is also included in the first `audio` event. Existing
`first_pcm_ready_ms`, per-segment `synthesis_ms` and final `total_ms` remain.
None of these measures speaker audibility. Network arrival, browser buffering,
AudioContext resume and acoustic output can add delay. Incomplete/failed streams
do not become successful timing observations.

The existing benchmark saves these fields under
`stream.server_metrics.live_stages_ms`, along with independent client-received
first-PCM timing and provider health/configuration. The response policy is part
of the configuration receipt and must remain unchanged within a measured run.

## Mac before/after protocol

1. Keep the exact model revision, quantization, device, reference audio,
   transcript, corpus, retrieval settings and generation settings fixed.
2. Keep the language/voice services running while restarting only the API to
   toggle the flag. Record the first cold run separately. Warm both providers
   with one complete, supported question before collecting the warm runs.
3. Use five fixed supported questions covering factual answers, values and
   follow-ups. Run each three times under each condition in separate sessions.
   Alternate the order of conditions across batches. Retain failures as failures.
4. Compare median and range of first client-received PCM, total delivery time,
   audio duration, number of TTS segments and sentence-level source support.
   Listen for incomplete caveats, awkward sentence joins and speaker quality.
5. Measure browser first playback separately with the same browser, audio
   device and avatar. A faster server result alone does not establish a faster
   or more natural live demonstration.

With this repository's local language and voice services on their documented
ports, one real run uses the existing client below. Substitute actual ports;
the client's identity checks require these local health contracts, so an
unrelated hosted/Ollama health endpoint is not interchangeable.

```bash
uv run python -m tools.live_benchmark \
  --base-url http://127.0.0.1:8000 \
  --language-health-url http://127.0.0.1:8020/health \
  --voice-health-url http://127.0.0.1:8010/health \
  --message "What did you mean by AI factories at COMPUTEX 2025?" \
  --language en --mode rag \
  --output data/local/yaicon-latency/baseline-01.json
```

Choose a new output filename for each run. Use `concise-01.json` for the enabled
condition. Missing models, unavailable services or insufficient retrieved
evidence must be reported; the client does not fabricate benchmark answers.

## Acceptance and remaining work

Functional checks use fake language/voice providers and synthetic PCM. They
test endpoint scoping, prompt budgeting, citation retention, Korean review,
cancellation and event timing order. They do **not** measure Jensen voice
similarity, model speed, answer faithfulness, GPU/MPS performance or browser
playback. The focused suite passed 154 tests on this implementation checkpoint.

Before enabling for the final demo, collect the Mac results above and accept the
policy only if first-audio latency improves without dropping necessary answer
content or citations. If TTS still dominates, evaluate a faster supported TTS
runtime/device while keeping the reference voice and test questions fixed.
Replacing only the LLM cannot remove time spent in the voice model.

The prepared presentation script can be synthesized and reviewed in advance.
Keep its cached playback measurement separate from live unseen-question Q&A.
