# Measure live HTTP text and PCM delivery

`python -m tools.live_benchmark` measures an actual `/api/chat/live` request
against separately running application, local language and local Qwen3-TTS
services. It fills the gap between the existing complete-WAV latency harness
and the browser's synthetic live-PCM tests.

The runner is implemented and tested. A separately recorded bounded neural
execution completed on November 11, 2026; its receipts and limitations are in
[the execution report](../reports/live-stream/2026-v1/report.md). The fixture
tests are independent of that neural measurement. A real run requires
downloaded model weights, an explicitly selected reference-voice configuration
and reachable services. Results remain tied to the machine, model identities,
exact question and warm state that produced them.

## Run an actual measurement

Start the local services using [language setup](language-runtime.md),
[text-adapter serving](text-adapter-runtime.md) when applicable, and
[voice setup](voice-evaluation.md). Configure the application's actual LLM and
speech endpoints; an evidence-only preview cannot pass this benchmark.

```bash
uv run python -m tools.live_benchmark \
  --base-url http://127.0.0.1:8000 \
  --language-health-url http://127.0.0.1:8020/health \
  --voice-health-url http://127.0.0.1:8010/health \
  --message "Why reuse rockets?" \
  --language en --mode behavior \
  --timeout 600 \
  --output reports/local/live-http-cold.json
```

Use ports matching the processes you actually started. Use a question whose
evidence exists in the selected corpus. Even with a configured LLM, a temporal
abstention or empty-evidence response can take the source-preview path. Such a
response is recorded as a failed *inference measurement*, not a fabricated
model timing. The separate multi-turn evaluation covers those legitimate
abstention behaviors.

Run another request into a new output file to observe warm behavior. The runner
records `model_loaded` before and after each request and retrieval prewarm
metadata; it does not infer a cold run from a filename. Every attempt receives
a new application session so that previous conversation history is not silently
carried into the measurement.

If an endpoint requires a bearer credential, pass an environment variable name
with `--api-key-env`, `--language-key-env` or `--voice-key-env`. Values are read
in memory and never included in reports. Credentials and query strings are
rejected in endpoint URLs. Remote endpoints require HTTPS; loopback HTTP is
supported for local development.

The default deadline covers the actual streaming request, including queue and
model waits. It is an absolute asynchronous deadline rather than a new timeout
for every received chunk. Health checks and the subsequent session cleanup
probe are measured separately. Reports are written to a new file even after a
protocol, provider or deadline failure; old reports are not overwritten.

### Launch all local services together

`tools.live_benchmark_local` starts the actual language service, voice service
and application as child processes, then runs the same HTTP recorder in their
network namespace. This is useful in execution environments where separately
started commands cannot share loopback listeners. All three processes are
stopped after the single bounded request, including failures.

```bash
uv run python -m tools.live_benchmark_local \
  --language-cache /absolute/path/to/language-hf-cache \
  --voice-cache /absolute/path/to/voice-hf-cache \
  --reference-wav /absolute/path/to/reference/audio.wav \
  --reference-text /absolute/path/to/reference/reference.txt \
  --reference-provenance /absolute/path/to/reference/restoration.json \
  --threads 2 --max-output-tokens 96 --max-voice-tokens 384 \
  --timeout 180 \
  --output reports/local/live-neural-attempt-01
```

The reference provenance must declare `audio_sha256` and `text_sha256` matching
the actual files. Its review status is retained, never promoted by the harness.
Use the [reference preparation instructions](research/voice-reference.md) and
preserve any remaining review limitations. Separate local Python environments
default to `.venv-model` for language/application and `.venv-voice` for speech;
explicit `--app-python`, `--language-python` and `--voice-python` override them.
Dependencies and model acquisition remain separate preflight steps.

The launcher pins Qwen3-0.6B at
`c1899de289a04d12100db370d81485cdf75e47ca`, Qwen3-TTS-12Hz-0.6B-Base at
`5d83992436eae1d760afd27aff78a71d676296fc`, and the speech tokenizer at
`7dd38ad4e9bad454aae9cd937d0cd577604fe229`. It hashes every downloaded
snapshot file and checks the tokenizer actually embedded in the TTS snapshot
against the separately pinned tokenizer. Runtime inference uses local files
only. It records package versions, source/corpus/reference hashes and sampled
process memory; changed application source during execution invalidates the
completed measurement. Use a cache on a volume with sufficient capacity rather
than assuming the default home-directory cache can hold the models.

The default tunnel-cost question resolves to evidence in the stored original
Q&A corpus. The launcher checks for lexical evidence before starting services;
an absent passage cannot waste a measured request on a source-preview answer.
The question asks for one short answer, with a 96-token output budget.
The actual model must finish normally; a token-limit interruption is retained
as a failed attempt. One cold request is not a warm latency distribution.

## What the report measures

All client receipt timestamps use the same monotonic `time.perf_counter`
origin immediately before the HTTP request. Server metrics retain their own
declared origin and are not subtracted from client timestamps.

| Field | Meaning |
| --- | --- |
| `headers_received_ms` | Response headers received by the HTTP client |
| `stream.timings.first_text_received_ms` | First non-whitespace text token received and validated |
| `stream.timings.first_pcm_received_ms` | First format/sequence/byte-valid PCM chunk received |
| `stream.timings.result_received_ms` | Final text result received, with matching turn/session and actual LLM provider |
| `stream.timings.audio_end_received_ms` | All announced PCM segments and totals validated |
| `stream.timings.done_received_ms` | Normal protocol completion observed after text and audio |
| `request_closed_ms` | HTTP request context closed, including cancellation/failure |
| `stream.delivered_audio_duration_ms` | Duration represented by the delivered frame count and sample rate; no playback is performed |
| `stream.first_pcm_preceded_text_result` | Whether PCM arrived before final text result; null if either event was unobserved |
| `session_release` | Whether deletion of the runner's own session was accepted after closing the stream |

Browser first-visible text, browser playback, speaker onset and spoken
completion remain **null**. A transport `done` is not audible completion.
The runner does not render an avatar, measure a device speaker, run a microphone
loopback or establish perceptual speech quality. Use the browser's live timing
export and a separate acoustic setup for those measurements.

`health_before` and `health_after` retain an explicit allowlist of safe model
aliases, immutable revision/checkpoint/tokenizer/adapter identities when the
language service reports them, observed language device/dtype/backend, thread
count, context, parameter count, package versions and generation defaults,
voice runtime settings, load state and current busy state. Unknown health
fields, credentials and local adapter paths are not
copied. The voice service exposes its configured alias and revision; it does
not currently expose a full checkpoint-file digest manifest. That missing
identity is not invented. The benchmark host description is the HTTP client's
machine and does not claim to describe remote inference hardware.

A completed stream also requires stable before/after application, language and
voice identities. A changed alias, base checkpoint, tokenizer, adapter or
observed runtime makes the measurement fail even if the replacement reports
`model_loaded=true`. Cold services may fill previously unknown hashes/runtime
fields and complete their configured adapter load; values already observed
before the request must remain unchanged. This checks service-reported
continuity, not independent attestation that the underlying process was never
restarted. Dynamic load timing, busy state and last-generation measurements are
excluded from the identity comparison.

The submitted question is retained. Generated text and PCM are represented by
hashes and bounded metadata; raw provider text, source excerpts and base64 audio
are not included in the report. Each accepted event has an observation time and
canonical payload hash. The report also records the local benchmark code hash;
this is not independent attestation of a remote server's code revision.

## Protocol integrity

The parser incrementally decodes UTF-8 and SSE across arbitrary network chunk
boundaries. It limits individual frames, total event count, generated text and
delivered PCM. Invalid JSON, truncated frames and an HTTP EOF before validated
`done` fail the measurement.

The validator requires one meta event and stable session/turn/segment
identities. Committed text must match already received tokens and declared
character offsets. Audio segments follow committed order with one stable
format, contiguous frame offsets and exact integer chunk/segment sequences.
Base64 is decoded strictly; every payload must contain exactly the declared
number of 16-bit PCM frames. Segment endings and final totals must match the
actual received data. Late text, repeated/unknown events, premature completion,
provider error events and stale-turn audio fail closed.

`audio_segment` now includes an additive `pcm_sha256` field calculated from the
actual decoded PCM. The runner hashes each segment incrementally and verifies
it before marking that segment complete. A valid base64 string with changed
sample bytes therefore cannot become a completed measurement. Existing clients
can continue ignoring this additional field.

The existing `audio_sha256` still identifies the original complete WAV. WAV
headers and optional chunks are not transmitted in this protocol, so that
digest cannot be independently reconstructed from PCM alone. The report
preserves this distinction: PCM content is verified; the original WAV digest
is recorded without claiming to verify unavailable header bytes. An older
server that does not provide `pcm_sha256` cannot pass the stricter benchmark.

The first chunk can arrive before its complete segment digest is available.
Its receipt time is retained, but a later digest failure makes the overall
attempt fail. This does not retrospectively label unverified partial data a
successful completed turn.

## Cancellation

```bash
uv run python -m tools.live_benchmark \
  --base-url http://127.0.0.1:8000 \
  --language-health-url http://127.0.0.1:8020/health \
  --voice-health-url http://127.0.0.1:8010/health \
  --message "Explain why rocket reuse matters and then describe the engineering tradeoffs." \
  --cancel-after-first-pcm --cleanup-timeout 10 \
  --output reports/local/live-http-cancelled.json
```

The HTTP response is closed immediately after the first valid PCM chunk. The
attempt is recorded as `cancelled`, never `completed`, and
`measured_inference_complete` stays false. Pending text/audio totals, unfinished
segment hashes and unobserved result events stay incomplete or null.

The runner then tries to delete only the newly created benchmark session. A
temporary 409 means the application still owns the turn; a successful deletion
confirms that session ownership was released. It does not prove that a separate
synchronous voice inference thread has stopped. Post-run voice `busy` status is
retained, and no acoustic cancellation latency is claimed. An unconfirmed
cleanup remains explicit in the report.

## Verification

```bash
uv run pytest tests/demo_latency/test_live_benchmark.py -q
```

The focused tests use synthetic PCM and a fixture LLM/voice backend. They cover
real production `live_events` output compatibility, actual PCM digest emission,
interleaved early audio, corruption with otherwise valid base64, stale
identities, sample/sequence/format drift, truncated frames, false completion,
source-preview rejection, incomplete cancellation, health redaction, cold
identity discovery, runtime/adapter/tokenizer replacement rejection and
credential-free report URLs. They establish the recorder's integrity; they do
not establish neural inference speed or Musk voice similarity.

## Recorded bounded execution

The actual base Qwen3-0.6B + Qwen3-TTS run used two CPU Torch threads per model,
bfloat16, the documented pinned snapshots and a restored engineering reference
candidate. Both model services were cold before the request. HTTP first text
arrived at **7,093.753 ms**, final text at **9,888.115 ms**, first PCM at
**52,715.145 ms**, and verified normal completion at **52,721.974 ms**.
One PCM segment contained 132,480 mono frames at 24 kHz, representing 5.52
seconds of audio. All 65 chunks and the complete PCM digest validated.

These are client transport measurements on the assistant execution host. The
parent's final four-case model comparison ran concurrently with two CPU
threads; training finished before the neural request. PCM arrived after the
final text in this example. This result therefore establishes actual end-to-end
execution, not overlapping delivery, real-time performance, warm latency or
the latency of the user's device. Browser/acoustic measurements and perceptual
voice ratings remain unmeasured.

The report preserves a failed acquisition caused by the default cache volume
running out of space and a failed first HTTP setup question that selected a
source-preview response. The latter was correctly refused as language-model
inference; it is not included in the successful neural timing. Detailed model,
reference, code and corpus hashes, exact package versions, service health,
process logs and the corrected concurrency context are retained together.
