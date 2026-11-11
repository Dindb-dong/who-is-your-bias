# Live language → voice measurement

The application exposes two distinct measurement surfaces. Its **This turn → Export** button records actual browser events through `LiveTurnClock`. The `tools.demo_latency` command measures the real server path through streamed conversation, Qwen speech preparation, and complete WAV acquisition. These reports have different schemas and must not be merged into a browser latency claim.

## What the HTTP harness measures

The command launches three real processes in one parent execution: llama.cpp with a local Qwen3-4B Q4_K_M file, the existing Qwen3-TTS voice service, and the production application. It sends requests to the application endpoints used by the browser. No provider, response, audio file, or network request is mocked during a run.

Every repetition starts a new conversation, records each SSE event, preserves the returned answer and evidence, removes only the citation tokens using the same rule as the browser, requests `/api/speech/prepare`, and downloads that package's audio. A SHA-256 check binds the saved WAV to the package; a PCM decoder verifies its duration and locates the first 10 ms window with normalized RMS above 0.002.

| Field | Meaning |
| --- | --- |
| `first_token_received_ms` | First nonempty text token reaches the HTTP client after submission. A whitespace token can qualify. |
| `first_content_text_received_ms` | First token containing non-whitespace text reaches the client. |
| `answer_received_ms` | The complete application `result` event reaches the client. |
| `stream_complete_ms` | The SSE response finishes. |
| `server_speech_timing.synthesis_ms` | Application-measured wait for the actual voice service, including its cold load when applicable. |
| `server_speech_timing.alignment_ms` | Application-measured optional mouth-cue analysis time. With no Rhubarb configured, this is only the fallback check. |
| `audio_available_ms` | The entire generated WAV has reached the HTTP client. |
| `offline_pcm_onset_offset_ms` | Signal onset within that saved waveform, relative to its first sample. This is not relative to question submission. |
| `browser.*` and `acoustic_first_audible_ms` | Always `null` or `not_measured` in this harness. |

All transport times use the same client monotonic clock, starting immediately before the question request. Source and model hashes, application source files, full corpus hashes, exact server arguments, voice settings, initial health responses, and voice cold/warm state accompany the timings. The process monitor samples direct server RSS every second; summed RSS can count shared memory twice and does not capture every transient peak.

Complete WAV availability is a lower bound on first audible output for the application's current full-buffer playback path. The harness deliberately does not add a waveform offset to that value and label the sum as observed playback. There is no browser paint, Web Audio, microphone, input ASR, sound-card, or loudspeaker measurement here.

## Reproduce on a CPU host

Use a prepared application environment and the separate voice environment described in [the repository README](../README.md). The model cache must contain the exact Qwen revisions pinned in the command implementation. The harness stays offline and refuses to overwrite an existing output directory. It does not download weights.

The LLM metadata JSON must contain `path`, `bytes`, and `sha256` for the actual GGUF. The recorded run uses the project's existing Qwen3-4B Q4_K_M metadata with its official source-model revision and local requantization history. Both the declared byte count and a fresh full-file hash must match before any server starts.

```bash
python -m tools.demo_latency \
  --app-repo "$PWD" \
  --app-python "$PWD/.venv/bin/python" \
  --voice-python /absolute/inference-venv/bin/python \
  --llama-server /absolute/llama.cpp/llama-server \
  --llm-metadata /absolute/gguf-q4-model.json \
  --voice-cache /absolute/hf-cache \
  --reference-wav /absolute/reference/audio.wav \
  --reference-text /absolute/reference/reference.txt \
  --corpus-dir "$PWD/data/local/qna/elon-musk-v2" \
  --question 'How does tunnel diameter affect cost? In 12 words.' \
  --repeats 2 \
  --output "$PWD/artifacts/demo-latency/my-cpu-run"
```

Hydrate the full original Q&A corpus first using the corpus workflow. The requested 12-word answer keeps this CPU pilot bounded, but the harness preserves all generated text and does not silently shorten an overlong answer. Default budgets are 96 LLM tokens and 384 voice tokens; the voice backend currently does not expose its termination reason. A successful file transfer does not prove the spoken content is complete or faithful. The earlier one-sentence instruction produced a 43-word answer whose CPU voice request exceeded the application's 180-second timeout; that attempt is preserved separately.

Default CPU settings use four llama threads, two voice threads, a 4,096-token context, llama logical batch 128 and physical batch 64, BF16 Qwen voice, and a 180-second provider request timeout. These settings are recorded so results cannot be mistaken for a different model run or a target-device measurement. The first LLM request follows llama.cpp's enabled startup warmup. The first voice request includes lazy model loading and reference-prompt preparation. The second repetition reuses warm processes and the same question; it can benefit from a repeated prompt cache. Two repetitions are an execution pilot, not a latency distribution.

Run the entire command in one shell execution when the execution environment isolates network namespaces between tool calls. The orchestrator handles readiness, cleanup, bounded requests, partial failure preservation, and per-process logs.

## Measure the browser on its intended machine

Start the actual configured application and voice service, enable **Connected voice** and **Read new responses aloud**, submit a supported question, wait for speech to finish, and use **This turn → Export**. Retain the JSON together with the question, actual answer, model revisions, machine specifications, and whether the model processes were already warm.

`live-turn.v1` records a post-paint opportunity for the first visible text and an actual Web Audio analyser signal for provider speech. The latter detects digital playback samples, not sound in the room. Browser speech synthesis exposes only its start event and therefore leaves the PCM signal field empty. A microphone-to-loudspeaker latency claim requires an additional measurement setup.

The browser export is the appropriate artifact for target-machine live-demo latency. The server report supports diagnosing generation and transfer costs while that target measurement remains outstanding.

## Validation

`pytest tests/demo_latency` verifies framed SSE parsing, invalid stream rejection, citation-only stripping, silent-waveform handling, truncated and swapped WAV rejection, and the invariant that HTTP results cannot populate browser or acoustic timing fields. Its fixtures are clearly identified as unit tests and are never written into an actual model report.
