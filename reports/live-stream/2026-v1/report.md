# One bounded live neural execution

The real `/api/chat/live` path completed one cold base Qwen3-0.6B plus
Qwen3-TTS-12Hz-0.6B-Base request on November 11, 2026. The strict recorder
validated normal termination, final text identity, all PCM chunk sequences and
sample offsets, complete segment hashes and stable service identities.
The measured response was produced by the configured neural language and
speech services. No fixture response or synthetic PCM was used.

## Observed result

Question: “How does tunnel diameter affect cost? In 12 words.”
The application used lexical retrieval over `data/qna/elon-musk-v1`; the
confirmed evidence ID was `qa-ted-2017-cost`. It used no trained text adapter.

| Observation | Result |
| --- | ---: |
| First client text receipt | 7,093.753 ms |
| Final text result receipt | 9,888.115 ms |
| First client PCM receipt | 52,715.145 ms |
| All audio totals validated | 52,721.938 ms |
| Normal protocol `done` | 52,721.974 ms |
| Server first PCM ready | 52,695.772 ms |
| Server total | 52,699.863 ms |
| Voice service elapsed time | 42.816622 s |
| Delivered audio | 5.52 s; 132,480 mono frames at 24 kHz |
| Validated PCM chunks / segments | 65 / 1 |
| Peak sampled sum of the three services' RSS | 4,855,287,808 bytes |

These values have different declared measurement origins. Client timings start
immediately before the HTTP request; server metrics retain their server
origin. The process memory figure is a sampled concurrent sum, not a sum of
each process's independent high-water mark or a whole-host maximum.

Both services reported `model_loaded=false` before and `true` after the
request. The language service observed 596,049,920 base parameters, CPU
bfloat16, SDPA attention and two Torch threads. The voice service also used CPU
bfloat16, SDPA and two Torch threads. The request had a 180-second deadline,
96 text tokens and 384 speech tokens as upper bounds. The language service
used greedy decoding; speech used the documented seeded sampling settings.
The exact configuration and package versions are retained in
[execution.json](attempt-02/execution.json), and health/timing observations in
[http.json](attempt-02/http.json).

The parent executor's final four-case model comparison ran concurrently using
two CPU threads. Training completed before this neural request. The launch
note was broader than the ultimately observed overlap; that correction is
explicit in [run-context.json](run-context.json), while the original launch
receipt remains unchanged. This was not an isolated performance measurement.

## Model and reference identity

| Component | Immutable revision |
| --- | --- |
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen/Qwen3-TTS-12Hz-0.6B-Base | `5d83992436eae1d760afd27aff78a71d676296fc` |
| Qwen/Qwen3-TTS-Tokenizer-12Hz | `7dd38ad4e9bad454aae9cd937d0cd577604fe229` |

Every downloaded snapshot file has a SHA-256 receipt. The four tokenizer files
actually embedded in the TTS snapshot exactly match the four independently
pinned tokenizer files. The language service's observed base-checkpoint
manifest digest is
`052d8c0b8c31f61d79bcbbcca49c85414118746986773d57fb6a56f58e5a0c09`;
its tokenizer contract digest is
`483ee203c64a06c6b7e569aaaaae7aebc3db459b029892ce79e537d578904ee6`.
The detailed receipts bind model weights, tokenizer configuration, code files
and the actual corpus bytes used in the run.

The reference was restored from the publisher's TED2013 native MP4 using the
documented 21.45–31.280042-second interval. The reacquired opening and extracted
WAV exactly match the prior canonical source/audio hashes. Its reference text
comes from the four original publisher cues and matches the separately
documented publisher-derived reference text. The reference audio SHA-256 is
`f20f897fde67d14d2fb77d42d98b0890bf3c1d1fd121b41879dfd448f0d47e4c`;
the text SHA-256 is
`0e75c3c62963a9a06932a87113f14d8ac206d1cad4fbb409a484b552fcc5f83d`.
[reference-restoration.json](reference-restoration.json) binds source identity,
publisher attribution, interval, prior evidence and current file hashes.

The reference remains an **engineering candidate**. No new ASR or listening
review was performed, and there is no new claim about acoustic speaker
isolation, intelligibility, perceptual quality or named-speaker similarity.
Prior ASR evidence is cited as prior work, not relabeled as a new measurement.

## Failed attempts remain visible

The first model acquisition targeted the default home-directory cache and
failed with `ENOSPC`. Only this attempt's incomplete TTS blobs were removed,
including a 40,235,008-byte partial weight file. The parent cleared reproducible
package cache; the successful retry used an explicit cache on the workspace
volume. Model environments and existing LLM checkpoints were preserved.
[acquisition.json](acquisition.json) retains the failed attempt and exact
successful acquisition timestamps/revisions.

The first HTTP setup question asked about rocket reuse, for which this selected
lexical corpus returned no matching evidence. The application correctly used a
source-preview response. The recorder refused to label it language-model
inference: language stayed unloaded and no PCM was received. Voice lazy loading
had begun for the preview and the harness stopped all children. Its failed
[execution receipt](attempt-01/execution.json) and
[HTTP receipt](attempt-01/http.json) are retained. A preflight evidence check
now prevents this configuration mistake before service startup.

## What this does and does not establish

The completed attempt establishes real local model execution through the
application's committed-text/PCM protocol. The final received PCM digest is
`69e7f795dbb1c790e9bdb138e6a878412a3d5ca69de99c1469e643cd8239a311`.
The complete-WAV hash reported by the server is retained separately; a receiver
of PCM-only chunks cannot independently reconstruct unseen WAV headers.
Reference media remains local. The recorder stores generated text/audio hashes
and metadata, not raw generated waveform or base64 payloads.

PCM arrived after the final text in this one example, so it does not establish
overlapping text/audio delivery. This is per-sentence full-WAV synthesis,
not native neural audio-token streaming. A complete returned waveform does
not prove TTS end-of-speech-token termination because the backend does not
expose that reason. Browser text visibility, playback onset/completion,
acoustic output, avatar behavior and spoken completion remain null/unmeasured.
There is no warm-latency distribution, target-device performance conclusion,
speech training or human quality result.

The focused validation command passed **112 tests** across held-out isolation,
live protocol behavior and benchmark recording. Synthetic tests establish
corruption, identity and cancellation handling; the separate actual model
receipts establish this execution. A reviewer then strengthened the launcher's
future tokenizer preflight from matching subset to exact file-map equality.
The four-file match recorded in this run already satisfies the stronger check;
no new neural execution was performed for that preflight-only change.

The machine-readable [summary](summary.json) and
[receipt manifest](receipt-manifest.json) index the measurements and retained
files. Reproduction instructions are in
[the live benchmark documentation](../../../docs/live-stream-benchmark.md).
