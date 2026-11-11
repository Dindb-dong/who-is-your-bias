# Voice quality evaluation

The previous integration smoke established that one five-word English sentence
could pass through reference-conditioned synthesis, lip-sync analysis, and an
independent recognizer. It did not establish sustained intelligibility, Korean
output, perceptual similarity, or conversational latency. This workflow keeps
those questions separate and records failed or incomplete runs as well as
successful WAV responses.

## What is measured

`tools/voice_evaluation/cases.json` declares four fixed cases before generation:
a short English disclosure, three English sentences, three Korean sentences,
and an English sentence pair containing numbers and pauses. Each case specifies
both the TTS language and the independent recognizer language. The expected text
is never passed to ASR as an initial prompt, prefix, or hotword list.

The synthesis report records request wall time, response-header latency, WAV
duration, real-time factor (`request seconds / audio seconds`), signal levels,
and SHA-256 hashes. The first case includes lazy model loading and reference
prompt preparation; subsequent cases reuse the same process and cached prompt.
The report captures service health before the run and after each successful
case so cold/warm behavior and runtime settings are visible. Header latency is
not a streaming time-to-first-audio measurement.

For English, the primary lexical diagnostic is word error rate (WER):
`(substitutions + deletions + insertions) / reference words`. It can exceed 1.
For Korean, the primary diagnostic is character error rate (CER), using NFC
normalization and retaining letters/numbers while ignoring spaces and
punctuation. Korean word spacing remains visible in secondary WER. Neither
metric rewrites spoken numbers to digits; recognizer formatting can therefore
produce lexical errors even when a person would hear equivalent content.

Missing expected suffixes, substantial deletions, and excess repeated n-grams
are screens for follow-up listening. They are not proof of audio truncation or
repetition: the recognizer can make mistakes. Tail signal energy is also only
descriptive. The Qwen public inference API returns waveforms without an EOS or
generated-token completion record, so every synthesized case explicitly reports
`termination_reason: not_exposed_by_backend`. A valid, nonempty WAV does not
prove that the model said the complete requested text.

## Actual CPU measurement: 2026-11-11

The [measured result](../tools/voice_evaluation/results/qwen06-cpu-2026-11-11.json)
records real Qwen synthesis followed by a separate multilingual faster-whisper
process. All four declared cases returned valid 24 kHz mono PCM16 WAVs. The
model used CPU bfloat16, four threads, seed 42, an explicit 512-token codec cap,
and the audio-and-transcript reference mode. No human listening was performed.

| Case | Requested units | Audio | HTTP wall time | Wall/audio ratio | Independent ASR result |
| --- | ---: | ---: | ---: | ---: | --- |
| Short English, cold process | 5 words | 1.52 s | 18.584 s | 12.23 | 0/5 word edits |
| Three English sentences, warm | 33 words | 10.24 s | 38.320 s | 3.74 | 0/33 word edits |
| Three Korean sentences, warm | 58 NFC characters | 9.680 s | 33.950 s | 3.51 | 2/58 character edits, CER 3.45% |
| English numbers, warm | 21 words | 6.640 s | 25.075 s | 3.78 | 5/21 literal word edits, WER 23.81% |

For Korean, the recognizer returned `연구형` for requested `연구용` and
`시뮬레이신` for requested `시뮬레이션`. These are two character substitutions;
this automated measurement cannot assign them to the synthesizer or recognizer
without listening evidence. Korean secondary WER was 2/17, or 11.76%.

For the number case, every observed literal token difference occurs at numeric
formatting: `nine thirty` → `9.30`, `twelve` → `12`, and `twenty four` → `24`.
The raw WER is retained. This explains the observed string differences, without
replacing the measurement with an invented human intelligibility score.

No case triggered the missing-suffix or excess-repeat screens, and no generated
WAV had clipped PCM samples. EOS/completion remains unknown because the backend
does not return that information. The four-case synthesis run took **116.187 s**
in total. The voice process reached **2.653 GiB** high-water RSS and was stopped
before ASR; this is one process's peak, not simultaneous deployment memory.
The separate ASR run took **9.261 s**, including **1.671 s** model loading. These
results support further testing of understandable multi-sentence speech, while
the CPU wall/audio ratios remain above real time.

The reference was re-extracted from the official TED2013 source at native
21.45–31.280042 seconds. Its restored WAV lasts 9.830042 seconds but is **not
byte-identical** to the previous candidate. A new, unprompted multilingual ASR
decode matched all 32 normalized publisher-caption tokens. That reference check
took 5.517 seconds alongside permitted language-model/light vision work, so its
timing is not an isolated benchmark. The restored artifact remains a candidate:
single-speaker acoustic review, perceptual quality, and likeness are unverified.
The earlier 44.563-second smoke used the main API and Rhubarb and a different
reference file, so it is not a controlled latency comparison with this direct
voice-service run.

The [repository listening bundle](../tools/voice_evaluation/results/qwen06-cpu-2026-11-11-listening/README.txt)
preserves the four actual synthesized WAVs (1.29 MiB combined), an offline review
page, and a separate answer key. Open `listener/index.html` after checkout; no
model server is required to play these saved outputs. This bundle contains only
generated audio. It omits the publisher's reference recording and therefore
does not offer a reference-similarity score. The original local bundle, including
that comparison recording, remains in ignored `reports/local/`.

Every score remains null and every listening attestation remains false. These
are ready review artifacts, not a completed listening study. The recovered
synthesis, ASR, execution records, and generated WAVs were checked against their
saved hashes. JavaScript syntax is checked, but browser verification remains
incomplete: the available Chromium cannot start because the execution environment
rejects its process-singleton socket operation.

## Reproducible voice runtime

The voice service resolves a requested model/revision to its local Hugging Face
snapshot before loading Qwen or its processor. `BIAS_VOICE_LOCAL_FILES_ONLY=true`
is the default; download the model separately. An unavailable cache produces a
service-unavailable response instead of initiating a download from an HTTP
speech request. Revisions, when supplied, must be immutable commit hashes.

The measured 0.6B model revision is
[`5d83992436eae1d760afd27aff78a71d676296fc`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/5d83992436eae1d760afd27aff78a71d676296fc).
The optional dependency remains `qwen-tts==0.1.1`. Runtime settings are exposed
in `/health` without reference text, local paths, or credentials:

```bash
export ORT_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BIAS_VOICE_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export BIAS_VOICE_REVISION=5d83992436eae1d760afd27aff78a71d676296fc
export BIAS_VOICE_CACHE_DIR=/absolute/path/to/huggingface/hub
export BIAS_VOICE_LOCAL_FILES_ONLY=true
export BIAS_VOICE_DEVICE=cpu
export BIAS_VOICE_DTYPE=bfloat16
export BIAS_VOICE_ATTENTION=sdpa
export BIAS_VOICE_THREADS=4
export BIAS_VOICE_SEED=42
export BIAS_VOICE_DO_SAMPLE=true
export BIAS_VOICE_TEMPERATURE=0.9
export BIAS_VOICE_TOP_K=50
export BIAS_VOICE_TOP_P=1.0
export BIAS_VOICE_REPETITION_PENALTY=1.05
export BIAS_VOICE_MAX_NEW_TOKENS=512
export BIAS_VOICE_MAX_AUDIO_SECONDS=60
export BIAS_VOICE_X_VECTOR_ONLY_MODE=false
export BIAS_VOICE_REFERENCE_WAV=/absolute/path/to/candidate/audio.wav
export BIAS_VOICE_REFERENCE_TEXT_FILE=/absolute/path/to/candidate/reference.txt
# Use the separate environment containing PyTorch and services/voice/requirements.txt.
python -m uvicorn services.voice.app:app --host 127.0.0.1 --port 8092
```

The production backend resets the configured Torch seed before each generation
while its single-generation lock is held. The same sampling settings are passed
explicitly to the talker and sub-talker. A seed improves repeatability on the
same stack; it does not promise identical bytes across library versions,
hardware, or different numeric precision. These controls are not a claim that
one sampling setting is universally better.

Qwen's [official inference implementation](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py)
states that `non_streaming_mode=false` simulates streaming text input. This API
still returns the whole waveform. The service keeps `streaming: false`; neither
that setting nor the language-model SSE stream makes the voice path stream.
The [official model documentation](https://github.com/QwenLM/Qwen3-TTS) lists
English and Korean among its supported languages. Support is not a quality
guarantee for a particular reference or cross-language output.

In the main application's environment, enable `BIAS_TTS_LANGUAGE_HINT=true`
when `BIAS_TTS_BASE_URL` points to this local Qwen service. The application then
maps request language `en` to `English` and `ko` to `Korean` for both
`/api/speech` and `/api/speech/prepare`. The simple endpoint accepts an optional
language; omitting it preserves the provider's automatic behavior. The prepared
endpoint retains its existing `en` default. The flag defaults to `false`, so
generic OpenAI-compatible speech endpoints receive their existing four-field
payload without this nonstandard extension.

## Run synthesis, then independent ASR

With the voice server available, run from the repository checkout in the core
environment. The runner accepts only loopback origins, never follows redirects,
does not inherit environment proxies, limits each WAV response to 20 MB, and
writes an atomic checkpoint after each case. Output directories are never
reused. `BIAS_VOICE_API_KEY`, if present, is used as a bearer credential and is
not included in the report.

```bash
python -m tools.voice_evaluation synthesize \
  --base-url http://127.0.0.1:8092 \
  --condition qwen06-seed42-cap512 \
  --output reports/local/voice-quality-qwen06 \
  --per-case-timeout 300 --total-timeout 900
```

The deadline includes HTTP response streaming. On a timeout, invalid response,
or busy/error response, later cases are not admitted. Disconnecting a client
does not stop the underlying synchronous model generation. Stop the voice
server and wait for its process to exit before starting ASR on a memory-limited
CPU machine. The ASR workflow does not share the TTS model's weights or process.

Install `tools/voice_evaluation/requirements-asr.txt` in a separate environment
and cache the multilingual recognizer explicitly. The checkpoint used here is
[`Systran/faster-whisper-small`](https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120)
at revision `536b0662742c02347bc0e980a01041f333bce120`. Its stored CTranslate2
weights are loaded using CPU `int8` compute, four threads, one worker, beam size
five, no VAD, and `condition_on_previous_text=false`. See the
[official faster-whisper usage](https://github.com/SYSTRAN/faster-whisper).

```bash
ORT_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 \
python -m tools.voice_evaluation asr \
  reports/local/voice-quality-qwen06/synthesis.json \
  --model-directory /absolute/path/to/cached/multilingual-small-snapshot \
  --model-repo Systran/faster-whisper-small \
  --revision 536b0662742c02347bc0e980a01041f333bce120 \
  --cpu-threads 4 \
  --output reports/local/voice-quality-qwen06/asr.json
```

ASR validates each synthesis WAV hash, resamples the verified PCM audio through
FFmpeg to mono 16 kHz, and passes a NumPy waveform to faster-whisper. This avoids
the file-decoder incompatibility observed with PyAV 19 in the earlier reference
workflow. Output includes recognized text, word timings, per-case metrics,
recognizer versions, the model-weight hash, and actual elapsed time. Model load
time is recorded separately. Automatic recognition cannot measure voice
likeness, naturalness, acoustic speaker overlap, or a human listener's judgment.

## Blind listening bundles and review records

Create a bundle from one or more measured synthesis reports:

```bash
python -m tools.voice_evaluation bundle \
  reports/local/voice-quality-qwen06/synthesis.json \
  --reference /absolute/path/to/candidate/audio.wav \
  --output reports/local/voice-quality-listening --seed 42
```

The bundle has two directories. Share only `listener/` with reviewers. Its local
`index.html` contains opaque randomized sample IDs, audio controls, a rating
rubric, and an export button. Model condition labels, expected transcripts, and
the randomization seed are kept in `private/answer-key.json`. A single-condition
bundle blinds the condition label but is not an A/B comparison. No human review
has happened merely because a bundle exists.

All scores begin `null`, each sample begins `listened: false`, and
`human_review_performed` begins `false`. A reviewer records the heard text,
language understanding, intelligibility, naturalness, unexpected repetition,
and audible cutoff. Optional reference similarity is a subjective acoustic
comparison requiring explicit listening to the hash-bound comparison clip.
The workflow does not infer identity or automatically approve a reference.

The page exports a local JSON review file. Validate it before recording results:

```bash
python -m tools.voice_evaluation review \
  reports/local/voice-quality-listening/listener/manifest.json \
  /absolute/path/to/listening-review.json \
  --output reports/local/voice-quality-listening/review-summary.json
```

Validation requires a reviewer identifier and timestamp, explicit personal
listening attestation, exact sample and reference hashes, and scores only for
listened samples. Language-dependent scores require understanding that language.
A reference-similarity score requires listening to the comparison recording.
Partial reviews remain partial. The attestation is self-reported; a validator
cannot prove that a person actually listened. It never changes the original
reference's candidate/review status.

## Verification and artifact scope

Protocol and metric tests use synthetic signals and fake recognizers; they are
not model-quality measurements:

```bash
python -m pytest tests/voice_evaluation tests/test_voice_quality_runtime.py \
  tests/media/test_voice_service.py tests/inference/test_voice_reference_modes.py \
  tests/test_voice_reference_file.py tests/test_tts_language_hint.py \
  tests/test_speech_timing.py tests/test_providers_and_api.py -q
```

Actual run summaries belong in `tools/voice_evaluation/results/`. The four
generated WAVs from the reported run are also retained there in the explicit
listening bundle so the saved measurements can be inspected after checkout.
Additional synthesis runs, complete ASR outputs, local review bundles, reference
recordings, and full publisher text stay in ignored `reports/local/` or
`data/local/`. Review bundles must stay unrated unless a person supplies a real
review. The previous reference and integration measurements are documented in
[reference validation](../tools/references/validation.json) and
[integration validation](integration-validation.md).
