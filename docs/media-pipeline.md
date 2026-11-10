# Media preparation, observation, and reference voice

The repository includes executable tools for local media preparation, optional speech transcription, optional facial/body measurements, and an optional Qwen3-TTS service. Heavy models and recordings are kept outside Git. These tools extend the working research application; they do not imply that a model has been trained or that a faithful voice or motion clone has been validated.

## 1. Prepare a short, traceable clip

Install `ffmpeg` and `ffprobe` as system tools. Run the commands from the repository checkout with the project's Python environment active.

```bash
python -m tools.media prepare /path/to/interview.mp4 data/raw/interview-clip \
  --source-id lex-400-2023 \
  --source-url https://lexfridman.com/elon-musk-4-transcript/ \
  --source-timeline transcript_video \
  --start 2087 --duration 12 --sample-rate 16000
```

The input must be an existing local audio/video file. The tool does not log into services, bypass access controls, or download a supplied URL. The provenance URL is recorded without being fetched. Use `--audio-only` when video is unnecessary.

The new output directory contains:

- `audio.wav`: mono PCM16 audio, resampled to 16, 24, or 48 kHz.
- `video.mp4`: a video-only H.264 clip when the input contains video and video was requested; dimensions are limited to 1280 pixels in width.
- `manifest.json`: source identity, timeline, local input hash, derived artifact hashes, clock mapping, and review fields.

Clips are limited to 120 seconds per invocation. Existing output directories are rejected. Input media are never modified. FFmpeg is invoked with argument arrays and local-file protocols; URLs and playlists are not valid media inputs.

`--start` is relative to the local input. If that input is already an excerpt beginning at source time 600 seconds, add `--input-source-offset 600`. The manifest then records:

```text
source_seconds = clip_seconds + source_start_seconds
source_start_seconds = input_source_offset + start
```

Choose the clock from the exact media version. Lex podcast audio can include advertisements absent from the video/transcript timeline. Matching an episode title does not establish alignment. The Henry Ford transcript's production timecodes must not be treated as published-video elapsed time. See [source provenance](research/sources.md).

The initial review fields remain unknown. A reviewer should check the speaker, overlapping speech, background music, editorial cuts, source-clock alignment, and suitability of a proposed voice reference. The pipeline does not identify a person from their face or voice.

## 2. Produce a draft transcript

Use an isolated ASR environment to avoid dependency conflicts with the optional voice model.

```bash
pip install -r tools/media/requirements-asr.txt
python -m tools.media transcribe data/raw/interview-clip/manifest.json \
  --engine faster-whisper --model small --device cpu --compute-type int8 --language en
```

For forced word alignment:

```bash
pip install -r tools/media/requirements-whisperx.txt
python -m tools.media transcribe data/raw/interview-clip/manifest.json \
  --engine whisperx --model small --device cpu --compute-type int8 --language en
```

Both adapters use the publishers' Python APIs: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) provides iterable segments and optional word timestamps; [WhisperX](https://github.com/m-bain/whisperX) adds a language-specific alignment stage. First execution may download model weights. CUDA installations must match the relevant framework requirements; the CLI defaults to CPU/int8. No speed or accuracy claim has been measured for these adapters in this project.

The output `transcript.json` includes:

- Engine, model, package version, configuration, and the prepared audio hash.
- Segment and word `start`/`end` values relative to the prepared clip.
- Separate `source_start`/`source_end` values in the declared source clock.
- `speaker: null` and `review_status: draft`.

Words without alignment keep null timestamps. Diarization and speaker identity matching are not performed. Review the transcript before passing it to the main application's evidence importer. Automatic transcription is not automatically admitted to the trusted corpus. Re-running a transcription requires `--overwrite` explicitly.

## 3. Measure visible facial and body movement

```bash
pip install -r tools/media/requirements-vision.txt
python -m tools.media observe data/raw/interview-clip/manifest.json \
  --face-model /path/to/face_landmarker.task \
  --pose-model /path/to/pose_landmarker_lite.task \
  --sample-fps 10
```

Obtain model assets from the official [Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python) and [Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) documentation. No model binary is bundled or silently substituted.

The observer decodes video frames and uses FFprobe presentation timestamps, including variable-frame-rate timing. It invokes the MediaPipe Tasks video APIs with monotonic milliseconds. Face outputs contain normalized landmarks, blendshape coefficients, and the model's facial transform; optional pose outputs include normalized and world landmarks. These are geometric model estimates, not measured emotional states.

`observations.jsonl` stores sampled frames, relative/source times, and every detected face/pose. `observations-manifest.json` records model hashes, package version, sample count, and the observation-file hash. Detection indexes are local to a frame; they are not persistent identities. Multiple people must be reviewed and associated with speaking turns before building person-specific behavior examples. Missing detection means unobserved, not that a movement did not occur.

These measurements do not directly determine when a conversational avatar should use a gesture. That later step needs reviewed situations, explicit timing, target-rig mapping, and evaluation of the resulting animation.

## 4. Run the optional local Qwen3-TTS service

The service uses the official [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) Base-model API: it loads the configured model lazily, constructs one reference prompt using `create_voice_clone_prompt`, then reuses it through `generate_voice_clone`. The service does not fine-tune weights. A missing reference produces HTTP 503; it does not substitute a generic voice and call it a clone.

Create a separate Python 3.12 environment and install a PyTorch build suited to the target hardware, then:

```bash
pip install -r services/voice/requirements.txt
export BIAS_VOICE_REFERENCE_WAV=/absolute/path/to/reviewed-reference.wav
export BIAS_VOICE_REFERENCE_TEXT='The accurate words spoken in that specific reference clip.'
export BIAS_VOICE_DEVICE=cuda:0
export BIAS_VOICE_DTYPE=bfloat16
uvicorn services.voice.app:app --host 127.0.0.1 --port 8010
```

The reference must be a non-silent, mono PCM16 WAV at 16–48 kHz lasting 1–30 seconds. These are input bounds, not a speaker-quality guarantee. Reference identity and transcription accuracy must be checked separately. Reference files and transcripts are read only from server configuration, never from HTTP request paths or URLs.

The default `BIAS_VOICE_X_VECTOR_ONLY_MODE=false` requires an accurate reference transcript and uses Qwen's audio-and-transcript conditioning. A server operator can explicitly set `BIAS_VOICE_X_VECTOR_ONLY_MODE=true` to use the official speaker-embedding-only mode; this ignores `BIAS_VOICE_REFERENCE_TEXT` and removes only the transcript requirement. WAV validation, the generation lock, and input/output limits remain active. Clients cannot select this mode in an HTTP request. The option does not verify reference identity or establish voice similarity. `/health` exposes the active `reference_mode` without revealing the reference file or transcript.

Default model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`. `BIAS_VOICE_MODEL` may select a downloaded local model directory or another compatible Base model. The default attention implementation is `sdpa`; FlashAttention is optional rather than a mandatory import. Match PyTorch and torchaudio builds to the hardware; the isolated CPU smoke uses the official `torch==2.11.0+cpu` / `torchaudio==2.11.0+cpu` pair. See [inference validation](inference-validation.md) for the exact tested path and remaining limitations. MPS and target-machine performance require separate validation. Model loading and prompt creation occur on the first valid request, so first-response latency includes initialization. Restart the service after changing the reference or conditioning mode.

Connect the main app using:

```bash
export BIAS_TTS_BASE_URL=http://127.0.0.1:8010/v1
export BIAS_TTS_MODEL=qwen3-tts
export BIAS_TTS_VOICE=reference
```

If `BIAS_VOICE_API_KEY` is configured on the voice service, set the same value in the main app's `BIAS_TTS_API_KEY`. Bind to loopback for local use. The HTTP interface provides a narrow compatible subset:

```http
POST /v1/audio/speech
Content-Type: application/json

{"model":"qwen3-tts","voice":"reference","input":"Hello.","response_format":"wav","language":"English"}
```

Only WAV, the fixed `reference` voice, and speed `1.0` are supported. The model ID in a request is the service alias, not a downloadable model path. `language` defaults to `Auto`; `Korean` and `English` are accepted. The response is complete generated WAV audio, **not streamed audio** or a measured phoneme-alignment track.

The service bounds request bytes, input characters, generation tokens, and returned audio duration. It permits one in-flight generation; concurrent requests receive 429. A synchronous inference lock remains held if an HTTP caller disconnects. Model failures return a generic 502 without exposing local file paths or credentials. `/health` reports configuration/load state without triggering downloads.

## Validation status

The lightweight tests in `tests/media/` execute real FFmpeg clipping/resampling and validate source offsets, hashes, traversal boundaries, ASR normalization, missing-dependency failures, and the voice HTTP contract. `tests/inference/` also checks the default transcript requirement and server-only embedding-mode opt-in. Voice-model calls in these tests use explicit protocol fakes; they verify request handling, mode boundaries, and prompt caching, not synthesized speech or similarity.

The development environment also fetched a four-second excerpt from the publisher-linked [Lex #49 audio](https://lexfridman.com/elon-musk-2/) through ordinary public HTTPS access for a local extraction smoke test. Its acquisition record and audio are under ignored `data/raw/lex-049-media-smoke/`. The excerpt remains unreviewed and was used only by an explicit speaker-embedding-only inference smoke; it is not configured as the normal service or web application's voice reference. It is not included in Git or required for tests.

Actual Qwen3-TTS model loading, reference encoding, and waveform generation through `VoiceEngine` completed on CPU bfloat16: the short test produced 2.24 seconds of 24 kHz mono PCM16 audio in 21.91 seconds of generation. The reference identity was not verified, and intelligibility or voice similarity was not evaluated. A separate real Qwen3-0.6B text generation also completed but failed to emit required citation markers. See [the full inference validation record](inference-validation.md) for pinned revisions, timing, reproducible commands, and the earlier float32 loading failure.

Actual WhisperX/faster-whisper transcription, MediaPipe model inference, transcript-conditioned synthesis from a reviewed target-speaker reference, live voice-service playback through the app, and end-to-end avatar likeness have **not** been validated in this environment. Run those stages with the required assets and hardware, review the outputs, and record separate quality/latency results before making such claims.

```bash
python -m pytest tests/media tests/inference -q
```
