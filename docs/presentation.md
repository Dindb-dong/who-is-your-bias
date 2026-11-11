# Prepared presentation with Qwen3-TTS and MuseTalk

This path implements proposal requirements **R5–R7**. The prepared clip uses
new Qwen3-TTS Base speech and real MuseTalk 1.5 neural inference. Its script
introduces the A/B/C experiment; it is original project narration, not an
attributed interview answer. The live conversational path remains separate.

## Completed pilot

The [self-contained player](../artifacts/presentation/yaicon-intro-v1/index.html)
and [MP4](../artifacts/presentation/yaicon-intro-v1/presentation.mp4) were generated
on 2026-11-11. The complete package includes the generated speech, captions,
input portrait, exact script, checksums, model revisions, and execution evidence.

| Measured property | Result |
| --- | --- |
| Prepared speech | 12.24 seconds; 293,760 mono PCM16 samples at 24 kHz |
| Qwen3-TTS synthesis | Three new segments; 79.27 seconds including the cold first call; CPU bfloat16, two threads |
| MuseTalk render | All 306 frames generated at 25 fps; 491.07 seconds including feature/model preparation; CPU bfloat16, four threads |
| MuseTalk UNet parameters | 849,947,844 |
| Delivered video | H.264/AAC, 1280×720, 416,948 bytes |
| Audio/video duration | 12.24 seconds each; measured difference 0 seconds |
| Audio/video start offset | 0 seconds |
| Full media decode | Passed |
| Unprompted ASR | WER 3/33 = 9.09%; CER 1/168 = 0.60% |
| Human voice, lip-sync, and likeness ratings | Unrated |

The recognizer joins the synthesis spelling `A I` into `AI`, accounting for two
word-level edits; it also omits the article before “profile.” Its raw output,
word timestamps, model hash/revision, and normalization are retained. This
small automated check does not establish naturalness or voice resemblance.

An assistant inspected rendered frames at 0.8, 5.8, and 10.8 seconds for layout,
visible disclosure, captions, and changing generated mouth shapes. This is
separate from a human audiovisual synchronization judgment. The successful run
and earlier failed/partial probes remain distinct in `attempts.json`; the
initial cache-path loader failure was corrected before inference succeeded.

The video preparation rate was about **0.63 frames per second** on this CPU.
That is appropriate for a prepared pilot, not evidence of real-time video
generation. The later independent ASR check briefly shared the CPU with video
preparation, so these times are execution observations rather than controlled
cross-backend benchmarks.

## What is implemented

`tools.presentation.speech` synthesizes bounded script segments, preserves each
WAV, joins the exact samples with 200 ms gaps, and writes timed SRT/WebVTT
captions. It uses the existing reference-conditioned voice engine and its
configured immutable model revision. Nothing silently falls back to prerecorded
audio or system text-to-speech.

`tools.presentation.musetalk` encodes the generated speech with Whisper-tiny,
maps its 50 Hz hidden states to 25 fps audio windows, adds positional encoding,
encodes the full and lower-half-masked face with the SD VAE, runs the actual
MuseTalk 1.5 UNet at timestep zero, and decodes the predicted face latents. Model
files are pinned and checksum-verified before loading; PyTorch pickle weights
use `weights_only=True`.

The CPU adaptation supplies a visually checked fixed face crop and blends the
generated lower face with a geometric feather. It replaces the upstream
landmark detection and face-parsing dependencies; it does **not** replace the
neural lip-sync model. The portrait stays fixed, and this path does not infer
personal head or body movements. The mask and crop are recorded in each run.
This is a suitable bounded preparation path on the available CPU, not a claim
of parity with the upstream tracking/blending pipeline or its GPU performance.

`tools.presentation.package` accepts only completed matching runs. It rejects a
changed script, swapped audio/portrait/video, and a partial neural probe. The
package includes a persistent simulation disclosure, burned-in captions,
optional WebVTT captions, the script/source links, original generated WAVs,
generation reports, a browser player, and a fingerprint manifest. It verifies
H.264/AAC stream timing and decodes the complete output once.

## Pinned upstream inputs

| Component | Repository and immutable revision |
| --- | --- |
| Qwen3-TTS Base 0.6B | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` at `5d83992436eae1d760afd27aff78a71d676296fc` |
| Qwen speech tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` at `7dd38ad4e9bad454aae9cd937d0cd577604fe229` |
| MuseTalk 1.5 weights | `TMElyralab/MuseTalk` at `2bcb936e2fddb4d86db4c62fd45b387d0c061571` |
| MuseTalk upstream implementation | `TMElyralab/MuseTalk` at `0a89dec45a0192b824e3cf4daf96c239440c5ed8` |
| SD VAE | `stabilityai/sd-vae-ft-mse` at `31f26fdeee1355a5c34592e401dd41e45d25a493` |
| Whisper-tiny | `openai/whisper-tiny` at `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` |

The [official MuseTalk implementation](https://github.com/TMElyralab/MuseTalk/tree/0a89dec45a0192b824e3cf4daf96c239440c5ed8)
documents its 256×256 face region, audio conditioning, model files, and 25 fps
recommendation. Its reported GPU throughput is not a measurement of this CPU
adapter. The upstream license and attribution are retained in
[`LICENSE-MuseTalk.txt`](../tools/presentation/LICENSE-MuseTalk.txt).

The source portrait is the existing project's CC0 authored anatomical proxy;
see [human avatar provenance](humanoid-avatar.md). It is not a verified likeness
of the target person. The reference audio is the existing TED2013 engineering
candidate, with its pending listening/overlap review status preserved. See
[reference preparation](research/voice-reference.md) for the source clock and
review boundaries. Source/reference recordings and model weights are not
redistributed in the prepared package.

## Reproduction

Use a dedicated Python environment, install
`tools/presentation/requirements.txt`, and make FFmpeg/FFprobe available. The
caption/package stage uses the DejaVu Sans font. The verified CPU stack is
recorded with the delivered artifact. The renderer uses four CPU threads by
default in the command below; leave enough memory for the roughly 850 million
parameter UNet, VAE, and working tensors. Run large language-model benchmarks
separately to avoid contaminating latency measurements.

Fetch the three video model components and create a local hash lock:

```bash
python -m tools.presentation.fetch \
  --cache models/presentation-cache \
  --output models/presentation-lock.json
```

For an offline repeat, add `--offline`. The lock contains local model paths and
is intentionally not committed. Each fetch is pinned to the revisions above.

Prepare a reference WAV and accurate transcript using the existing reference
workflow. Set the voice service environment and synthesize the new script:

```bash
export BIAS_VOICE_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export BIAS_VOICE_REVISION=5d83992436eae1d760afd27aff78a71d676296fc
export BIAS_VOICE_CACHE_DIR=/path/to/complete/qwen/cache
export BIAS_VOICE_LOCAL_FILES_ONLY=true
export BIAS_VOICE_DEVICE=cpu
export BIAS_VOICE_DTYPE=bfloat16
export BIAS_VOICE_ATTENTION=sdpa
export BIAS_VOICE_REFERENCE_WAV=/path/to/reviewed/reference.wav
export BIAS_VOICE_REFERENCE_TEXT_FILE=/path/to/accurate/reference.txt
export BIAS_VOICE_THREADS=2
export BIAS_VOICE_SEED=42
export BIAS_VOICE_MAX_NEW_TOKENS=384
export BIAS_VOICE_MAX_AUDIO_SECONDS=30
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m tools.presentation.speech \
  --plan data/presentation/intro.v1.json \
  --output reports/local/prepared-speech
```

Render the included fixed portrait with the generated audio. Crop values are
pixels in the supplied 720×900 image; a new portrait requires its own crop
inspection. The output directory must be new so an earlier run is preserved.

```bash
python -m tools.presentation.musetalk \
  --portrait artifacts/presentation/yaicon-intro-v1/portrait.png \
  --audio reports/local/prepared-speech/speech.wav \
  --model-lock models/presentation-lock.json \
  --crop 266 250 454 460 --threads 4 --timeout 1800 \
  --output reports/local/prepared-video

python -m tools.presentation.package \
  --plan data/presentation/intro.v1.json \
  --speech-dir reports/local/prepared-speech \
  --render-dir reports/local/prepared-video \
  --portrait artifacts/presentation/yaicon-intro-v1/portrait.png \
  --output reports/local/prepared-presentation
```

`--max-frames` exists only for a bounded neural probe. Such a run is explicitly
`partial_probe` and cannot be packaged as a completed presentation.

Run the unprompted recognition check in the existing separate ASR environment
(`tools/voice_evaluation/requirements-asr.txt`). This reuses the hash validation,
language hints, word timestamps, lexical diagnostics, and empty human ratings
from the established voice evaluator:

```bash
python -m tools.presentation.inspect \
  --plan data/presentation/intro.v1.json \
  --speech-dir reports/local/prepared-speech \
  --model-directory /path/to/faster-whisper-small/snapshot \
  --revision 536b0662742c02347bc0e980a01041f333bce120 \
  --threads 1 --output reports/local/prepared-asr
```

## Read the verification correctly

Container checks compare video/audio duration and start offsets within one
frame or AAC packet. Full decoding catches corrupted media. These checks do
not establish perceptual lip synchronization, intelligibility, naturalness,
speaker resemblance, or preference. Independent human ratings remain empty
until people supply them. ASR, when recorded, is an automated intelligibility
check and is kept separate from listening judgments.

The preparation process is offline and expensive on this CPU. Playing the
result needs no model execution. Live Q&A should use the existing streaming
answer and speech path; use measured question-to-text, question-to-audio, and
turn-completion timings on the actual demonstration computer to choose its
settings.
