# Local model inference validation

This document records an execution smoke test on **2026-11-10 UTC**, separate from the application's protocol tests and browser checks. Model inference, citation compliance, speaker identity, and perceptual similarity are separate outcomes. Reports and generated audio stay in ignored `reports/local/`; downloaded weights stay outside the repository.

These are the initial direct-engine measurements. Subsequent real HTTP/SSE integration, transcript-conditioned speech, ASR, mouth-cue recognition, and video observations are recorded separately in [integration validation](integration-validation.md). Historical limitations below describe the initial runs and are not silently upgraded by later results.

## Language model: real generation completed; citation compliance failed

The actual [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) weights at commit `c1899de289a04d12100db370d81485cdf75e47ca` generated a response on CPU. The runner used the repository's `Corpus`, `Retriever`, `build_messages`, and `validate_citations`; it did not use a canned model response. The prompt used the behavior condition, English, no conversation history, and four reviewed evidence records. Qwen thinking was disabled through the official chat-template option.

| Measured item | Result |
| --- | --- |
| Device / precision / threads | CPU / bfloat16 / 4 |
| Actual parameter count | 596,049,920 |
| Published safetensors download | 1,503,300,328 bytes |
| Input / generated tokens | 840 / 29 |
| Cached model load | 2.71 seconds |
| First visible text chunk | 9.12 seconds after generation began |
| Generation duration | 19.89 seconds |
| Output rate | 1.46 tokens per second |
| Completion | EOS reached, within the 160-token and 300-second bounds |
| Citation markers | **None emitted; citation compliance failed** |

The question was `Why did Tesla start with an expensive sports car?`. The generated response was:

> Tesla started with an expensive sports car to address market entry and scale, framing the first model as a stepping stone toward more affordable electric vehicles.

This is newly generated model output, not a quotation attributed to Musk. Inspection against the retrieved records found the general explanation consistent with the curated summaries of the expensive initial model, limited manufacturing scale, and a path toward cheaper vehicles. This was an assistant review of one example, not an independent human evaluation. The answer is brief and omits the more specific capital/proceeds explanation. It provided no citation despite the prompt's requirement. `validate_citations` therefore returned empty valid and invalid ID lists; the application's existing engine would classify this output as `unverified`. Successful execution does not make this a successful grounded-answer or persona-fidelity evaluation.

The retrieved IDs were `elon-tesla06-sequence`, `elon-tesla16-low-volume`, `senate-2007-roadster-limits`, and `elon-ted22l-production`. The run's sources/evidence SHA-256 was `a70a8d34881467bfd236d3f109d07fe4c88900f4dd9608a5143302334f1c31c0`; `reports/local/llm-smoke.json` contains the complete metadata and output.

This run used the expanded research corpus delivered in the dependent corpus-expansion PR. Running the same smoke on the earlier foundation corpus selects fewer records and changes the prompt and token count. The recorded corpus hash identifies the exact input snapshot; a later explicit-review-state migration also changes file bytes without changing these evidence texts.

An earlier float32 attempt exited with code **137 during model loading**, before a `model_loaded` or text event. The cause was not independently established. Available memory observed afterward was approximately 3.15 GB, and the CPU exposed AVX512 BF16 support; a single bfloat16 retry completed. The failed-attempt metadata is retained as `reports/local/llm-float32-attempt.json`. These observations are not a measured peak-memory requirement or a cross-machine latency benchmark.

This test called Transformers directly. It does **not** establish that an OpenAI-compatible model server is connected to the running web application, or that the browser completed a live model turn. The application still requires explicit provider configuration for that path.

## Voice model: real reference-conditioned WAV generated

The bounded voice smoke uses [`Qwen/Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) at commit `5d83992436eae1d760afd27aff78a71d676296fc`, the actual `services.voice.engine.VoiceEngine`, and a short statement: `This is a research simulation.` It uses CPU bfloat16, four threads, at most 96 generated codec tokens, and a 300-second generation deadline. The base model and bundled speech tokenizer together publish 2,511,637,364 safetensors bytes.

The four-second reference came from a publisher-linked Lex #49 audio excerpt prepared by the media pipeline. Its manifest remains **unreviewed**: speaker identity, single-speaker status, source-clock alignment, and reference suitability are not verified. No reference transcript was invented. The smoke explicitly opts into Qwen's speaker-embedding-only mode. This is not a verified Musk voice reference and provides no evidence of Musk voice similarity.

The real model loaded, encoded the reference, and generated a non-silent WAV through `VoiceEngine.synthesize`. The synchronous service lock was released afterward and the engine reported ready. No model or waveform fake participated in this run.

| Measured item | Result |
| --- | --- |
| Device / precision / threads | CPU / bfloat16 / 4 |
| Model and speech-tokenizer download | 103.50 seconds |
| Model load | 12.68 seconds |
| Reference encoding | 1.01 seconds |
| Generation including waveform decode | 21.91 seconds |
| Complete first `synthesize` call, excluding download | 35.61 seconds |
| Generated audio | 2.24 seconds, 24 kHz, mono PCM16 |
| Audio frames | 53,760 |
| Normalized peak / RMS | 0.707 / 0.108 |
| Generation real-time factor | 9.78; slower than real-time on this CPU |

The run's result is recorded in `reports/local/voice-smoke.json`, and the generated output is `reports/local/voice-smoke.wav` with SHA-256 `52d4d2811844293064c0fc124ba2fb2a9b127a4eb5d15d9e67b8bd36a999bc3a`. The smoke validates waveform shape, PCM encoding, duration, sample rate, and non-silence. Listening evaluation, intelligibility, perceptual similarity, and naturalness were **not** assessed. The default transcript-conditioned mode was not exercised with a real reviewed reference. The upstream waveform API did not expose EOS status, so natural completion and absence of truncation have not been independently confirmed.

The run used the actual engine directly; the HTTP route remains covered separately by protocol-fake tests. It does not establish live reference-voice playback through the web application. Neither the repository's normal service environment nor the application's voice settings were changed to use the unreviewed excerpt. Upstream imports printed missing-SoX and missing-FlashAttention warnings; this tested synthesis path completed without installing either package.

## Reproduce the bounded smokes

Use a separate Python 3.12 environment. Install a matching PyTorch/torchaudio build for the hardware before the optional voice requirements. The CPU pair successfully imported in this environment was `torch==2.11.0+cpu` and `torchaudio==2.11.0+cpu`, obtained from the official PyTorch CPU index. An initial combination of torch 2.15.0+cpu and torchaudio 2.11.1 failed because torchaudio expected `libcudart.so.13`; no torchaudio 2.15.0+cpu wheel was available in the checked official index.

```bash
python3.12 -m venv .venv-inference
.venv-inference/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.11.0+cpu' 'torchaudio==2.11.0+cpu'
.venv-inference/bin/python -m pip install -e . \
  -r tools/inference/requirements.txt -r services/voice/requirements.txt
.venv-inference/bin/python -c \
  'import torch, torchaudio; print(torch.__version__, torchaudio.__version__)'
```

The tested Python packages also included Transformers 4.57.3, qwen-tts 0.1.1, accelerate 1.12.0, huggingface-hub 0.36.2, NumPy 2.5.4, and SoundFile 0.14.0. The full package versions are recorded automatically per run. Network requests use normal public publisher endpoints and inherit the existing proxy configuration. No authentication token, alternate model mirror, or remote model code is required for these pinned checkpoints.

Run the language model from the repository root:

```bash
PYTHONPATH=src:. .venv-inference/bin/python -m tools.inference.llm \
  --cache-dir /path/outside/repository/hf-cache \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --dtype bfloat16 --threads 4 --max-new-tokens 160 --timeout-seconds 300
```

The default precision is float32 for portability; the shown bfloat16 setting matches the completed CPU test and requires suitable hardware. Download time is separate from generation time. The first visible chunk metric measures decoded text emission, not the first internal token. Sampling used seed 42, temperature 0.7, top-p 0.8, and top-k 20, following the publisher's non-thinking example; output may still vary with package versions and hardware. A completed script status records execution, while citation fields report the separate grounding result.

For normal transcript-conditioned voice testing, supply a reviewed reference WAV and an accurate transcript file:

```bash
PYTHONPATH=src:. .venv-inference/bin/python -m tools.inference.voice \
  --cache-dir /path/outside/repository/hf-cache \
  --revision 5d83992436eae1d760afd27aff78a71d676296fc \
  --reference /path/to/reviewed-reference.wav \
  --reference-text-file /path/to/accurate-reference-transcript.txt \
  --dtype bfloat16 --threads 4 --max-new-tokens 96 --timeout-seconds 300
```

The execution-only smoke without a reference transcript instead adds `--x-vector-only` and omits `--reference-text-file`. Pass `--reference-manifest /path/to/manifest.json` when a preparation manifest exists: its audio SHA-256 must match, and its supplied review status is preserved. This opt-in does not verify a speaker or improve the review status. Do not interpret an unreviewed-reference output as a target-person voice result.

The voice runner's generation deadline uses POSIX signals and is intended for Linux/macOS execution in the main thread. It reports complete WAV output; it does not implement audio streaming or phoneme alignment. The upstream waveform API does not expose whether generation stopped at EOS, so a token-capped result must still be reviewed for truncation. The service's synchronous single-generation lock and normal output validation remain active during this test.

## What remains to validate

- Citation compliance over a representative held-out question set, including unsupported questions and adversarial source content.
- Factual entailment of each cited claim; valid identifiers alone are insufficient.
- A configured provider's live HTTP/SSE turn through the application and its actual audio playback clock.
- Reviewed reference identity and accurate transcription, then transcript-conditioned voice synthesis and human listening evaluation.
- Speech intelligibility, target-voice similarity, naturalness, latency distribution, memory use, and Korean-language behavior.
- Actual ASR and MediaPipe inference, learned situation-to-gesture behavior, phoneme alignment, and likeness of a production avatar asset.

The protocol tests remain independent of these model downloads:

```bash
python -m pytest tests/media/test_voice_service.py tests/inference -q
```

Those tests use synthetic reference audio and explicit protocol fakes. They verify HTTP request boundaries, prompt reuse, the default transcript requirement, server-only embedding-mode opt-in, invalid-reference rejection, and lock behavior; they do not synthesize speech.
