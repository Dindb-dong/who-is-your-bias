# Local Qwen3-TTS voice on Apple Silicon

This opt-in provider uses the existing persona/RAG backend and studio. It starts Qwen3-TTS when the first English sentence is committed. The conversation can keep writing while voice synthesis runs. All sentences share one browser audio clock. Longer answers can still be generated in text-only mode. The browser buffers audio adaptively to reduce gaps at sentence boundaries. The regular Listen button generates one WAV from the completed text. Qwen MLX uses the existing EMAGE integration when speech-driven gestures are enabled.

## Start the local services

On an Apple Silicon Mac, use Python 3.11, FFmpeg, and the Node/uv setup in the [main README](../README.md).

```sh
qwen_env="$HOME/.venvs/yaicon-qwen-mlx"
python3.11 -m venv "$qwen_env"
"$qwen_env/bin/python" -m pip install -r tools/realtime/requirements-qwen-mlx.txt
"$qwen_env/bin/hf" download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --revision 5d83992436eae1d760afd27aff78a71d676296fc --local-dir /path/to/local-qwen-tts

"$qwen_env/bin/python" tools/realtime/qwen_mlx_voice.py \
  --model /path/to/local-qwen-tts --reference-dir /path/to/reviewed-voice-reference
```

Provide a short, reviewed recording suitable for synthesis as `reference.wav`, its exact UTF-8 transcript as `reference.txt`, and a `reference.json` object containing `sha256` and `transcript_sha256` with the two files' SHA-256 digests. Run `shasum -a 256 reference.wav reference.txt` to calculate them. The worker verifies both before loading the reference. Keep the files outside Git. Use only a reference you are authorized to use and set the corresponding persona binding. The worker listens on `127.0.0.1:8108` and warms the model before reporting ready. The local demo's +1 semitone adjustment is applied during streaming with tempo preserved.

Set these in your local `.env`, then start the studio as shown in the main README.

```dotenv
BIAS_TTS_PROTOCOL=qwen-mlx
BIAS_TTS_BASE_URL=http://127.0.0.1:8108
BIAS_TTS_MODEL=qwen3-tts-mlx
BIAS_TTS_VOICE=reference
BIAS_TTS_PERSONA_ID=jensen-huang
```

The voice worker is separate from the answer model. Point the studio at the team's existing LLM endpoint or run a local OpenAI-compatible MLX model. The original Mac trial used `mlx-community/Qwen3-4B-Instruct-2507-4bit`. To reuse it, set `BIAS_LLM_BASE_URL=http://127.0.0.1:8103/v1`, `BIAS_LLM_MODEL=default_model` and `BIAS_LLM_MAX_TOKENS=280` if the MLX server was launched with that local model alias. The source-evidence preview works with no LLM configured.

Open the presentation mode with `?present=1` and choose provider voice with autoplay. Use `&gesture=emage` if your [EMAGE runtime](speech-gesture.md) is configured. English speech can accompany Korean captions when the studio is configured for translation. Korean text replies remain available with this English voice worker.

## Verify playback

Try one- and two-sentence English questions. The timing panel reports first audible PCM on the browser clock and counts interruptions in the playback queue. The server's first PCM timing measures when bytes were emitted, before buffering and speaker output. Run the live-stream regression tests with `uv run pytest -q tests/test_live_streaming.py`. This tests early audio delivery, exact packet order, and preservation of the generated answer if the voice worker fails. Qwen checkpoints, reference recordings, generated audio and avatar weights are separate local inputs.
