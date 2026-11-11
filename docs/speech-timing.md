# Audio-driven mouth timing

The optional speech path now connects the main application, a configured TTS server, an audio recognizer, and the browser's actual audio transport. `/api/speech/prepare` generates one audio clip, analyzes its mouth cues, and returns an opaque clip URL with SHA-256-bound timestamps. The browser downloads and verifies the exact bytes before playback. The existing `/api/speech` binary endpoint remains available.

## Configure and run

Install an official [Rhubarb Lip Sync release](https://github.com/DanielSWolf/rhubarb-lip-sync/releases) for the host platform. Keep its executable beside its bundled resource files. The repository does not download or execute a binary on behalf of an HTTP client. Set the absolute executable path in the main application's environment:

```dotenv
BIAS_TTS_BASE_URL=http://127.0.0.1:8010/v1
BIAS_TTS_MODEL=qwen3-tts
BIAS_TTS_VOICE=reference
BIAS_REQUEST_TIMEOUT=180
BIAS_LIPSYNC_BIN=/absolute/path/to/Rhubarb-Lip-Sync/rhubarb
BIAS_LIPSYNC_TIMEOUT=30
BIAS_LIPSYNC_THREADS=2
```

Run the optional voice service as described in [the media pipeline](media-pipeline.md). The service accepts either `BIAS_VOICE_REFERENCE_TEXT_FILE` pointing to a UTF-8 transcript or `BIAS_VOICE_REFERENCE_TEXT`, alongside the reference WAV. Supplying both is rejected. The reviewed-reference workflow is documented in [voice reference preparation](research/voice-reference.md); its review ledger must distinguish a candidate from a listened-to, approved reference.

Start the main application with its environment, open Studio settings, choose the connected speech provider, and select **Listen** on a response. A prepared clip uses audio-derived timing when analysis succeeds; the status shows approximate movement when it falls back. Optional automatic playback uses the same preparation and verification path.

For a repeatable HTTP smoke against the running API:

```bash
uv run python -m tools.inference.speech_package \
  --base-url http://127.0.0.1:8000 \
  --text 'This is a research simulation.' \
  --output reports/local/voice-http.wav \
  --report reports/local/voice-http.json
```

This runs the real configured service. Its JSON report separates successful HTTP audio retrieval, mouth-cue availability, and unassessed voice similarity. It does not claim to have exercised browser playback or human listening. Generated audio and reports remain local and ignored by Git.

## Timing and interpretation

The adapter uses [Rhubarb's documented JSON format and recognizers](https://github.com/DanielSWolf/rhubarb-lip-sync). English input uses PocketSphinx with the requested text as a recognition hint. Other supported response languages use the language-independent phonetic recognizer. Neither path is described as forced alignment. The output carries `timing_basis: audio_recognition` and the recognizer's A–H/X labels; the renderer maps those predictions to supported mouth shapes.

Each `speech-timing.v1` document includes an audio SHA-256, the decoded duration, ordered non-overlapping segments, a label set, and analysis provenance. Times are milliseconds relative to the generated audio. Its transport clock is `audio.currentTime * 1000`, separate from the scaled behavior-plan clock. Browser speech synthesis has no waveform or mouth-cue data and continues to use explicitly estimated articulation.

Imported rigs can expose distinct visemes; a generic rig approximates their appearance. The calibration inspector lists unsupported controls. Additional timing sources can use the same contract with `forced_alignment`, `tts_model`, or `manual`, but no such data is invented by this adapter. Correct byte binding and timestamp playback do not establish perceptual lip-sync quality or identity similarity.

## Resource and failure behavior

Analysis accepts PCM16 WAV, mono or stereo, 8–96 kHz, at most 180 seconds and 20 MB. Subprocess arguments never go through a shell. A bounded deadline kills and reaps analysis on timeout; cancellation also reaps the process before its temporary files are removed. English dialogue is passed through a UTF-8 file. Audio in another supported playback format is preserved with an `unsupported_audio` timing status.

Prepared clips live in an in-memory cache: at most eight clips, 40 MB total, and 120 seconds per clip, with a maximum of two active preparation calls per API process. Old entries are evicted to maintain these bounds. Responses use `Cache-Control: no-store`; process shutdown clears the cache. An expired clip returns 404 and must be synthesized again. Requests should reach the same API process for preparation and download; multiple workers need shared storage or sticky routing. The default development deployment uses one process.

The frontend rejects unexpected or external clip URLs, malformed timing, hash mismatches, empty audio, and downloads exceeding 20 MB. A late response after Stop or navigation cannot revive a canceled playback. If Rhubarb is missing, fails, or exceeds its deadline, the valid audio still plays with measured amplitude-based mouth movement. Model generation and analysis happen before playback, so this path adds preparation latency and does not stream speech.
