# Live call (Sophon-Realtime)

**Status: experimental, not committed to the default flow.** A **Live call**
button appears in the conversation composer when `BIAS_SOPHON_URL` is set.
You talk, and the avatar answers out loud in real time.

## How it works

```
browser mic (PCM16, 16 kHz, 100 ms frames)
  → WS /api/realtime (this API; holds the key)
  → Sophon-Realtime /v1/realtime (listens, decides turns, speaks)
  → reply audio (PCM16, 24 kHz) → browser playback on its own clock
                                  └→ EMAGE streaming (docs/speech-gesture.md) → gesture frames
```

- **Model.** Sophon is a full-duplex speech model, so there is no separate
  speech-to-text, language model or TTS step. Barge-in (`audio.cancelled`)
  stops the reply and its gestures.
- **Instructions.** At the start of a call, the API sends the persona's
  description, notice and style rules, plus every reviewed training-split
  source note. `bias.sophon.build_instructions` builds this text. The Jensen
  corpus is about 11k characters, small enough to send whole.
- **Not source-checked.** The model writes the replies itself. They are not
  validated against citations the way the text conversation is, and the UI
  labels them that way.
- **Upstream.** The hosted API and a self-hosted engine (vgi, VESSL) speak the
  same protocol. To switch, change `BIAS_SOPHON_URL` (and the key).

```
BIAS_SOPHON_URL=wss://api.adventists.ai/sophon/v1/realtime
BIAS_SOPHON_MODEL=sophon-realtime-tag
BIAS_SOPHON_API_KEY=...            # .env only; never commit
BIAS_SOPHON_VOICE=ryan_en
```

## Measured on 2026-11-21

Hosted `sophon-realtime-tag`, laptop CPU for EMAGE, Research human rig, and a
fake microphone playing an English question.

| Step | Time |
| --- | --- |
| Call connected | 2.9 s after the click |
| End of question to first reply audio | 0.5 s with the API alone. About 1.3 s in the browser, including turn detection. |
| Reply audio delivery | About 6× faster than real time (29 s of audio in 4.7 s) |
| EMAGE taking over from designed movement | 1.9 s after the reply started, then to the end without a stall |

The reply drew on the COMPUTEX 2025 source note in the instructions.

## Limits

- **Trial account.** One concurrent call, 10 minutes per call, and a 90 s idle
  timeout. The API allows one call at a time.
- **Data leaves the machine.** Microphone audio goes to the hosted service
  when it is used.
- **Voice cloning.** The hosted API supports cloning (`POST /sophon/v1/voices`
  with consent). It is not wired in yet.
- **Tools and per-question retrieval.** The hosted Realtime endpoint has no
  tool calling, so retrieval cannot run per question yet.
- **Echo.** Browser echo cancellation is on, but headphones avoid the avatar
  interrupting itself.
