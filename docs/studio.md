# Conversation and avatar studio

## Start

Install Node 22.12+ (Node 24 is used in CI), Python 3.11+, and uv.

```bash
uv sync --extra dev
npm ci
npm run build
uv run bias serve
```

Open `http://127.0.0.1:8000`. Build before starting the API so the server mounts `apps/web/dist`.

For hot-reload development, run `uv run bias serve --reload` and `npm run dev` in separate terminals. The frontend on port 5173 proxies `/api` to port 8000.

## Conversation

The default **Source preview** answers using curated evidence notes. It is deliberately labeled and does not claim LLM generation. Enable a provider through the root `.env.example` and restart the API to use generated dialogue. The source drawer shows the original publisher URL, locator, and paraphrased evidence behind a response.

The conversation approach selector exposes three experimental conditions. The research lab makes real independent requests under each condition, with separate sessions and measured server timing. It does not invent a likeness score.

## Audio

Voice is off initially. Choose a generic browser voice or a configured speech service in Studio settings. Playback verifies a prepared audio package's SHA-256, uses the actual audio clock for its mouth cues, and separately retimes the estimated behavior plan to the waveform duration. With [Rhubarb configured](speech-timing.md), generated WAVs receive audio-recognized mouth cues. The status distinguishes that path from amplitude-based fallback. These cues are predictions, not forced-aligned phonemes. Browser speech has no accessible waveform and uses explicitly approximate articulation.

Listen to any completed answer or enable automatic playback. Starting another turn, switching pages, stopping audio, changing voice, or resetting the session interrupts old playback. Dictation uses the browser's speech-recognition capability when available and places the transcript in the composer for editing before submission.

## Avatar

The bundled portrait is a stylized procedural reference rig with independent facial controls and articulated head, neck, eyes, arms, and hands. It demonstrates the control pipeline; it is not a photorealistic scan or a verified likeness. Initial behavior cues are designed, not observed Musk mannerisms.

In **Research lab → Motion & rig calibration**, select **Load measured sample** to load a six-second face/head track extracted from native publisher video. It stays behind the explicit **Preview geometry without speaker attribution** control. The measured coefficients and source clock are preserved; this does not assign a personality trait or verify a visible person's identity. Other `avatar-clip.v1` files can be imported locally, paused, scrubbed, and replayed independently of conversation timing.

**Load arm-motion sample** loads a separate five-second pose-only track from a fixed single-person crop of a wider camera view. Chest, upper arms, forearms, and hand directions are measured; fingers and limb twist are not reconstructed. The inspector reports face and body provenance independently, so this pose-only replay leaves its facial animation labeled designed. Both included samples retain their geometry-preview review state.

Open **Rig inspection & calibration** to inspect the controls actually supported by the current asset. Manual pose preview can isolate a face channel, viseme, or joint. Import/export a rig-mapping JSON to set target names, neutral/range calibration, and local quaternion bases. Missing channels remain visible as unsupported instead of claiming full facial coverage. See [avatar calibration](avatar-calibration.md) for the data contract, source-review states, and measured sample limitations.

Studio settings accepts a local, self-contained GLB up to 50 MB. The file stays in the browser. The runtime supports morph-target and bone mappings; see [avatar retargeting](../apps/web/src/avatar/README.md). Assets with external resource references are rejected. A finished human asset needs its own calibration and visual evaluation.

## Validation

```bash
npm run check
npx playwright install chromium
npm run test:e2e
```

Unit tests cover deterministic cue sampling, real rig movement, mesh validity, GLB retargeting, stream protocol validation, audio cancellation, hash-bound timing, and recognition ownership. Browser tests exercise conversations, evidence, comparison, settings, measured-sample loading, calibration, and mobile overflow against the actual API. Speech browser tests use a clearly synthetic WAV/timing transport fixture; [real model HTTP runs](integration-validation.md) are separate. The CI workflow runs these tests and stores screenshots/traces. Check the workflow result rather than assuming that a configured test has passed.
