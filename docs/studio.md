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

Voice is off initially. Choose a generic browser voice or a configured speech service in Studio settings. Playback uses actual audio time for server-generated audio, and retimes the estimated behavior plan to the waveform duration. Mouth movement is driven by audio amplitude; it is not a phoneme-aligned viseme system. Browser speech has no accessible waveform and uses explicitly approximate articulation.

Listen to any completed answer or enable automatic playback. Starting another turn, switching pages, stopping audio, changing voice, or resetting the session interrupts old playback. Dictation uses the browser's speech-recognition capability when available and places the transcript in the composer for editing before submission.

## Avatar

The bundled portrait is a stylized procedural reference rig with independent facial controls and articulated head, neck, eyes, arms, and hands. It demonstrates the control pipeline; it is not a photorealistic scan or a verified likeness. Initial behavior cues are designed, not observed Musk mannerisms.

Studio settings accepts a local, self-contained GLB up to 50 MB. The file stays in the browser. The runtime supports morph-target and bone mappings; see [avatar retargeting](../apps/web/src/avatar/README.md). Assets with external resource references are rejected. A finished human asset needs its own calibration and visual evaluation.

## Validation

```bash
npm run check
npx playwright install chromium
npm run test:e2e
```

Unit tests cover deterministic cue sampling, real rig movement, mesh validity, GLB retargeting, stream protocol validation, audio cancellation and recognition ownership. Browser tests exercise conversations, evidence, comparison, settings, and mobile overflow against the actual API. The CI workflow runs these tests and stores screenshots/traces. Check the workflow result rather than assuming that a configured test has passed.
