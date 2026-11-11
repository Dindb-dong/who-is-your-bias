# Quality completion and recovery

Status snapshot: 2026-11-11. PRs #7, #8, and #9 remain open. The baseline browser
suite completed, screenshots were inspected, and the recovery fixes below are
included on `feat/human-motion-studio`. The [PR #9 checks](https://github.com/Dindb-dong/who-is-your-bias/pull/9/checks)
and PR description record the latest pushed head's result; a baseline result
does not establish that a later commit passed.

## Review and merge dependencies

| PR | Scope | Current base | Head at initial recovery |
| --- | --- | --- | --- |
| [#7](https://github.com/Dindb-dong/who-is-your-bias/pull/7) | Grounding diagnostics and measured prompt comparison | `main` | `69db99a7359a3d611c79a2f59715c8368c65f982` |
| [#8](https://github.com/Dindb-dong/who-is-your-bias/pull/8) | Language propagation and measured voice/listening outputs | `feat/grounding-quality` | `afbb9581524ec2f37533eb465a9ea4b0a359fe89` |
| [#9](https://github.com/Dindb-dong/who-is-your-bias/pull/9) | Human avatar, measured motion, and browser coverage | `feat/voice-quality` | `3c10b3b627716e972bd09a535b3ee5dddd49f52c` |

The full implementation is on `feat/human-motion-studio`. When merging is
requested, merge #7 first, retarget #8 to `main` and merge it, then retarget #9 to
`main`. Recheck diffs and CI after each base change. This handoff does not merge
or approve the three PRs.

## Browser evidence and recovery fixes

- [Baseline push run 42943071040](https://github.com/Dindb-dong/who-is-your-bias/actions/runs/42943071040)
  passed **16/16 browser cases without retries** at `3c10b3b`; the complete job
  took 9m 20s. Artifact `13822460220` has SHA-256
  `367c9c18a632ad8dec74880accf533eacdaa754a4192bee378ced7e167080055`.
- The matching [PR run 42943116261](https://github.com/Dindb-dong/who-is-your-bias/actions/runs/42943116261)
  also succeeded, with 13 first-attempt passes and 3 mobile retries. Its failure
  logs show a GLB still loading after the 5 s assertion budget and two cumulative
  60 s test timeouts. The suite now gives real-rig readiness 15 s, full CI
  scenarios 120 s, and the CI job 20 minutes. Assertions and retry count are retained.
- Actual desktop/mobile screenshots show the textured human, portrait/full-body
  framing, measured face/arm previews, and calibration controls without visible
  overflow. These are browser frames, not the Blender build preview.
- Selecting the procedural portrait after a failed initial GLB now acknowledges
  readiness. The added desktop/mobile regression verifies that both loading
  indicators and the error clear. Configured browser coverage is **18 cases**.
- Screenshot review exposed an existing retrieval false positive: `more` plus
  `power` let a tunnel-cost note answer an electric-car efficiency question.
  The narrow stopword correction makes that unsupported question abstain and
  select no motion. The source-motion browser scenario now uses a supported
  energy-comparison query and asserts the relevant evidence text.
- The correction passed 69 focused retrieval/motion/grounding tests; all 20
  legacy and 100 published comparison prompt hashes reproduce unchanged.
  Graded retrieval remains **39/39**, with no detected split leaks and two
  separately ungraded injection cases. Historical model outputs and ratings
  were not regenerated or rewritten.
- Full local verification after these fixes passed **361 Python tests**, **102
  frontend tests**, TypeScript checking, and the production build.
- The listening test verifies all four saved WAV hashes/metadata, then exercises
  play, pause, and completion on the shortest English sample per viewport, plus
  JSON export. Muted automated playback and synthetic form entries are not human
  listening or perceptual quality scores.

## Application defaults and measured limits

- **Grounding:** compact v1 remains the live prompt; `grounded_v2` is an explicit
  evaluation candidate. Generated content remains semantically unreviewed even
  when every sentence-like unit has a valid citation marker.
- Five studies preserve 120 executions: 99 complete model responses, one
  incomplete response, and 20 deterministic application gates. For the 11 paired
  complete 4B responses, compact→v2 citation coverage improved 4→8 and supported
  relevant answers 4→5. The full v2 run has 9/12 complete citations and 6/12
  supported answers. Review was nonblind and model-assisted; these are development
  judgments, not held-out accuracy or persona fidelity. See [grounding quality](grounding-quality.md).
- **Voice:** English short and three-sentence cases had ASR WER 0/5 and 0/33;
  Korean CER was 2/58. The number case retains literal WER 5/21, with differences
  at number formatting. Warm CPU synthesis took about 3.5–3.8 times audio duration.
  Four generated WAVs are preserved; human listening, naturalness, likeness, and
  backend EOS completion remain unverified. See [voice evaluation](voice-evaluation.md).
- **Avatar/motion:** Research human is the default; procedural portrait and local
  GLB upload remain available. The bundled CC0 anatomical reference is not a
  verified likeness. Conversation movement defaults to authored motion; source
  geometry requires opt-in. The two-event library has 672 sampled / 648 accepted
  frames and 40 primitives; only 12 training-event primitives enter preview
  selection. Speaker association and a learned personal gesture policy remain
  unverified. See [human avatar](humanoid-avatar.md) and [motion library](motion-library.md).

## Start the studio

```bash
uv sync --locked --extra dev
npm ci
npm run build
BIAS_MOTION_CATALOG=data/motion/elon-musk-ted-v1.json uv run bias serve
```

Open `http://127.0.0.1:8000`. Enable **Studio settings → Conversation movement →
Source geometry preview** to opt into eligible measured windows. Configure a
language provider separately for generated conversation; see [language runtime](language-runtime.md).

Voice is a separate optional service. In its PyTorch/Qwen environment, configure
the cached checkpoint and local reference using [voice evaluation](voice-evaluation.md),
then run `python -m uvicorn services.voice.app:app --host 127.0.0.1 --port 8092`.
Set the main API's `BIAS_TTS_BASE_URL=http://127.0.0.1:8092/v1`,
`BIAS_TTS_MODEL=qwen3-tts`, `BIAS_TTS_VOICE=reference`, and
`BIAS_TTS_LANGUAGE_HINT=true`; load `.env` explicitly with
`uv run --env-file .env bias serve` and retain `BIAS_MOTION_CATALOG` there or in the environment.
Without rerunning synthesis, open the saved listening page at
`tools/voice_evaluation/results/qwen06-cpu-2026-11-11-listening/listener/index.html`.

## Verification commands

```bash
uv run pytest
uv run bias evaluate --fail-on-regression --output reports/local/retrieval.json
npm run check
npm run test:e2e
```

Current-head workflow links and test totals are recorded in the PR description.
Keep original listening ratings unrated until an actual listener supplies them.
