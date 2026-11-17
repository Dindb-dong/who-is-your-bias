# Jensen studio quickstart

Use this guide to run the current Gaussian face with the articulated body and
compare the initial and expanded gesture candidates. It applies to `main` after
[PR #25](https://github.com/Dindb-dong/who-is-your-bias/pull/25), merged at
`973627da0dc668c8544d60fd092da2e88bdb2307`. Commands below run from the repository root.

After installing the files and starting the app, open:

**[Current Gaussian face + expanded gestures](http://127.0.0.1:8000/?head=gaussian&headBundle=jensen_a_c1c3_240k_sh&gesture=learned&gestureBundle=jensen-gesture-basis-expanded)**

## 1. Update and install dependencies

Use Node **22.12+**, Python **3.11+**, and `uv`. Starting with your work saved:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
uv sync --locked --extra dev
npm ci
```

## 2. Install the two private asset packages

### Gaussian face

The supplied `gaussian-head.zip` contains a `gaussian-head/` directory with six
files. Extract it **into `apps/web/public/models/`**:

```bash
uv run python -m zipfile -e /path/to/gaussian-head.zip apps/web/public/models
```

The final directory must be `apps/web/public/models/gaussian-head/`, with these
three pairs directly inside it:

| Basename: both `.json` and `.bin` are required | Role |
| --- | --- |
| `jensen_a_c1c3_240k_sh` | Current default, SH-baked face |
| `jensen_a_sh` | Earlier SH-baked comparison |
| `jensen_a` | Earlier DC-only comparison |

Avoid an extra nested `gaussian-head/gaussian-head/` directory. Keep the original
names and capitalization. The supplied ZIP's SHA-256 is
`8f33e6b37da63674c43331bd87d899f92b3f6b39e9e111713618c7fc62a24793`;
individual file hashes are in [the local asset inventory](local-only-assets.md).

### Learned gestures

Extract `jensen-gesture-candidates-2026-11-17.zip` to a separate local directory:

```bash
uv run python -m zipfile -e /path/to/jensen-gesture-candidates-2026-11-17.zip data/local/gesture-candidates
```

Copy these two files from its `models/gesture/` directory into
`apps/web/public/models/gesture/`:

| Filename | Candidate |
| --- | --- |
| `jensen-gesture-basis.json` | Initial fitted arm/finger model; default learned selection |
| `jensen-gesture-basis-expanded.json` | Expanded fitted model; selected explicitly |

Keep both filenames. The face ZIP does not contain these gesture models. The
gesture ZIP also contains datasets, checkpoints and reports; those stay outside
the web asset directory. These private assets remain ignored by Git.

## 3. Start the studio

### Built app: one server

Install the assets **before building**, then run:

```bash
npm run build
uv run bias serve --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. The API serves the built `apps/web/dist/` directory,
including the copied model files. The first build provisions a pinned local face
detector and needs network access if that detector is not already cached.

### Development: two terminals

Terminal 1:

```bash
uv run bias serve --host 127.0.0.1 --port 8000 --reload
```

Terminal 2:

```bash
npm run dev
```

Open the Vite URL, normally `http://127.0.0.1:5173/`. Its `/api` requests proxy to
`http://127.0.0.1:8000`; model files come directly from `apps/web/public/`.
Use port **5173** instead of **8000** in the preview URLs below. If 5173 is busy,
use the URL Vite prints. Keep the API on 8000 unless you also change the proxy.

### Persona and optional providers

Jensen is the default when `BIAS_CORPUS_DIR` and `BIAS_PERSONA_SNAPSHOT` are unset.
For an existing `.env` with older settings, set both explicitly:

```dotenv
BIAS_CORPUS_DIR=data/personas/jensen-huang
BIAS_PERSONA_SNAPSHOT=data/snapshots/jensen-huang-2026-11-12.json
```

Load that file explicitly when starting the API:

```bash
uv run --env-file .env bias serve --host 127.0.0.1 --port 8000
```

Use `--reload` as well for development. Merely creating `.env` does not load it.
Check `http://127.0.0.1:8000/api/persona`: `id` should be `jensen-huang`.
`BIAS_TTS_PERSONA_ID` binds a voice; it does not select the conversation persona.

No LLM or TTS server is required for an initial playback check: the API can return
labeled source notes and the browser can read them. For generated dialogue or a
speech provider, use [`.env.example`](../.env.example). **Connected voice** requires
the configured TTS endpoint/model and `BIAS_TTS_PERSONA_ID=jensen-huang` for a
matching voice, or `generic` for a generic voice.

## 4. Select the face and gesture candidate

Parameters are case-sensitive. Supply **basenames**, without `.json`, `.bin`,
directory separators or URLs. Reload the page after changing the query.

| Parameter | Default when omitted | Behavior |
| --- | --- | --- |
| `head=gaussian` | Gaussian face off; Jensen's stylized portrait is the default | Requests the Gaussian face on the Jensen source-fitted body. Requires the Jensen persona and that built-in avatar, with no custom GLB selected. |
| `headBundle=<basename>` | `jensen_a_c1c3_240k_sh` | Chooses the `.json`/`.bin` pair under `/models/gaussian-head/`. Does not enable the face by itself. Letters, digits, `_` and `-` are accepted; invalid syntax falls back to the default. A valid but missing name produces a loading error. |
| `gesture=learned` | Learned gestures off; designed motion is the default | Enables **Learned gesture preview** and the geometry-preview gate. Requires Jensen, Behavior mode, active voice playback and reduced motion off. Works independently of `head`. |
| `gestureBundle=<basename>` | `jensen-gesture-basis` | Chooses `/models/gesture/<basename>.json` when learned preview is enabled. Accepts 1–80 letters, digits, `_` or `-`. Invalid names produce an error; this parameter alone does not enable gestures. |

Useful comparisons on the built app:

| View | URL |
| --- | --- |
| Current face + expanded learned gestures | `http://127.0.0.1:8000/?head=gaussian&headBundle=jensen_a_c1c3_240k_sh&gesture=learned&gestureBundle=jensen-gesture-basis-expanded` |
| Same face + initial learned gestures | `http://127.0.0.1:8000/?head=gaussian&gesture=learned` |
| Same face + designed movement | `http://127.0.0.1:8000/?head=gaussian` |
| Default stylized portrait + designed movement | `http://127.0.0.1:8000/` |

To compare an earlier face, keep the desired gesture parameters and change only
`headBundle` to `jensen_a_sh` or `jensen_a`.

## 5. Make it speak and move

1. Open **Conversation**. Set **Conversation approach** to
   **C · Topic & situation Q&A**. This is the code's `behavior` mode; there is no
   separate button labeled “Behavior.”
2. In **Studio settings → Avatar asset**, keep **Jensen source-fitted candidate**
   selected for the Gaussian composite. Set **Camera framing → Upper body & hands**.
   Both are selected automatically on a fresh `head=gaussian` page.
3. Turn off **Reduce automatic motion** for this comparison. It starts from the
   operating system's reduced-motion preference; learned gestures are suppressed
   while it is on.
4. In **Conversation movement**, enable **Learned gesture preview**, or use one of
   the learned URLs above. The checkbox also enables the required geometry-preview
   mode. A server-side source-motion catalog is not required for this checkbox.
5. In **Voice playback**, choose **Built-in voice** for a generic browser voice,
   or **Connected voice** when your configured provider is available. Enable
   **Read new responses aloud** for automatic playback.
6. Ask a question, for example “What are AI factories?”, then click **Listen** on
   the completed response if automatic playback is off. **Listen** also uses the
   browser voice when **Text only** is selected. Learned gestures run while speech
   plays; generating or displaying text alone does not start them.

The square audio control is **Stop audio**. Stopping voice stops learned gesture
sampling and lets the body settle. **Stop response** cancels a generating response;
**New conversation** or leaving the Conversation page cancels playback and source
replay too.

If a measured source clip is playing, it takes precedence over learned gestures.
Source clips run once on their own clock, so stopping audio alone may leave that
replay running. Choose **Designed motion**, start **New conversation**, or leave
the page to stop it. A **No eligible source movement matched this answer** notice
describes source-clip selection; it does not mean the learned model failed.

## 6. Understand the three separate assets

| Asset | Purpose and expected behavior |
| --- | --- |
| Gaussian face `.json` + `.bin` | Trained face appearance, FLAME data and recorded expression coefficients. The current known bundle can replay its recorded training-split expressions without an expression driver. |
| Gesture basis `.json` | Fitted arm/finger motion, selected separately and sampled during speech. It does not supply the face, semantic facial controls or a learned torso policy. |
| `<headBundle>.expression-driver.json` | Optional calibrated mapping from named facial controls to this exact face bundle's FLAME expression space. The loader looks beside the face files and verifies the bundle binding. |

The supplied six-file face ZIP has no expression driver. **Recorded expression
preview; semantic calibration pending** is therefore an expected status, not a
missing-face error. Recorded lip-expression residuals may remain; replay does not
establish that eyebrows or expressions match the current sentence's meaning.
Unrecognized bundles without a verified replay split use a static reference.
See [the paired expression workflow](../tools/gaussian_head/expression/README.md)
for the separate calibration inputs. Installing these packages does not establish
visual acceptance or completed semantic-expression training.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| **Photoreal face unavailable** or a model 404 | Confirm the exact `.json`/`.bin` pair under `apps/web/public/models/gaussian-head/`, without an extra nested directory. Check spelling and case. The body remains available on a face-load failure. |
| Files exist, but the API-served app still cannot load them | Re-run `npm run build` after adding or moving files, then restart the API and reload the page. Port 8000 serves `dist`, not the current `public` directory. |
| Face disappears after changing avatar settings | Select **Jensen source-fitted candidate** again. **Stylized portrait**, **Research human** and a custom GLB do not enable this Gaussian composite. |
| Wrong subject, or no Jensen gesture control | Check `/api/persona`, correct the corpus and snapshot settings together, and restart the API. Face and gesture controls are Jensen-bound. |
| **Install the local Jensen gesture model…** | Check `/models/gesture/jensen-gesture-basis.json` or `/models/gesture/jensen-gesture-basis-expanded.json` on the same origin as the studio. Use the matching basename without an extension. Rebuild if serving from port 8000. |
| Model loaded, but arms do not move | Use **C · Topic & situation Q&A**, start **Listen** or automatic voice playback, enable **Learned gesture preview**, and check reduced motion and source-clip precedence. Use **Upper body & hands** framing. |
| `.expression-driver.json` returns 404 | The default optional driver may be absent; the known bundle uses labeled recorded-expression replay. A malformed or mismatched driver is an error, so use only one fitted for the exact bundle. |
| **Connected voice** is disabled | Configure the TTS endpoint, model and explicit voice-persona binding, then restart the API. **Built-in voice** can be used without a TTS server. |
| Vite shows **Service offline** | Start the API on port 8000. The frontend's `/api` proxy does not start the Python server. |
| Build/dev fails while provisioning the face detector | Restore access to its pinned model download and retry `npm run prepare:vision`, then repeat build/dev. |

## References

- [Gesture fitting, model hashes and installation](jensen-gesture-training.md)
- [Expanded dataset and measured comparison](jensen-gesture-expansion.md)
- [Private asset inventory](local-only-assets.md)
- Implementation: [URL/avatar selection](../apps/web/src/App.tsx),
  [settings labels](../apps/web/src/components/SettingsDialog.tsx),
  [gesture loading](../apps/web/src/hooks/useLearnedGesture.ts),
  [performance precedence](../apps/web/src/avatar/performance.ts),
  [API environment settings](../src/bias/config.py), and
  [Vite port/proxy](../apps/web/vite.config.ts).
