# Explicit Jensen runtime candidate

This opt-in configuration binds the restored original-dialogue-v4 corpus, the
three-event motion catalog and the source-fitted avatar. The application default
is unchanged. It delivers original interview excerpts in **evidence-preview**
mode; it does not run a language model or generate a new persona response.

The separate runtime snapshots retain the frozen M6 corpus digest and cutoff.
They label the actual preview, optional voice and avatar roles without changing
the condition-neutral M6 comparison snapshot. Historical interviews do not
establish present-day positions. The optional retrieval candidate did not meet
the frozen retrieval acceptance gate, the motion remains `geometry_preview`,
the avatar has not passed human likeness review, and the 16-step voice adapter
requires separate listening evaluation. None is silently promoted here.

## Launch

Restore the durable artifacts described by the replication and modality reports,
install the locked application dependencies and build the web app with
`npm ci` and `npm run build`. From the integrated repository:

```bash
PYTHONPATH=src:. .venv/bin/python -m tools.runtime_checks verify \
  --artifacts /absolute/path/to/jensen-2026-11-13 --allow-candidates

PYTHONPATH=src:. .venv/bin/python -m tools.runtime_checks serve \
  --artifacts /absolute/path/to/jensen-2026-11-13 --allow-candidates
```

Open `http://127.0.0.1:8000`, then select **Jensen source-fitted candidate** in
Studio settings. The launcher checks the exact corpus, snapshot, motion catalog,
source assets and built avatar bytes before starting. It does not inherit a
language model, speech provider or persona snapshot from unrelated environment
settings.

Add `--with-voice --inference-python /absolute/path/to/.venv-inference/bin/python`
to start the separately pinned candidate voice service as a child process. This
loads the actual model on the first speech request and needs sufficient memory;
HTTP health before synthesis is not a completed model-load or quality check.
The voice runtime binding and reference verification are owned by
`tools.voice_evaluation.serve_candidate`. The launcher keeps API, provider and
test clients in the same process tree because this execution environment does
not share loopback networking across independent command sessions.

## Frozen functional protocol

`protocol.json` was written before execution. Its real TCP checks cover source
inspection, unchanged citation content, four same-session turns, current and
private-information controls, endorsement, Korean preview, SSE and disconnect
cleanup. Timings in this configuration measure evidence delivery and API work,
not language-model inference. The optional short speech probe is an integration
check excluded from the frozen voice quality comparison.

```bash
PYTHONPATH=src:. .venv/bin/python -m tools.runtime_checks smoke \
  --artifacts /absolute/path/to/jensen-2026-11-13 --allow-candidates \
  --output /absolute/path/outside-the-repository/runtime-preview-check
```

The runner starts the real API on an ephemeral loopback TCP port, executes the
protocol, writes raw HTTP evidence and a source-text-free report, and stops its
servers. The output directory must be new and outside the repository because
the raw trace contains original interview answers. Its report does not claim
that a quickly completed preview was interrupted neural inference.

The first actual HTTP attempt completed 90 assertions; seven failed because the
runner compared serialized URLs with Pydantic URL objects and expected streaming
citation markers to remain in the final display text. Direct inspection found
the original citation text, source identity, URL serialization and locator
unchanged. The runner now compares canonical URL strings, marker identities and
all display content after explicit citation-marker removal. The initial run is
retained as `preview-attempt-1.json`; no shared API or corpus change was made.

## Measured preview result

The corrected real HTTP run passed **91 of 91 assertions**. The result and exact
imported API module hashes are recorded in `preview-result.json`. The raw traces
from both runs were uploaded and freshly retrieved with matching SHA-256;
`artifact-receipts.json` records the stable artifact identifiers and hashes.

| Check | Observed result |
| --- | --- |
| Corpus and source inspection | Nine train sources, 554 original exchanges, 410 factual-eligible records; source records and returned citation text, locators and URLs match the frozen corpus. |
| Avatar asset | Actual HTTP response is 2,530,256 bytes with SHA-256 `dea0c921fb7d5668527bd8ee9082f382b14e8d300c43936842e15356dc7d90cd`. |
| Motion | Six clips from three events; 580 frames, 36 preview primitives, zero reviewed runtime primitives. Each representative clip returned 404 without preview permission and 200 with it. No uncontextualized clip was selected by chat. |
| Same-session conversation | CUDA question, follow-up, explicit correction and AI-factory topic switch retained four inspectable user turns with the expected reference provenance. Deletion returned 200 and the subsequent inspection returned 404. |
| Current/private/endorsement controls | Each returned `insufficient` with zero citations. No present-day coverage or private information was inferred. |
| Korean request | Returned an explicitly labeled original-source preview. This is not newly generated Korean persona speech. |
| SSE | Real TCP `meta → status → status → token → status → result → done`; metadata at 2.172 ms, original-source text at 62.673 ms, complete at 64.336 ms in this one run. |
| Client disconnect | The client closed after `meta`; zero turns were stored and session deletion succeeded. This tests peer disconnect and session release with a fast preview, not cancellation of neural inference. |

These measurements establish functional wiring and provenance. Citation binding
does not establish that every retrieved exchange answers the question well.
The retrieval acceptance gate remains unmet. The reported milliseconds measure
preview assembly and HTTP delivery, not language-model latency, and are one
observation rather than a throughput benchmark. Live generated speech/avatar
synchronization, two simultaneous neural providers on this host, generated text
quality and human voice/likeness acceptance remain outside this preview result.

## Voice gateway diagnostic

The first actual main-API speech request returned HTTP 502; the child provider
returned 503 before loading a model. The launcher had resolved the supplied
virtualenv Python symlink to the base interpreter, which lacks `qwen_tts`.
Direct interpreter probes established that the supplied virtualenv contains
the module and the dereferenced base interpreter does not. Both services exited
cleanly. `voice-attempt-1.json` preserves this failure; it produced no waveform.

The launcher now preserves the executable path and checks dependency discovery
in that exact interpreter before starting a server. This changes no shared API,
voice service, frozen M6 input or model setting.

The corrected run passed **95 of 95 assertions**, including an actual main-API
`/api/speech` request to the adapted provider in the same invocation. Its
`voice-result.json` records the response and post-generation provider identity;
`voice-signal.json` records a separate PCM signal measurement. The failed and
successful raw traces, service logs and actual generated waveform are preserved
in the freshly retrieved, hash-verified gateway artifact package.

| Measurement | Actual result |
| --- | --- |
| Requested integration text | “This is a simulation.”, English; excluded from the frozen voice quality comparison |
| HTTP response | 200, `audio/wav`, configured-provider header; 73,004 bytes |
| Waveform | Mono PCM16, 24,000 Hz, 36,480 frames, **1.52 seconds** |
| Waveform SHA-256 | `fb29a43a125eb26f12b5c9fc2ff3d2abff998f33efe2223f1fd9178ed92b6f1a` |
| Signal | Peak 0.4630, RMS 0.0990, 35,573 nonzero samples |
| Cold HTTP duration | **34.835 seconds**, including provider model load and synthesis |
| Provider duration | 34.752 seconds; real-time factor 22.863 |
| Actual adaptation | 16 optimizer steps, adapter SHA-256 `56d01017b1d464ea384d60661fd44a97de72ebde884b09adbf70a90f4d20477f` |
| Loaded checkpoint | `9286167534283c576c329203f010b44bbdd31ef20f679f4693875b79a91dbf5b` |
| Subject binding | Jensen persona, publisher-attributed Goldman reference, candidate review status |

The provider used the pinned local model, CPU float32, four threads and seed 42.
The actual command also selected `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`.
The request returned a complete buffered WAV; its header explicitly reports
speech timing as unavailable. This is successful gateway integration with a
slow cold request, not real-time speech or a simultaneous neural text/voice
demonstration. Both child services exited before the text-training slot resumed.
Human similarity and end-of-sentence completion remain the separate voice
evaluation's responsibility; the backend does not expose an EOS reason.

## Recorded generated-audio browser check

The bounded browser check replays the already generated 9.92-second adapted English
sample on the actual fitted GLB. `recorded-browser-protocol.json` freezes its
exact WAV/case hashes, replaced HTTP boundaries and exclusions. The browser will
decode actual saved audio with its normal media clock and analyser; no new
synthesis or invented phoneme labels are involved. The authored gesture cues
are declared fixtures, separate from measured source motion.

The added output reader observes actual mapped morph weights and joint angles
after the rig applies its pose. Node references are cached and observations do
not change animation. Its focused actual-GLB composition/reset check passed,
web type checking passed, and Playwright discovered the new case for desktop
and mobile. The first actual CI run passed the existing 20 Jensen browser cases
and 30 historical cases, but failed both new recorded-audio cases. Its retained
[failure report](recorded-browser-attempt-1.json) and freshly retrieved archive
record the actual measurements: desktop maximum clock difference 898.537 ms
against the 750 ms gate, and a mobile screenshot that took 18.189 seconds and
outlasted the 9.92-second recording. Cancellation was not established by that run.

The revised acquisition order collects the playback trace and clicks Stop before
capturing the full studio screenshot. It keeps the 750 ms gate and actual runtime
unchanged. Raw observations are attached before channel/timing assertions, so a
subsequent failure retains its measurements. Screenshots now document the state
after cancellation; frame traces document animation during playback. An
[intermediate run](recorded-browser-attempt-2.json) passed the new desktop case
but retained the mobile screenshot-expiry failure. Both attempts remain saved.

The corrected [actual CI run](https://github.com/Dindb-dong/who-is-your-bias/actions/runs/43289898996)
at `a5cdcdffc1c5b4bf088baccae8f9469a9ad6f501` **passed both new cases**:
242 web tests, 96 focused Python tests, 30 historical browser tests and all 22
Jensen browser cases passed. The [qualification report](recorded-browser-result.json)
binds the measured result to the exact workflow, saved archive, waveform and GLB.
The complete archive was uploaded and freshly retrieved with matching SHA-256.

| Actual observation | Desktop | Mobile emulation |
| --- | ---: | ---: |
| Playing frame observations | 24 | 406 |
| Frames with concurrent mouth, smile, arm and hand output | 23 | 404 |
| Maximum absolute renderer/media clock difference | 618.667 ms | 580.950 ms |
| Media clock immediately before Stop | 7.433 seconds | 7.483 seconds |
| Actual media elements / AudioContexts | 1 / 1 | 1 / 1 |
| Cancelled before natural completion and cleared clock, source and speech/gesture channels | Passed | Passed |

Jaw, blink, smile, arm, hand and finger outputs were nonzero in both projects.
The unchanged 750 ms functional clock gate passed. These are actual software
WebGL observations, with different layouts and scheduling; the frame counts are
not a controlled FPS comparison, and the clock gate is not a perceptual lip-sync
acceptance test. No physical phone, acoustic output timing, live inference or
simultaneous neural providers were tested in this browser run.

The [desktop trace](recorded-browser-desktop.json),
[mobile trace](recorded-browser-mobile.json) and
[plotted observations](renders/recorded-browser-trace.png) retain the measured
composition during playback. The unmodified
[desktop screenshot](renders/recorded-browser-desktop.png) and
[mobile screenshot](renders/recorded-browser-mobile.png) show the fitted avatar
and explicit replay label **after cancellation**. The screenshots were deliberately
captured after the timed interval. Regenerate the trace plot with
`python3 -m tools.runtime_checks.plot_browser_trace` in an environment with
Matplotlib; that command reads committed observations and runs no model.
