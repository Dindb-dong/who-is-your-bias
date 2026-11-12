# Jensen attention interaction — PR #16

Status: **plan committed before implementation**. Created 2026-11-12 (Asia/Seoul).
Base: `2e5e82809fe3a413dc4775195c9dd6a612fc7488`, the merge of PR #15.

## Goal and scope

The active project subject is **Jensen Huang**, replacing the earlier Musk demo.
Add a live interaction in which the avatar notices a selected visible participant,
turns its eyes first, and then brings its head around to follow that participant.
The interaction is authored entertainment, not a claim about Jensen's actual
habits or a learned personal behavior.

PR #15 remains the full photo/video reconstruction and character-production plan.
It did not deliver a Jensen mesh, training inputs, a qualified voice, or a Jensen
corpus. This PR delivers the live interaction, a Jensen-bound conversation seed,
and an explicitly labeled functional avatar preview. It must not rename old Musk
evidence, recorded motion or voice weights and present them as Jensen material.
The mesh-generation acceptance gates in [the pipeline](avatar-generation-pipeline.md)
remain separate.

The user's proposed trigger referred to women visible on screen. This implementation
uses **explicit participant selection**, without inferring gender from appearance.
A presenter can select the intended participant in the preview. Detection locates
faces only; it does not classify gender, identify people or retain biometric records.

## User flow

1. Open the Jensen studio. Attention is off; no capture permission is requested.
2. Choose **Camera** or **Share screen** using an explicit start button. Request
   video only, never microphone/system audio.
3. Show the live preview and detected face boxes. Let the user select a numbered
   participant with an accessible button or by selecting its preview box.
4. Eyes orient promptly toward the selected face. After an initial artistic delay
   (starting value 220 ms), the head follows more slowly. Eyes compensate for the
   head's turn so total gaze does not overshoot the target.
5. Maintain the selected geometric track through small motion and brief gaps.
   Do not switch to a different face just because detection order or size changes.
   Ambiguous/crossing tracks require reselection instead of identity guesses.
6. Show searching/lost/paused/error states. After a bounded missing-target interval,
   ease back toward neutral. Stop and navigation release all capture resources.
7. Provide a small pointer-driven preview so the eyes/head sequence can be tried
   without a camera or screen-sharing permission. Clearly label it as a demo.

## Architecture and contracts

### Local capture and detection

- Browser `getUserMedia` / `getDisplayMedia` runs only after an explicit action.
  Screen capture must be called before asynchronous model initialization to keep
  transient user activation.
- Face detection uses a pinned MediaPipe FaceDetector dependency/model, loaded
  lazily. Inference runs in a worker with at most one frame in flight.
- Scale detector frames to a bounded size and process fresh video frames at a
  capped rate. Drop work rather than accumulate latency. Close transferred image
  bitmaps on all paths.
- Produce normalized source-image boxes and confidence, with monotonic capture
  timestamps. Do not transmit, save or log camera/screen frames.
- Use same-origin model/WASM assets with documented provisioning or bundled pinned
  assets. A missing model produces an actionable error; never fake detections.
- A generation token invalidates late permission, model and worker results.
  Canceling startup must also stop a stream that resolves after cancellation.
- Explicit stop, external track end, unmount, navigation and hidden-page handling
  share cleanup. Stop tracks explicitly: `track.stop()` does not emit `ended`.
- Camera preview may be mirrored; source coordinates, boxes, clicks and gaze must
  apply mirroring exactly once. Preserve aspect ratio and account for letterboxing.

### Target tracking

Keep only session-local numbered geometric tracks. Match boxes using bounded
overlap/center displacement, confidence and age, with one-to-one assignment.
Track IDs are not identities. Ignore nonfinite, degenerate or out-of-range input.
Target selection does not change on detector-array reordering. Prefer loss and
reselection over reassignment when multiple candidates are ambiguous.

The render target contract contains a selected session track ID, normalized
display-space center, observation timestamp and an enabled/valid state.
Coordinates use `x: left -> right`, `y: top -> bottom`.
Fresh detections update position without resetting the eye-before-head timer.
A changed selected target or capture session restarts acquisition.

### Animation

- Advance attention on the renderer's monotonic wall clock, independently of audio
  time, which may be zero when idle or jump on seek.
- Use separate eye and head responses with bounded yaw/pitch and capped time steps.
  Artistic delay/smoothing values are configurable constants, not physiological
  measurements.
- Compute remaining eye angle after total head/neck contribution. Maintain a
  short loss grace, then a smooth return to the underlying idle pose.
- Compose after designed/recorded performance and speech. While attention owns
  gaze/head/neck, recorded eye/head rotations cannot silently override it.
- Preserve blink, mouth/visemes, speech timing, facial expression and body/arm
  gestures. Calibration should remain deterministic.
- Fix/bypass the current implicit `gaze * 0.2` head coupling, which otherwise
  violates the eyes-first phase.
- Verify both procedural and GLB paths, including rigs with eye morphs and rigs
  with eye bones. Avoid applying the same eye motion twice.
- Reduced motion disables or strongly limits head tracking and keeps UI status
  understandable. Disabling attention clears the ownership state.

### Subject binding

- Add a Jensen profile and a small dated, source-grounded seed from primary NVIDIA
  publications. Clearly label editorial summaries and avoid invented quotations.
- Bind default startup, displayed subject and API profile to Jensen; preserve Musk
  data and pinned historical evaluation as historical assets.
- Make frontend labels derive from the actual API persona; an offline fallback
  must not silently claim a different backend persona.
- Prevent Jensen sessions from selecting Musk-specific motion, corpus snapshots
  or voice configurations. Generic/browser speech can remain explicitly generic.
- A designed Jensen preview can use glasses, gray hair and a dark jacket, but is
  not a photorealistic reconstruction or a qualified PR #15 generated mesh.
  Imported custom GLBs retain their own diagnostics and mapping limits.

## Ordered work packages

1. **Plan** — commit this document alone on the feature branch before implementation.
2. **Subject transition** — Jensen seed/profile/default configuration, correct API
   metadata, isolation of historical Musk artifacts and focused backend tests.
3. **Attention runtime** — pure state/controller, pose composition, procedural/GLB
   convention fixes, deterministic tests for onset, direction, loss and speech.
4. **Capture/targeting** — worker, fresh-frame scheduling, track selection, cleanup,
   permissions/errors, preview controls and lifecycle tests.
5. **Studio integration** — input controls, pointer demo, Jensen preview and
   data-driven labels; pause/stop behavior and responsive accessibility.
6. **Validation and PR** — run relevant frontend/backend suites, build/typecheck,
   browser scenarios, inspect actual rendered behavior, document measured results
   and limitations, and finish PR #16 for review.

## Acceptance checks

| Area | Required evidence |
| --- | --- |
| Plan-first | First feature commit changes this plan only |
| Subject | Default profile/evidence identify Jensen; no Musk voice/motion substitution |
| Eyes lead | Before head onset, gaze changes while attention head/neck delta stays zero |
| Head follows | After onset, head progresses without exceeding angle limits |
| Direction | Left/right/up/down agree in procedural and actual GLB retarget transforms |
| Composition | Visemes, blink, facial expression and arm channels survive attention |
| Idle | Tracking progresses with audio elapsed time fixed at zero |
| Selection | Box-order changes retain selection; ambiguity does not switch people |
| Loss | Short gap tolerated; prolonged loss clears selection and returns to neutral |
| Lifecycle | Denial, startup cancellation, external ended, Stop and unmount release resources |
| Freshness | Late results cannot update stopped/new sessions; inference never queues unbounded |
| UI | Keyboard-selectable participant controls, mobile layout and reduced-motion behavior |
| Build | TypeScript, frontend tests, relevant Python tests and production build pass |

Browser tests may use deterministic synthetic detections/media to exercise lifecycle
and rendered movement. Such tests are not claims about real-camera detection
accuracy, physical-device latency, avatar likeness or learned motion quality.
Report those boundaries separately and retain an actual manual-camera checklist.

## Primary implementation references

- [MediaPipe Face Detector for web](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/web_js)
- [MediaPipe bounding boxes](https://developers.google.com/edge/api/mediapipe/js/tasks-vision.boundingbox)
- [W3C Screen Capture](https://www.w3.org/TR/screen-capture/)
- [Media Capture and Streams](https://w3c.github.io/mediacapture-main/getusermedia.html)
- [Apple face blendshape definitions](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation/)

## Implementation record

The sections above retain the pre-implementation plan. The first feature commit,
[`26c6c60`](https://github.com/Dindb-dong/who-is-your-bias/commit/26c6c60d23086d8147bb98d54deac87086e1dbc0),
changed this document alone, before any implementation was pushed.

### Delivered behavior

- The default API, source library, topics and studio labels use Jensen Huang's
  own profile, eight editorial evidence summaries and three dated NVIDIA sources.
  An exact corpus snapshot binds that seed; it does not assert 2026 coverage.
- Voice configuration must declare the active subject or a generic voice.
  Foreign/unbound speech and historical Musk motion cannot enter a Jensen turn.
  Historical corpora, presentation media and reproducibility tools retain their
  original subject, with explicit archival labels and test configurations.
- The authored portrait has gray hair, glasses and a black jacket. The avatar
  production/likeness gates from PR #15 remain separate from this preview.
- Pointer/keyboard preview and opt-in camera/screen input feed a selected target
  to the avatar. Selection is explicit and uses geometric face boxes only; there
  is no gender inference, identity recognition or automatic participant choice.
- Eyes lead the head by 220 ms. The compositor bounds angles, compensates eye
  direction for the head/neck contribution and retains speech, blink, expression
  and body channels. Input timestamps use the live monotonic clock rather than a
  potentially earlier RAF timestamp. Bone-route telemetry records the actual
  composed eye rotations.
- Real Chromium exposed a flaw in the initially planned capped frame delta:
  at low render rates, capping elapsed time made head following and release run
  slower than the wall clock. The final controller uses bounded exponential
  smoothing over actual elapsed time, with only the part after the eye-lead
  deadline contributing to head movement. This is a measured correction to the
  earlier capped-time-step plan, not a relaxation of the browser assertions.
- A classic worker runs the pinned local MediaPipe model. One transferred frame
  may be in flight; obsolete results, stale targets and ambiguous tracks are
  released. Stop, navigation, page hiding, ended tracks and late permission grants
  all have explicit cleanup. Reduced motion suppresses new head turns.
- Normal `make dev`, production build and Docker packaging provision the same
  verified assets. Compose passes the voice-subject binding.

See the [implementation and usage guide](jensen-attention.md) and
[asset provenance](../apps/web/public/vision/README.md) for exact settings and paths.

### Validation record

The implementation tree is checked against the remote branch, using exact Git
blob hashes for every changed frontend/deployment file. No unrelated baseline
files or generated model binaries are included in the PR.

Local execution on 2026-11-12:

- Python regression suite: **805 passed, 2 skipped**. The skips require optional
  Torch/Pillow presentation dependencies that are absent from this environment.
- Frontend suite after the browser-discovered timing fix: **230 passed across
  21 files**. Coverage includes actual-eye telemetry and consistent following/
  neutral return at 1–120 FPS and irregular rendering intervals.
- TypeScript and production build passed; avatar audition fixtures: **4 passed**.
- Historical retrieval smoke: **39 graded cases passed**, two explicitly
  ungraded cases; this does not establish Jensen fidelity.
- Actual `make dev`: API/profile, Vite HTML, worker, bundle, WASM and model served
  HTTP 200; profile identified Jensen, the model hash matched, and Ctrl-C released
  both servers.

Local Chromium could not start because its process-singleton socket was rejected
by the execution host. A temporary workflow was scoped to PR #16 and its feature
branch to run real Chromium and the container build. The final documentation
commit removes it, preserving the existing local-validation default. Browser scenarios
keep a real worker/WASM blank-frame smoke separate from synthetic capture tests.

The [second verification run](https://github.com/Dindb-dong/who-is-your-bias/actions/runs/43156887553)
at `e5f766a33e96e9ef879333bdb59c53fd433eb0bc` completed successfully:

| Check | Result |
| --- | --- |
| Python suite | 804 passed, 3 skipped |
| Frontend unit suite | 230 passed across 21 files |
| Avatar audition fixtures | 4 passed |
| TypeScript / production build | Passed |
| Jensen browser scenarios | 18 passed, desktop and mobile viewports |
| Historical studio browser regressions | 30 passed, desktop and mobile viewports |
| Actual MediaPipe worker/WASM/model | Two blank frames processed in each viewport, same-origin assets |
| Container build and startup | Passed with read-only filesystem; Jensen profile, exact snapshot, UI, worker and model digest verified |

The three remote Python skips require optional Torch, Pillow and Blender. Blender
was present for the local run, explaining its one additional passing test. No
browser case was skipped or reported flaky. Camera/screen lifecycle scenarios
use synthetic acquisition and detection boundaries; real worker loading/inference
is exercised separately by the blank-frame smoke.

[Browser reports and desktop/mobile screenshots](https://github.com/Dindb-dong/who-is-your-bias/actions/runs/43156887553/artifacts/13915764986)
are retained as artifact `13915764986` (10,334,375 bytes, configured seven-day
retention). Downloaded ZIP SHA-256:
`cb22d76c75a2404863edfc35b1c5eb50ab6667dfed667fdf9d9cce75856e46fc`.
The final commit changes documentation and removes the temporary runner only;
the runtime, tests and deployment configuration are the tested `e5f766a` tree.

The [initial Chromium run](https://github.com/Dindb-dong/who-is-your-bias/actions/runs/43155921858)
at `202de25` ran all 18 Jensen cases: 11 passed and seven failed. Both actual
worker/WASM/model tests passed. Six failures showed slow neutral return, and one
showed slow head following. These led to the elapsed-time fix in
[`e5f766a`](https://github.com/Dindb-dong/who-is-your-bias/commit/e5f766a33e96e9ef879333bdb59c53fd433eb0bc).
The browser thresholds and timeouts were retained. Desktop and mobile captures
were visually inspected; a clipped pointer indicator and overlapping helper
text were also corrected.

No real-camera accuracy, physical-device latency, trained Jensen voice, learned
personal mannerisms or photorealistic likeness result is claimed. The real-device
presentation checklist remains in the usage guide.

