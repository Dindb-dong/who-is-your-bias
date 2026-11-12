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

To be filled after execution; the sections above record the pre-implementation plan.
