# Jensen studio: eyes first, then head

PR #16 adds an authored attention interaction to the Jensen Huang studio. A selected
participant's position directs the avatar's eyes; the head follows after a short
delay. This is a designed interaction, not a learned or documented personal habit.
The [plan-first record](jensen-attention-plan.md) was committed before implementation.

The studio uses the active API persona for its name and source data. The default
Jensen portrait is a **stylized procedural preview** with glasses, gray hair and a
dark jacket. It is not a photorealistic reconstruction or a qualified likeness.
The separately planned [photo/video avatar pipeline](avatar-generation-pipeline.md)
still defines that larger character-production deliverable.

## Try it

1. Start the application and open **Conversation**. Find **Avatar attention** below
   the avatar. Capture starts off; opening the studio does not request permission.
2. Choose **Try pointer preview** to test the motion immediately. Move the point,
   use the arrow keys while the preview has focus, or adjust the horizontal and
   vertical sliders. Shift increases the arrow-key step; Home centers the point.
3. For live input, choose **Camera** or **Share screen**. Allow the browser to use a
   camera, or choose the tab/window/screen in its picker. These requests use video
   only: no microphone or system audio.
4. Select a numbered face box or its **Participant N** button. Detection alone
   does not select anyone. **Clear selection** releases the current target while
   keeping the preview open; **Stop** releases the entire capture session.

The original requested scene referred to women appearing on screen. This feature
uses an explicit presenter-selected participant. It has **no gender classifier,
face recognition or identity labels**. Participant numbers are temporary geometric
tracks within one capture session, not persistent identities.

Camera previews are mirrored; screen previews are not. The avatar follows the
participant's position as shown in that preview. Boxes are drawn over the actual
video area, accounting for letterboxing. Selecting a shared tab that contains the
studio itself can produce a recursive image; choose the intended participant
explicitly rather than relying on the largest detected face.

## How the motion behaves

[AttentionController](../apps/web/src/avatar/attention.ts) runs on the renderer's
monotonic wall clock. It continues to work while audio time is zero and is unaffected
by an audio seek. The following are artistic settings, not human-behavior measurements:

| Setting | Current value | Effect |
| --- | --- | --- |
| Head onset | 220 ms after acquiring a target | Eyes lead the attention-driven head turn |
| Eye response time constant | 55 ms | Fast, smoothed gaze changes |
| Head response time constant | 280 ms | Slower head following |
| Return time constant | 240 ms | Smoothly releases attention into the underlying pose |
| Maximum target age | 500 ms | Stale positions stop driving the avatar |
| Eye yaw/pitch limits | 0.35 / 0.25 radians | Bounds eye motion |
| Head yaw/pitch limits | 0.42 / 0.22 radians | Bounds total head/neck attention motion |
| Reduced-motion eye limit | 0.08 radians | Small eye movement; no new attention head turn |

Fresh coordinates for the same participant do not restart the 220 ms delay.
Selecting another participant starts a new eye-leading transition. Existing idle
or recorded head movement may continue before attention takes over. As the head
turns, the eyes use the remaining angle so both contributions do not overshoot.
Exponential smoothing uses actual elapsed time, so a low frame rate does not make
following or release take proportionally longer. A frame that crosses the 220 ms
deadline advances the head only for the elapsed portion after that deadline.

Attention is composed after ordinary playback and damping. It owns eye/head/neck
direction while active and preserves speech timing, visemes, mouth expression,
blink and arm/body channels. Both the procedural preview and mapped GLBs are
supported. A custom GLB needs usable eye bones or directional eye morphs and a
head/neck joint. Unsupported controls are reported in attention telemetry; the
feature cannot create missing deformation or repair an incorrect rig mapping.

Reduced motion keeps the smaller eye response and releases attention-driven head
rotation. The rest of the performance retains the application's existing
reduced-motion handling.

## Detection, selection and loss

The [local detector worker](../apps/web/public/vision/attention.worker.js) runs
MediaPipe FaceDetector with the CPU delegate. It returns bounding boxes and detection
confidence, not demographic or identity predictions. Input images remain in the
browser; the capture path does not upload, record or persist them.

| Processing rule | Current value |
| --- | --- |
| Input resize | Longest side at most 640 pixels |
| Inference cadence | At most 10 requests per second, using fresh video frames |
| In-flight work | One transferred frame; later frames are skipped while busy |
| Accepted result age | At most 350 ms from frame capture |
| Minimum detection confidence | 0.60 |
| Maximum detections per frame | 8 |
| Selected geometric track retention | 700 ms since its last observation |
| Detector initialization timeout | 25 seconds |
| Detector inference timeout | 8 seconds |

[FaceTracker](../apps/web/src/attention/tracker.ts) matches nearby boxes using size,
position and overlap, with one-to-one assignment. It preserves a selection through
small motion, reordered detection results and brief gaps. It drops ambiguous
assignments or substantial face overlap rather than silently attaching an old
participant number to a different person. Box tracking cannot establish identity
continuity through arbitrary occlusion, sudden jumps or crossing people.

The 500 ms animation freshness limit and 700 ms selection retention are distinct:
a stale target can begin returning to neutral before the selection expires. After
expiration or ambiguity, a visible participant must be selected again. No face is
automatically selected when it appears or when another participant disappears.

The bundled short-range model is intended for nearby, selfie-like camera images.
Small faces in distant footage or large shared screens may not be detected well;
see Google's [Face Detector model description](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector)
and [short-range model card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20%28Short%20Range%29.pdf).
Their published benchmarks are not measurements of this application.

## Permissions and lifecycle

Camera capture requires HTTPS or localhost and browser/device support. Screen
sharing depends on browser support and an explicit click each time. The code calls
`getDisplayMedia()` before awaiting model initialization to preserve user activation.
Where supported, `CaptureController.setFocusBehavior('no-focus-change')` asks the
browser to keep the studio focused when sharing another tab or window. This is
best effort; unsupported browsers use the ordinary picker.

Stop, changing input, navigation away from Conversation, hiding the page, external
capture termination and component unmount all clear the target, stop media tracks,
terminate the worker and release the video element. A permission request granted
after cancellation is immediately stopped. Returning to a hidden page does not
automatically restart capture; choose an input again.

Permission denial, missing/busy cameras, missing detector assets and worker
failures appear in the panel. The pointer preview remains available without a
camera, screen-sharing support or detector model loading.

## Install and run

Run commands from the repository root:

```sh
npm ci
npm run prepare:vision
npm run build
uv run bias serve --host 127.0.0.1 --port 8000
```

For Vite development, `npm run dev` runs asset preparation first and proxies API
requests to port 8000. `npm run build` also prepares assets automatically. Calling
a workspace command directly bypasses these root-level hooks, so run
`npm run prepare:vision` explicitly in that case.

The [vision asset guide](../apps/web/public/vision/README.md) records the exact
package, model URL, size, SHA-256 and upstream license sources. The first setup
needs access to the npm registry and Google's model host. Subsequent setup reuses
a correctly hashed model, and live detection loads assets from this application's
own origin.

## Validation

```sh
npm run typecheck
npm run test --workspace @bias/web -- src/attention src/avatar/attention.test.ts
npm run build
uv run pytest tests/test_jensen_persona.py
npm run test:attention
```

The attention browser configuration starts a Jensen-bound backend on port 8001,
separate from historical Musk browser fixtures. Install a Playwright Chromium
browser for the last command, or set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to an
available compatible Chromium executable.

The focused unit tests cover detector-order stability, crossings, stale results,
loss, mirrored coordinates, late permission grants, external track termination,
focus control, single-frame ownership, worker timeout and cancellation. Runtime
tests cover eye-leading timing, direction, limits, pose composition and reduced
motion. Running a command above is not itself evidence that the corresponding
suite passed; record the actual result with the PR's validation evidence.

The initial local Chromium smoke could not start because the execution host
rejected Chromium's process-singleton socket. The first scoped remote run executed
the real worker/WASM/model smoke successfully on desktop and mobile and exposed
a slow-frame animation bug, which was fixed with elapsed-time regressions. The
final remote run passed all 18 Jensen scenarios and all 30 historical studio
scenarios, including the real detector smoke in both viewports. The
[plan's implementation record](jensen-attention-plan.md#validation-record) tracks
the final tested commit and complete results. Do not treat blank input, synthetic
geometry or mocked media as a measurement of real-camera accuracy, end-to-end
device latency, avatar likeness or learned Jensen behavior.

Before a live presentation, check a real camera and shared tab on the target device:
select a participant, move them through all four directions, briefly occlude them,
try two crossing participants, toggle reduced motion, and confirm that Stop and
page hiding release the camera/sharing indicator. The existing evidence does not
establish those real-device acceptance results.

## Primary implementation references

- [MediaPipe Face Detector for web](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/web_js)
- [W3C Screen Capture](https://www.w3.org/TR/screen-capture/)
- [W3C Media Capture and Streams](https://w3c.github.io/mediacapture-main/getusermedia.html)
- [Chrome conditional focus](https://developer.chrome.com/docs/web-platform/conditional-focus)
- [Apple directional eye blendshapes](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation/)
