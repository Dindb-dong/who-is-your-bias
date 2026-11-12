# Local face-detector assets

This directory contains the committed classic worker, `attention.worker.js`, and
runtime assets prepared by [scripts/prepare-vision.mjs](../../../../scripts/prepare-vision.mjs).
The detector estimates face locations for explicit participant selection. It does
not identify people or classify gender. Camera and screen images stay in the
browser; only local geometric coordinates reach the avatar controller.

## Provisioning

From the repository root:

```sh
npm ci
npm run prepare:vision
```

Root-level `npm run dev` and `npm run build` call the preparation script through
`predev` and `prebuild`. A direct workspace build/dev command bypasses those hooks.

The script verifies the installed package version, copies its ESM bundle and WASM
files, and downloads the fixed model artifact when a valid local copy is absent.
It checks both byte count and SHA-256 before atomically installing the model. A
matching cached model can be reused without contacting the model host.

| Item | Pinned value |
| --- | --- |
| Package | `@mediapipe/tasks-vision` |
| Package version | `0.10.32` |
| npm package license metadata | `Apache-2.0` |
| Model | BlazeFace short-range, float16, version `1` |
| Model size | `229746` bytes |
| Model SHA-256 | `b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f` |

Model download:

<https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite>

Generated files, excluded from Git:

- `vision_bundle.mjs`: the pinned package's ESM entry bundle.
- `wasm/`: the pinned package's WASM loaders and binaries.
- `blaze_face_short_range.tflite`: the verified model.
- `assets.json`: installed package/version/license metadata and the model URL,
  size and digest. This is a provisioning record, not a detection benchmark.

The Vite build copies these public assets into its output. Deploy that complete
output so the worker, bundle, WASM and model remain on the application origin.
The detector does not fetch a CDN bundle or model from Google's host during live
capture. If a required file is absent, the panel shows an actionable error and
stops capture rather than substituting fake detections.

## Worker and ownership

The worker is intentionally **classic**, not a module worker. It dynamically
imports the same-origin `vision_bundle.mjs`; MediaPipe's WASM loading path can then
use `importScripts`. This also avoids different Vite development and production
worker-loader behavior.

The main thread sends one transferable `ImageBitmap` at a time. The worker closes
that bitmap in a `finally` block after processing. Capture stop terminates the
worker, invalidates late results and stops all media tracks. See
[the attention guide](../../../../docs/jensen-attention.md) for cadence, stale
result limits, selection and lifecycle details.

## License and model scope

The installed npm package records Apache-2.0 for its code. Separately, Google's
[BlazeFace short-range model card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20%28Short%20Range%29.pdf)
lists Apache License 2.0 for the short-range model. Keep those code/model records
separate: `assets.json`'s `packageLicense` field refers to package metadata, not an
independent model-license audit. Preserve applicable upstream notices when
redistributing bundled assets. These terms do not license unrelated source
photos, captured footage, avatar likenesses or other third-party models.

The [official model description](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector)
focuses on nearby, front-facing/selfie-like images. It is not an identity tracker,
and its published measurements do not establish this project's accuracy or
latency for shared screens, tiny faces, occlusions or crowded scenes.

## Verification

```sh
npm run prepare:vision
npm run test --workspace @bias/web -- src/attention
npm run build
npm run test:attention
```

Unit fixtures test the capture/tracking protocol and worker client; they do not
run the actual MediaPipe model. The browser suite must execute successfully before
claiming worker/WASM integration validation. A real camera and the target device
are separately required for detection-accuracy and latency evaluation.
