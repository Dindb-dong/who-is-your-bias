# Avatar runtime

The bundled human is an **authored anatomical proxy** with a complete native
face/viseme set; the stylized procedural portrait is also available. Neither is
a scan or a validated likeness. The renderer composes authored animation, imported
geometry estimates, and audio-bound mouth cues while reporting their separate
origins. A geometry measurement does not establish identity, emotion, or a
characteristic personal habit.

For source provenance, actual measurement results, import commands, and detailed
calibration instructions, see [Avatar calibration](../../../../docs/avatar-calibration.md).
For the humanoid's exact asset measurements, licenses, build commands, and native
deformation checks, see [Bundled humanoid](../../../../docs/humanoid-avatar.md).
For local package conversion, actual geometry inspection, and fixed-camera
same-audio recording/export, see [Avatar audition](../../../../docs/avatar-audition.md).
For the planned new identity, face/body/finger authoring, multi-photo/video
reconstruction and required runtime extensions, see
[New avatar generation](../../../../docs/avatar-generation-pipeline.md).
The runtime now supports 30 articulated finger controls and hand-inclusive upper-body framing. New source-fitted identity production remains a separate asset qualification.

## Runtime contract

`AvatarStage` accepts `speaking`, `timeline`, `elapsedMs`, `amplitude`, optional
`modelUrl`, `rigMapping`, and `reducedMotion`. `onReady` returns controls actually
available on the current rig; `onError` receives an `Error`. New measurement props:

| Prop | Meaning |
| --- | --- |
| `audioElapsedMs`, `audioSha256`, `speechTiming` | Actual audio clock and digest, plus a validated `speech-timing.v1` track. |
| `performanceClip`, `observationElapsedMs` | An `avatar-clip.v1` and its independent source-motion clock. |
| `allowGeometryPreview` | Explicit permission to display unreviewed geometry. Default false. |
| `showCalibration` | Show supported controls, actual asset targets, mapping import/export, and manual pose preview. |
| `framing` | `portrait`, `upper_body` (complete hand envelope), or `full_body`; changes the camera without reloading geometry. |
| `onTelemetry` | Face, body/joint, and speech provenance plus unsupported channels and warnings. |

`elapsedMs` is the behavior-plan clock. The transport can map audio playback time
into an estimated plan duration, but it must pass the **unscaled** audio clock as
`audioElapsedMs`. Observations use their own clock. Seek jumps reset temporal
smoothing rather than slowly chasing an obsolete pose.

`amplitude` is normalized audio energy in `[0,1]`; zero means silence. Omitting
it while speaking enables authored articulation. Neither fallback is phoneme
recognition. Imported speech timing requires the matching audio SHA-256 and
actual clock; the current Rhubarb adapter declares `audio_recognition`, not
forced alignment. Unsupported phonemes retain a warning and the fallback.

`parseAvatarClip`, `parseSpeechTiming`, and `parseRigMapping` are exported from
`validation.ts`. They validate and construct bounded typed inputs rather than
trusting arbitrary JSON. The renderer validates imported timelines again and
caches them by object identity. Do not mutate a validated input in place.

## Animation layers

1. Authored breathing, blink, gaze, and text-timed behavior cues establish a base.
2. A reviewed clip, or explicitly enabled geometry preview, overrides its
   measured face channels and local joint rotations. Missing channels retain
   the base. Missing frames and excessive gaps are not interpolated across.
3. Matching audio timing drives mouth articulation. Native viseme blends are
   used only when the active blend is supported; otherwise ARKit-style mouth
   controls provide the fallback.

Face and joint provenance are independent. A body-only sample reports a measured
body source while its authored face remains `designed`. A manual calibration
pose reports `calibration`. The provenance describes source estimates, not
identity verification or the quality of an asset's deformation.

Reduced-motion preferences disable idle motion and reduce nonverbal movement,
including observed joint deltas. Mouth articulation remains available.

Supported authored cue names:

| Channel | Names |
| --- | --- |
| `face` | `smile`, `thoughtful`, `brow_raise` |
| `gaze` | `center`, `aversion`, `left`, `right`, `down` |
| `head` | `nod`, `tilt` |
| `gesture` | `open_palm`, `emphasis`, `shrug`, `wave` |

Each cue retains its `provenance`, `evidence_ids`, time, duration, and intensity.
An authored cue name is not upgraded into an observation by the renderer.

## Importing and calibrating a rig

Use a self-contained binary glTF 2.0 (`.glb`) with embedded textures
and buffers. Uncompressed and Meshopt geometry are supported; the Meshopt decoder
is bundled. Draco/KTX2 decoders are not bundled. Models must be local object URLs
or same-origin resources; external asset and decoder URLs are not fetched. The
runtime rejects files above 64 MiB; the upload UI may impose a smaller limit.

`channels.ts` enumerates 52 ARKit-style face channels, 15 native visemes, and
46 joint controls, including three joints per finger on each hand. Existing `avatar-clip.v1` files remain supported; the optional bone dictionary admits the new names. `retarget.ts` resolves actual morph targets and node names,
preserves authored rest transforms, and reports unsupported channels explicitly.
Loading an asset does not create missing shapes or joints.

A custom mapping supports per-mesh targets, neutral/range calibration, joint
neutral offsets, source-to-target local bases, and bounded angular scaling:

```ts
const mapping = {
  morphs: {
    jawOpen: { target: 'Jaw_Open', mesh: 'Face', inputNeutral: 0.04,
      inputMax: 0.8, outputNeutral: 0, scale: 0.8 },
    eyeBlinkLeft: 'Blink_L',
    eyeBlinkRight: 'Blink_R',
  },
  bones: {
    head: { name: 'Head', neutral: [0, 0, 0, 1], basis: [0, 0, 0, 1], maxAngle: 1.22 },
  },
  visemes: { aa: 'viseme_aa', PP: 'viseme_PP' },
  yaw: 0,
};
```

Quaternions are normalized XYZW. The joint operation is authored rest × neutral
× (basis × bounded delta × inverse basis), using shortest-path interpolation.
Every frame begins from the authored rest pose. A jaw morph or native mouth
shape takes precedence over jaw-bone opening to prevent double deformation.

The procedural portrait approximates 46 named face controls through articulated parts
and authored mesh morphs. It has 36 morph targets across meshes, covering 20
unique named channels; its procedural eye scaling is not a GLB morph. Exporting
it does not magically produce a complete 52-target face. The calibration panel
and real GLB roundtrip diagnostics retain that distinction.

## Lifecycle and verification

The stage owns one renderer per mount, supports React StrictMode cleanup, aborts
obsolete model downloads, disposes decoded images and GPU resources, pauses
rendering while hidden, and reports WebGL context interruption with a reload
action. Pixel ratio is capped. Dragging changes the view within a limited
portrait range; wheel scrolling is not intercepted.

Tests cover import bounds/review gates, source/audio clock separation, hash
binding, missing detections, quaternion antipodes and neutral offsets, partial
viseme mappings, deterministic replay, reduced motion, geometry, actual joint
movement, rest-pose preservation, and model URL restrictions. Integration tests
apply the two actual measured sample clips through a binary GLTFExporter /
GLTFLoader roundtrip. Optional offline geometry exports are controlled by
`BIAS_AVATAR_QA_DIR`; normal CI writes no artifacts. Offline Blender inspection
of those exports does not validate browser appearance or likeness.

References:

- [glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
- [Three.js GLTFLoader](https://threejs.org/docs/pages/GLTFLoader.html)
- [Three.js Quaternion](https://threejs.org/docs/pages/Quaternion.html)
- [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
- [Rhubarb mouth cues](https://github.com/DanielSWolf/rhubarb-lip-sync)


## Hands and upper-body framing

Finger names are subject-relative: `left`/`right` + `ThumbMetacarpal`,
`ThumbProximal`, `ThumbDistal`, and `Index`/`Middle`/`Ring`/`Little` +
`Proximal`/`Intermediate`/`Distal`. Each rotation is a normalized local-delta
XYZW quaternion, with the same neutral and basis calibration as other joints.
Unobserved channels remain authored; missing frames are not held across gaps.
The generic MPFB calibration maps all 30 real deform bones and restores the
authored rest transform every frame. Named channels alone do not establish
Jensen-specific movement or likeness.

`upper_body` fits a declared 2.6-unit-wide envelope around the normalized
2.4-unit human. Distance adapts to the viewport aspect, including mobile
resize. Geometry beyond this envelope requires a wider view. The stylized
bust continues using its existing framing because it is not a complete body.
