# Measured avatar motion, calibration, and speech timing

The avatar can replay actual MediaPipe estimates and mouth cues inferred from an
actual audio file. Those paths are separate from the authored idle movements and
text-timed gestures. The built-in asset is still a stylized procedural prototype;
this work does not supply a photorealistic portrait, a verified likeness, or a
learned model of someone's characteristic gestures.

## What was actually run

On 2026-11-10, MediaPipe **0.10.35** ran locally on two clips from the native TED
2013 publisher video. No landmark values were invented to replace a failed
measurement. The committed samples contain bounded normalized coefficients and
joint rotations, source digests, calibration, and review limitations. Raw video,
audio, model weights, full landmark dumps, and exported GLBs remain local.

| Sample | Actual measurement | Exported animation | Important limit |
| --- | --- | --- | --- |
| `apps/web/public/samples/ted-2013-geometry-preview.json` | 64 sampled frames from an 8-second prepared clip; 48 frames selected after excluding a camera cut | 6 seconds; 51 named face coefficients and head rotation | Requested native window 14–20 seconds. Elbows/wrists were outside the usable image; no arm or hand tracks were exported. |
| `apps/web/public/samples/ted-2013-body-geometry-preview.json` | 40 sampled frames from a 5-second fixed spatial crop | 5 seconds; chest, both upper arms, both forearms, and both hand directions in all 40 frames | Requested native window 65–70 seconds. Only 7/40 frames had a face detection, so this sample deliberately exports pose only. |

The requested sampling rate was 10 fps; the frame-stride implementation and
source cadence produced approximately 8 fps. Replay uses the recorded timestamps,
not the requested rate. The face sample's largest reference-relative head
rotation was **69.04°**, and its `jawOpen` coefficient ranged from **0.000209** to
**0.039445**. The body sample's maximum reference-relative rotations were:

| Joint | Maximum rotation |
| --- | ---: |
| Chest | 13.56° |
| Left upper arm | 20.30° |
| Left forearm | 55.56° |
| Left hand direction | 33.67° |
| Right upper arm | 14.43° |
| Right forearm | 27.83° |
| Right hand direction | 17.69° |

These are differences from the selected reference frame, not clinical joint
angles. No body frame reached the exporter's 70° bound. Hand direction comes
from the wrist-to-index-landmark direction; finger bending and forearm axial
twist are not measured.

Both samples remain **`geometry_preview`**. Speaker association, continuous
person tracking, and precise source-clock verification are not approved. The
body crop selects the left seated *image position*, without identifying that
person. Caption times are not interchangeable with native video times. The
65-second source offset also inherits uncertainty in the first frame of the
parent 60–80-second acquisition.

### Reproducible source lineage

The body input was an 854×480 frame. Its fixed ROI is `x=170, y=50, width=280,
height=350`, applied before scaling, without mirroring. This excludes the second
seated person and retains visible shoulders, elbows, and wrists. Cropping the
video does not isolate a person's voice in the accompanying audio.

| Artifact | SHA-256 |
| --- | --- |
| Face-window prepared video | `9623d217112de891147327b2328e96b6a6d224caef7e300da928c0471da3e3a8` |
| Face-window observations | `d785cba8cb6e0548a83f74f92a6bb89e89b4dcf71eeeae5ec7d16fb9820ed9e6` |
| Body-window parent input | `698365fcb362f82c93f14c120a878886785d69608be81f76e4b4ac53af93bdb4` |
| Body-window prepared video | `47e3cb910e00a8f5b3f430a0f38e86355df8d19fb4631fbb8c1a565ee2ef164e` |
| Body-window observations | `aea97e1e1c932717046840bcf84b64fbf077aff6affb7f2ef498223c189ba0e5` |
| MediaPipe face task | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| MediaPipe pose-lite task | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |

Both clips preserve selection-review digests and observation offsets. The body
clip additionally preserves the preparation-manifest digest and crop coordinate
system. Hashes bind these artifacts to one another; they do not establish
identity or approve attribution.

## Try the measured samples

Open the research lab, explicitly enable geometry preview, and load the face or
arm-motion sample. Its dedicated observation clock supports replay and seeking.
Open **Rig inspection & calibration** to see the channels the current asset can
actually animate. The body-only sample should report `Face: designed` and
`Body: geometry_preview`; a missing face track never upgrades the authored face
animation's provenance.

The two examples are separate source windows. Playing a geometry clip alongside
new speech is a compositing experiment, not evidence that the observed gesture
occurred with those generated words. To replay a synchronized source performance,
use matching source media and a reviewed common clock.

## Prepare observations and export a clip

The optional vision dependencies are in `tools/media/requirements-vision.txt`.
Use the official [MediaPipe Face Landmarker][face-docs] and
[Pose Landmarker][pose-docs] task assets. The actual run used the version-1 face
landmarker and the float16 pose-landmarker-lite task, with the digests above.
The task assets are explicit local arguments; the application does not download
models automatically.

Prepare a new review bundle from a local video. For the acquired 60–80-second
parent clip, local `--start 5` plus `--input-source-offset 60` records the requested
native 65-second offset; it must not seek an additional 65 seconds.

```sh
python -m tools.media prepare \
  data/raw/voice-reference/ted-2013-native/publisher-60-80.mp4 \
  data/local/references/body-window \
  --source-id ted-2013 \
  --source-url https://www.ted.com/talks/elon_musk_the_mind_behind_tesla_spacex_solarcity \
  --source-timeline ted_native_mp4 \
  --start 5 --duration 5 --input-source-offset 60 \
  --crop 170 50 280 350

ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2 python -m tools.media observe \
  data/local/references/body-window/manifest.json \
  --face-model /absolute/path/face_landmarker.task \
  --pose-model /absolute/path/pose_landmarker_lite.task \
  --sample-fps 10
```

The crop helper accepts only bounded even-pixel coordinates and dimensions, at
least 32×32, wholly inside the source frame. It rejects rotated source metadata
until the source orientation is normalized. It accepts local media rather than
arbitrary FFmpeg filter expressions, and it never overwrites an existing bundle.

The first local MediaPipe attempt failed before inference because the environment
lacked `libGLESv2.so.2`. A standard Debian `libgles2` package was downloaded from
the official Debian repository and extracted into an isolated runtime directory;
no package-maintainer scripts ran. Providing that directory through
`LD_LIBRARY_PATH` allowed the ordinary CPU runtime to load. This is an environment
dependency, not a model failure. Successful measurements above used that runtime.

### Review explicit detections

Detection indexes are frame-local. Two independently selected face/pose indexes
in a multi-person frame do not establish that they describe the same individual.
Use an explicitly reviewed single-person ROI or independently reviewed tracks,
and preserve the review limitations.

Create a `motion-review.v1` JSON beside the observations. Bind it to the exact
`observations.sha256` in the observation manifest:

```json
{
  "schema_version": "motion-review.v1",
  "observations_sha256": "REPLACE_WITH_THE_EXACT_64_CHARACTER_DIGEST",
  "status": "geometry_preview",
  "reviewer": "Your review name",
  "track_id": "fixed-roi-preview",
  "track_continuity_reviewed": false,
  "source_clock_reviewed": false,
  "speaker_association_reviewed": false,
  "attribution_basis": "Fixed spatial ROI; identity and audio-speaker association unknown.",
  "selections": [
    { "start_ms": 0, "end_ms": 5000, "face_index": null, "pose_index": 0 }
  ]
}
```

Then export the bounded animation:

```sh
python -m tools.animation \
  data/local/references/body-window/observations-manifest.json \
  data/local/references/body-window/motion-review.json \
  data/local/references/body-window/avatar-clip.json \
  --start-ms 0 --end-ms 5000 --neutral-ms 0 \
  --max-gap-ms 250 --visibility 0.65 --max-rotation-degrees 70 \
  --allow-geometry-preview
```

`--neutral-ms` chooses a rotation reference, not an automatically detected neutral
expression. The first speaking frame is not subtracted from face coefficients.
Optional reviewed `face_neutral` values can explicitly calibrate those channels.
The output includes `bone_neutral` for inspection, while frame rotations contain
local deltas from that reference. Matching the source's initial seated posture
requires target-specific neutral offsets; the default replay transfers changes
onto the target's authored rest pose.

Exports require at most 120 seconds and 3,601 frames. Observations must have
strictly increasing times and matching source IDs, timelines, and hashes. Empty
selected intervals preserve gaps. Channels missing from the reference pose,
landmarks outside the image, or landmarks below the visibility/presence threshold
remain absent. The runtime does not bridge gaps above `max_gap_ms`.

To publish a `speaker_reviewed` clip, a reviewer must approve continuity, source
clock, and speaker association separately. The converter and importer both reject
that status unless all three review fields are true. The checked-in samples have
not received that approval.

## Calibrate a GLB asset

Use a self-contained, uncompressed binary glTF 2.0 model with embedded textures
and buffers. The renderer accepts local object URLs or same-origin assets and
caps data at 64 MiB; the surrounding upload UI may have a lower cap. It does not
fetch external decoder libraries or model subresources.

The rig contract enumerates **52 ARKit-style channels**, **15 visemes**, and
**16 joint controls**. Enumeration is not a promise that an asset supports them.
The inspector lists actual morph target names, actual joint names, supported
controls, and unsupported channels. The [ARKit specification][arkit-docs] defines
blendshape names, and [glTF 2.0][gltf-docs] defines node transforms, morph weights,
and unit XYZW quaternions. Names use the subject's left/right conventions; keep
source mirroring explicit.

The procedural fallback currently implements 46 named face-control
approximations. Its 36 authored mesh morph targets cover 20 distinct named face
channels; the remaining supported controls use articulated parts and transforms.
Exporting it as a GLB does not turn those procedural controls into morph targets.
The roundtrip correctly reports, for example, that the exported asset lacks an
`eyeBlinkLeft` morph. A finished production asset needs independently authored
facial topology, skinning, eyelids, teeth, tongue, and calibrated expression
shapes. Loading a model without those features cannot create them.

A mapping can be entered or exported in the inspector:

```json
{
  "morphs": {
    "jawOpen": {
      "mesh": "Face",
      "target": "Jaw_Open",
      "inputNeutral": 0.04,
      "inputMax": 0.8,
      "outputNeutral": 0,
      "scale": 0.8
    },
    "eyeBlinkLeft": "Blink_L",
    "eyeBlinkRight": "Blink_R"
  },
  "bones": {
    "head": {
      "name": "Head",
      "neutral": [0, 0, 0, 1],
      "basis": [0, 0, 0, 1],
      "scale": 1,
      "maxAngle": 1.22
    }
  },
  "visemes": { "aa": "viseme_aa", "PP": "viseme_PP" },
  "yaw": 0
}
```

For a morph, source-neutral subtraction and range normalization happen before
scale and target-neutral addition. Target weights are clamped to `[0,1]`. If two
controls address the same physical target, the largest absolute contribution
wins and a diagnostic explains the shared mapping; contributions are not added.

For a joint, the operation is:

`authoredRest × targetNeutral × (basis × boundedSourceDelta × inverse(basis))`

`neutral` and `basis` are normalized XYZW quaternions; `maxAngle` is in radians.
`basis` converts source-local delta axes into target-local axes. The legacy `axis`
option only redirects an authored Euler cue; use a quaternion basis for measured
motion. Quaternions use shortest-path interpolation and normalization, following
[Three.js Quaternion][quaternion-docs]. Every frame starts from the authored rest
rotation, so repeated application does not accumulate drift.

Enable manual pose preview, test one channel at a time, and inspect neutral,
intermediate, and extreme values. Joint controls allow ±60° previews on each
local axis. Closing the inspector releases the preview. Mapping changes reload
the asset so they do not use an already-animated pose as its rest pose.

## Speech timing and the three clocks

| Runtime prop | Meaning |
| --- | --- |
| `elapsedMs` | Authored behavior-plan time; the transport may scale it to an estimated text duration. |
| `audioElapsedMs` | Actual playback time for the current decoded audio. Never scale this into the behavior plan. |
| `observationElapsedMs` | Source geometry clip time, controlled independently for replay and seeking. |

The exported `speech-timing.v1` contract includes a SHA-256 digest of its audio,
actual audio duration, timing basis, label vocabulary, engine provenance, and
ordered non-overlapping intervals. The renderer only applies these intervals
when the supplied digest matches and an actual audio clock is available.

| `timing_basis` | Appropriate claim |
| --- | --- |
| `audio_recognition` | Mouth labels inferred from actual audio, including Rhubarb. |
| `forced_alignment` | Labels from a real forced-aligner execution. |
| `tts_model` | Timings actually emitted by the speech model. |
| `manual` | A human-authored timing track. |
| Amplitude fallback | Jaw opening driven by actual normalized audio energy; not phoneme recognition. |
| Estimated fallback | Authored articulation used when timing and energy are unavailable. |

[Rhubarb][rhubarb-docs] produces the A–H/X mouth-cue vocabulary. The importer maps
A→PP, B→kk, C→E, D→aa, E→O, F→U, G→FF, H→nn, and X→sil. It also accepts known
`viseme_15` labels and supported ARPAbet/IPA labels. Empty tracks, unknown Rhubarb
labels, and unknown native-viseme labels are rejected. Unmapped IPA/ARPAbet
segments retain an explicit warning and use the amplitude/estimated fallback.
The short interpolation around adjacent cues is authored coarticulation, not an
additional measurement or a forced-alignment claim.

On a target with a complete active native-viseme blend, native shapes take
precedence over ARKit mouth articulation. A partial native mapping preserves the
full ARKit fallback so a supported `sil` cannot erase an unsupported `aa`. A
jaw-open morph takes precedence over a jaw bone; an `aa`-only fallback also avoids
opening both a morph and the jaw joint. These rules are tested with deliberately
incomplete assets.

## Validation and remaining work

Focused validation completed with **29 TypeScript avatar tests** and TypeScript
typechecking, plus **19 Python animation/crop tests**. The latter include real
FFmpeg crop execution. Both committed measured clips passed a real
`GLTFExporter → GLTFLoader → retargeted rig` roundtrip. The face sample changed
actual morph influences and head quaternions; the body sample changed all seven
mapped joint rotations. Tests also cover provenance gates, malformed imports,
quaternion antipodes, missing detections, speech-clock/hash mismatch, sparse
viseme assets, and release back to authored motion.

Optional local geometry artifacts can be produced without writing files during
normal CI:

```sh
BIAS_AVATAR_QA_DIR=/absolute/path/to/ignored/qa \
  npm run test --workspace @bias/web -- src/avatar/replay.integration.test.ts
```

The same exported articulation and measured-body geometry was inspected in an
offline Blender CPU render. This checks basic geometry and deformation; it is
not a browser screenshot, motion-quality study, or likeness validation. The
source windows are short, single-camera estimates. Production work still needs
a licensed high-quality target asset, target-specific calibration, reviewed
speaker tracks and clock alignment, longer observations across settings, and
independent evaluation of animation and voice similarity.

[face-docs]: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
[pose-docs]: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
[arkit-docs]: https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation
[gltf-docs]: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
[quaternion-docs]: https://threejs.org/docs/pages/Quaternion.html
[rhubarb-docs]: https://github.com/DanielSWolf/rhubarb-lip-sync

## Optional articulated hand observations

`python -m tools.media observe` accepts `--hand-model` alongside the existing
face and pose model paths. It records 21 image/world landmarks and the model's
handedness labels, with a separate model SHA-256. A detection index is local to
one frame; handedness does not associate it with a person or a voice.

A hash-bound `motion-review.v1` selection may add `left_hand_index` and
`right_hand_index`. They must explicitly refer to the subject's anatomical
hands; the same detection cannot serve both sides. Existing reviews that omit
these fields continue exporting face/body-only clips. Review changes remain
bound to the original observation hash.

The converter exports 15 local-delta finger rotations per selected hand,
retaining the existing `avatar-clip.v1` schema. Hand landmarks outside the image
or degenerate segment geometry stay unavailable. Each finger's swing is
estimated from the measured segment direction; axial twist is unobserved.
No implicit interpolation crosses missing detections or camera cuts. Source
neutral calibration and the usual reviewed/geometry-preview gates still apply.

Upstream API reference: [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python).
This extension does not itself establish a Jensen-specific motion dataset.
