# New avatar generation from photographs and video

Status: **implementation design; no new avatar has been generated or qualified**.
Recorded on **2026-11-12, Asia/Seoul**. Repository audit baseline:
[`0ef4e8b`](https://github.com/Dindb-dong/who-is-your-bias/commit/0ef4e8bf2e90e2aefdc0601d346f75af9f069275).

## 1. Decision and completion definition

Build a **new Jensen Huang likeness from multiple photographs and continuous
video**, with a complete upper body, controllable expressions, speech animation,
and new arm/hand gestures. Previously suggested ready-made celebrity assets are
excluded from this work. The input is not restricted to one photograph.

The product deliverable is an editable character master and a **skinned,
morph-enabled, self-contained GLB** for this repository's browser renderer. A
full-body base is preferred when it simplifies fitting and proportions; the
product view must show the upper body and the complete gesture envelope. An
upper-body-only mesh is acceptable after its scale and camera anchors are handled
explicitly. A talking-head video or a static textured bust cannot satisfy this
deliverable.

Treat these as three separate acceptance decisions:

1. **Identity and appearance:** recognizable neutral geometry, skin, hair, glasses,
   clothing, and likeness preserved during expression.
2. **Animation capability:** facial deformation, eye/jaw controls, body/arm/hand
   skinning, and independently articulated fingers.
3. **Performance:** fresh dialogue drives lip sync, expression, gaze and gestures
   with correct timing, blending and interruption behavior.

The required visible anatomy is head, neck, both shoulders, torso through the
waist, both complete arms and hands, and all fingers. Eyes, eyelids, teeth and the
visible mouth interior need usable geometry. Glasses, hair and a jacket are
separate authoring tasks. A texture that resembles the subject in one frontal
image is insufficient evidence of these capabilities.

This document defines the Jensen **visual-avatar workstream**. The running
application currently represents Musk. A conversational Jensen release also
needs a matching persona, evidence corpus, voice configuration and motion
library; visual upload alone does not create that persona. Section 7 specifies
that binding. This documentation change does not claim any of those new assets
or runtime additions have been implemented.

Read the [method and license evidence](research/avatar-generation-methods.md)
for upstream capabilities. The decisions below are project engineering choices,
not measured results from those methods.

## 2. What can be reused, and what needs implementation

The following is verified against the audit baseline, rather than inferred from
the bundled model's appearance or skeleton size.

| Area | Present in this repository | Required for the new avatar |
| --- | --- | --- |
| Geometry | Embedded GLB; uncompressed or Meshopt; same-origin resources | Newly fitted identity and browser-compatible materials |
| Face | 52 ARKit-style control names and 15 native visemes | Correct deformation for the new identity, including combined expressions |
| Body | 16 runtime joint controls, ending at left/right wrists | Articulated fingers, joint limits and hand-specific mapping |
| Asset skeleton | Bundled proxy has 163 joint names | Map and exercise required joints; this count does not establish finger control |
| Speech | Prepared audio can use hash-bound `speech-timing.v1` | Calibrated mouth shapes and measured audiovisual alignment |
| Live PCM | Current streaming path omits speech timing and audio digest | Timed streaming articulation; current mouth movement is amplitude-based |
| Expression composition | Native visemes suppress overlapping mouth/jaw/tongue face channels | A policy for speaking smiles and other compound lower-face expressions |
| Camera | Fixed `portrait` and `full_body` presets; visible model height normalized to 2.40 stage units | Hand-inclusive upper-body framing and declared anatomical scale anchors |
| Motion | Authored cues and reviewed `avatar-clip.v1` playback | New gestures with finger motion and Jensen-specific observations where supported |
| Embedded animations | GLB animation tracks are inventoried | They are not played by an AnimationMixer; runtime controls must drive motion |
| Identity selection | Custom GLB can be selected independently of `/api/persona` | Explicit avatar/persona/voice/motion binding before a conversational release |
| Qualification | GLB inspection, calibration and same-audio browser audition | Human likeness, deformation, timing and target-device evaluation |

Code evidence: [channel vocabulary](../apps/web/src/avatar/channels.ts),
[types](../apps/web/src/avatar/types.ts),
[validation](../apps/web/src/avatar/validation.ts),
[retargeting](../apps/web/src/avatar/retarget.ts),
[performance composition](../apps/web/src/avatar/performance.ts),
[camera presets](../apps/web/src/avatar/builtin.ts),
[studio selection and streaming](../apps/web/src/App.tsx), and
[AvatarStage](../apps/web/src/components/AvatarStage.tsx).
See also the [runtime contract](../apps/web/src/avatar/README.md) and
[existing audition measurements](avatar-audition.md).

## 3. Source preparation: use multiple references deliberately

### 3.1 Choose a consistent target appearance

Start with an appearance brief: reference period, hair, glasses, jacket, body
proportions, expected camera distances and lighting. Select a coherent appearance
group before fitting. Photographs from different years or different outfits may
help understand identity, but should not be averaged into one texture or body
silhouette without explicit treatment of those changes.

Organize sources by their role:

| Source class | Use | Handling |
| --- | --- | --- |
| Multi-event photographs | Identity, profile, ears, hair, glasses and material references | Estimate camera/expression independently; record appearance group |
| A continuous single-person shot | Temporal face/body fitting and expressive-video pilot | Preserve source timestamps, cuts, zooms and occlusion flags |
| Compatible quasi-static multiview images | Optional photogrammetric geometry constraints | Check real viewpoint baseline, overlap, pose and lighting consistency first |
| Gesture footage | Handedness, range, timing, rest poses and reusable motion observations | Keep motion estimation separate from neutral identity fitting |
| Development events | Pilot selection, loss/rig tuning and appearance review | Separate events from fitting data; record every use in model selection |
| Final held-out events | Frozen-candidate likeness and motion assessment | Exclude entire events/recordings and reuploads from fitting, tuning and method selection |
| Synthesized views, if used | Initialization or filling unobserved regions | Label as synthesized; never count as independent evidence |

Unrelated photographs are not a calibrated simultaneous scan. Rotating a person
with image generation does not recover unseen measured geometry. A talking
person is also not a rigid object for ordinary photogrammetry. Use a rigid
reconstruction only on a compatible subset; use articulated fitting for changing
poses and expressions. COLMAP's [capture guidance](https://colmap.github.io/tutorial.html)
supports overlapping, differently viewed images with similar illumination and
selective video downsampling, rather than retaining every adjacent frame.

### 3.2 Coverage and preparation output

Prefer coverage over an arbitrary photo/frame count. Prepare frontal, left/right
three-quarter and available profile references; neutral and expressive faces;
open/closed eyelids; and visible teeth during speech. For gestures, retain shots
showing shoulders, elbows, wrists and fingers. Track the visibility of each hand,
including palm/back views, separated digits and curled digits.

Preparation should perform shot segmentation, duplicate/reupload detection,
quality ranking, person masks and face/body/hand landmark estimation. Keep
original images and timebases, plus crop-to-original transforms. Reject or
downweight motion blur, tiny hands, severe compression, occlusion and uncertain
landmarks. Reflection-contaminated glasses regions need their own masks.

The first output is a **coverage report and contact sheets**, not a claimed
complete reconstruction. Mark missing profiles, hidden fingertips, ears, back
surfaces and mouth detail. Regions completed from a generic prior or sculpting
must remain distinguishable from observed regions.

The proposed source manifest is a new contract, not an existing parser:

| Fields | Purpose |
| --- | --- |
| `subject_id`, `source_id`, original URL, source SHA-256 | Bind each reference to a subject and exact original bytes |
| `event_id`, `recording_id`, `shot_id`, `appearance_group`, `wardrobe_group` | Prevent appearance mixing and source-level evaluation leakage |
| Media dimensions, video timebase, frame PTS/seconds, frame SHA-256 | Recover the exact image or source-video frame |
| Camera group, estimated camera parameters, confidence, mirror flag | Keep unknown parameters unknown; preserve subject-left/right |
| Crop transform, masks, face/body/hand landmarks, visibility by region | Reproduce fitting and identify missing evidence |
| Blur/exposure/occlusion scores and estimator versions | Explain inclusion, exclusion and weighting |
| Use tags, split, rejection reason, `derived_from` | Separate identity, texture, motion and evaluation uses |

Split by event/recording **before extracting frames** into fitting, development
and final-test sets. Use development events for pilot selection, loss settings,
manual corrections and rubric calibration. Freeze the candidate asset, mapping,
performance configuration and thresholds before opening final-test events.
Adjacent frames from one clip do not create an independent test. If a failed
final test informs later tuning, retain that result and retire the exposed events
from the next unseen-test claim. If independent references are unavailable,
report the gap and keep final likeness acceptance incomplete.

## 4. Reconstruction strategy and selection gate

### 4.1 Primary product path: fit a controllable mesh

Use a neutral human topology with a body/hand rig and facial action basis, then
fit a **new identity** to the selected references. The existing MakeHuman/MPFB
tooling is a practical free foundation for Blender finishing. Anny is an optional
differentiable fitting target, subject to a small compatibility trial; it is not
an implemented photo-to-person solver in this repository. Upstream evidence and
asset conditions are recorded in the [method ledger](research/avatar-generation-methods.md).

The planned fitting model separates shared identity shape from per-frame pose
and expression, per-shot camera parameters and per-image illumination. Use
landmark reprojection, silhouettes, masked appearance, temporal consistency and
shape regularization as candidate objectives. Loss weights and model choices
must be selected on development references and saved in the run configuration;
this document does not invent trained settings.

| Stage | Work | Inspectable output |
| --- | --- | --- |
| Initialize | Estimate cameras, body pose, hand pose and face expression with visibility masks | Per-source overlays and uncertainty report |
| Fit coarse identity | Jointly fit shared neutral shape across reliable viewpoints | Neutral mesh, parameters and profile comparisons |
| Fit temporal clips | Solve changing expressions/poses while keeping identity shared | Frame overlays, temporal residuals and rejected frames |
| Refine regions | Improve face, ears, neck and hands with reliable references | Regional geometry and evidence coverage |
| Correct manually | Sculpt identity errors, refine topology and adjust anatomy | Saved editable master and before/after comparisons |
| Verify neutral | Review shape from several angles before relying on textures | Neutral turntable and development-reference panels |

The fitting optimizer, preparation manifest and automated export orchestration
are **planned implementation**, not current CLI commands. Manual sculpting,
texture cleanup and rig correction are explicit production steps. A free
template removes neither those steps nor the need to validate identity.

Before running the first mesh-fit pilot, complete this configuration contract:

| Contract | Required decision or output |
| --- | --- |
| Inputs | Canonical template revision and asset hashes; fitting observation manifest; declared camera model and coordinate convention |
| Backend selection | Choose the fitting backend, face/body/hand estimators, differentiable silhouette/appearance renderer and parameter bounds in a compatibility trial |
| Optimization order | Camera/pose initialization; shared neutral identity; per-frame expression/pose refinement; then detail and texture finishing |
| Outputs | Neutral mesh, shared identity parameters, per-frame camera/pose/expression, objective residuals, source overlays and resumable checkpoint |
| Success boundary | Demonstrate geometry fitted from the observations; texturing an unchanged generic body does not pass identity fitting |

Backend selection is an exit condition of the first contracts/corpus stage.
Record an unresolved selection as blocked work rather than calling the pilot
ready to run. This permits choosing Anny after its compatibility trial without
pretending MPFB alone provides a differentiable reconstruction optimizer.

### 4.2 Bounded neural reconstruction experiments

Run these on the same target identity and development reference set, with each
method's actual input requirements recorded. Different input subsets must be
visible in the comparison report.

| Experiment | Input and reason to try it | Output boundary and stop condition |
| --- | --- | --- |
| ExAvatar | A coherent monocular video with adequate body/hand visibility; evaluate expressive face/body/hand reconstruction | Gaussian/mesh-hybrid avatar with its own deformation/rendering path; stop if sources, dependency access, motion quality or reproducibility fail |
| LHM++ | Several compatible pose-free photos; evaluate quick multi-image identity/appearance recovery | Animatable Gaussian representation; pin an available checkpoint before starting; stop if the required weights are unavailable or hands/face are inadequate |
| MetaHuman authoring alternative | A reconstructed/sculpted human mesh; evaluate production facial rigging and material quality | Complete Assembly/DCC export is required; browser conversion and rig mapping need a separate proof |

These are upstream-supported directions, **not results on Jensen**. Method
details, current checkpoint caveats, hardware and distinct dependency licenses
are in the [evidence ledger](research/avatar-generation-methods.md).

A Gaussian PLY is not a portable skinned GLB. Do not draw an automatic
`ExAvatar/LHM++ -> MetaHuman -> GLB` chain. Using a neural reconstruction as a
mesh source requires a separately implemented extraction/fitting, retopology,
texture projection and rig-transfer stage, followed by the same deformation
tests. Rendered neural views can instead be labeled visual references.

Keep the conventional mesh deliverable as the product baseline. A Gaussian
runtime is an optional separate renderer experiment, with its own deformation
state, loading, material/lighting behavior, memory budget and playback adapter.
It may be adopted only after it meets the same face, hands, novel performance,
camera and interruption requirements. A high-quality fixed-view demo is not
enough to make that decision.

### 4.3 First pilot decision

Before full production, produce one small **mesh identity-fit pilot** and, when
the required environment/assets are available, one ExAvatar video pilot and one
LHM++ multi-image pilot. Record configuration, actual inputs, checkpoint hashes,
hardware, elapsed time, peak memory, manual work and all failures.

Before execution, fill the run configuration with `max_trials`, an optimizer-step
or wall-time budget, actual device and memory ceiling, checkpoint cadence,
resume policy and stop conditions. Select the numbers after environment preflight;
they are not estimates of achieved performance. Preserve failed runs and resume
lineage. Missing weights, incompatible dependencies, exhausted budget or a failed
quality gate are explicit outcomes, not reasons for unrecorded extra attempts.

Compare matching development viewpoints, neutral geometry, speaking identity,
hand visibility and new poses. Select the production authoring route by visible
quality and integration effort. Do not select a winner from a paper's timing,
an upstream demo, a frontal screenshot or an unavailable checkpoint. If both
neural experiments fail their gates, continue fitting and authoring the mesh;
the project does not depend on either becoming a complete exporter.

## 5. Appearance, topology and rig authoring

### 5.1 Appearance master

Project compatible source images with visibility/confidence weights, normalize
exposure, and avoid baking stage lighting or glasses reflections into skin.
Refine diffuse appearance, roughness and normal detail separately. Inspect the
face, ears and neck under multiple lighting setups. Record which texture regions
come from references and which are prior-filled or painted.

Build glasses frames/lenses as head-attached geometry; fit hair and eyebrows;
create usable eyes, teeth and mouth interior; fit the jacket to a plausible body
underneath. Preserve identity with and without these accessories. Do not turn
loose jacket volume into body anatomy. Clothing, hair transparency, lens shading
and eye highlights must be checked in the browser, where DCC shaders differ.

Keep the editable high-resolution master and maps outside the runtime bundle.
Retopologize or fit a deformation-friendly mesh, create UVs, bake supported
details and prepare browser PBR materials. DCC hair curves, constraints, drivers
and complex correctives need explicit conversion, baking or runtime equivalents;
their presence in a Blender file does not establish GLB support.

### 5.2 Face and speech shapes

Author and calibrate the runtime's facial controls and native visemes for the
**new neutral identity**. A transferred shape basis can lose lip closure or
distort likeness after sculpting; inspect and correct it. Test asymmetric brows,
cheeks, smiles, eye closure/gaze, jaw movement, lip rounding/pressing and mouth
corners, plus the speech combinations used in fresh English/Korean dialogue.

The acceptance target is functional deformation, not a target-name count.
Check nonzero shapes except the deliberately neutral `sil` viseme, attached
teeth/tongue, watertight-looking lip closure, believable eyelids and stable eyes.
Define whether each viseme contains jaw opening; avoid applying the same motion
through a jaw bone, a jaw morph and the viseme simultaneously.

### 5.3 Body, fingers and correctives

Retain spine/chest, neck/head, shoulders, upper arms, forearms and wrists. Plan
**15 articulated digit joints per hand**, conventionally three per digit: 30
additional channels if the existing 16 remain. Freeze names, thumb conventions,
rest poses, rotation limits and aliases before production rigging; this is a
proposed contract, not the current vocabulary.

Use subject-left/right, right-handed Y-up coordinates and normalized XYZW
quaternion deltas consistent with the runtime. Record authored rest transforms,
neutral offsets and source-to-target local bases. Current retargeting applies
rest, neutral and basis-converted bounded deltas; arbitrary DCC driver graphs
are not supported. Adjust joint positions and skin weights after identity fitting.

Inspect fingers individually and in open palms, pointing, counting, a loose
fist and two-handed framing. Correct shoulder collapse, elbow pinching, wrist
twist, finger webbing and jacket intersections. Repeat gesture cycles and return
to rest to catch cumulative drift. The existing hand-direction measurement does
not measure finger curls or reliable forearm axial twist.

## 6. New performances: speech, expression and gestures

The reusable asset must perform new dialogue without replaying its source video.
Keep identity fitting, source-motion observation and performance generation as
separate stages.

Start with a small parameterized gesture vocabulary: open-palm explanation,
emphasis, counting, pointing and two-handed framing. Each gesture has preparation,
stroke, optional hold and recovery phases. Dialogue intent and actual speech
timing schedule the gesture; bounded variation changes amplitude and timing.
Apply IK/joint limits and collision correction before rendering. Support
interruption and a smooth return to rest.

Estimate Jensen-specific motion tendencies only from reviewed, visible footage:
rest pose, handedness, amplitude, cadence and holds. Mark designed motion as
designed. Do not relabel the existing Musk motion library as Jensen observations.
A learned speech-to-gesture network can be evaluated later; it is not required
to prove the first controllable asset.

### Layer ownership

| Layer | Owns | Required composition rule |
| --- | --- | --- |
| Speech | Lip articulation and the declared jaw/tongue contribution | Follow actual audio time; prevent duplicate jaw movement |
| Expression | Brows, cheeks and compatible lower-face expression | Define masks/correctives for smiling while speaking |
| Gaze | Eyes and constrained head following | Blend around blinks and preserve eye limits |
| Gesture | Arms, hands/fingers and bounded torso/head movement | Resolve overlapping cues and enforce reach/joint limits |
| Idle | Breathing, blinks and rest variation | Yield to explicit controls and reduced-motion settings |

The current retargeter suppresses all `mouth*`, `jaw*` and `tongueOut` face
controls during a supported native-viseme blend. That prevents one class of
double deformation but can erase a speaking smile. Plan a region/corrective
policy in the asset and runtime together, then test compound poses.

Preserve three clocks: behavior-plan `elapsedMs`, actual unscaled
`audioElapsedMs`, and source replay `observationElapsedMs`. Source-video offsets
are lineage data, not interchangeable clocks. Prepared timing must match the
exact audio digest. For live PCM, design chunk/utterance identity and timing
anchored to the playback clock, including rescheduling and cancellation; adding
a GLB does not provide this timing. Audio-to-face tools are optional coefficient
generators, not geometry builders or automatic learners of personal mannerisms.

## 7. Runtime integration and artifact contracts

### 7.1 Planned implementation map

| Work package | Existing modules to extend | Required outcome |
| --- | --- | --- |
| Finger controls | `apps/web/src/avatar/channels.ts`, `types.ts`, `validation.ts`; Python observation/export schemas | Versioned finger vocabulary and bounds; preserve v1 readers and reject unknown controls |
| Retarget and calibration | `retarget.ts`, `timeline.ts`, `motion.ts`, `components/AvatarStage.tsx` | Explicit bases/limits, supported-control reporting and per-finger inspection |
| Hand observations | `tools/media/`, `tools/animation/`, `tools/motion/`, `src/bias/motion.py` | Hand landmarks/curls with visibility, source timestamps and review state |
| Speech/expression composition | `performance.ts`, `retarget.ts`, `speech.ts` | Defined lower-face ownership, correctives and bounded combinations |
| Live speech alignment | `apps/web/src/App.tsx` and speech transport/timing components | Playback-clock timing for streamed speech; clear fallback provenance |
| Upper-body framing | `builtin.ts`, `retarget.ts`, `types.ts`, `components/AvatarStage.tsx` | Anatomical anchors and a camera containing the full gesture envelope |
| Subject binding | App selection, `/api/persona` configuration, asset/motion/voice manifests | Match visual subject to session persona, corpus, voice and motion |
| Qualification | `tools/avatar_audition.mjs`, native integration and browser audition scenarios | Exact-file structural evidence plus human and timing evaluation |

Plan an explicit next version of `avatar-clip` for finger motion, keeping current
`avatar-clip.v1` import compatibility. Update TypeScript/Python validation and
export together. A v1 comparison may deliberately use wrist-only motion, but
must not silently discard required finger controls and claim equivalent quality.
No new schema or parser is introduced by this documentation change.

Model normalization currently scales visible height as if the mesh were a whole
person. For a bust, use declared shoulder/waist/head anchors and physical scale,
or retain a full-body base. Test the whole gesture envelope on desktop and mobile;
a neutral screenshot cannot prove that moving elbows or fingertips stay in frame.

### 7.2 Proposed build and release records

The following records are **to be implemented**. Their fields describe the
intended artifact boundary, not an existing accepted JSON format.

| Record | Minimum contents |
| --- | --- |
| Source manifest | Fields from section 3, source-level splits and coverage |
| Build receipt | Source-manifest digest; code revision; environment; fitting config; dependency/model versions and hashes; seeds; checkpoints; measured runtime/memory; manual edits |
| Asset manifest | `subject_id`, asset version/hash, mapping hash, rig/control schema, topology, units/orientation, anatomical anchors, material/texture inventory, appearance period |
| Compatibility record | Code, weights, parametric models, textures and source-media conditions separately; retained license-text hashes and allowed distribution path |
| Quality report | Structural checks; held-out reference inventory; renders; deformation tests; human ratings; speech timing; device profile; unresolved defects |
| Session binding | Persona snapshot, evidence corpus, voice configuration, motion library and visual-asset identities; explicit animation-only audition mode |

Before a conversational Jensen session, reject mismatched subject bindings or
enter a clearly labeled **animation-only audition**. The current UI does not
enforce that rule; implement it before connecting the new visual to live persona
dialogue. A neutral test voice is sufficient for asset/lip-sync auditions and
does not establish Jensen voice likeness.

Store raw media, weights, intermediate meshes, textures and checkpoints under
untracked `data/local/` or an explicitly managed artifact store. Publish small
manifests and review reports in the repository. Preserve editable `.blend`
masters; use FBX as optional interchange. USD is an optional DCC intermediate,
not a current browser or converter input. Exporting any format still requires
verifying that skins, shapes, materials and mappings survived.

### 7.3 Packaging limits and existing commands

Current limits: **64 MiB** for runtime GLB loading, conversion output and offline
inspection; **50 MiB** for the Studio upload UI; **16 MiB** for audition WAV.
Target the tighter 50 MiB model limit for a locally uploaded release candidate.
Use embedded resources and uncompressed or Meshopt geometry. Draco and KTX2
decoders are not bundled; VRM semantics and Gaussian rendering are not supplied
by the GLB loader. Sources: [renderer](../apps/web/src/avatar/renderer.ts),
[Studio](../apps/web/src/App.tsx), [audio importer](../apps/web/src/avatar/auditionAudio.ts),
[converter](../tools/avatar_convert.py) and [inspector](../tools/avatar_audition.mjs).

Optimize the offline master before passing the current converter if its raw GLB
would exceed 64 MiB. A larger intermediate needs a separately bounded staging
path; a later compression step cannot rescue a file the converter already
rejected. Preserve the master and record texture/mesh reductions and their
visible costs.

These existing commands are usable **after** the named local files have been
created; they were not run on a new avatar for this document:

```bash
uv run python tools/avatar_convert.py \
  data/local/jensen-avatar/run-001/master.blend \
  data/local/jensen-avatar/run-001/candidate.glb \
  --report data/local/jensen-avatar/run-001/conversion.json

npm ci --prefix tools/avatar --ignore-scripts

node tools/avatar_audition.mjs inspect \
  data/local/jensen-avatar/run-001/candidate.glb \
  data/local/jensen-avatar/run-001/qualification.json \
  --provenance data/local/jensen-avatar/run-001/provenance.json \
  --audio data/local/jensen-avatar/run-001/audition.wav
```

Create the provenance record using the existing [audition instructions](avatar-audition.md),
with actual generated-asset inputs and license records. Supply a real local WAV
and preserve its digest. Blender and the documented tool dependencies are
prerequisites; output/report paths must be new. Conversion and inspection do not
grade likeness or render quality. For browser calibration and same-audio export,
follow the [audition workflow](avatar-audition.md#3-calibrate-and-audition-in-the-studio).

## 8. Acceptance protocol

All gates below are **proposed and unmeasured for the new avatar**. Record a
failed gate as a failure; do not fill it with upstream demo results or proxy
measurements.

| Gate | Required evidence and pass condition |
| --- | --- |
| Data | Exact source/frame lineage, appearance groups, coverage and independent event-level holdouts; unknown regions identified |
| Identity | Neutral and expressive renders matched to held-out frontal/three-quarter/profile references; human review of likeness under several lighting setups |
| Complete anatomy | Torso through waist, both entire arms/hands and all fingers; no missing geometry hidden by the camera |
| Face | Eye closure/gaze, lip seal, jaw range, teeth/tongue and individual/compound expressions without severe tearing, detached parts or identity collapse |
| Hands/body | Every required digit joint moves; open palms, pointing, counting, loose fist, crossing the torso, hands near face and raised-arm poses retain plausible deformation |
| Fresh speech | Unseen English/Korean utterances where supported; measured lip events against actual audio; inspect closure, silence, drift and talking expressions |
| Gestures | Multiple new gesture categories with finger articulation, plausible transitions, no severe intersections and smooth interruption/recovery |
| Framing | Both elbows/wrists/fingertips remain visible throughout required gestures at agreed desktop/mobile viewports |
| Package | Exact candidate hashes; embedded resources; required nonzero shapes/skins/mappings; supported encoding and size limits; actual browser appearance |
| Persona | Correct subject binding or explicit animation-only audition; motion provenance and test voice stated |
| Runtime | Fixed device/browser/resolution/settings; sustained frame intervals, loading, memory observations and cancellation under concurrent LLM/TTS workload |

Prepare a blinded comparison against the existing neutral proxy and between new
candidate revisions. At least three independent reviewers should score identity,
speaking naturalness and hands/body separately on a predefined 1–5 rubric, with
matching views/audio and retained comments. A proposed promotion gate is a median
of at least 4 in each category with no unresolved anatomy/control blocker.
These are project targets, not reported ratings. Automated identity similarity
may support review but cannot replace it.

For an initial desktop performance target, propose a 60-second 1280×720 audition
with p95 browser frame intervals at or below 33.3 ms, recording the exact machine
and concurrent workload. Freeze this profile before comparing assets; define a
separate mobile target when the intended device is selected. Browser frame
interval is not GPU render duration, and texture dimensions are not measured
VRAM. Record cold-load time and stop/interrupt behavior separately.

For speech timing, annotate expected visible articulation events and report
signed offset, drift and uncertainty against actual audio playback. Freeze the
event-specific offset/drift tolerances, interruption recovery criteria and review
rubric after development calibration and before inspecting final candidate
results. Do not interpret the existing
100 ms same-audio comparison-window tolerance as a lip-sync quality score. The
audition tool enforces comparable conditions; it does not select a quality winner.

## 9. Implementation order and release evidence

| Order | Reviewable deliverable | Exit decision |
| --- | --- | --- |
| 1. Contracts and corpus | Appearance brief, source manifests, independent splits, coverage report, rig vocabulary, fitting-backend selection and device profile | Compatible backend selected; enough observed evidence and explicit gaps to begin fitting |
| 2. Bounded pilots | Mesh fit; eligible ExAvatar/LHM++ experiments; input/config/runtime/failure reports | Choose an available authoring route from actual results |
| 3. Identity and appearance | Editable neutral master, texture lineage, accessories and development renders | Likeness review supports further rigging |
| 4. Complete rig | Face/visemes, eyes/mouth, arms/hands/fingers, correctives and deformation test renders | Required anatomy and controls work in the candidate |
| 5. Runtime completion | Finger schema, mapping, framing, expression composition, subject binding and speech alignment | Actual application can exercise the candidate correctly |
| 6. Performance and qualification | Fresh dialogue/gestures, same-audio auditions, human ratings and target-device measurements | All required gates pass or remain explicitly blocked |
| 7. Promotion | Versioned asset/mapping/report manifest, distribution decision and rollback reference | Release the exact qualified bytes and compatible configuration |

Stages 4 and 5 can proceed in parallel after control names and neutral conventions
are frozen. Keep each implementation PR reviewable; preparation, fitting,
runtime changes and promotion have different evidence requirements. None of the
implementation rows is completed by this documentation PR.

At the first implementation step, select actual compute and pin versions before
estimating training time or cost. Blender authoring and the browser are distinct
from CUDA research environments; do not infer ExAvatar/LHM++ support from the
project's existing MPS language-model experiments. A no-cost tool license does
not account for GPU time, manual correction or restrictive model dependencies.

The completion report must identify the exact asset, dataset and tool revisions;
show new-dialogue facial/hand motion in the actual application; retain raw timing
and review evidence; and list remaining defects. That is the point at which the
project can claim a newly generated, controllable Jensen avatar.
