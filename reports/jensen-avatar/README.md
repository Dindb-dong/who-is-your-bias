# Jensen source-fitted avatar and hand performance pilot

The new, editable Jensen **candidate v3c** now has a concrete
[2.53 MB runtime GLB](../../apps/web/public/models/jensen-source-fitted-v3.glb),
[attribution and license](../../apps/web/public/models/jensen-source-fitted-v3.LICENSE.md),
and [actual neutral render](renders/textured-v3c-neutral.png). Choose **Jensen
source-fitted candidate** in Studio settings to use it, then open Motion Lab for
the measured hand clips. It is an explicit optional choice; the designed Jensen
default remains unchanged. Human likeness acceptance has not passed.

The candidate combines new source-measured facial proportions and dense surface
deformation with a two-photo licensed skin bake, a complete articulated body,
independent glasses, and authored dark irises, gray hair and subtle eyebrows.
The [packaging report](runtime-asset-report.json) binds the exact shipped bytes to
the uncompressed source: all 67 native face/speech target names, 66 nonzero
targets, and 163 skeletal joints survive compression. The actual decoded asset
passes finger deformation/reset, facial deformation, and Jensen clip consent and
cancellation checks. This is a fitted animated mesh, not a static portrait.

The completed dense-v2 attempt is **rejected for promotion**: it reduced landmark
RMSE from 0.0125784 to 0.00632817, but remained too generic and moved the rigid
glasses inside the head. The exact failed model and editable master are retained
in the `jensen-dense-fit-v2-rejected` artifact package. Its
[measured result and visual findings](dense-v2-rejected-results.json),
[coarse render](renders/coarse-v1-neutral.png), and
[rejected dense render](renders/dense-v2-neutral-rejected.png) are in the repository.
Commit `570ad5b` implements rigid-accessory preservation; the corrected dense-v2b
render now shows the glasses in front of the face. Its complete package was
uploaded and freshly retrieved with matching SHA-256; see the
[derived artifact receipts](derived-artifact-receipts.json).

The first licensed photo texture bake also **ran**: 221,259 skin UV pixels received
source appearance, producing a 17.5 MB GLB and 28 MB editable master. The
[textured-v3 render](renders/textured-v3-neutral-rejected.png) is rejected because
authored hair compression exposed the scalp and source eye pixels overlapped the
rig's eyes. The [bake result and failure findings](textured-v3-rejected-results.json)
preserve actual output hashes, attribution, and limitations. The corrected v3b
bake **ran**, and the subsequent v3c appearance pass **ran**. Their
[texture result](texture-bake-results.json) and
[appearance result](appearance-results.json) distinguish 214,578 skin UV pixels
with source contribution from authored iris/hair/brow changes. Complete v3b and
v3c editable masters were uploaded and freshly retrieved with matching SHA-256.
The v3c master is 28,495,448 bytes; its durable receipt is in the artifact manifest
linked above. Pale irises and white hair were corrected, but the hair silhouette,
face geometry and baked lighting still limit likeness.

## Actual inputs and outputs

The initial identity measurement used seven accepted images across three event
families: two official NVIDIA portraits, two Stanford CS153 photos, and three
frames from the Goldman Sachs 2024 interview. One additional Stanford photograph
was excluded for occlusion/multiple people. Repeated portraits from one session
receive one event-family vote. All sources belong to training families; the
held-out WEF Davos 2026 and No Priors 2023 sources do not enter fitting.

- NVIDIA [headshot](https://nvidianews.nvidia.com/file/jensen-huang-headshot-black-background300738)
  and [portrait](https://nvidianews.nvidia.com/file/jensen-1920x1920-305868): official
  reference media, with no open redistribution license asserted.
- Stanford photos [008](https://commons.wikimedia.org/wiki/File:Jensen_huang_stanford_2026-04-30_008.jpg)
  and [010](https://commons.wikimedia.org/wiki/File:Jensen_huang_stanford_2026-04-30_010.jpg):
  Anders Eidesvik, CC BY-SA 4.0, taken 2026-04-30. Photo 024 is excluded.
- [Goldman Sachs interview](https://www.goldmansachs.com/insights/talks-at-gs/jensen-huang):
  recorded 2024-09-11, published 2024-11-11; reference frames at publisher elapsed
  130, 150 and 170 seconds. This pilot used the hash-bound 180-second acquisition.

The initial coarse/dense candidates copied no source photograph pixels. The
later texture experiment uses only the two credited Stanford photographs under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The source texture
and derived textured candidate are distributed under that license, with warping,
blending, UV baking and other changes identified in the bake report. The foundation
is the repository's generic CC0 anatomical rig, hash
`aee550c9cb5b20c6afccfb038c42c2eb8d7aab2c8fb74ca977aefed56f633519`.
It is not a ready-made celebrity model. Hair/glasses/garment styling remains
authored and must be evaluated separately from measured geometry.

The completed coarse run produced a new 17 MB uncompressed GLB, an editable
Blender master, a neutral render, every tenth fitting state, and a full fit report.
The source-bound six-parameter objective fell from 0.0130520 to 0.0000264972 in 80
iterations. These are geometric optimization units, not a percentage likeness
score. [The measured report](coarse-fit-results.json) contains all parameters,
source hashes, mesh-anchor bindings and output hashes.

The actual step-20 state was uploaded, freshly retrieved from durable storage,
and resumed to step 80. The resumed final JSON was byte-identical to an
uninterrupted 80-step run, including optimizer step size and the full history.
[Resume evidence](resume-check.json) records the exact input and output hashes.

## Hands and runtime composition

The runtime vocabulary now adds 30 optional finger joints to the existing
16 body/face joints. Existing `avatar-clip.v1` files remain valid. Explicit bone
mapping covers the foundation's actual MPFB finger chains. Missing observations
remain absent; measured controls override only their own designed channels.
The upper-body camera considers both viewport dimensions so that the declared
arm/hand envelope remains visible on narrow and wide screens.

The first measured motion batch contains three eight-second Goldman windows at
10 Hz: **240 actual observation frames**, with face, pose and hand model hashes.
[Observation counts](motion-pilot-observations.json) preserve missing/multiple
detections rather than calling them usable frames. The raw observations do not
establish speaker identity or continuous track association.

`tools.motion.preview_selection` requires an explicit selected face/pose index
and an inspected bounded window. It skips multi-person frames and associates
hands only when wrists are unambiguously close to the selected pose. It always
emits `geometry_preview`; a model's handedness label cannot promote identity.
Finger segment swing is measured, while axial twist and hidden joints remain
unavailable. The second completed event adds two twelve-second Stripe Sessions
2024 windows: 240 more actual observation frames. Their source video and raw
observations have both passed fresh durable readback. Fingers are retained on
98/120 and 108/120 frames; arm observations are less complete because of source
cropping and confidence gates. A third WIRED event measurement is in progress.

Three actual exported clips are now versioned under
`apps/web/public/motion/jensen/`, with hash-bound selections under
`data/motion/reviews/jensen/`. Each clip carries 30 finger channels; fingers are
available on 68, 59 and 71 frames respectively. The standalone
`data/motion/jensen-goldman-v1.json` library binds explicitly to `jensen-huang`.
It contains no text-context assertions and does not enable automatic conversation
selection. The new `data/motion/jensen-multievent.v1.json` catalog contains five
clips, two events, 480 observed frames and 28 preview primitives. It also has no
context assertions, reviewed runtime primitives or held-out quality result.
The Jensen Motion Lab offers **Load Jensen face & hands**, a source-event
selector, and an explicit geometry-preview checkbox. Imported clips and audio reset when the
persona changes. Historical Musk clip/narration buttons are not offered in the
Jensen lab.

The focused browser test was attempted on desktop and mobile, but Chromium exited
before creating a page: `socket() failed: Operation not permitted` in
`process_singleton_posix.cc`. Those two attempts are environment-blocked, not
passing UI tests. Web type checking and the production build passed; actual
browser rendering/replay acceptance remains open. Six actual Blender CPU renders
of v3c cover front, three-quarter, profile, an observed frame and desktop/mobile
hand articulation stress. All 3,586 selected hand skin vertices remain in frame.
Visual review found excessive mobile framing margin; a tighter camera check is
the next bounded correction. These renders are offline evidence, not browser
screenshots or proof of WebGL material parity.

## Reproduction and remaining gates

The code is in `tools/avatar/{identity,prepare_fit,fit_mesh,surface_fit,refine_surface}.py`,
`tools/avatar/{decode,optimize}.mjs`, `tools/media/observe.py`,
`tools/animation/convert.py`, and `tools/motion/preview_selection.py`.
Raw images, videos, model task files, masters and reusable intermediate packages
are retained through the parent execution record's artifact receipts, not in Git.
The code checkpoints and artifact receipts are separate preservation claims.

The exact optimized asset's native geometry checks passed (including every
finger, three face controls, measured clip consent and reset). The native and
attention test files passed 26 tests together. Remaining acceptance gates are
browser rendering, the tighter mobile camera, continuous source/speaker review,
and independent human likeness/performance assessment. The observed optimizer
residual cannot pass those gates by itself.

### Next commands after restoring artifact packages

The following reproduce completed stages. Set `AVATAR_ARTIFACT_ROOT`
to the directory containing the restored packages. Use a fresh output directory;
completed candidates are immutable.

```bash
OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 blender --background --threads 3 \
  --python tools/avatar/refine_surface.py -- \
  "$AVATAR_ARTIFACT_ROOT/coarse-fit-v1/jensen-landmark-candidate-v1.blend" \
  "$AVATAR_ARTIFACT_ROOT/template-binding/template-binding.json" \
  "$AVATAR_ARTIFACT_ROOT/template-binding/template-landmarks.json" \
  "$AVATAR_ARTIFACT_ROOT/multimodal-measurements.json" \
  "$AVATAR_ARTIFACT_ROOT/dense-fit-v2b"

OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 blender --background --threads 3 \
  --python tools/avatar/bake_reference_texture.py -- \
  "$AVATAR_ARTIFACT_ROOT/dense-fit-v2b/jensen-landmark-candidate-v2.blend" \
  "$AVATAR_ARTIFACT_ROOT/template-binding/template-binding.json" \
  "$AVATAR_ARTIFACT_ROOT/template-binding/template-landmarks.json" \
  "$AVATAR_ARTIFACT_ROOT/multimodal-measurements.json" \
  "$AVATAR_ARTIFACT_ROOT/references" \
  reports/jensen-avatar/texture-reference-plan.json \
  "$AVATAR_ARTIFACT_ROOT/textured-fit-v3b"

OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 blender --background --threads 3 \
  --python tools/avatar/refine_appearance.py -- \
  "$AVATAR_ARTIFACT_ROOT/textured-fit-v3b/jensen-source-fitted-candidate-v3.blend" \
  "$AVATAR_ARTIFACT_ROOT/textured-fit-v3c"

node tools/avatar/optimize.mjs \
  "$AVATAR_ARTIFACT_ROOT/textured-fit-v3c/jensen-source-fitted-candidate-v3.glb" \
  apps/web/public/models/jensen-source-fitted-v3.glb \
  reports/jensen-avatar/runtime-asset-report.json jensen-source-fitted-v3

PYTHONPATH=src:. python -m tools.motion check \
  data/motion/jensen-multievent.v1.json apps/web/public
```
