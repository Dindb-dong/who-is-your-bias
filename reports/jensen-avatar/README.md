# Jensen source-fitted avatar and hand performance pilot

This is a new, editable Jensen **candidate**, not accepted likeness. The first
coarse render is anatomically coherent but still resembles the generic foundation
too strongly. It must not replace the designed default under a claim that Jensen
appearance has been reproduced. Dense landmark refinement and visual comparison
are the next gate.

The completed dense-v2 attempt is **rejected for promotion**: it reduced landmark
RMSE from 0.0125784 to 0.00632817, but remained too generic and moved the rigid
glasses inside the head. The exact failed model and editable master are retained
in the `jensen-dense-fit-v2-rejected` artifact package. Its
[measured result and visual findings](dense-v2-rejected-results.json),
[coarse render](renders/coarse-v1-neutral.png), and
[rejected dense render](renders/dense-v2-neutral-rejected.png) are in the repository.
Commit `570ad5b` implements rigid-accessory preservation; rerender verification
is still pending. Source-derived texture from the licensed Stanford photographs
is proposed, not yet implemented or baked.

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

No source photograph pixels were copied into the runtime texture. The foundation
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
unavailable. This initial event is a pilot; broader full-video acquisition and
additional event windows are still being prepared.

Three actual exported clips are now versioned under
`apps/web/public/motion/jensen/`, with hash-bound selections under
`data/motion/reviews/jensen/`. Each clip carries 30 finger channels; fingers are
available on 68, 59 and 71 frames respectively. The standalone
`data/motion/jensen-goldman-v1.json` library binds explicitly to `jensen-huang`.
It contains no text-context assertions and does not enable automatic conversation
selection. Import a clip in Motion & rig calibration and explicitly enable
geometry preview to inspect its measured motion.

## Reproduction and remaining gates

The code is in `tools/avatar/{identity,prepare_fit,fit_mesh,surface_fit,refine_surface}.py`,
`tools/avatar/{decode,optimize}.mjs`, `tools/media/observe.py`,
`tools/animation/convert.py`, and `tools/motion/preview_selection.py`.
Raw images, videos, model task files, masters and reusable intermediate packages
are retained through the parent execution record's artifact receipts, not in Git.
The code checkpoints and artifact receipts are separate preservation claims.

Remaining acceptance gates are dense-fit visual review, expression/skin/finger
deformation checks on the exact final asset, desktop/mobile rendered evidence,
more than one measured video event, source-bound clip qualification, and human
likeness/performance assessment. The observed optimizer residual cannot pass any
of those gates by itself.

### Next commands after restoring artifact packages

The following are proposed next execution commands. Set `AVATAR_ARTIFACT_ROOT`
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

PYTHONPATH=src:. python -m tools.motion check \
  data/motion/jensen-goldman-v1.json apps/web/public
```
