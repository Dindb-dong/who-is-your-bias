# Upper-body Gaussian avatar (head + torso + arms + hands)

Status: **work in progress**. `body_model.py` is verified, and `keypoints.py` + `fit_body.py` were checked locally on a
1-in-12 frame subset (body ~28 px, hands ~7.5 px reprojection). Nothing has been trained yet.
`upperbody_prep_cpu.sbatch` is the same prep job without a GPU (the export without mesh mask, keypoints and the fit all
run on CPU).

## Why

The head-only GaussianAvatars model is attached to a separate body. The neck–body seam looks wrong, so this pipeline
binds gaussians to one continuous SMPL-X surface instead.

- **Head:** FLAME tracking from VHAP, transplanted with `../smplx/smplx_head.py`.
- **Body:** SMPL-X torso, arms and hands, fitted per frame to 2D whole-body keypoints.

The head stays exactly on the FLAME tracking (`body_model.py`: the neck rotation and body translation are solved from
it). So the existing face quality and tracking are kept, and the neck and shoulders become one surface.

The output is driven by SMPL-X body pose plus FLAME expression. That is the format EMAGE and StreamTalk produce
(`../gesture/`).

## Pipeline

| Step | Script | Where |
|---|---|---|
| 0 | `../smplx/smplx_head.py`: FLAME head → SMPL-X transplant (`jensen_c1c3_smplx_head_b0.06.npz`) | laptop (done) |
| 1 | VHAP re-export **without** the neck mask: `export_as_nerf_dataset.py ... --no-create-mask-from-mesh` | cluster (`upperbody_prep.sbatch`) |
| 2 | `keypoints.py`: RTMPose whole-body 2D keypoints per timestep (resumable) | cluster, CPU or GPU |
| 3 | `fit_body.py`: per-frame SMPL-X torso/arm/hand pose. Head locked to FLAME; robust 2D reprojection with pose priors, elbow hinge and temporal smoothness; writes an overlay image | cluster, GPU |
| 4 | `prepare_dataset.py`: training folder; whitens pixels below the fitted hip line (lap and sofa are not modelled) | cluster |
| 5 | `ga_patch/apply_patch.py` + `train_upperbody.sbatch`: GaussianAvatars with `SmplxGaussianModel` | cluster, GPU |
| 6 | `export_upperbody.py`: browser export (`smplx-head-preview.v1` with body faces bound): **written, untested** | cluster |

## Details

`SmplxGaussianModel`:
- **Bound faces.** The FLAME head and teeth faces come first, in the same order as `FlameGaussianModel`. They are
  followed by the SMPL-X faces that are not skinned to the legs.
- **Fine-tuning.** FLAME pose, translation and expression are fine-tuned as in upstream. The body pose is also
  fine-tuned, at learning rate `GA_BODY_POSE_LR` (default 1e-5).
- **Switching it on.** The class is only used when `GA_UPPERBODY=<transplant npz>` is set. Without it, `train.py`
  behaves exactly as upstream.

## Check

`check_body_model.py <bundle_stem> <smplx_head.npz>` compares the torch model with the FLAME bundle:
- Head vertices with pure FLAME skinning deviate at most **0.24 mm** with the body at rest.
- They deviate at most **0.28 mm** with random torso and arm poses.
- The skin-weight blend near the neck seam follows the torso on purpose.

## Files needed on the cluster (`$AVATAR_ROOT/upperbody/`)

- `jensen_c1c3_smplx_head_b0.06.npz`: the transplant. It is derived from FLAME, so it is research-licensed and must
  not go into Git.
- `SMPLX_NEUTRAL_2020.npz`: the SMPL-X model. Each user needs their own licensed copy.
- `tools/`: a copy of this folder.

## Face + neck bundle for the studio (`upperbody_to_head_bundle.mjs`)

The full upper body is not usable in the studio as is. The arms are blurry
fragments, because the source crop rarely shows them, and the torso edges
carry white-background haze.

What the studio uses instead is the jointly trained head and neck:

```
node upperbody_to_head_bundle.mjs <export>/jensen_upperbody_120000 \
  apps/web/public/models/gaussian-head/jensen_a_c1c3_240k_sh \
  apps/web/public/models/gaussian-head/jensen_ub_head_120k --keep 12 --white 0.5
```

The script writes a normal `gaussian-head.v1` bundle:

- **Base.** It copies the studio's FLAME arrays and expression replay.
- **Gaussians.** It takes the upper-body model's FLAME-bound Gaussians. They
  include the neck and the black T-shirt neckline.
- **Collar cut.** It writes a negative `collar_cut_m`, so the Gaussians below
  the FLAME boundary are kept. The neck seam still starts at the FLAME boundary
  (`seamStartY`), so the neck follows the neck joint and the face follows the
  head joint.

Open the result with `?head=gaussian&headBundle=jensen_ub_head_120k`.

`--keep 12,13,14` also re-binds the SMPL-X collar Gaussians to the nearest
FLAME faces. On 2026-11-21 that left a grey wisp over one shoulder, so
`--keep 12` is preferred. With `--keep 12`, no body Gaussians pass the filter
and only the FLAME-bound ones are used.

The outputs are research-licensed and are not committed.
