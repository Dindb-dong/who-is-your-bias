# Jensen Gaussian head avatar (VHAP + GaussianAvatars)

Status record for the photoreal real-time head avatar track. Last updated 2026-11-17.
Diagnosis report with charts: [reports/gaussian-head/avatar-diagnosis-2026-11-16.html](../reports/gaussian-head/avatar-diagnosis-2026-11-16.html).

Current follow-up: [expression and upper-body performance execution](jensen-performance-2026-11-17.md).

## Scope and boundaries

- **Goal.** A FLAME-bound 3D Gaussian head that renders in the browser studio in real time, trained from one
  monocular clip. The optional `?head=gaussian` path now attaches that head to the existing articulated,
  clothed body through the shared studio renderer. The default studio avatar is unchanged.
- **Not in Git.** Trained weights, exported bundles (`apps/web/public/models/gaussian-head/*`, ignored), source
  video, rendered comparison images, and the FLAME / SMPL-X model files (research licences, no redistribution).
  Reports in Git contain numbers only, no face images.
  Inventory, hashes and sharing rules for these files: [local-only-assets.md](local-only-assets.md).
- **Where things live.** Training runs on the lab Slurm cluster under a private work root (`$AVATAR_ROOT` in the
  scripts). Checkpoints and logs stay there. Cluster access details are internal and are not recorded here.

## Pipeline

| Stage | Tool | Command / file |
|---|---|---|
| Tracking | VHAP (monocular) | `python vhap/track.py --data.root_folder data/monocular --exp.output_folder output/monocular/jensen_a_whiteBg_staticOffset --data.sequence jensen_a --data.n_downsample_rgb 2` |
| Export | VHAP | `python vhap/export_as_nerf_dataset.py --src_folder output/monocular/jensen_a_whiteBg_staticOffset --tgt_folder export/monocular/jensen_a_whiteBg_staticOffset_maskBelowLine --background-color white` |
| Patch | GaussianAvatars | [`slurm/patch_min_extent.sh`](../tools/gaussian_head/slurm/patch_min_extent.sh) adds the `GA_MIN_EXTENT` lower bound on `cameras_extent` |
| Training | GaussianAvatars | [`slurm/train_exp.sbatch`](../tools/gaussian_head/slurm/train_exp.sbatch) (300-iteration smoke run, then full run) |
| Bundle export | this repo | [`bundle/export_bundle.py`](../tools/gaussian_head/bundle/export_bundle.py), run inside the GaussianAvatars checkout. Env: `GA_ITER=<iteration>`, `GA_BAKE_SH=1` |
| Browser | this repo | [`AvatarStage.tsx`](../apps/web/src/components/AvatarStage.tsx), [`renderer.ts`](../apps/web/src/avatar/renderer.ts), and [`avatar/gaussian/`](../apps/web/src/avatar/gaussian/) |

Data: 512×512, 15 fps, 3,319 frames. The first 70% (2,324 frames) is the train split and the last 30% (995 frames)
the test split.

Studio preview: put `<name>.json/.bin` in `apps/web/public/models/gaussian-head/` and open
`/?head=gaussian&headBundle=<name>` (default bundle name `jensen_a_c1c3_240k_sh`).
Install local assets before running `npm run build`; the API serves that built copy.
The head and body share the camera, depth buffer, speech/performance clocks and attention controls.
Without a qualified expression driver, the UI labels recorded/static expression fallback explicitly.

For optional learned arm/finger geometry, install the private `jensen-gesture-basis.json` in
`apps/web/public/models/gesture/` before building, then enable **Learned gesture preview** in settings
or use `/?head=gaussian&gesture=learned`. This is an experimental preview, not an accepted behavioral
model. See [training and measured limits](jensen-gesture-training.md).
Semantic brow/mouth calibration requires the paired observations described in
[the expression workflow](../tools/gaussian_head/expression/README.md).

## Code map (`tools/gaussian_head/`)

- `bundle/`: bundle format (`gaussian-head.v1`) and its tooling.
  - `export_bundle.py`: the server-side exporter. It validates the bundle against GaussianAvatars' own FlameHead.
    With `GA_BAKE_SH=1` it averages the SH colour over 64 train views and stores the result as DC.
  - `head_bundle.py`: the numpy reference for the bundle math.
  - JS reference: `gaussian-head-core.js`, `gaussian-head-renderer.js`, `viewer.html`.
  - Checks: `test_core.mjs` (JS vs numpy), `bench.mjs` (fps), `studio_shot.mjs`.
- `analysis/`: one-off diagnosis scripts behind the report. They expect the working layout used at the time
  (`analysis/<run>/…`, `cmp/<variant>/…`) and are kept for provenance.
  - `analyze_ga.py`: run in GaussianAvatars. It renders saved checkpoints and computes PSNR by iteration.
    Known bug: the last iteration repeats the previous one, so the 600k values come from the training log.
- `cmp/`: renders a bundle in WebGL at the exact analysis camera views, for server-vs-browser comparison.
  Run with `MSYS_NO_PATHCONV=1` under Git Bash.
- `smplx/`: transplants the head onto SMPL-X (see below), with a browser preview (`preview.html`).
  - `test_core.mjs` + `ref_forward.py` check the JS forward pass against numpy: 0.0003 mm, PASS.
- `slurm/`: the training job scripts. Pass the node and log path to `sbatch` as `-w <node> -o …`.
  Request `--mem=16G`; the measured MaxRSS is ~4.7 GB.

## Experiments and results

| Run | Change | Best test PSNR | 600k test PSNR | 600k LPIPS | Gaussians |
|---|---|---|---|---|---|
| baseline | official defaults (monocular → `cameras_extent = 0`) | 23.06 (300k) | 22.33 | 0.0786 | 552,553 |
| c1_ext1 | `GA_MIN_EXTENT=1.0` | 23.06 (120k) | 22.69 | 0.0623 | 76,923 |
| **c1c3** | c1_ext1 + `--flame_expr_lr 1e-4` | **23.30 (240k)** | 22.83 | 0.0638 | 92,267 |

c1c3 test PSNR at 60k / 120k / 180k / 240k: 23.02 / 23.12 / 23.15 / 23.30. Each run takes ~11.6 h to 600k on
one RTX 4090. Each run was trained once, so treat differences of 0.1–0.2 dB as noise.

### Findings (details and evidence labels in the report)

1. **Browser blur came from dropping SH degrees 1–3.**
   - The WebGL renderer matches the server's DC-only render at 35 dB.
   - The DC-only render matches the full-SH render at only 18 dB.
   - Monocular training has no view variation, so the higher SH bands store colour itself.
2. **SH bake fix (E1), with no retraining.**
   - WebGL vs server full SH improved from 18.5 to 27.5 dB (train views) and from 19.0 to 26.4 dB (test views).
   - The studio now uses c1c3 at 240k, baked: 80,014 gaussians, 34 fps on the laptop (was 29 fps).
3. **77% of gaussians below 0.005 opacity in the baseline.**
   - Cause: `cameras_extent = 0` makes `big_points_ws` true for every point after 60k.
   - The rule that keeps at least one gaussian per face then blocks all pruning, and clone can never trigger.
   - Fixed by the extent lower bound, which brings the share down to 10–13%.
4. **Test PSNR drops after the peak.**
   - The drop after the peak is mainly overfitting: it is uniform across pose-distance terciles, while train PSNR
     keeps rising.
   - The overall level is set by pose distribution shift: far frames score 1.6–1.9 dB lower.
   - Recommended stop point: about 240k.
5. **Remaining visible issues.**
   - A white rim and a white sheet below the mesh, from training on a white background only. Proposed fixes:
     B2 (randomise the background) and D1 (head-only mask).
   - Gaussians are flat discs: the shortest axis is near zero, so they can split into lines at side views.
   - 54% of gaussians sit on the neck and lower boundary, acting as a shoulder/clothes sheet.

### SMPL-X transplant (preview only)

[`smplx/smplx_head.py`](../tools/gaussian_head/smplx/smplx_head.py) moves the trained head onto an SMPL-X body.

- **Shape and joints.** It fits 100 betas plus a translation to the head. Head and teeth keep FLAME's exact shape,
  expression basis, pose correctives and skinning. The joints are re-parented as root→12, neck→15, jaw→22,
  eyes→23/24.
- **Gaussians.** All 9,976 FLAME faces exist in SMPL-X with the same winding, so the gaussian bindings carry over
  unchanged.
- **Neck seam.** A 6 cm smoothstep blend of the skinning weights removes tearing at the seam. Edge stretch is
  1.07 (neck turn), 1.01 (nod and spine) and 1.12 (shrug), down from up to 8.7 without the blend.
- **Checks.** The transplant matches FLAME exactly at rest (0.0001 mm on the face).
- **Expression basis.** The SMPL-X expression basis differs from FLAME's (median R² 0.62), so the FLAME basis is
  carried over.

[`smplx/preview_export.py`](../tools/gaussian_head/smplx/preview_export.py) applies a collar cut (`CUT_M`, default
5 cm) and writes the `smplx-head-preview.v1` bundle. The preview renders at 42–47 fps.

Preview issues still open:
- Side-view artifacts: training only covered about ±30° of head yaw.
- Dark jacket remnants behind the neck.
- The body is the neutral SMPL-X model; the male model could be used instead.

## In progress

**Data-quantity experiment** (job commands prepared 2026-11-16; submission and results not yet confirmed in this record).

- c1c3 settings, trained on the first 50% (1,162) and 25% (581) of train frames, stopped at 240k.
- The test split is the same 995 frames.
- GaussianAvatars fixes `position_lr_max_steps` and `densify_until_iter` at 600k. These runs therefore follow
  the full-data c1c3 schedule up to 240k and compare directly with its 60k–240k checkpoints.
- Scripts: [`make_train_subset.sh`](../tools/gaussian_head/slurm/make_train_subset.sh) and
  [`train_scale.sbatch`](../tools/gaussian_head/slurm/train_scale.sbatch).
- Expectation (not yet measured): more frames of the same frontal clip mainly help overfitting and part of the
  pose shift. Sharper detail needs a higher-resolution source, and side views need multi-angle footage.

## Follow-up status and remaining steps

- Implemented on the current follow-up branch: exact-hash-bound train-frame expression replay
  (0–2323), removal of global idle damping, shared articulated-body playback, and a calibrated semantic
  expression-driver route. Real-bundle visual acceptance is still pending the private assets.
- The body composite uses the existing clothed GLB, with bounded collar-to-neck transform blending.
  It does not assert completion of the separate SMPL-X identity/texture training track.
- D1 + B2 retrain to remove the white rim and the shoulder sheet.
- SMPL-X preview: filter the neck remnants, switch to the male body, and integrate it with the clothed body GLB in
  the three.js studio.
- Evaluation: report the near-pose tercile separately. For future captures, split train/test in alternating
  1-second blocks.
