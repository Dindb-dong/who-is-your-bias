# Local-only assets (not in Git)

This repository is **public**. The files below are needed for some workflows but must not be committed or pushed.
The reasons are licences, internal source footage, or real-person likeness. This page records what exists, where
it lives, how to check a copy (SHA-256), and how to obtain or rebuild it. Last updated 2026-11-17.

## Rules

- **Share privately.** Only share with project members, over a private channel such as direct transfer or a team
  drive. Never use a public link, a public repo, or a release asset.
- **Licences apply to recipients too.** FLAME and SMPL-X require each user to register for the model themselves.
  GaussianAvatars outputs are CC BY-NC-SA (non-commercial). The trained heads come from internal footage and are
  for internal research use only.
- **Don't force-add.** These paths are already in `.gitignore` (`apps/web/public/models/*`, `models/`, `data/raw/`,
  `data/local/`, `.env`). Never override that with `git add -f`. Git history is permanent and this repo is public.
- **Keep this page current.** When you add a new local-only asset, add a row here with its path, size, SHA-256 and
  how to rebuild it.

## 1. Gaussian head bundles (browser studio)

**Location:** `apps/web/public/models/gaussian-head/` in each user's checkout (ignored). Open the studio with
`/?head=gaussian&headBundle=<name>`.

**Format:** schema `gaussian-head.v1`.
- Format and loader code: [`tools/gaussian_head/bundle/head_bundle.py`](../tools/gaussian_head/bundle/head_bundle.py)
  and [`apps/web/src/avatar/gaussian/gaussianHeadCore.ts`](../apps/web/src/avatar/gaussian/gaussianHeadCore.ts).
- Each bundle holds FLAME-derived data: the shaped template, 100 expression directions, pose correctives and
  skinning weights. It also holds the trained gaussians and the tracked per-frame FLAME parameters.

| Bundle (`.json` + `.bin`) | What it is | `.bin` bytes | SHA-256 `.json` | SHA-256 `.bin` |
|---|---|---|---|---|
| `jensen_a_c1c3_240k_sh` | **Current studio bundle.** c1c3 run at 240k, SH baked, 80,014 gaussians | 10,856,460 | `3891190a06c81e1968a625b868f6e41f5651cb86a41ef3749991a81155c1f753` | `32d5bc15901a8673901c6234ae7af946d2bfcf2c0aa5e7ed9c816d1c3aff599c` |
| `jensen_a_sh` | Baseline run at 600k, SH baked, 132,438 gaussians | 14,001,900 | `b3681b793b0137388277d26ad6f21ee1144e026de0bb689c625f4626d13dab3f` | `1a8c3bed73cdc93553257887bd682b3d87ea5c9dd1d9b2d6f4abed730e0605c7` |
| `jensen_a` | Baseline run at 600k, DC colour only (looks blurry; kept for comparison) | 14,001,900 | `8a108dafead696d82f6a03505832f3e764e5b6ad79a4e99bbe766cea83d888fd` | `727ac52904b7a610b8b3bd59594ed85ef9990fd4b2105e8dc5a57079bc852c0d` |

A zip of this folder for hand-off also exists at `apps/web/public/models/gaussian-head.zip`
(SHA-256 `8f33e6b37da63674c43331bd87d899f92b3f6b39e9e111713618c7fc62a24793`).

**Rebuild from a checkpoint:**
1. Run [`export_bundle.py`](../tools/gaussian_head/bundle/export_bundle.py) inside the GaussianAvatars checkout on
   the cluster:
   `GA_ITER=240000 GA_BAKE_SH=1 python export_bundle.py <model_dir> <out_stem>`
2. Copy the `.json` and `.bin` into the folder above.

## 2. SMPL-X transplant outputs (preview)

These are made by [`smplx_head.py`](../tools/gaussian_head/smplx/smplx_head.py) and
[`preview_export.py`](../tools/gaussian_head/smplx/preview_export.py). They contain SMPL-X data as well as FLAME
data, so a recipient must be registered for both.

**Where they are now:** a temporary working folder on the author's laptop. That folder can be wiped, so if they
need to be kept, move them to the team drive and update this row.

| File | What it is | Bytes | SHA-256 |
|---|---|---|---|
| `jensen_c1c3_smplx_head_b0.06.npz` | Head transplanted onto SMPL-X with a 6 cm seam blend (used for the preview) | 10,304,775 | `1e36123929706661785f0b330a1c10d1a8bdea64eeb8f8eddf3319ecd4497433` |
| `preview_cut0.05.json` / `.bin` | Browser preview bundle (`smplx-head-preview.v1`) with a 5 cm collar cut | 9,043,896 (`.bin`) | `7cb585d96ee95f5feb3f093d8a1ff4df0c9f2aed5b31421c7bcaf7ef6a7ac6a5` / `13ba5d96ec7159febc110775157495214cb3054e03a2629746a04bbf71acba97` |

The same folder also holds variants:
- seam blends of 0, 0.04 and 0.09 m
- collar cuts of 3 cm and 7 cm

To rebuild, use the commands in each script's docstring. Inputs:
- the bundle `jensen_a_c1c3_240k_sh`
- `SMPLX_NEUTRAL_2020.npz`
- `SMPL-X__FLAME_vertex_ids.npy`

## 3. On the lab cluster (training side)

All of these live under the avatar work root on the cluster (`$AVATAR_ROOT` in
[`tools/gaussian_head/slurm/`](../tools/gaussian_head/slurm/)). Ask the track owner for access; cluster
connection details are internal and are not written down here.

| Path under `$AVATAR_ROOT` | What it is |
|---|---|
| `VHAP/data/monocular/jensen_a*` | Source clip and preprocessed frames (internal footage) |
| `VHAP/output/monocular/jensen_a_whiteBg_staticOffset` | VHAP FLAME tracking result |
| `VHAP/export/monocular/jensen_a_whiteBg_staticOffset_maskBelowLine` | Training dataset (512×512, 3,319 frames; 2,324 train / 995 test) |
| `VHAP/export/monocular/…_train50`, `…_train25` | Data-quantity subsets: symlinks plus a truncated `transforms_train.json` |
| `GaussianAvatars/output/jensen_a_c1_ext1`, `jensen_a_c1c3` | Checkpoints every 60k up to 600k, with logs |
| `GaussianAvatars/output/jensen_a` | Baseline run checkpoints (60k–600k) |
| `GaussianAvatars/output/jensen_a_c1c3_train50`, `…_train25` | Data-quantity runs to 240k (in progress / pending) |
| `logs/` | Slurm job logs (`%x_%j.log`) |

## 4. Licensed body/face models

Each user obtains these themselves after registering on the official sites. Do not pass copies around.

| File | Licence / source | Used by |
|---|---|---|
| FLAME 2023 model (`flame2023.pkl` / VHAP assets), FLAME masks | FLAME licence (flame.is.tue.mpg.de) | VHAP, GaussianAvatars, analysis |
| `SMPLX_NEUTRAL_2020.npz` (also MALE / FEMALE) | SMPL-X licence (smpl-x.is.tue.mpg.de) | `smplx_head.py` |
| `SMPL-X__FLAME_vertex_ids.npy` | SMPL-X download (FLAME correspondence) | `smplx_head.py`, `preview_export.py` |

## 5. Other ignored local files

| Path | What it is | How to get it |
|---|---|---|
| `.env` | API keys and local settings (secrets) | Create from `.env.example`; never share in chat or Git |
| `apps/web/public/vision/*` | Pinned face-detector runtime | Downloaded by `npm run prepare:vision` |
| `apps/web/dist/` | Web build output | `npm run build` |
