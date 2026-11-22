"""Build the upper-body training folder from a VHAP export made WITHOUT the FLAME neck mask.

    python prepare_dataset.py <export_no_mesh_mask> <body_fit.npz> <smplx_head.npz> <out_dir> [--margin_px 12]

<export_no_mesh_mask>: `vhap/export_as_nerf_dataset.py ... --no-create-mask-from-mesh` (person matte only).
Everything is symlinked except images/ and fg_masks/, which are rewritten so that pixels below the fitted hip line
(seated lap, sofa, legs: not modelled) become background. smplx_body_param.npz (the fit) is copied in, where
SmplxGaussianModel reads it.
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_model import TransplantedSmplx  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('export'); ap.add_argument('fit'); ap.add_argument('transplant'); ap.add_argument('out')
ap.add_argument('--margin_px', type=float, default=12)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
for name in os.listdir(a.export):
    if name in ('images', 'fg_masks'):
        continue
    dst = os.path.join(a.out, name)
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(os.path.join(a.export, name)), dst)
for d in ('images', 'fg_masks'):
    os.makedirs(os.path.join(a.out, d), exist_ok=True)
shutil.copy(a.fit, os.path.join(a.out, 'smplx_body_param.npz'))

fit = np.load(a.fit)
pose_by_t = dict(zip(fit['timesteps'].tolist(), fit['body_pose']))
model = TransplantedSmplx(np.load(a.transplant))
n = 0
for split in ('train', 'val', 'test'):
    p = os.path.join(a.export, f'transforms_{split}.json')
    if not os.path.exists(p):
        continue
    for fr in json.load(open(p))['frames']:
        t = int(fr['timestep_index'])
        fp = np.load(os.path.join(a.export, fr['flame_param_path']))
        flame = {k: torch.tensor(np.asarray(fp[k], np.float32).reshape(1, -1)) for k in
                 ('expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation')}
        with torch.no_grad():
            J = model(torch.tensor(pose_by_t[t][None]), flame, return_verts=False)['joints'][0].numpy()
        c2w = np.array(fr['transform_matrix'], np.float64); c2w[:3, 1:3] *= -1; w2c = np.linalg.inv(c2w)
        hips = J[[1, 2]] @ w2c[:3, :3].T + w2c[:3, 3]
        u = fr['fl_x'] * hips[:, 0] / hips[:, 2] + fr['cx']; v = fr['fl_y'] * hips[:, 1] / hips[:, 2] + fr['cy']
        img = cv2.imread(os.path.join(a.export, fr['file_path']), cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        xs = np.arange(w)
        k = (v[1] - v[0]) / (u[1] - u[0]) if abs(u[1] - u[0]) > 1e-3 else 0.0
        yline = v[0] + k * (xs - u[0]) - a.margin_px                          # hip line, slightly above the joints
        below = np.arange(h)[:, None] > yline[None, :]
        img[below] = 255
        cv2.imwrite(os.path.join(a.out, fr['file_path']), img)
        if 'fg_mask_path' in fr and os.path.exists(os.path.join(a.export, fr['fg_mask_path'])):
            m = cv2.imread(os.path.join(a.export, fr['fg_mask_path']), cv2.IMREAD_UNCHANGED)
            m[below] = 0
            cv2.imwrite(os.path.join(a.out, fr['fg_mask_path']), m)
        n += 1
print(f'prepared {n} frames into {a.out}')
