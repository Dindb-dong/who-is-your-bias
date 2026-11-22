"""Find upper-body training frames to drop: empty/cut-away frames and frames where the body fit failed.

    python frame_audit.py <export_no_mesh_mask> <smplx_head.npz> <keypoints.npz> <body_fit.npz> <dataset_dir>
                          [--min_fg 0.03] [--max_arm_px 80] [--max_head_px 80] [--apply]

Per frame (camera 0) it measures:
  fg       share of non-white pixels in the export image (white background), so a frame without the person is ~0
  arm px   mean reprojection error of shoulders, elbows and wrists (confident keypoints inside the image)
  hip px   the same for the hips (reported separately; seated hips are often guessed by the detector)
  head px  distance between the fitted SMPL-X head joint and the detected nose, i.e. whether the FLAME head
           tracking and the 2D person agree (a different person or a lost track gives a large value)
It prints per-joint statistics, the flagged frames as runs, and writes <dataset_dir>/audit/audit_worst.jpg
(the 16 worst flagged frames; red = detected keypoints, cyan = fitted joints).

--apply rewrites <dataset_dir>/transforms_{train,val,test}.json WITHOUT the flagged frames. The originals in the
export are never touched: the dataset files are symlinks to them, so they are replaced by real files built from the
export's copy every time (re-running with other thresholds starts from the full list again).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_model import TransplantedSmplx  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('export'); ap.add_argument('transplant'); ap.add_argument('keypoints'); ap.add_argument('fit'); ap.add_argument('dataset')
ap.add_argument('--min_fg', type=float, default=0.03)
ap.add_argument('--max_arm_px', type=float, default=80)
ap.add_argument('--max_head_px', type=float, default=80)
ap.add_argument('--apply', action='store_true')
a = ap.parse_args()
torch.set_grad_enabled(False)

frames = {}
for split in ('train', 'val', 'test'):
    p = os.path.join(a.export, f'transforms_{split}.json')
    if os.path.exists(p):
        for fr in json.load(open(p))['frames']:
            if fr.get('camera_index', 0) == 0:
                frames[int(fr['timestep_index'])] = fr
fit = np.load(a.fit)
pose_by_t = dict(zip(fit['timesteps'].tolist(), fit['body_pose']))
ts = np.array([t for t in sorted(frames) if t in pose_by_t]); T = len(ts)
kp = np.load(a.keypoints)['kp'][ts]                                               # (T,133,3)
print(f'{T} timesteps with a body fit (of {len(frames)} in the export)')

keys = ('expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation')
model = TransplantedSmplx(np.load(a.transplant))
BODY = [('L shoulder', 5, 16), ('R shoulder', 6, 17), ('L elbow', 7, 18), ('R elbow', 8, 19),
        ('L wrist', 9, 20), ('R wrist', 10, 21), ('L hip', 11, 1), ('R hip', 12, 2)]
uv = np.zeros((T, 55, 2), np.float32)
for s in range(0, T, 256):
    chunk = ts[s:s + 256]
    fl = {k: torch.tensor(np.stack([np.asarray(np.load(os.path.join(a.export, frames[t]['flame_param_path']))[k], np.float32).reshape(-1)
                                    for t in chunk])) for k in keys}
    J = model(torch.tensor(np.stack([pose_by_t[t] for t in chunk])), fl, return_verts=False)['joints'].numpy()
    for i, t in enumerate(chunk):
        fr = frames[t]
        c2w = np.array(fr['transform_matrix'], np.float64); c2w[:3, 1:3] *= -1; w2c = np.linalg.inv(c2w)  # OpenGL -> OpenCV
        pc = J[i] @ w2c[:3, :3].T + w2c[:3, 3]
        z = np.maximum(pc[:, 2], 1e-3)
        uv[s + i] = np.stack([fr['fl_x'] * pc[:, 0] / z + fr['cx'], fr['fl_y'] * pc[:, 1] / z + fr['cy']], -1)

wh = np.array([[frames[t]['w'], frames[t]['h']] for t in ts], np.float32)
err = np.full((T, len(BODY)), np.nan, np.float32)
for b, (_, c, j) in enumerate(BODY):
    xy = kp[:, c, :2]
    ok = (kp[:, c, 2] > 0.3) & (xy > 10).all(-1) & (xy < wh - 10).all(-1)        # same rule as fit_body.py
    e = np.linalg.norm(uv[:, j] - xy, axis=-1)
    err[ok, b] = e[ok]
with np.errstate(all='ignore'):
    arm = np.nanmean(err[:, :6], 1); hip = np.nanmean(err[:, 6:], 1)
nose_ok = kp[:, 0, 2] > 0.3
head = np.where(nose_ok, np.linalg.norm(uv[:, 15] - kp[:, 0, :2], axis=-1), np.nan)

fg = np.zeros(T, np.float32)
for i, t in enumerate(ts):
    img = cv2.imread(os.path.join(a.export, frames[t]['file_path']), cv2.IMREAD_UNCHANGED)
    if img is None:
        fg[i] = 0.0
    elif img.ndim == 3 and img.shape[2] == 4:                                      # RGBA export: use the matte
        fg[i] = float((img[..., 3] > 128).mean())
    else:
        fg[i] = float((img[..., :3].min(2) < 235).mean())

print('\nper joint reprojection px (confident, inside image):   median    p90   frames')
for b, (name, _, _) in enumerate(BODY):
    col = err[:, b]; n = int(np.isfinite(col).sum())
    print(f'  {name:11s} {np.nanmedian(col):8.1f} {np.nanpercentile(col, 90):7.1f} {n:7d}' if n else f'  {name:11s}      -')
pct = lambda x: ' '.join(f'p{q} {np.nanpercentile(x, q):.3g}' for q in (1, 5, 50, 95, 99))  # noqa: E731
print(f'arm px  {pct(arm)}\nhip px  {pct(hip)}\nhead px {pct(head)}\nfg      {pct(fg)}')

empty = fg < a.min_fg
bad_arm = np.nan_to_num(arm) > a.max_arm_px
bad_head = np.nan_to_num(head) > a.max_head_px
drop = empty | bad_arm | bad_head


def runs(mask):
    out, start = [], None
    sel = ts[mask]
    for i, t in enumerate(sel):
        start = t if start is None else start
        if i == len(sel) - 1 or sel[i + 1] != t + 1:
            out.append(f'{start}-{t}' if t != start else f'{t}'); start = None
    return out


for name, m in (('empty (fg < %g)' % a.min_fg, empty), ('arm fit > %g px' % a.max_arm_px, bad_arm),
                ('head/nose > %g px' % a.max_head_px, bad_head)):
    r = runs(m)
    print(f'\n{name}: {int(m.sum())} frames in {len(r)} runs: {" ".join(r[:60])}{" ..." if len(r) > 60 else ""}')
print(f'\nTOTAL flagged: {int(drop.sum())} of {T} frames ({drop.mean():.1%})')

os.makedirs(os.path.join(a.dataset, 'audit'), exist_ok=True)
score = np.nan_to_num(arm) / a.max_arm_px + np.nan_to_num(head) / a.max_head_px + empty * 10
worst = [i for i in np.argsort(-score) if drop[i]][:16]
tiles = []
for i in worst:
    img = cv2.imread(os.path.join(a.export, frames[ts[i]]['file_path']))
    img = np.full((512, 512, 3), 128, np.uint8) if img is None else img
    for _, c, j in BODY + [('nose', 0, 15)]:
        if kp[i, c, 2] > 0.3:
            cv2.circle(img, tuple(int(v) for v in kp[i, c, :2]), 4, (0, 0, 255), -1)
        cv2.circle(img, tuple(int(np.clip(v, -1e4, 1e4)) for v in uv[i, j]), 3, (255, 255, 0), -1)
    cv2.putText(img, f't{ts[i]} fg{fg[i]:.2f} arm{np.nan_to_num(arm[i]):.0f} head{np.nan_to_num(head[i]):.0f}',
                (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    tiles.append(cv2.resize(img, (256, 256)))
if tiles:
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % 4)
    sheet = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
    cv2.imwrite(os.path.join(a.dataset, 'audit', 'audit_worst.jpg'), sheet)
    print('wrote', os.path.join(a.dataset, 'audit', 'audit_worst.jpg'))
np.savez_compressed(os.path.join(a.dataset, 'audit', 'frame_audit.npz'), timesteps=ts, fg=fg, arm_px=arm, hip_px=hip,
                    head_px=head, joint_px=err, drop=drop)

if a.apply:
    dropped = set(ts[drop].tolist())
    for split in ('train', 'val', 'test'):
        src = os.path.join(a.export, f'transforms_{split}.json')
        dst = os.path.join(a.dataset, f'transforms_{split}.json')
        if not os.path.exists(src):
            continue
        meta = json.load(open(src))
        before = len(meta['frames'])
        meta['frames'] = [fr for fr in meta['frames'] if int(fr['timestep_index']) not in dropped]
        if os.path.lexists(dst):
            os.remove(dst)                                                        # symlink to the export, or our earlier file
        with open(dst, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'{split}: {before} -> {len(meta["frames"])} frames')
    with open(os.path.join(a.dataset, 'audit', 'dropped_timesteps.json'), 'w') as f:
        json.dump(dict(min_fg=a.min_fg, max_arm_px=a.max_arm_px, max_head_px=a.max_head_px, timesteps=sorted(dropped)), f)
    print('applied; list in', os.path.join(a.dataset, 'audit', 'dropped_timesteps.json'))
else:
    print('\n(dry run: nothing changed; add --apply to rewrite the dataset transforms)')
