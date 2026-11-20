"""Fit SMPL-X upper-body pose (torso, arms, hands) for every timestep of a VHAP export, head locked to FLAME tracking.

    python fit_body.py <export_dir> <smplx_head.npz> <SMPLX_NEUTRAL_2020.npz> <keypoints.npz> <out.npz>
                       [--iters 600 600] [--device cpu|cuda] [--overlay_dir DIR]

Inputs: VHAP NeRF export (transforms_*.json with cameras + per-frame flame_param npz), smplx_head.py transplant,
the SMPL-X model (hand PCA only), keypoints.py output (COCO-WholeBody 133 points in export-image pixels).
Optimises all frames jointly: robust 2D reprojection of shoulders/elbows/wrists/hips/hand joints, light pose
priors, elbow-hinge limits and temporal smoothness. Head joints (12, 15, 22, 23, 24) always come from FLAME.
Output: body_pose (T,55,3) axis-angle (head joints zero), per-frame reprojection error, settings.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_model import TransplantedSmplx  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('export'); ap.add_argument('transplant'); ap.add_argument('smplx'); ap.add_argument('keypoints'); ap.add_argument('out')
ap.add_argument('--iters', type=int, nargs=2, default=[600, 600])
ap.add_argument('--device', default='cpu')
ap.add_argument('--hand_pca', type=int, default=12)
ap.add_argument('--overlay_dir', default='')
a = ap.parse_args()
dev = torch.device(a.device)
torch.manual_seed(0)

# ---------------- data ----------------
frames = {}
for split in ('train', 'val', 'test'):
    p = os.path.join(a.export, f'transforms_{split}.json')
    if os.path.exists(p):
        for fr in json.load(open(p))['frames']:
            if fr.get('camera_index', 0) == 0:
                frames[int(fr['timestep_index'])] = fr
ts = np.array(sorted(frames)); T = len(ts)
keys = ('expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation')
fl = {k: [] for k in keys}
K_cam, W2C = [], []
for t in ts:
    fr = frames[t]
    fp = np.load(os.path.join(a.export, fr['flame_param_path']))
    for k in keys:
        fl[k].append(np.asarray(fp[k], np.float32).reshape(-1))
    c2w = np.array(fr['transform_matrix'], np.float64); c2w[:3, 1:3] *= -1       # OpenGL -> OpenCV (as GaussianAvatars)
    W2C.append(np.linalg.inv(c2w)); K_cam.append([fr['fl_x'], fr['fl_y'], fr['cx'], fr['cy']])
flame = {k: torch.tensor(np.stack(v), device=dev) for k, v in fl.items()}
W2C = torch.tensor(np.stack(W2C), dtype=torch.float32, device=dev); K_cam = torch.tensor(K_cam, dtype=torch.float32, device=dev)
kpz = np.load(a.keypoints)
kp_all = kpz['kp']
kp = torch.tensor(kp_all[ts], device=dev)                                         # (T,133,3)

S = np.load(a.transplant)
model = TransplantedSmplx(S).to(dev)
M = np.load(a.smplx, allow_pickle=False)
NH = a.hand_pca
hc = {s: torch.tensor(np.asarray(M[f'hands_components{s}'][:NH], np.float32), device=dev) for s in 'lr'}
hm = {s: torch.tensor(np.asarray(M[f'hands_mean{s}'], np.float32), device=dev) for s in 'lr'}

# COCO-WholeBody -> SMPL-X joints (weights reflect how well the 2D point matches the 3D joint)
BODY = [(5, 16, .5), (6, 17, .5), (7, 18, 1.), (8, 19, 1.), (9, 20, 1.), (10, 21, 1.), (11, 1, .5), (12, 2, .5)]
HAND = [(0, None)] + [(1, 0 + 12), (2, 1 + 12), (3, 2 + 12)] + [(5, 0), (6, 1), (7, 2)] + [(9, 3), (10, 4), (11, 5)] \
    + [(13, 9), (14, 10), (15, 11)] + [(17, 6), (18, 7), (19, 8)]                 # coco hand idx -> offset in 15 finger joints
pairs = [(c, j, w) for c, j, w in BODY]
for base_c, wrist, first in ((91, 20, 25), (112, 21, 40)):                       # left hand, right hand
    for ci, off in HAND:
        pairs.append((base_c + ci, wrist if off is None else first + off, .5))
P_COCO = torch.tensor([p[0] for p in pairs], device=dev)
P_JOINT = torch.tensor([p[1] for p in pairs], device=dev)
P_W = torch.tensor([p[2] for p in pairs], device=dev)
IS_HAND = torch.tensor([p[1] >= 25 for p in pairs], device=dev)

# ---------------- parameters ----------------
TORSO = [0, 3, 6, 9]; ARMS = [13, 14, 16, 17, 18, 19, 20, 21]
init = torch.zeros(T, 55, 3, device=dev)
init[:, 16, 2], init[:, 17, 2] = -1.15, 1.15                                     # arms down from the T-pose
init[:, 18, 1], init[:, 19, 1] = -0.4, 0.4                                       # elbows slightly bent
torso = torch.nn.Parameter(init[:, TORSO].clone()); arms = torch.nn.Parameter(init[:, ARMS].clone())
hands = torch.nn.Parameter(torch.zeros(T, 2, NH, device=dev))


def full_pose():
    P = torch.zeros(T, 55, 3, device=dev)
    P[:, TORSO] = torso; P[:, ARMS] = arms
    P[:, 25:40] = (hm['l'] + hands[:, 0] @ hc['l']).reshape(T, 15, 3)
    P[:, 40:55] = (hm['r'] + hands[:, 1] @ hc['r']).reshape(T, 15, 3)
    return P


def project(J):                                                                   # (T,55,3) -> (T,55,2) pixels
    pc = (W2C[:, None, :3, :3] @ J[..., None])[..., 0] + W2C[:, None, :3, 3]
    z = pc[..., 2:3].clamp_min(1e-3)
    return torch.stack([K_cam[:, None, 0] * pc[..., 0] / z[..., 0] + K_cam[:, None, 2],
                        K_cam[:, None, 1] * pc[..., 1] / z[..., 0] + K_cam[:, None, 3]], -1)


conf = kp[:, P_COCO, 2]
# joints outside the crop (seated hips, elbows at the frame edge) are clamped to the border by the detector: ignore them
IMG_WH = torch.tensor([[frames[t]['w'], frames[t]['h']] for t in ts], dtype=torch.float32, device=dev)
xy = kp[:, P_COCO, :2]; BORDER = 10.0
inside = (xy > BORDER).all(-1) & (xy < IMG_WH[:, None] - BORDER).all(-1)
conf = torch.where(inside, conf, torch.zeros_like(conf))
print(f'confident keypoints dropped at the image border: {float(((kp[:, P_COCO, 2] > 0.3) & ~inside).sum()) / max(1.0, float((kp[:, P_COCO, 2] > 0.3).sum())):.1%}')
wconf = torch.where(conf > 0.3, conf, torch.zeros_like(conf)) * P_W
RHO_END = 20.0


def losses(stage, rho=RHO_END):
    P = full_pose()
    J = model(P, flame, return_verts=False)['joints']
    uv = project(J)[:, P_JOINT]
    d2 = ((uv - kp[:, P_COCO, :2]) ** 2).sum(-1)
    gm = rho ** 2 * d2 / (d2 + rho ** 2)                                         # Geman-McClure
    if stage == 0:
        gm = gm * (~IS_HAND)
    L = {'kp': (wconf * gm).sum() / wconf.sum().clamp_min(1) / rho ** 2}
    L['torso'] = 1.0 * (torso[:, 1:] ** 2).mean() + 0.2 * (torso[:, 0] ** 2).mean()   # seated speakers lean; keep it loose
    L['collar'] = 2.0 * (arms[:, :2] ** 2).mean()
    el = arms[:, 4:6]                                                              # elbows 18, 19: hinge about y
    L['elbow'] = 2.0 * (el[..., 0] ** 2 + el[..., 2] ** 2).mean() + 10.0 * (torch.relu(el[:, 0, 1]) + torch.relu(-el[:, 1, 1])).mean()
    L['hand'] = 0.02 * (hands ** 2).mean()
    smooth = torch.cat([torso, arms], 1)
    L['vel'] = 20.0 * ((smooth[1:] - smooth[:-1]) ** 2).mean() + 5.0 * ((hands[1:] - hands[:-1]) ** 2).mean()
    L['acc'] = 20.0 * ((smooth[2:] - 2 * smooth[1:-1] + smooth[:-2]) ** 2).mean()
    return L, d2


t0 = time.time()
for stage, n in enumerate(a.iters):
    params = [torso, arms] + ([hands] if stage == 1 else [])
    opt = torch.optim.Adam(params, lr=0.03 if stage == 0 else 0.01)
    for it in range(n):
        # robust radius anneals from wide (far-off start) to RHO_END px, so distant joints still get pulled in
        rho = float(np.exp(np.log(300.0) + (np.log(RHO_END) - np.log(300.0)) * min(1.0, it / (0.7 * n))))
        L, _ = losses(stage, rho)
        tot = sum(L.values())
        opt.zero_grad(); tot.backward(); opt.step()
        if it % 100 == 0 or it == n - 1:
            print(f'stage {stage} it {it:4d} rho {rho:5.0f} ' + ' '.join(f'{k} {v.item():.4f}' for k, v in L.items()) + f'  ({time.time() - t0:.0f}s)')

with torch.no_grad():
    _, d2 = losses(1)
    err = d2.sqrt(); valid = wconf > 0
    body_err = torch.where(valid & ~IS_HAND, err, torch.nan); hand_err = torch.where(valid & IS_HAND, err, torch.nan)
    pf_body = torch.nanmean(body_err, 1).cpu().numpy(); pf_hand = torch.nanmean(hand_err, 1).cpu().numpy()
    pose = full_pose().cpu().numpy().astype(np.float32)
pose[:, [12, 15, 22, 23, 24]] = 0
print(f'reprojection (px, confident points): body median {np.nanmedian(pf_body):.1f} p90 {np.nanpercentile(pf_body, 90):.1f} | '
      f'hands median {np.nanmedian(pf_hand):.1f} p90 {np.nanpercentile(pf_hand, 90):.1f}')
np.savez_compressed(a.out, body_pose=pose, timesteps=ts, reproj_body_px=pf_body, reproj_hand_px=pf_hand,
                    settings=json.dumps(dict(iters=a.iters, hand_pca=NH, rho_end=RHO_END, keypoints=os.path.basename(a.keypoints))))
print('wrote', a.out, pose.shape)

if a.overlay_dir:
    import cv2
    os.makedirs(a.overlay_dir, exist_ok=True)
    sel = np.linspace(0, T - 1, 8).round().astype(int)
    with torch.no_grad():
        sub = {k: v[sel] for k, v in flame.items()}
        out = model(torch.tensor(pose[sel], device=dev), sub)
        J, V = out['joints'], out['verts']
        pc = lambda X: (W2C[sel][:, None, :3, :3] @ X[..., None])[..., 0] + W2C[sel][:, None, :3, 3]  # noqa: E731
        def uvf(X):
            p = pc(X); return torch.stack([K_cam[sel][:, None, 0] * p[..., 0] / p[..., 2] + K_cam[sel][:, None, 2],
                                           K_cam[sel][:, None, 1] * p[..., 1] / p[..., 2] + K_cam[sel][:, None, 3]], -1).cpu().numpy()
        uvJ, uvV = uvf(J), uvf(V)
    upper = np.where(np.asarray(S['v_rest'])[:, 1] > np.asarray(S['j_rest'])[3, 1])[0]
    tiles = []
    for i, k in enumerate(sel):
        img = cv2.imread(os.path.join(a.export, frames[ts[k]]['file_path']))
        for x, y in uvV[i, upper[::6]].astype(int):
            cv2.circle(img, (x, y), 1, (80, 200, 80), -1)
        for c, j, _ in pairs:
            if kp_all[ts[k], c, 2] > 0.3:
                cv2.circle(img, tuple(kp_all[ts[k], c, :2].astype(int)), 3, (0, 0, 255), -1)
            cv2.circle(img, tuple(uvJ[i, j].astype(int)), 2, (255, 255, 0), -1)
        tiles.append(cv2.resize(img, (256, 256)))
    cv2.imwrite(os.path.join(a.overlay_dir, 'fit_overlay.jpg'), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
    print('overlay', os.path.join(a.overlay_dir, 'fit_overlay.jpg'))
