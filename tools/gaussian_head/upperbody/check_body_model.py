"""Check TransplantedSmplx against the FLAME bundle (needs the licensed local files, so it is not a CI test).

    python check_body_model.py <gaussian_bundle_stem> <smplx_head.npz>

A: body at rest -> head vertices equal FLAME forward on tracked frames.
B: torso/arm poses change -> head vertices still equal FLAME forward (head anchored to tracking).
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'bundle')); sys.path.insert(0, HERE)
from head_bundle import read_bundle, flame_forward  # noqa: E402
from body_model import TransplantedSmplx  # noqa: E402

H, B = read_bundle(sys.argv[1])
S = np.load(sys.argv[2])
m = TransplantedSmplx(S).double()
NF = 5023; NS = 10475
ids = np.where(np.isin(np.arange(len(S['v_rest'])), np.unique(S['faces_gaussian'])))[0]
vmap = np.unique(S['faces_gaussian'][:9976])   # not the same order as FLAME; recover the map from faces instead
fg = S['faces_gaussian'].astype(np.int64); ff = B['faces'].astype(np.int64)
vm = np.zeros(H['n_verts'], np.int64); vm[ff.reshape(-1)] = fg.reshape(-1)


def flame_dict(f):
    p = B['m_pose'][f].astype(np.float64)
    t = lambda a: torch.from_numpy(np.asarray(a, np.float64))[None]  # noqa: E731
    return {'expr': t(B['m_expr'][f]), 'rotation': t(p[0:3]), 'neck_pose': t(p[3:6]), 'jaw_pose': t(p[6:9]),
            'eyes_pose': t(p[9:15]), 'translation': t(B['m_trans'][f])}


frames = np.linspace(0, H['n_frames'] - 1, 8).round().astype(int)
rng = np.random.default_rng(0)
for name, scale in (('A rest body', 0.0), ('B random torso+arms', 0.4)):
    worst_core, worst_all = 0.0, 0.0
    for f in frames:
        fv = flame_forward(B, B['m_expr'][f].astype(np.float64), B['m_pose'][f].astype(np.float64), B['m_trans'][f].astype(np.float64))
        pose = torch.zeros(1, 55, 3, dtype=torch.float64)
        if scale:
            for j in (0, 3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21):
                pose[0, j] = torch.from_numpy(rng.normal(0, scale, 3))
        sv = m(pose, flame_dict(f))['verts'][0].numpy()
        d = np.linalg.norm(sv[vm] - fv, axis=1)
        # skinning blend near the neck seam follows the torso on purpose; judge the rest of the head separately
        core = S['weights'][vm][:, [12, 15, 22, 23, 24]].sum(1) > 0.999
        worst_core = max(worst_core, d[core].max()); worst_all = max(worst_all, d.max())
    print(f'{name}: head vertices (pure FLAME skinning) max err {worst_core*1e3:.4f} mm | incl. blended neck {worst_all*1e3:.1f} mm')
