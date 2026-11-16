"""Build the browser preview bundle (schema smplx-head-preview.v1) from smplx_head.py output + the gaussian bundle.

    python preview_export.py <gaussian_bundle_stem> <smplx_head.npz> <SMPL-X__FLAME_vertex_ids.npy> <out_stem>

Collar cut: gaussians bound to FLAME triangles whose centroid lies within CUT_M of the head/body seam
(measured on the rest mesh), or whose rest centre sits below the seam, are dropped: the shirt collar hides
that band and those gaussians were painting the shoulders/clothes in training.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bundle'))
from head_bundle import read_bundle, face_frames  # noqa: E402

stem, head_npz, ids_path, out = sys.argv[1:5]
CUT_M = float(os.environ.get('CUT_M', '0.05'))
H, B = read_bundle(stem)
S = np.load(head_npz)
ids = np.load(ids_path).astype(np.int64)
NF = 5023; Vf = H['n_verts']; NS = 10475; NT = Vf - NF
vmap = np.concatenate([ids, NS + np.arange(NT)])
Vr = S['v_rest'].astype(np.float64); N = len(Vr)
faces_g = S['faces_gaussian'].astype(np.int64)

# distance of each FLAME triangle centroid to the seam (head vertices that touch a body vertex)
ish = np.zeros(N, bool); ish[vmap] = True
fb = np.concatenate([S['faces_body'], S['faces_head']]).astype(np.int64)
e = np.concatenate([fb[:, [0, 1]], fb[:, [1, 2]], fb[:, [2, 0]]])
seam = np.unique(e[ish[e[:, 0]] & ~ish[e[:, 1]], 0])
cen = Vr[faces_g].mean(1)
dseam = np.min(np.linalg.norm(cen[:, None] - Vr[seam][None], axis=2), axis=1)
dseam[9976:] = 1.0                                                   # teeth: never cut
cut_face = dseam < CUT_M

bind = B['g_binding'].astype(np.int64)
c, Rf, s = face_frames(Vr, faces_g)
g_rest = np.einsum('nab,nb->na', Rf[bind], B['g_xyz'].astype(np.float64)) * s[bind, None] + c[bind]
seam_y = Vr[seam, 1].max()
keep = ~cut_face[bind] & ~(g_rest[:, 1] < seam_y)
print(f'collar cut {CUT_M*100:.0f} cm: dropped {int((~keep).sum())} of {len(keep)} gaussians '
      f'({int(cut_face[bind].sum())} on cut triangles, {int(((g_rest[:,1] < seam_y) & ~cut_face[bind]).sum())} more with centre below seam); kept {int(keep.sum())}')

# mesh to draw opaquely: SMPL-X body + the cut band of the head (it will sit under the collar)
cut_tris = faces_g[:9976][cut_face[:9976]]
mesh_faces = np.concatenate([S['faces_body'].astype(np.int64), cut_tris])

# compact head-only deformation data (body vertices: skinning only in the preview)
E = S['expr_dirs'].shape[0]
expr_head = S['expr_dirs'][:, vmap].astype(np.float32)               # (E, Vf, 3)
PDH = S['pose_dirs'][vmap].astype(np.float32)                         # (Vf, 3, 486)
cols = []
for sj in (15, 22, 23, 24):
    cols += list(range((sj - 1) * 9, (sj - 1) * 9 + 9))
pose_head = PDH[:, :, cols].transpose(2, 0, 1)                        # (36, Vf, 3)
other = np.delete(PDH, cols, axis=2)
print('head pose-corrective energy outside joints 15/22/23/24:', float(np.abs(other).max()))
W = S['weights'].astype(np.float32)
nz = (W > 1e-6).sum(1); K = int(nz.max())
order = np.argsort(-W, axis=1)[:, :K]
widx = order.astype(np.uint32); wval = np.take_along_axis(W, order, 1).astype(np.float32)
print('max skinning influences per vertex', K, '| dropped weight mass', float(1 - wval.sum(1).min()))
mapped = np.array([12, 15, 22, 23, 24])
train_T = 2324                                                        # first 70% of frames were the training split

arrays = {
    'v_rest': ('f32', Vr), 'vmap': ('u32', vmap), 'expr_head': ('f16', expr_head.reshape(E, -1)),
    'pose_head': ('f16', pose_head.reshape(36, -1)), 'w_idx': ('u32', widx), 'w_val': ('f32', wval),
    'j_rest': ('f32', S['j_rest']), 'j_expr': ('f32', S['j_expr_dirs'][:, mapped].reshape(E, 15)),
    'parents': ('u32', np.where(S['parents'] < 0, 0xFFFFFFFF, S['parents']).astype(np.uint32)),
    'faces_g': ('u32', faces_g), 'mesh_faces': ('u32', mesh_faces),
    'g_xyz': ('f32', B['g_xyz'][keep]), 'g_rot': ('f32', B['g_rot'][keep]), 'g_scale': ('f32', B['g_scale'][keep]),
    'g_opacity': ('f32', B['g_opacity'][keep]), 'g_color': ('f32', B['g_color'][keep]), 'g_binding': ('u32', bind[keep]),
    'm_expr': ('f32', B['m_expr'][:train_T]), 'm_pose': ('f32', B['m_pose'][:train_T]),
}
DT = {'f32': np.float32, 'f16': np.float16, 'u32': np.uint32}
header = dict(schema='smplx-head-preview.v1', buffers={}, n_verts=int(N), n_head_verts=int(Vf), n_joints=55, n_expr=int(E),
              n_faces_g=int(len(faces_g)), n_gauss=int(keep.sum()), n_frames=train_T, fps=15, influences=K,
              mapped_joints=mapped.tolist(), cut_m=CUT_M,
              source=H.get('source', ''), note='research preview; SMPL-X/FLAME are research-licensed; do not redistribute')
blobs, off = [], 0
for name, (dt, a) in arrays.items():
    a = np.ascontiguousarray(a, dtype=DT[dt]); bts = a.tobytes(); pad = (-len(bts)) % 4
    header['buffers'][name] = dict(dtype=dt, shape=list(a.shape), offset=off, bytes=len(bts)); blobs.append(bts + b'\0' * pad); off += len(bts) + pad
open(out + '.bin', 'wb').write(b''.join(blobs)); json.dump(header, open(out + '.json', 'w'), indent=1)
print(f'wrote {out}.json/.bin  {off/1e6:.1f} MB  gaussians {int(keep.sum())}  mesh faces {len(mesh_faces)}')
