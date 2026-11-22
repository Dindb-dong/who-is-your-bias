"""Export a trained upper-body GaussianAvatars model (SmplxGaussianModel) for the browser SMPL-X preview.

Run inside the GaussianAvatars checkout (needs its utils.sh_utils / utils.graphics_utils, roma, plyfile):
    GA_ITER=<iter> python export_upperbody.py <model_dir> <smplx_head.npz> <SMPL-X__FLAME_vertex_ids.npy> <out_stem>

Writes smplx-head-preview.v1 (the format of ../smplx/preview_export.py) with:
  * faces_g = every bound face (head + upper body); mesh_faces = the unbound (leg) faces;
  * gaussian colours with SH 1-3 baked over 64 training views (as export_bundle.py GA_BAKE_SH=1);
  * m_body (T,165) full SMPL-X pose (solved neck + FLAME head joints) and m_body_trans (T,3) for tracked replay,
    plus m_expr / m_pose (FLAME) for compatibility.
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
import roma
import torch
from plyfile import PlyData

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_model import TransplantedSmplx  # noqa: E402
from utils.graphics_utils import compute_face_orientation  # noqa: E402
from utils.sh_utils import eval_sh  # noqa: E402

model_dir, tp_path, ids_path, out = sys.argv[1:5]
its = sorted(glob.glob(f'{model_dir}/point_cloud/iteration_*'), key=lambda p: int(p.rsplit('_', 1)[1]))
pc = f"{model_dir}/point_cloud/iteration_{int(os.environ['GA_ITER'])}" if os.environ.get('GA_ITER') else its[-1]
print('using', pc)
ply = PlyData.read(f'{pc}/point_cloud.ply')['vertex']; names = [p.name for p in ply.properties]
cols = lambda pre: np.stack([np.asarray(ply[k]) for k in sorted([n for n in names if n.startswith(pre)], key=lambda s: int(s.rsplit('_', 1)[1]))], 1).astype(np.float32)  # noqa: E731
g_xyz = np.stack([ply['x'], ply['y'], ply['z']], 1).astype(np.float32)
g_dc, g_scale, g_rot, g_rest = cols('f_dc_'), cols('scale_'), cols('rot_'), cols('f_rest_')
g_op = np.asarray(ply['opacity'], np.float32); g_bind = np.asarray(ply['binding_0']).astype(np.int64)
keep = 1 / (1 + np.exp(-g_op)) > 0.004
g_xyz, g_dc, g_scale, g_rot, g_rest, g_op, g_bind = (a[keep] for a in (g_xyz, g_dc, g_scale, g_rot, g_rest, g_op, g_bind))
faces = np.load(f'{pc}/bound_faces.npy').astype(np.int64)
P = {k: torch.from_numpy(v).float() for k, v in np.load(f'{pc}/flame_param.npz').items()}
S = np.load(tp_path); model = TransplantedSmplx(S)
T = P['expr'].shape[0]; E = model.n_expr
print(f'gaussians kept {keep.sum()} / {len(keep)} | bound faces {len(faces)} | frames {T}')


def run(idx):
    fl = {k: P[k][idx] for k in ('expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation')}
    with torch.no_grad():
        return model(P['body_pose'][idx], fl)


# ---- tracked motion as full SMPL-X pose (local rotations incl. the solved neck) + body translation
m_body = np.zeros((T, 165), np.float32); m_bt = np.zeros((T, 3), np.float32)
for s in range(0, T, 256):
    idx = list(range(s, min(T, s + 256))); o = run(idx)
    m_body[idx] = roma.rotmat_to_rotvec(o['rotmats']).reshape(len(idx), 165).numpy(); m_bt[idx] = o['trans'].numpy()

# ---- SH bake over training views
C0 = 0.28209479177387814
K = g_rest.shape[1] // 3; deg = int(round(math.sqrt(K + 1))) - 1
src = re.search(r"source_path='([^']+)'", open(f'{model_dir}/cfg_args').read()).group(1)
frs = json.load(open(f'{src}/transforms_train.json'))['frames']
pick = np.linspace(0, len(frs) - 1, min(64, len(frs))).round().astype(int)
sh = torch.from_numpy(np.concatenate([g_dc[:, :, None], g_rest.reshape(-1, 3, K)], 2))
acc = np.zeros((len(g_dc), 3))
ft = torch.from_numpy(faces); b = torch.from_numpy(g_bind)
for i in pick:
    fr = frs[int(i)]; t = int(fr['timestep_index'])
    v = run([t])['verts'][0]
    R, s = compute_face_orientation(v, ft, return_scale=True)
    ctr = (R[b] @ torch.from_numpy(g_xyz)[..., None])[..., 0] * s.reshape(-1)[b][:, None] + v[ft].mean(1)[b]
    d = ctr - torch.tensor(np.array(fr['transform_matrix'])[:3, 3], dtype=torch.float32); d = d / d.norm(dim=1, keepdim=True)
    with torch.no_grad():
        acc += torch.clamp_min(eval_sh(deg, sh, d) + 0.5, 0.0).numpy()
g_color = ((acc / len(pick) - 0.5) / C0).astype(np.float32)
print(f'SH bake: degree {deg}, {len(pick)} views')

# ---- preview arrays (as ../smplx/preview_export.py)
ids = np.load(ids_path).astype(np.int64); NS = 10475
Vr = np.asarray(S['v_rest'], np.float64); N = len(Vr); NT = N - NS
vmap = np.concatenate([ids, NS + np.arange(NT)])
expr_head = np.asarray(S['expr_dirs'], np.float32)[:, vmap]
PDH = np.asarray(S['pose_dirs'], np.float32)[vmap]
cols_pd = sum([list(range((j - 1) * 9, (j - 1) * 9 + 9)) for j in (15, 22, 23, 24)], [])
pose_head = PDH[:, :, cols_pd].transpose(2, 0, 1)
W = np.asarray(S['weights'], np.float32); Kw = int((W > 1e-6).sum(1).max())
order = np.argsort(-W, axis=1)[:, :Kw]
fb = np.asarray(S['faces_body'], np.int64)
bound = {tuple(sorted(f)) for f in faces.tolist()}
legs = np.array([f for f in fb.tolist() if tuple(sorted(f)) not in bound], np.int64).reshape(-1, 3)
mapped = np.array([12, 15, 22, 23, 24])
m_pose = torch.cat([P['rotation'], P['neck_pose'], P['jaw_pose'], P['eyes_pose']], 1).numpy()
arrays = {
    'v_rest': ('f32', Vr), 'vmap': ('u32', vmap), 'expr_head': ('f16', expr_head.reshape(E, -1)),
    'pose_head': ('f16', pose_head.reshape(36, -1)), 'w_idx': ('u32', order), 'w_val': ('f32', np.take_along_axis(W, order, 1)),
    'j_rest': ('f32', S['j_rest']), 'j_expr': ('f32', np.asarray(S['j_expr_dirs'])[:, mapped].reshape(E, 15)),
    'parents': ('u32', np.where(np.asarray(S['parents']) < 0, 0xFFFFFFFF, S['parents']).astype(np.uint32)),
    'faces_g': ('u32', faces), 'mesh_faces': ('u32', legs),
    'g_xyz': ('f32', g_xyz), 'g_rot': ('f32', g_rot), 'g_scale': ('f32', g_scale), 'g_opacity': ('f32', g_op),
    'g_color': ('f32', g_color), 'g_binding': ('u32', g_bind),
    'm_expr': ('f32', P['expr'].numpy()), 'm_pose': ('f32', m_pose), 'm_body': ('f32', m_body), 'm_body_trans': ('f32', m_bt),
}
DT = {'f32': np.float32, 'f16': np.float16, 'u32': np.uint32}
header = dict(schema='smplx-head-preview.v1', buffers={}, n_verts=int(N), n_head_verts=int(len(vmap)), n_joints=55, n_expr=int(E),
              n_faces_g=int(len(faces)), n_gauss=int(len(g_op)), n_frames=int(T), fps=15, influences=Kw, mapped_joints=mapped.tolist(),
              upper_body=True, iteration=int(pc.rsplit('_', 1)[1]), sh_baked=True,
              note='research preview; SMPL-X/FLAME are research-licensed; trained on internal footage; do not redistribute')
blobs, off = [], 0
for name, (dt, arr) in arrays.items():
    bts = np.ascontiguousarray(arr, dtype=DT[dt]).tobytes(); pad = (-len(bts)) % 4
    header['buffers'][name] = dict(dtype=dt, shape=list(np.shape(arr)), offset=off, bytes=len(bts)); blobs.append(bts + b'\0' * pad); off += len(bts) + pad
open(out + '.bin', 'wb').write(b''.join(blobs)); json.dump(header, open(out + '.json', 'w'), indent=1)
print(f'wrote {out}.json/.bin  {off / 1e6:.1f} MB')
