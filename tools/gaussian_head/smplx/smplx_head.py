"""Transplant a FLAME-bound GaussianAvatars head (gaussian-head.v1 bundle) onto SMPL-X.

    python smplx_head.py <bundle_stem> <SMPLX_NEUTRAL_2020.npz> <SMPL-X__FLAME_vertex_ids.npy> <out.npz>

Head + teeth vertices keep FLAME's identity shape, expression basis, pose correctives and skinning,
re-parented into the SMPL-X kinematic tree (FLAME root -> neck 12, neck -> head 15, jaw -> 22, eyes -> 23/24).
Body vertices keep SMPL-X skinning, with betas fitted to the head. Gaussians keep their binding: FLAME face i
becomes the same triangle on the new mesh, so no gaussian parameter changes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bundle'))
from head_bundle import read_bundle, flame_forward, face_frames  # noqa: E402

stem, smplx_path, ids_path, out = sys.argv[1:5]
H, B = read_bundle(stem)
M = np.load(smplx_path, allow_pickle=False)
ids = np.load(ids_path).astype(np.int64)
NF = 5023
Vf = H['n_verts']; E = H['n_expr']; NT = Vf - NF                     # teeth vertices appended by GaussianAvatars
NS = M['v_template'].shape[0]; NJ = M['J_regressor'].shape[0]
parents = M['kintree_table'][0].astype(np.int64); parents[0] = -1
JMAP = {0: 12, 1: 15, 2: 22, 3: 23, 4: 24}                          # FLAME joint -> SMPL-X joint
assert [int(parents[JMAP[c]]) for c in (1, 2, 3, 4)] == [12, 15, 15, 15], 'unexpected SMPL-X tree'
vmap = np.concatenate([ids, NS + np.arange(NT)])                    # FLAME(+teeth) vertex -> new vertex
N = NS + NT

Fv = B['v_shaped'].astype(np.float64)                               # (Vf,3) person-specific FLAME rest shape
# ---- 1) fit SMPL-X betas + translation so the SMPL-X head lands on the FLAME head (sets the body proportions)
K = 100
A = np.concatenate([M['shapedirs'][ids, :, :K].reshape(-1, K), -np.tile(np.eye(3), (NF, 1))], 1)
b = (Fv[:NF] - M['v_template'][ids]).reshape(-1)
x = np.linalg.solve(A.T @ A + 1e-4 * np.diag([1.0] * K + [0.0] * 3), A.T @ b)
beta, t = x[:K], x[K:]
Vr = M['v_template'] + M['shapedirs'][:, :, :K] @ beta
fit_res = np.linalg.norm(Vr[ids] - (Fv[:NF] + t), axis=1)
Vr = np.concatenate([Vr, np.zeros((NT, 3))]); Vr[vmap] = Fv + t   # head + teeth exact

# ---- 2) expression basis: FLAME's on head/teeth rows, zero elsewhere
ED = np.zeros((E, N, 3), np.float32); ED[:, vmap] = B['expr_dirs'].astype(np.float32).reshape(E, Vf, 3)
# ---- 3) pose correctives: SMPL-X for body rows; FLAME's (joints 1..4 -> 15,22,23,24) for head rows
PD = np.zeros((N, 3, (NJ - 1) * 9), np.float32); PD[:NS] = M['posedirs']; PD[vmap] = 0
FPD = B['pose_dirs'].astype(np.float32).reshape(4, 9, Vf, 3)
for fj in (1, 2, 3, 4):
    sj = JMAP[fj]; PD[vmap, :, (sj - 1) * 9:(sj - 1) * 9 + 9] = FPD[fj - 1].transpose(1, 2, 0)
# ---- 4) skinning: SMPL-X for body, FLAME's mapped joints for head/teeth
W = np.zeros((N, NJ), np.float32); W[:NS] = M['weights']; W[vmap] = 0
for fj, sj in JMAP.items():
    W[vmap, sj] = B['lbs_weights'][:, fj]
# FLAME skins its whole neck rigidly to the root joint; SMPL-X skins the neck base to spine3/collars.
# Near the seam, blend from SMPL-X's own weights (at the seam) to FLAME's (BLEND_M metres up the neck).
BLEND_M = float(os.environ.get('BLEND_M', '0.06'))
fe0 = M['f'].astype(np.int64); e0 = np.concatenate([fe0[:, [0, 1]], fe0[:, [1, 2]], fe0[:, [2, 0]]])
ish0 = np.zeros(NS, bool); ish0[ids] = True
seam_head = np.unique(np.where(ish0[e0[:, 0]] & ~ish0[e0[:, 1]], e0[:, 0], -1)); seam_head = seam_head[seam_head >= 0]
d = np.min(np.linalg.norm(Vr[ids][:, None] - Vr[seam_head][None], axis=2), axis=1)
alpha = np.clip(d / BLEND_M, 0, 1) if BLEND_M > 0 else np.ones(NF)
alpha = alpha * alpha * (3 - 2 * alpha)                              # smoothstep
W[ids] = alpha[:, None] * W[ids] + (1 - alpha[:, None]) * M['weights'][ids]
ALPHA = np.concatenate([alpha, np.ones(NT)])
# ---- 5) joints: SMPL-X regressor for body joints; FLAME joints (expression dependent) for the mapped ones
Jr = M['J_regressor'] @ Vr[:NS]
JE = np.zeros((E, NJ, 3)); Jf = B['j_base'].astype(np.float64) + t
Jfe = B['j_expr_dirs'].astype(np.float64).reshape(E, 5, 3)
for fj, sj in JMAP.items():
    Jr[sj] = Jf[fj]; JE[:, sj] = Jfe[:, fj]
faces_g = vmap[B['faces'].astype(np.int64)]                           # gaussian binding triangles on the new mesh
key = {tuple(sorted(f)): i for i, f in enumerate(M['f'].tolist())}
head_faces_mask = np.zeros(M['f'].shape[0], bool)
for f in faces_g[:9976]:
    head_faces_mask[key[tuple(sorted(f.tolist()))]] = True


def rod(aa):
    th = np.linalg.norm(aa)
    if th < 1e-12:
        return np.eye(3)
    k = aa / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


EDf = ED.astype(np.float64); PDf = PD.astype(np.float64).reshape(N * 3, -1); Wf = W.astype(np.float64)


def smplx_forward(pose55, expr, trans):
    v = Vr + np.einsum('e,evc->vc', expr, EDf)
    J = Jr + np.einsum('e,ejc->jc', expr, JE)
    R = np.stack([rod(p) for p in pose55])
    v = v + (PDf @ (R[1:] - np.eye(3)).reshape(-1)).reshape(N, 3)
    G = np.zeros((NJ, 4, 4))
    for j in range(NJ):
        L = np.eye(4); L[:3, :3] = R[j]; L[:3, 3] = J[j] - (J[parents[j]] if j else 0)
        G[j] = L if j == 0 else G[parents[j]] @ L
    Arel = G.copy(); Arel[:, :3, 3] -= np.einsum('jab,jb->ja', G[:, :3, :3], J)
    T = np.einsum('vj,jab->vab', Wf, Arel)
    return np.einsum('vab,vb->va', T[:, :3, :3], v) + T[:, :3, 3] + trans


def flame_pose_to_smplx(p15):
    P = np.zeros((NJ, 3))
    for fj, sj in JMAP.items():
        P[sj] = p15[3 * fj:3 * fj + 3]
    return P


bind = B['g_binding'].astype(np.int64)


def centres(verts, faces):
    c, Rf, s = face_frames(verts, faces)
    return np.einsum('nab,nb->na', Rf[bind], B['g_xyz'].astype(np.float64)) * s[bind, None] + c[bind]


# ---- validation A: body at rest -> SMPL-X head must reproduce FLAME (vertices and gaussian centres)
Tn = H['n_frames']
exact_v = ALPHA[np.arange(Vf)] >= 1.0                               # vertices that keep pure FLAME skinning
fexact = exact_v[B['faces'].astype(np.int64)].all(1)[bind]          # gaussians on such triangles
ev = {'exact': 0.0, 'blend': 0.0}; eg = {'exact': [], 'blend': []}
for f in np.linspace(0, Tn - 1, 12).round().astype(int):
    ex, po, tr = (B[k][f].astype(np.float64) for k in ('m_expr', 'm_pose', 'm_trans'))
    fv = flame_forward(B, ex, po, tr)
    sv = smplx_forward(flame_pose_to_smplx(po), ex, tr - t)
    dv = np.abs(sv[vmap] - fv).max(1)
    ev['exact'] = max(ev['exact'], dv[exact_v].max()); ev['blend'] = max(ev['blend'], dv[~exact_v].max() if (~exact_v).any() else 0)
    dg = np.linalg.norm(centres(sv, faces_g) - centres(fv, B['faces'].astype(np.int64)), axis=1)
    eg['exact'].append(dg[fexact].max()); eg['blend'].append(np.percentile(dg[~fexact], 95) if (~fexact).any() else 0)
print(f'betas fit: head residual mean {fit_res.mean()*1e3:.2f} mm, max {fit_res.max()*1e3:.2f} mm (head then replaced by exact FLAME shape)')
print(f'neck blend {BLEND_M*100:.0f} cm: {int((~exact_v[:NF]).sum())} of {NF} head vertices blended; gaussians on blended triangles {int((~fexact).sum())} of {len(bind)}')
print(f'VALIDATION A (body at rest, 12 tracked frames): pure-FLAME part max vertex err {ev["exact"]*1e3:.4f} mm, '
      f'gaussian-centre err {max(eg["exact"])*1e3:.4f} mm | blended neck part max vertex dev {ev["blend"]*1e3:.1f} mm, gaussian p95 {max(eg["blend"])*1e3:.1f} mm')

# ---- validation B: seam when the body/neck moves (edges joining FLAME-skinned and SMPL-X-skinned vertices)
ishead = np.zeros(N, bool); ishead[vmap] = True
fe = M['f'].astype(np.int64); edges = np.concatenate([fe[:, [0, 1]], fe[:, [1, 2]], fe[:, [2, 0]]])
seam = edges[ishead[edges[:, 0]] != ishead[edges[:, 1]]]
rest = smplx_forward(np.zeros((NJ, 3)), np.zeros(E), np.zeros(3))
L0 = np.linalg.norm(rest[seam[:, 0]] - rest[seam[:, 1]], axis=1)
Ls = np.linalg.norm((M['v_template'] + M['shapedirs'][:, :, :K] @ beta)[seam[:, 0]] - (M['v_template'] + M['shapedirs'][:, :, :K] @ beta)[seam[:, 1]], axis=1)
print(f'seam: {len(seam)} edges; rest length median {np.median(L0)*1e3:.1f} mm; vs pure SMPL-X fit ratio median {np.median(L0/Ls):.3f} worst {np.max(L0/Ls):.3f}')
tests = {'neck turn 30deg (neck12)': {12: [0, np.radians(30), 0]},
         'head nod 20deg (head15)': {15: [np.radians(20), 0, 0]},
         'spine3 bend 20deg': {9: [np.radians(20), 0, 0]},
         'collars shrug 15deg': {13: [0, 0, np.radians(-15)], 14: [0, 0, np.radians(15)]}}
for name, d in tests.items():
    P = np.zeros((NJ, 3))
    for j, v in d.items():
        P[j] = v
    pv = smplx_forward(P, np.zeros(E), np.zeros(3))
    r = np.linalg.norm(pv[seam[:, 0]] - pv[seam[:, 1]], axis=1) / L0
    print(f'  {name:28s} seam edge stretch median {np.median(r):.3f}  min {r.min():.3f}  max {r.max():.3f}')

np.savez_compressed(out, v_rest=Vr.astype(np.float32), expr_dirs=ED.astype(np.float16), pose_dirs=PD.astype(np.float16),
                    weights=W, j_rest=Jr.astype(np.float32), j_expr_dirs=JE.astype(np.float32), parents=parents,
                    faces_gaussian=faces_g.astype(np.uint32), faces_body=M['f'][~head_faces_mask].astype(np.uint32),
                    faces_head=M['f'][head_faces_mask].astype(np.uint32), betas=beta.astype(np.float32),
                    flame_offset=t.astype(np.float32), joint_map=np.array([[k, v] for k, v in JMAP.items()]))
print('wrote', out, '| vertices', N, '| body faces', int((~head_faces_mask).sum()), '| head faces', int(head_faces_mask.sum()))
