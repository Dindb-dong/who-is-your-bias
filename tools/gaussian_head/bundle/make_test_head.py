"""Synthetic gaussian-head.v1 bundle from the open FLAME model (no trained avatar needed).
Used only to validate the browser math and measure rendering speed.
usage: make_test_head.py <flame2023_Open.pkl> <out_stem> [gauss_per_face=10]
"""
import sys, pickle
import numpy as np
from head_bundle import write_bundle, read_bundle, flame_forward, face_frames, gaussian_centers

src, out = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
m = pickle.load(open(src, "rb"), encoding="latin1")
V = m["v_template"].shape[0]; E = 100
faces = m["f"].astype(np.uint32)
v_shaped = m["v_template"].astype(np.float32)
expr_dirs = m["shapedirs"][:, :, 300:300 + E].transpose(2, 0, 1).reshape(E, V * 3)
pose_dirs = m["posedirs"].reshape(V * 3, 36).T
Jreg = m["J_regressor"].toarray()
j_base = Jreg @ v_shaped
j_expr = np.stack([Jreg @ expr_dirs[e].reshape(V, 3) for e in range(E)]).reshape(E, 15)

rng = np.random.default_rng(0)
F = len(faces)
binding = np.repeat(np.arange(F), K).astype(np.uint32)
bary = rng.dirichlet([1, 1, 1], size=F * K)
c, Rf, s = face_frames(v_shaped.astype(np.float64), faces.astype(np.int64))
tri = v_shaped[faces.astype(np.int64)]                       # F,3,3
p = np.einsum("nk,nkd->nd", bary, tri[binding])
local = np.einsum("nba,nb->na", Rf[binding], p - c[binding]) / s[binding, None]
normal = Rf[binding][:, :, 1]
# simple skin colour with normal-based shading, stored as SH DC
light = np.clip(normal @ np.array([0.3, 0.5, 0.8]) / 1.0, 0, 1)
rgb = np.array([0.85, 0.66, 0.55]) * (0.35 + 0.65 * light[:, None])
f_dc = (rgb - 0.5) / 0.28209479
scale = np.log(np.full((F * K, 3), 0.35) * np.array([1, 0.15, 1]))  # flat discs in face plane, in face-scale units
rot = np.tile([1.0, 0, 0, 0], (F * K, 1))
opacity = np.full(F * K, 3.0)

T = 120
t = np.arange(T) / 30.0
m_expr = np.zeros((T, E)); m_expr[:, 0] = 1.5 * np.sin(t * 2); m_expr[:, 1] = 1.0 * np.sin(t * 3 + 1)
m_pose = np.zeros((T, 15)); m_pose[:, 4] = 0.35 * np.sin(t * 1.5); m_pose[:, 6] = 0.25 * (0.5 + 0.5 * np.sin(t * 6))
m_trans = np.zeros((T, 3))

arrays = dict(v_shaped=("f32", v_shaped), expr_dirs=("f16", expr_dirs), pose_dirs=("f16", pose_dirs),
              lbs_weights=("f32", m["weights"]), j_base=("f32", j_base), j_expr_dirs=("f32", j_expr),
              faces=("u32", faces), g_xyz=("f32", local), g_rot=("f32", rot), g_scale=("f32", scale),
              g_opacity=("f32", opacity), g_color=("f32", f_dc), g_binding=("u32", binding),
              m_expr=("f32", m_expr), m_pose=("f32", m_pose), m_trans=("f32", m_trans))
meta = dict(n_verts=V, n_faces=F, n_expr=E, n_gauss=F * K, n_frames=T, fps=30, parents=[-1, 0, 1, 1, 1],
            source="synthetic test bundle from FLAME2023_Open (CC BY 4.0); not a trained avatar",
            camera=dict(position=[0, 0.02, 0.55], target=[0, 0.0, 0.0], fov_y_deg=25))
write_bundle(out, arrays, meta)

# reference values for the JS test: frame 37
H, B = read_bundle(out)
f = 37
verts = flame_forward(B, B["m_expr"][f].astype(np.float64), B["m_pose"][f].astype(np.float64), B["m_trans"][f])
cent = gaussian_centers(B, verts)
np.save(out + ".ref_verts37.npy", verts.astype(np.float32)); np.save(out + ".ref_centers37.npy", cent.astype(np.float32))
print("bundle", out, "gaussians", F * K, "verts", V, "bytes", sum(b["bytes"] for b in H["buffers"].values()))
