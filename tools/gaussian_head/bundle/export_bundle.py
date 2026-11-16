"""Export a trained GaussianAvatars model to the browser bundle (schema gaussian-head.v1).

Run inside the GaussianAvatars repo (conda env `avatar`):
    python export_bundle.py <model_dir> <out_stem>
Self-contained: the bundle writer and the numpy reference of the browser math are inlined.
The bundle is validated against GaussianAvatars' own FlameHead + compute_face_orientation.
"""
import sys, os, re, json, glob, math
import numpy as np
import torch
from plyfile import PlyData

sys.path.insert(0, os.getcwd())
from flame_model.flame import FlameHead

model_dir, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

# ---------------- bundle io + numpy reference (same as laptop head_bundle.py) ----------------
DTYPES = {"f32": np.float32, "f16": np.float16, "u32": np.uint32}


def write_bundle(stem, arrays, meta):
    header = dict(schema="gaussian-head.v1", buffers={}, **meta)
    blobs, off = [], 0
    for name, (dt, arr) in arrays.items():
        a = np.ascontiguousarray(arr, dtype=DTYPES[dt]); b = a.tobytes(); pad = (-len(b)) % 4
        header["buffers"][name] = dict(dtype=dt, shape=list(a.shape), offset=off, bytes=len(b))
        blobs.append(b + b"\0" * pad); off += len(b) + pad
    with open(stem + ".bin", "wb") as f:
        for b in blobs: f.write(b)
    with open(stem + ".json", "w") as f: json.dump(header, f, indent=1)
    return header


def read_bundle(stem):
    header = json.load(open(stem + ".json")); raw = open(stem + ".bin", "rb").read()
    return header, {k: np.frombuffer(raw, DTYPES[b["dtype"]], count=int(np.prod(b["shape"])), offset=b["offset"]).reshape(b["shape"])
                    for k, b in header["buffers"].items()}


def rodrigues(aa):
    th = np.linalg.norm(aa) + 1e-8; k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


PARENTS = [-1, 0, 1, 1, 1]


def flame_forward(B, expr, pose, trans):
    V = B["v_shaped"].shape[0]; E = expr.shape[0]
    v = B["v_shaped"].astype(np.float64) + (expr[:, None] * B["expr_dirs"][:E].astype(np.float64)).sum(0).reshape(V, 3)
    J = B["j_base"].astype(np.float64) + (expr[:, None] * B["j_expr_dirs"][:E]).sum(0).reshape(5, 3)
    R = [rodrigues(pose[3 * i:3 * i + 3]) for i in range(5)]
    feat = np.concatenate([(R[i] - np.eye(3)).ravel() for i in range(1, 5)])
    v = v + (feat[:, None] * B["pose_dirs"].astype(np.float64)).sum(0).reshape(V, 3)
    G = []
    for i in range(5):
        M = np.eye(4); M[:3, :3] = R[i]; M[:3, 3] = J[i] - (J[PARENTS[i]] if i else 0)
        G.append(M if i == 0 else G[PARENTS[i]] @ M)
    A = []
    for i in range(5):
        a = G[i].copy(); a[:3, 3] -= G[i][:3, :3] @ J[i]; A.append(a)
    T = np.einsum("vj,jab->vab", B["lbs_weights"], np.stack(A))
    return np.einsum("vab,vb->va", T[:, :3, :3], v) + T[:, :3, 3] + trans


def face_frames(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
    a0 = n(v1 - v0); a1 = n(np.cross(a0, v2 - v0)); a2 = -n(np.cross(a1, a0))
    s = (np.linalg.norm(v1 - v0, axis=-1) + np.abs((a2 * (v2 - v0)).sum(-1))) / 2
    return (v0 + v1 + v2) / 3, np.stack([a0, a1, a2], -1), s


def gaussian_centers(B, verts):
    c, Rf, s = face_frames(verts, B["faces"].astype(np.int64)); b = B["g_binding"].astype(np.int64)
    return np.einsum("nab,nb->na", Rf[b], B["g_xyz"]) * s[b, None] + c[b]


# ---------------- options (environment variables) ----------------
#   GA_ITER=240000   export this saved iteration instead of the latest one
#   GA_BAKE_SH=1     bake SH degree 1-3 into the colour: the browser renders DC only, so the view-dependent
#                    SH terms are evaluated from the training camera for sampled training frames and averaged
#                    per gaussian (the least-squares constant colour over the training views)
GA_ITER = os.environ.get("GA_ITER", "")
GA_BAKE_SH = os.environ.get("GA_BAKE_SH", "0") == "1"

# ---------------- load trained model ----------------
its = sorted(glob.glob(f"{model_dir}/point_cloud/iteration_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
assert its, f"no point_cloud/iteration_* under {model_dir}"
pc_dir = its[-1]
if GA_ITER:
    pc_dir = f"{model_dir}/point_cloud/iteration_{int(GA_ITER)}"
    assert os.path.isdir(pc_dir), f"GA_ITER={GA_ITER}: {pc_dir} not found; saved: {[p.rsplit('_', 1)[1] for p in its]}"
print("using", pc_dir, "| bake SH:", GA_BAKE_SH)
ply = PlyData.read(f"{pc_dir}/point_cloud.ply")["vertex"]
names = [p.name for p in ply.properties]


def cols(prefix):
    ks = sorted([n for n in names if n.startswith(prefix)], key=lambda s: int(s.rsplit("_", 1)[1]))
    return np.stack([np.asarray(ply[k]) for k in ks], 1).astype(np.float32)


g_xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float32)
g_dc, g_scale, g_rot = cols("f_dc_"), cols("scale_"), cols("rot_")
g_rest = cols("f_rest_") if any(n.startswith("f_rest_") for n in names) else np.zeros((len(g_xyz), 0), np.float32)
g_op = np.asarray(ply["opacity"], np.float32)
g_bind = np.asarray(ply["binding_0"]).astype(np.int64)
assert g_dc.shape[1] == 3 and g_scale.shape[1] == 3 and g_rot.shape[1] == 4, (g_dc.shape, g_scale.shape, g_rot.shape)
assert g_rest.shape[1] % 3 == 0 and int(round(math.sqrt(g_rest.shape[1] // 3 + 1))) ** 2 == g_rest.shape[1] // 3 + 1, g_rest.shape
keep = 1 / (1 + np.exp(-g_op)) > 0.004          # drop invisible gaussians
print(f"gaussians {len(g_op)}  kept {keep.sum()}")
g_xyz, g_dc, g_rest, g_scale, g_rot, g_op, g_bind = (a[keep] for a in (g_xyz, g_dc, g_rest, g_scale, g_rot, g_op, g_bind))

P = dict(np.load(f"{pc_dir}/flame_param.npz"))
print("flame_param keys", {k: v.shape for k, v in P.items()})
n_shape, E = P["shape"].shape[-1], P["expr"].shape[-1]
flame = FlameHead(n_shape, E, add_teeth=True)
np_ = lambda t: t.detach().cpu().numpy().astype(np.float64)
v_template, sd = np_(flame.v_template), np_(flame.shapedirs)
V = v_template.shape[0]
faces = flame.faces.detach().cpu().numpy().astype(np.int64)
assert g_bind.max() < len(faces), "binding index beyond FLAME faces (teeth mismatch?)"
assert flame.parents.tolist() == PARENTS, flame.parents.tolist()

so = np.zeros((V, 3))
if "static_offset" in P:
    s0 = P["static_offset"].reshape(-1, 3)
    so[:len(s0)] = s0[:V]
if "dynamic_offset" in P and np.abs(P["dynamic_offset"]).max() > 1e-6:
    print("WARNING: non-zero dynamic_offset is ignored by the browser bundle")
shape = P["shape"].reshape(-1).astype(np.float64)
v_shaped = v_template + np.einsum("vck,k->vc", sd[:, :, :n_shape], shape) + so
expr_dirs = sd[:, :, n_shape:].transpose(2, 0, 1).reshape(E, V * 3)
pose_dirs = np_(flame.posedirs)
assert pose_dirs.shape == (36, V * 3), pose_dirs.shape
Jreg = np_(flame.J_regressor)
j_base = Jreg @ v_shaped
j_expr = np.einsum("jv,evc->ejc", Jreg, expr_dirs.reshape(E, V, 3)).reshape(E, 15)
m_expr = P["expr"].reshape(-1, E)
m_pose = np.concatenate([P["rotation"], P["neck_pose"], P["jaw_pose"], P["eyes_pose"]], 1)
m_trans = P["translation"]
Tn = len(m_expr)
assert m_pose.shape == (Tn, 15) and m_trans.shape == (Tn, 3), (m_pose.shape, m_trans.shape)

# ---------------- camera from the training data ----------------
cam = dict(position=[0, 0.02, 0.55], target=[0, 0, 0], fov_y_deg=25)
try:
    src = re.search(r"source_path='([^']+)'", open(f"{model_dir}/cfg_args").read()).group(1)
    tj = json.load(open(f"{src}/transforms_train.json"))
    fr = tj["frames"][0]
    c2w = np.array(fr["transform_matrix"], np.float64)
    fl_y = fr.get("fl_y", tj.get("fl_y")); h = fr.get("h", tj.get("h"))
    v0 = flame_forward(dict(v_shaped=v_shaped, expr_dirs=expr_dirs, j_base=j_base, j_expr_dirs=j_expr, pose_dirs=pose_dirs,
                            lbs_weights=np_(flame.lbs_weights)), m_expr[0].astype(np.float64), m_pose[0].astype(np.float64), m_trans[0])
    cam = dict(position=c2w[:3, 3].tolist(), target=v0[:5023].mean(0).tolist(),
               fov_y_deg=float(2 * math.degrees(math.atan(h / (2 * fl_y)))) if fl_y and h else 25)
    print("camera from", src, cam)
except Exception as e:
    print("WARNING: camera fallback", repr(e))

# ---------------- optional: bake SH 1-3 into the DC colour ----------------
C0 = 0.28209479177387814
if GA_BAKE_SH:
    from utils.sh_utils import eval_sh
    K = g_rest.shape[1] // 3
    assert K > 0, "GA_BAKE_SH=1 but the ply has no f_rest_* (sh_degree 0) - nothing to bake"
    deg = int(round(math.sqrt(K + 1))) - 1
    src = re.search(r"source_path='([^']+)'", open(f"{model_dir}/cfg_args").read()).group(1)
    frs = json.load(open(f"{src}/transforms_train.json"))["frames"]
    pick = np.linspace(0, len(frs) - 1, min(64, len(frs))).round().astype(int)
    # ply layout: f_rest index = channel * K + coeff  ->  (N, 3, K); GA evaluates sh as (N, 3, (deg+1)^2)
    sh = torch.from_numpy(np.concatenate([g_dc[:, :, None], g_rest.reshape(-1, 3, K)], 2).astype(np.float32))
    Bw = dict(v_shaped=v_shaped, expr_dirs=expr_dirs, j_base=j_base, j_expr_dirs=j_expr, pose_dirs=pose_dirs,
              lbs_weights=np_(flame.lbs_weights), faces=faces, g_binding=g_bind, g_xyz=g_xyz)
    acc = np.zeros((len(g_dc), 3)); acc2 = np.zeros_like(acc)
    for i in pick:
        fr = frs[int(i)]; t = int(fr["timestep_index"])
        campos = np.array(fr["transform_matrix"], np.float64)[:3, 3]
        ctr = gaussian_centers(Bw, flame_forward(Bw, m_expr[t].astype(np.float64), m_pose[t].astype(np.float64), m_trans[t].astype(np.float64)))
        d = ctr - campos; d /= np.linalg.norm(d, axis=1, keepdims=True)
        with torch.no_grad():
            col = torch.clamp_min(eval_sh(deg, sh, torch.from_numpy(d.astype(np.float32))) + 0.5, 0.0).numpy().astype(np.float64)
        acc += col; acc2 += col ** 2
    mean = acc / len(pick); std = np.sqrt(np.maximum(acc2 / len(pick) - mean ** 2, 0))
    w = 1 / (1 + np.exp(-g_op.astype(np.float64)))
    dc_only = np.clip(C0 * g_dc + 0.5, 0, None)
    wm = lambda x: float((x.mean(1) * w).sum() / w.sum() * 255)
    print(f"SH bake: degree {deg}, {len(pick)} training views | opacity-weighted mean |baked - DC-only| {wm(np.abs(mean - dc_only)):.1f}/255"
          f" | remaining view-to-view variation (std) {wm(std):.1f}/255")
    g_dc = ((mean - 0.5) / C0).astype(np.float32)

arrays = dict(v_shaped=("f32", v_shaped), expr_dirs=("f16", expr_dirs), pose_dirs=("f16", pose_dirs),
              lbs_weights=("f32", np_(flame.lbs_weights)), j_base=("f32", j_base), j_expr_dirs=("f32", j_expr),
              faces=("u32", faces), g_xyz=("f32", g_xyz), g_rot=("f32", g_rot), g_scale=("f32", g_scale),
              g_opacity=("f32", g_op), g_color=("f32", g_dc), g_binding=("u32", g_bind),
              m_expr=("f32", m_expr), m_pose=("f32", m_pose), m_trans=("f32", m_trans))
meta = dict(n_verts=V, n_faces=len(faces), n_expr=E, n_gauss=int(len(g_op)), n_frames=Tn, fps=15, parents=PARENTS,
            source="GaussianAvatars (CC BY-NC-SA) trained on internal interview footage - internal research use only; not the real person",
            camera=cam, iteration=int(pc_dir.rsplit("_", 1)[1]), sh_baked=bool(GA_BAKE_SH))
H = write_bundle(out, arrays, meta)
mb = sum(b["bytes"] for b in H["buffers"].values()) / 1e6
print(f"bundle {out}  gaussians {len(g_op)}  verts {V}  faces {len(faces)}  frames {Tn}  {mb:.1f} MB")

# ---------------- validate the written (f16-quantized) bundle against GaussianAvatars ----------------
_, B = read_bundle(out)
from utils.graphics_utils import compute_face_orientation
fl = flame.float()
worst_v = worst_c = 0.0
for f in sorted({0, Tn // 3, (2 * Tn) // 3, Tn - 1}):
    t = lambda k: torch.from_numpy(np.asarray(P[k][f], np.float32))[None]
    with torch.no_grad():
        res = fl(torch.from_numpy(shape.astype(np.float32))[None], t("expr"), t("rotation"), t("neck_pose"), t("jaw_pose"),
                 t("eyes_pose"), t("translation"), zero_centered_at_root_node=False, return_landmarks=False,
                 static_offset=torch.from_numpy(so.astype(np.float32))[None])
        gv = res[0] if isinstance(res, (tuple, list)) else res
        orien, scale = compute_face_orientation(gv, fl.faces, return_scale=True)
        tri = gv[:, fl.faces]
        cen = tri.mean(-2)
        b = torch.from_numpy(B["g_binding"].astype(np.int64))
        gx = torch.bmm(orien[0][b], torch.from_numpy(B["g_xyz"].copy())[..., None]).squeeze(-1) * scale[0].reshape(-1)[b][:, None] + cen[0][b]
    gv = gv[0].numpy()
    mv = flame_forward(B, m_expr[f].astype(np.float64), m_pose[f].astype(np.float64), m_trans[f].astype(np.float64))
    mc = gaussian_centers(B, mv)
    ev, ec = np.abs(mv - gv).max(), np.abs(mc - gx.numpy()).max()
    worst_v, worst_c = max(worst_v, ev), max(worst_c, ec)
    print(f"frame {f}: max vertex err {ev:.2e} m, max gaussian-centre err {ec:.2e} m")
ok = worst_v < 5e-4 and worst_c < 5e-4
print("EXPORT VALIDATION", "PASS" if ok else "FAIL", f"(vertex {worst_v:.2e}, centre {worst_c:.2e})")
sys.exit(0 if ok else 1)
