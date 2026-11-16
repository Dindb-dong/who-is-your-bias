"""Write/read the browser bundle for a FLAME-bound Gaussian head (schema gaussian-head.v1).

Bundle = <name>.json (header) + <name>.bin (little-endian arrays, 4-byte aligned).
Arrays (V verts, F faces, E expr, N gaussians, T motion frames):
  v_shaped      f32 [V,3]     template + shape blend + static offset (identity is baked)
  expr_dirs     f16 [E,V*3]   expression blend shapes
  pose_dirs     f16 [36,V*3]  pose corrective blend shapes (4 non-root joints x 9)
  lbs_weights   f32 [V,5]
  j_base        f32 [5,3]     J_regressor @ v_shaped
  j_expr_dirs   f32 [E,15]    J_regressor @ expr_dirs
  faces         u32 [F,3]
  g_xyz f32 [N,3], g_rot f32 [N,4] (wxyz, unnormalized), g_scale f32 [N,3] (log),
  g_opacity f32 [N] (logit), g_color f32 [N,3] (SH DC), g_binding u32 [N]
  m_expr f32 [T,E], m_pose f32 [T,15] (global, neck, jaw, eyeL, eyeR axis-angle), m_trans f32 [T,3]
"""
import json
import numpy as np

DTYPES = {"f32": np.float32, "f16": np.float16, "u32": np.uint32}


def write_bundle(path_stem, arrays, meta):
    header = dict(schema="gaussian-head.v1", buffers={}, **meta)
    blobs, off = [], 0
    for name, (dt, arr) in arrays.items():
        a = np.ascontiguousarray(arr, dtype=DTYPES[dt])
        b = a.tobytes()
        pad = (-len(b)) % 4
        header["buffers"][name] = dict(dtype=dt, shape=list(a.shape), offset=off, bytes=len(b))
        blobs.append(b + b"\0" * pad); off += len(b) + pad
    with open(path_stem + ".bin", "wb") as f:
        for b in blobs:
            f.write(b)
    with open(path_stem + ".json", "w") as f:
        json.dump(header, f, indent=1)
    return header


def read_bundle(path_stem):
    header = json.load(open(path_stem + ".json"))
    raw = open(path_stem + ".bin", "rb").read()
    out = {}
    for k, b in header["buffers"].items():
        out[k] = np.frombuffer(raw, DTYPES[b["dtype"]], count=int(np.prod(b["shape"])), offset=b["offset"]).reshape(b["shape"])
    return header, out


# ---------------- numpy reference of the browser math (for tests) ----------------
def rodrigues(aa):
    th = np.linalg.norm(aa) + 1e-8
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


PARENTS = [-1, 0, 1, 1, 1]


def flame_forward(B, expr, pose, trans):
    V = B["v_shaped"].shape[0]
    E = expr.shape[0]
    v = B["v_shaped"].astype(np.float64) + (expr[:, None] * B["expr_dirs"][:E].astype(np.float64)).sum(0).reshape(V, 3)
    J = B["j_base"].astype(np.float64) + (expr[:, None] * B["j_expr_dirs"][:E]).sum(0).reshape(5, 3)
    R = [rodrigues(pose[3 * i:3 * i + 3]) for i in range(5)]
    feat = np.concatenate([(R[i] - np.eye(3)).ravel() for i in range(1, 5)])
    v = v + (feat[:, None] * B["pose_dirs"].astype(np.float64)).sum(0).reshape(V, 3)
    G = []
    for i in range(5):
        M = np.eye(4); M[:3, :3] = R[i]
        M[:3, 3] = J[i] - (J[PARENTS[i]] if i else 0)
        G.append(M if i == 0 else G[PARENTS[i]] @ M)
    A = []
    for i in range(5):
        a = G[i].copy(); a[:3, 3] -= G[i][:3, :3] @ J[i]; A.append(a)
    A = np.stack(A)
    T = np.einsum("vj,jab->vab", B["lbs_weights"], A)
    out = np.einsum("vab,vb->va", T[:, :3, :3], v) + T[:, :3, 3]
    return out + trans


def face_frames(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
    a0 = n(v1 - v0); a1 = n(np.cross(a0, v2 - v0)); a2 = -n(np.cross(a1, a0))
    Rf = np.stack([a0, a1, a2], -1)
    s = (np.linalg.norm(v1 - v0, axis=-1) + np.abs((a2 * (v2 - v0)).sum(-1))) / 2
    return (v0 + v1 + v2) / 3, Rf, s


def gaussian_centers(B, verts):
    c, Rf, s = face_frames(verts, B["faces"].astype(np.int64))
    b = B["g_binding"].astype(np.int64)
    return np.einsum("nab,nb->na", Rf[b], B["g_xyz"]) * s[b, None] + c[b]
