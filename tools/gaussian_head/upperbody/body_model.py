"""Torch SMPL-X body with the transplanted FLAME head (smplx_head.py output), driven by VHAP FLAME tracking.

The head stays exactly where FLAME tracking puts it, whatever the body pose:
  * the SMPL-X neck joint (12) takes FLAME's global rotation: its local rotation is solved as R_spine3^T R_flame;
  * the body translation is solved so joint 12 lands on FLAME's root joint (J0 + translation - flame_offset).
So with the body at rest this reproduces smplx_head.py's validation A, and torso/arm poses never move the head.

    model = TransplantedSmplx(np.load('jensen_c1c3_smplx_head_b0.06.npz'))
    out = model(body_pose, flame)      # body_pose (B,55,3) axis-angle; head joints 12/15/22/23/24 are ignored
    out['verts'] (B,N,3), out['joints'] (B,55,3)
flame: dict of (B,·) tensors with VHAP/GaussianAvatars keys: expr, rotation, neck_pose, jaw_pose, eyes_pose, translation.
"""
import numpy as np
import torch

HEAD_JOINTS = (12, 15, 22, 23, 24)


def axis_angle_to_matrix(aa):
    """(..., 3) -> (..., 3, 3), Rodrigues; safe at zero."""
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = aa / th
    kx, ky, kz = k.unbind(-1)
    z = torch.zeros_like(kx)
    K = torch.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], -1).reshape(*aa.shape[:-1], 3, 3)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    small = (aa.norm(dim=-1) < 1e-8)[..., None, None]
    R = eye + s * K + (1 - c) * (K @ K)
    return torch.where(small, eye + K * th[..., None], R)


class TransplantedSmplx(torch.nn.Module):
    def __init__(self, npz, bound_faces=None):
        super().__init__()
        f32 = lambda k: torch.from_numpy(np.asarray(npz[k], np.float32))  # noqa: E731
        self.register_buffer('v_rest', f32('v_rest'))                      # (N,3)
        self.register_buffer('expr_dirs', f32('expr_dirs'))                # (E,N,3)
        N = self.v_rest.shape[0]
        self.register_buffer('pose_dirs', f32('pose_dirs').reshape(N * 3, -1))  # (N*3, 54*9)
        self.register_buffer('weights', f32('weights'))                    # (N,55)
        self.register_buffer('j_rest', f32('j_rest'))                      # (55,3)
        self.register_buffer('j_expr', f32('j_expr_dirs'))                 # (E,55,3)
        self.register_buffer('flame_offset', f32('flame_offset'))          # (3,)
        self.parents = [int(p) for p in np.asarray(npz['parents'])]; self.parents[0] = -1
        self.faces_gaussian = torch.from_numpy(np.asarray(npz['faces_gaussian'], np.int64))   # FLAME faces + teeth, new index
        self.faces_body = torch.from_numpy(np.asarray(npz['faces_body'], np.int64))
        self.n_expr = self.expr_dirs.shape[0]
        # chain to spine3 (9), parent of the neck joint 12
        self.chain9 = []
        j = 9
        while j >= 0:
            self.chain9.insert(0, j); j = self.parents[j]

    def forward(self, body_pose, flame, return_verts=True):
        B, dev = body_pose.shape[0], body_pose.device
        expr = flame['expr'][:, :self.n_expr]
        if expr.shape[1] < self.n_expr:
            expr = torch.nn.functional.pad(expr, (0, self.n_expr - expr.shape[1]))
        J = self.j_rest[None] + torch.einsum('be,ejc->bjc', expr, self.j_expr)
        P = body_pose.clone()
        P[:, 15] = flame['neck_pose']; P[:, 22] = flame['jaw_pose']
        P[:, 23] = flame['eyes_pose'][:, :3]; P[:, 24] = flame['eyes_pose'][:, 3:6]
        P[:, 12] = 0
        R = axis_angle_to_matrix(P)                                        # (B,55,3,3)
        # global rotation of spine3 -> local neck rotation that yields FLAME's global head rotation
        G9 = None
        for j in self.chain9:
            G9 = R[:, j] if G9 is None else G9 @ R[:, j]
        R12 = G9.transpose(1, 2) @ axis_angle_to_matrix(flame['rotation'])
        R = torch.cat([R[:, :12], R12[:, None], R[:, 13:]], 1)
        # forward kinematics (translation 0), then shift so joint 12 sits on FLAME's root joint
        Gr, Gt = [None] * 55, [None] * 55
        for j in range(55):
            p = self.parents[j]
            off = J[:, j] - (J[:, p] if p >= 0 else 0)
            if p < 0:
                Gr[j], Gt[j] = R[:, j], off
            else:
                Gr[j] = Gr[p] @ R[:, j]
                Gt[j] = Gt[p] + (Gr[p] @ off[..., None])[..., 0]
        Gr = torch.stack(Gr, 1); Gt = torch.stack(Gt, 1)                    # (B,55,3,3), (B,55,3)
        trans = J[:, 12] + flame['translation'] - self.flame_offset - Gt[:, 12]
        joints = Gt + trans[:, None]
        out = {'joints': joints, 'rotmats': R, 'trans': trans}           # rotmats: local, with the solved neck (12)
        if not return_verts:
            return out
        v = self.v_rest[None] + torch.einsum('be,enc->bnc', expr, self.expr_dirs)
        eye = torch.eye(3, device=dev)
        pf = (R[:, 1:] - eye).reshape(B, -1)
        v = v + (pf @ self.pose_dirs.T).reshape(B, -1, 3)
        At = Gt - (Gr @ J[..., None])[..., 0]                               # relative transforms
        A = torch.cat([Gr, At[..., None]], -1).reshape(B, 55, 12)
        T = torch.einsum('nj,bjk->bnk', self.weights, A).reshape(B, -1, 3, 4)
        verts = (T[..., :3] @ v[..., None])[..., 0] + T[..., 3] + trans[:, None]
        out['verts'] = verts
        return out
