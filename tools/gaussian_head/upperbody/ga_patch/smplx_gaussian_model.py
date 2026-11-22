"""GaussianAvatars model bound to the SMPL-X body + transplanted FLAME head (upper-body avatar).

Drop-in for FlameGaussianModel (copied into GaussianAvatars/scene/ by apply_patch.py):
  * faces = FLAME head + teeth faces (same order as FlameGaussianModel, so head gaussians keep their binding ids)
    followed by the SMPL-X upper-body faces (torso, arms, hands; legs excluded);
  * the mesh for a timestep comes from TransplantedSmplx(body_pose[t], FLAME tracking[t]); the head stays exactly
    on the FLAME tracking, the body follows the fitted SMPL-X pose;
  * FLAME pose/translation/expression are fine-tuned as in FlameGaussianModel; the body pose can be fine-tuned too
    (GA_BODY_POSE_LR, default 1e-5; 0 disables).
Environment: GA_UPPERBODY=<smplx_head.npz>; the fitted body pose is read from <source_path>/smplx_body_param.npz.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz

from utils.graphics_utils import compute_face_orientation
from .gaussian_model import GaussianModel

sys.path.insert(0, os.environ.get('GA_UPPERBODY_TOOLS', os.path.dirname(os.path.abspath(__file__))))
from body_model import TransplantedSmplx  # noqa: E402

LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
FLAME_KEYS = ('expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation')


class SmplxGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int, transplant_npz: str, body_param_npz: str, not_finetune_flame_params=False):
        super().__init__(sh_degree)
        self.not_finetune_flame_params = not_finetune_flame_params
        S = np.load(transplant_npz)
        self.body = TransplantedSmplx(S).cuda()
        # upper-body faces: body faces whose vertices are mostly not skinned to the legs
        W = np.asarray(S['weights'])
        leg = W[:, LEG_JOINTS].sum(1) + 0.5 * W[:, 0]
        fb = np.asarray(S['faces_body'], np.int64)
        upper = fb[(leg[fb] < 0.5).all(1)]
        faces = np.concatenate([np.asarray(S['faces_gaussian'], np.int64), upper])
        self.n_head_faces = len(S['faces_gaussian'])
        self.faces_all = torch.from_numpy(faces).cuda()
        self.body_param_npz = body_param_npz
        self.flame_param = None
        self.flame_param_orig = None
        if self.binding is None:
            self.binding = torch.arange(len(faces)).cuda()
            self.binding_counter = torch.ones(len(faces), dtype=torch.int32).cuda()
        print(f'SmplxGaussianModel: {self.n_head_faces} head faces + {len(upper)} upper-body faces')

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        meshes = {**train_meshes, **test_meshes}
        tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
        pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
        self.num_timesteps = max(pose_meshes) + 1
        T = self.num_timesteps
        E = self.body.n_expr
        p = {'expr': torch.zeros([T, E]), 'rotation': torch.zeros([T, 3]), 'neck_pose': torch.zeros([T, 3]),
             'jaw_pose': torch.zeros([T, 3]), 'eyes_pose': torch.zeros([T, 6]), 'translation': torch.zeros([T, 3]),
             'body_pose': torch.zeros([T, 55, 3])}
        for i, mesh in pose_meshes.items():
            ex = torch.from_numpy(np.asarray(mesh['expr'], np.float32)).reshape(-1)[:E]
            p['expr'][i, :len(ex)] = ex
            for k in FLAME_KEYS[1:]:
                p[k][i] = torch.from_numpy(np.asarray(mesh[k], np.float32)).reshape(-1)
        bp = np.load(self.body_param_npz)
        ts = np.asarray(bp['timesteps'], np.int64)
        p['body_pose'][ts[ts < T]] = torch.from_numpy(np.asarray(bp['body_pose'], np.float32)[ts < T])
        missing = sorted(set(pose_meshes) - set(ts.tolist()))
        if missing:
            print(f'WARNING: {len(missing)} timesteps without a fitted body pose (rest pose used), e.g. {missing[:5]}')
        self.flame_param = {k: v.float().cuda() for k, v in p.items()}
        self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}

    def _forward(self, param, idx):
        flame = {k: param[k][idx] for k in FLAME_KEYS}
        return self.body(param['body_pose'][idx], flame)['verts']

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        param = self.flame_param_orig if original and self.flame_param_orig is not None else self.flame_param
        verts = self._forward(param, [timestep])
        self.update_mesh_properties(verts, verts)

    def update_mesh_by_param_dict(self, flame_param):
        flame = {k: flame_param[k if k in flame_param else {'neck_pose': 'neck', 'jaw_pose': 'jaw', 'eyes_pose': 'eyes'}[k]].cuda()
                 for k in FLAME_KEYS}
        B = flame['expr'].shape[0]
        body = flame_param['body_pose'].cuda() if 'body_pose' in flame_param else torch.zeros(B, 55, 3, device='cuda')
        verts = self.body(body, flame)['verts']
        self.update_mesh_properties(verts, verts)

    def update_mesh_properties(self, verts, verts_cano):
        faces = self.faces_all
        triangles = verts[:, faces]
        self.face_center = triangles.mean(dim=-2).squeeze(0)
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces, return_scale=True)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))
        self.verts = verts
        self.faces = faces
        self.verts_cano = verts_cano

    def compute_dynamic_offset_loss(self):
        return torch.zeros((), device='cuda')

    def compute_laplacian_loss(self):
        return torch.zeros((), device='cuda')

    def training_setup(self, training_args):
        super().training_setup(training_args)
        if self.not_finetune_flame_params:
            return
        fp = self.flame_param
        for k in ('rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation', 'expr'):
            fp[k].requires_grad = True
        self.optimizer.add_param_group({'params': [fp['rotation'], fp['neck_pose'], fp['jaw_pose'], fp['eyes_pose']],
                                        'lr': training_args.flame_pose_lr, 'name': 'pose'})
        self.optimizer.add_param_group({'params': [fp['translation']], 'lr': training_args.flame_trans_lr, 'name': 'trans'})
        self.optimizer.add_param_group({'params': [fp['expr']], 'lr': training_args.flame_expr_lr, 'name': 'expr'})
        body_lr = float(os.environ.get('GA_BODY_POSE_LR', '1e-5'))
        if body_lr > 0:
            fp['body_pose'].requires_grad = True
            self.optimizer.add_param_group({'params': [fp['body_pose']], 'lr': body_lr, 'name': 'body_pose'})

    def save_ply(self, path):
        super().save_ply(path)
        np.savez(str(Path(path).parent / 'flame_param.npz'), **{k: v.detach().cpu().numpy() for k, v in self.flame_param.items()})
        np.save(str(Path(path).parent / 'bound_faces.npy'), self.faces_all.cpu().numpy())

    def load_ply(self, path, **kwargs):
        super().load_ply(path)
        if not kwargs.get('has_target', False):
            fp = np.load(str(Path(path).parent / 'flame_param.npz'))
            self.flame_param = {k: torch.from_numpy(v).cuda() for k, v in fp.items()}
            self.num_timesteps = self.flame_param['expr'].shape[0]
