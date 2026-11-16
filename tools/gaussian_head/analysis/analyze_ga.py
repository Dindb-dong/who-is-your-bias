"""Read-only analysis of a trained GaussianAvatars model (no training, no writes to the model folder).

Run inside the GaussianAvatars repo:
    python analyze_ga.py <model_dir> <out_dir>
Writes to <out_dir>:
  cams.json                 per-view camera (OpenGL view matrix, fovY) + timestep for the WebGL comparison
  views/<split>_<t>_{gt,full,dc,noLowOp}.png   native renders (full SH, DC only, opacity<0.005 removed)
  psnr_by_iter.npz          per-frame PSNR for train/test frames at every saved iteration
  gauss_600k.npz            per-gaussian dump at timestep 0 (world centre, opacity, world scale, min-axis dir, binding, SH energy)
"""
import sys, os, json, glob
import numpy as np, torch
from argparse import ArgumentParser
from PIL import Image

model_dir, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.makedirs(out + "/views", exist_ok=True)
sys.argv = [sys.argv[0], "-m", model_dir]
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import render, FlameGaussianModel
from utils.general_utils import build_rotation

parser = ArgumentParser(); mp = ModelParams(parser, sentinel=True); pp = PipelineParams(parser)
args = get_combined_args(parser)
dataset, pipe = mp.extract(args), pp.extract(args)
bg = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.], device="cuda")
its = sorted(int(p.rsplit("_", 1)[1]) for p in glob.glob(model_dir + "/point_cloud/iteration_*"))
print("iterations", its)


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-10)))


def save(img, path):
    Image.fromarray((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(path)


def load(it):
    g = FlameGaussianModel(dataset.sh_degree)
    s = Scene(dataset, g, load_iteration=it, shuffle=False)
    return g, s


with torch.no_grad():
    # ---------- 1) per-frame PSNR at every saved iteration (same frames each time) ----------
    g, s = load(its[-1])
    train, test = s.getTrainCameras(), s.getTestCameras()
    tr_idx = [int(i) for i in np.linspace(0, len(train) - 1, 300).round()]  # CameraDataset only accepts python int
    te_idx = list(range(len(test)))
    res = {"iters": np.array(its), "train_t": np.array([train[i].timestep for i in tr_idx]),
           "test_t": np.array([test[i].timestep for i in te_idx])}
    gts = {("tr", i): train[i].original_image.cuda() for i in tr_idx}
    gts.update({("te", i): test[i].original_image.cuda() for i in te_idx})
    ps_tr, ps_te = [], []
    for it in its:
        if it != its[-1]:
            g, s = load(it)
            train, test = s.getTrainCameras(), s.getTestCameras()
        row_tr, row_te = [], []
        for i in tr_idx:
            v = train[i]; g.select_mesh_by_timestep(v.timestep)
            row_tr.append(psnr(render(v, g, pipe, bg)["render"], gts[("tr", i)]))
        for i in te_idx:
            v = test[i]; g.select_mesh_by_timestep(v.timestep)
            row_te.append(psnr(render(v, g, pipe, bg)["render"], gts[("te", i)]))
        ps_tr.append(row_tr); ps_te.append(row_te)
        print(f"iter {it}: train PSNR {np.mean(row_tr):.3f}  test PSNR {np.mean(row_te):.3f}", flush=True)
    res["train_psnr"], res["test_psnr"] = np.array(ps_tr), np.array(ps_te)
    np.savez(out + "/psnr_by_iter.npz", **res)
    del gts

    # ---------- 2) native renders of the final model for the WebGL comparison ----------
    g, s = load(its[-1])
    train, test = s.getTrainCameras(), s.getTestCameras()
    low = (g.get_opacity < 0.005).squeeze(-1)
    op_orig = g._opacity.data.clone()
    cams, stats = [], []
    for split, ds in (("train", train), ("test", test)):
        for i in np.linspace(0, len(ds) - 1, 8).round().astype(int):
            v = ds[int(i)]; t = int(v.timestep); g.select_mesh_by_timestep(t)
            gt = v.original_image.cuda()
            full = render(v, g, pipe, bg)["render"]
            g.active_sh_degree = 0; dc = render(v, g, pipe, bg)["render"]; g.active_sh_degree = g.max_sh_degree
            g._opacity.data[low] = -30.0; nolow = render(v, g, pipe, bg)["render"]; g._opacity.data.copy_(op_orig)
            stem = f"{out}/views/{split}_{t:05d}"
            for k, im in (("gt", gt), ("full", full), ("dc", dc), ("noLowOp", nolow)):
                save(im, f"{stem}_{k}.png")
            w2c_cv = v.world_view_transform.T.cpu().numpy().astype(np.float64)
            view_gl = np.diag([1., -1., -1., 1.]) @ w2c_cv
            cams.append(dict(split=split, timestep=t, view_colmajor=view_gl.T.reshape(-1).tolist(), fovY=float(v.FoVy),
                             width=int(v.image_width), height=int(v.image_height)))
            stats.append(dict(split=split, timestep=t, psnr_full_vs_gt=psnr(full, gt), psnr_dc_vs_gt=psnr(dc, gt),
                              psnr_dc_vs_full=psnr(dc, full), psnr_noLowOp_vs_full=psnr(nolow, full)))
            print(stats[-1], flush=True)
    json.dump(dict(cams=cams, stats=stats, n_low_opacity=int(low.sum()), n_total=int(low.numel())), open(out + "/cams.json", "w"), indent=1)

    # ---------- 3) per-gaussian dump at timestep 0 ----------
    g.select_mesh_by_timestep(0)
    sc = g.get_scaling; R = build_rotation(g.get_rotation)
    amin = sc.argmin(1)
    min_axis = R[torch.arange(len(R)), :, amin]
    np.savez_compressed(out + "/gauss_600k.npz",
                        xyz=g.get_xyz.cpu().numpy().astype(np.float32), opacity=g.get_opacity.squeeze(-1).cpu().numpy(),
                        scale=sc.cpu().numpy(), raw_log_scale=g._scaling.cpu().numpy(), min_axis=min_axis.cpu().numpy(),
                        binding=g.binding.cpu().numpy().astype(np.int32),
                        sh_dc=g._features_dc.squeeze(1).cpu().numpy(),
                        sh_rest_norm=g._features_rest.flatten(1).norm(dim=1).cpu().numpy(),
                        face_scaling=g.face_scaling.squeeze().cpu().numpy() if g.face_scaling.dim() > 1 else g.face_scaling.cpu().numpy(),
                        cam_center=np.linalg.inv(train[0].world_view_transform.T.cpu().numpy())[:3, 3])
    print("DONE", out)
