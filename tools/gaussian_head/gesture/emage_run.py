"""Pretrained EMAGE inference (speech -> SMPL-X + FLAME, BEAT2 format) with per-clip timing; no rendering.

    python emage_run.py <character-policy checkout> <audio_dir> <out_dir> [--device cpu|cuda] [--threads N]

Same model calls as the upstream test_emage_audio.py `inference()`, without its video/visualisation imports.
Prints the real-time factor (compute seconds / audio seconds) per clip; < 1 means faster than real time.
EMAGE code and weights are for research/education only.
"""
import argparse
import glob
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('repo'); ap.add_argument('audio_dir'); ap.add_argument('out_dir')
ap.add_argument('--device', default='')
ap.add_argument('--threads', type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.repo))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from models.emage_audio import EmageAudioModel, EmageVAEConv, EmageVQModel, EmageVQVAEConv  # noqa: E402

if a.threads: torch.set_num_threads(a.threads)
dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
t0 = time.time()
vq = {p: EmageVQVAEConv.from_pretrained('haiyangliu0/emage_audio', subfolder=f'emage_vq/{p}').to(dev) for p in ('face', 'upper', 'lower', 'hands')}
glob_ae = EmageVAEConv.from_pretrained('haiyangliu0/emage_audio', subfolder='emage_vq/global').to(dev)
motion_vq = EmageVQModel(face_model=vq['face'], upper_model=vq['upper'], lower_model=vq['lower'], hands_model=vq['hands'],
                         global_model=glob_ae).to(dev).eval()
model = EmageAudioModel.from_pretrained('haiyangliu0/emage_audio').to(dev).eval()
sr, fps = model.cfg.audio_sr, model.cfg.pose_fps
print(f'device {dev} threads {torch.get_num_threads()} | load {time.time() - t0:.1f}s | audio_sr {sr} pose_fps {fps}')


def run(audio):
    x = torch.from_numpy(audio).to(dev).unsqueeze(0)
    with torch.no_grad():
        L = model.inference(x, torch.zeros(1, 1).long().to(dev), motion_vq, masked_motion=None, mask=None)
        c = model.cfg
        pick = lambda part, l, cl: L[f'rec_{part}'] if l > 0 and cl == 0 else None  # noqa: E731
        idx = lambda part, cl: torch.max(F.log_softmax(L[f'cls_{part}'], dim=2), dim=2)[1] if cl > 0 else None  # noqa: E731
        out = motion_vq.decode(
            face_latent=pick('face', c.lf, c.cf), upper_latent=pick('upper', c.lu, c.cu), lower_latent=pick('lower', c.ll, c.cl),
            hands_latent=pick('hands', c.lh, c.ch), face_index=idx('face', c.cf), upper_index=idx('upper', c.cu),
            lower_index=idx('lower', c.cl), hands_index=idx('hands', c.ch), get_global_motion=True, ref_trans=torch.zeros(1, 3).to(dev))
    t = out['motion_axis_angle'].shape[1]
    return (out['motion_axis_angle'].cpu().numpy().reshape(t, -1), out['expression'].cpu().numpy().reshape(t, -1),
            out['trans'].cpu().numpy().reshape(t, -1))


os.makedirs(a.out_dir, exist_ok=True)
run(np.zeros(sr * 2, np.float32))  # warm-up
tot_a = tot_c = 0.0
for f in sorted(glob.glob(os.path.join(a.audio_dir, '*.wav'))):
    audio, _ = librosa.load(f, sr=sr)
    t1 = time.time(); poses, expr, trans = run(audio); dt = time.time() - t1
    up = 30 // fps
    if up > 1:  # upstream beat_format_save upsamples to 30 fps; nearest-repeat keeps this script dependency-free
        poses, expr, trans = (np.repeat(v, up, axis=0) for v in (poses, expr, trans))
    name = os.path.splitext(os.path.basename(f))[0]
    np.savez(os.path.join(a.out_dir, f'{name}_output.npz'), poses=poses.astype(np.float32), expressions=expr.astype(np.float32),
             trans=trans.astype(np.float32), betas=np.zeros(300, np.float32), mocap_frame_rate=np.int32(30),
             gender=np.str_('neutral'), model=np.str_('smplx2020'))
    dur = len(audio) / sr; tot_a += dur; tot_c += dt
    print(f'{name}: audio {dur:5.1f}s  compute {dt:5.2f}s  RTF {dt / dur:.3f}  frames {len(poses)}')
print(f'TOTAL audio {tot_a:.1f}s compute {tot_c:.2f}s RTF {tot_c / max(tot_a, 1e-9):.3f}')
