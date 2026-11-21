"""Resident EMAGE worker for the web API (speech WAV -> SMPL-X poses), one JSON request per line.

    <emage-env python> emage_worker.py <character-policy checkout> [--device cpu|cuda] [--threads N]

Started by bias.gesture.GestureEngine (BIAS_EMAGE_PYTHON / BIAS_EMAGE_REPO); it is not imported by the API,
so torch and the EMAGE code stay in their own environment. Protocol on stdin/stdout:

    -> {"ready": true, "fps": 30, "device": "cpu", "load_s": 12.3}                       once, after warm-up
    <- {"id": "...", "wav": "<path>", "out": "<path>"}
    -> {"id": "...", "ok": true, "frames": T, "fps": 30, "compute_ms": 812.5}           out = T*165 float32 (LE)

Model calls match emage_run.py (same as upstream test_emage_audio.py `inference()`). Everything else the
libraries print goes to stderr so stdout stays a clean protocol channel.
EMAGE code and weights are for research/education only.
"""
import argparse
import json
import os
import sys
import time

proto = os.fdopen(os.dup(1), 'w', buffering=1, encoding='utf-8')
os.dup2(2, 1)
sys.stdout = sys.stderr


def reply(**fields):
    proto.write(json.dumps(fields) + '\n')
    proto.flush()


ap = argparse.ArgumentParser()
ap.add_argument('repo')
ap.add_argument('--device', default='')
ap.add_argument('--threads', type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.repo))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from models.emage_audio import EmageAudioModel, EmageVAEConv, EmageVQModel, EmageVQVAEConv  # noqa: E402

if a.threads:
    torch.set_num_threads(a.threads)
dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
t0 = time.time()
vq = {p: EmageVQVAEConv.from_pretrained('haiyangliu0/emage_audio', subfolder=f'emage_vq/{p}').to(dev) for p in ('face', 'upper', 'lower', 'hands')}
glob_ae = EmageVAEConv.from_pretrained('haiyangliu0/emage_audio', subfolder='emage_vq/global').to(dev)
motion_vq = EmageVQModel(face_model=vq['face'], upper_model=vq['upper'], lower_model=vq['lower'], hands_model=vq['hands'],
                         global_model=glob_ae).to(dev).eval()
model = EmageAudioModel.from_pretrained('haiyangliu0/emage_audio').to(dev).eval()
sr, fps = model.cfg.audio_sr, model.cfg.pose_fps


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
    poses = out['motion_axis_angle'].cpu().numpy().reshape(t, -1)
    up = 30 // fps
    return np.repeat(poses, up, axis=0) if up > 1 else poses  # nearest-repeat to 30 fps, as in emage_run.py


run(np.zeros(sr * 2, np.float32))  # warm-up
reply(ready=True, fps=30, device=str(dev), load_s=round(time.time() - t0, 3))

for line in sys.stdin:
    if not line.strip():
        continue
    request_id = None
    try:
        request = json.loads(line)
        request_id = request['id']
        audio, _ = librosa.load(request['wav'], sr=sr)
        started = time.time()
        poses = run(audio.astype(np.float32))
        compute_ms = (time.time() - started) * 1000
        assert poses.ndim == 2 and poses.shape[1] == 165, poses.shape
        np.ascontiguousarray(poses, dtype='<f4').tofile(request['out'])
        reply(id=request_id, ok=True, frames=int(poses.shape[0]), fps=30, compute_ms=round(compute_ms, 3))
    except Exception as exc:  # report and keep serving; the API decides whether to restart
        reply(id=request_id, ok=False, error=f'{type(exc).__name__}: {exc}'[:500])
