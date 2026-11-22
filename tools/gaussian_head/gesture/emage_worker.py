"""Resident EMAGE worker for the web API (speech WAV -> SMPL-X poses), one JSON request per line.

    <emage-env python> emage_worker.py <character-policy checkout> [--device cpu|cuda] [--threads N] [--window 8..64]

Started by bias.gesture.GestureEngine (BIAS_EMAGE_PYTHON / BIAS_EMAGE_REPO); it is not imported by the API,
so torch and the EMAGE code stay in their own environment. Protocol on stdin/stdout:

    -> {"ready": true, "fps": 30, "device": "cpu", "load_s": 12.3}                       once, after warm-up
    <- {"id": "...", "wav": "<path>", "out": "<path>"}
    -> {"id": "...", "ok": true, "frames": T, "fps": 30, "compute_ms": 812.5}           out = T*165 float32 (LE)

Streaming (live voice). The model already runs in windows (pose_length 64 frames, 4 seed frames carried over),
so a session runs each window as soon as its audio exists and returns the new frames. Each window is decoded on
its own, so the 4 overlapping frames are cross-faded (slerp). --window shortens the windows for lower latency:
less audio is needed before the first motion, but the model was trained on 64.
    <- {"id", "op": "open", "session": "<hex>"}                                  -> {"id", "ok": true, "window"}
    <- {"id", "op": "push", "session", "pcm": "<base64 s16le>", "rate", "channels", "final": false}
    -> {"id", "ok": true, "chunks": [{"start": frame, "frames": n, "poses": "<base64 n*165 f32le>"}], "compute_ms"}
    <- {"id", "op": "close", "session"}                                          -> {"id", "ok": true}

Model calls match emage_run.py (same as upstream test_emage_audio.py `inference()`). Everything else the
libraries print goes to stderr so stdout stays a clean protocol channel.
EMAGE code and weights are for research/education only.
"""
import argparse
import base64
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
ap.add_argument('--window', type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.repo))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from models.emage_audio import EmageAudioModel, EmageVAEConv, EmageVQModel, EmageVQVAEConv  # noqa: E402
from models.emage_audio.modeling_emage_audio import axis_angle_to_rotation_6d  # noqa: E402

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


def latents(L):
    c = model.cfg
    pick = lambda part, l, cl: L[f'rec_{part}'] if l > 0 and cl == 0 else None  # noqa: E731
    idx = lambda part, cl: torch.max(F.log_softmax(L[f'cls_{part}'], dim=2), dim=2)[1] if cl > 0 else None  # noqa: E731
    return dict(face_latent=pick('face', c.lf, c.cf), upper_latent=pick('upper', c.lu, c.cu), lower_latent=pick('lower', c.ll, c.cl),
                hands_latent=pick('hands', c.lh, c.ch), face_index=idx('face', c.cf), upper_index=idx('upper', c.cu),
                lower_index=idx('lower', c.cl), hands_index=idx('hands', c.ch))


def run(audio):
    x = torch.from_numpy(audio).to(dev).unsqueeze(0)
    with torch.no_grad():
        L = model.inference(x, torch.zeros(1, 1).long().to(dev), motion_vq, masked_motion=None, mask=None)
        out = motion_vq.decode(**latents(L), get_global_motion=True, ref_trans=torch.zeros(1, 3).to(dev))
    t = out['motion_axis_angle'].shape[1]
    poses = out['motion_axis_angle'].cpu().numpy().reshape(t, -1)
    up = 30 // fps
    return np.repeat(poses, up, axis=0) if up > 1 else poses  # nearest-repeat to 30 fps, as in emage_run.py


# ---------------- streaming sessions ----------------
PRE = model.cfg.seed_frames
WIN = a.window or model.cfg.pose_length
HOP = sr // 30                                   # audio samples per pose frame, sliced as upstream does (533 at 16 kHz)
if fps != 30 or not PRE + 4 <= WIN <= model.cfg.pose_length:
    raise SystemExit(f'streaming needs 30 fps and {PRE + 4} <= window <= {model.cfg.pose_length}')
SPK = torch.zeros(1, 1).long().to(dev)


def rotvec_to_quat(v):
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-8, v / np.maximum(angle, 1e-12), 0.0)
    return np.concatenate([np.cos(angle / 2), axis * np.sin(angle / 2)], -1)


def quat_to_rotvec(q):
    q = np.where(q[..., :1] < 0, -q, q)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(s, q[..., :1])
    return np.where(s > 1e-8, q[..., 1:] / np.maximum(s, 1e-12) * angle, 2 * q[..., 1:])


def blend(a, b, w):
    """Per-joint slerp from frames a to frames b (n x 165) with weights w (n,)."""
    qa, qb = rotvec_to_quat(a.reshape(len(a), 55, 3)), rotvec_to_quat(b.reshape(len(b), 55, 3))
    dot = np.sum(qa * qb, -1, keepdims=True)
    qb = np.where(dot < 0, -qb, qb)
    theta = np.arccos(np.clip(np.abs(dot), 0, 1))
    s = np.sin(theta)
    t = w[:, None, None]
    small = s < 1e-6
    safe = np.where(small, 1.0, s)
    q = np.where(small, (1 - t) * qa + t * qb, (np.sin((1 - t) * theta) * qa + np.sin(t * theta) * qb) / safe)
    return quat_to_rotvec(q / np.linalg.norm(q, axis=-1, keepdims=True)).reshape(len(a), 165).astype(np.float32)


def window(audio_slice, seed, n):
    zero = axis_angle_to_rotation_6d(torch.zeros(1, n, 55, 3, device=dev)).reshape(1, n, -1)
    motion = torch.cat([zero, torch.zeros(1, n, 7, device=dev)], -1)
    if seed is not None:
        motion[:, :PRE] = seed
    mask = torch.ones_like(motion)
    mask[:, :PRE] = 0
    with torch.no_grad():
        net = model.forward(torch.from_numpy(audio_slice).to(dev).unsqueeze(0), SPK, masked_motion=motion, mask=mask, use_audio=True)
        dec = motion_vq.decode(**latents(net))
    return dec['all_motion4inference'][:, -PRE:], dec['motion_axis_angle'][0].cpu().numpy().reshape(n, 165).astype(np.float32)


class Stream:
    """The windows of EmageAudioModel.inference, emitted as soon as each window's audio exists."""

    def __init__(self):
        self.audio = np.zeros(0, np.float32)
        self.start = 0
        self.seed = None
        self.pending = None
        self.closed = False

    def _emit(self, aa, last):
        if self.pending is not None:
            k = min(PRE, len(aa))
            aa = aa.copy()
            aa[:k] = blend(self.pending[:k], aa[:k], (np.arange(k) + 0.5) / PRE)
        frames, self.pending = (aa, None) if last else (aa[:-PRE], aa[-PRE:])
        chunk = (self.start, frames)
        self.start += len(frames)
        return chunk

    def push(self, audio, final):
        if self.closed:
            raise ValueError('stream already finished')
        self.audio = np.concatenate([self.audio, audio])
        out = []
        while len(self.audio) >= (self.start + WIN) * HOP:
            self.seed, aa = window(self.audio[self.start * HOP:(self.start + WIN) * HOP], self.seed, WIN)
            out.append(self._emit(aa, last=False))
        if final:
            n = len(self.audio) * 30 // sr - self.start
            if n >= PRE + 4:
                self.seed, aa = window(self.audio[self.start * HOP:(self.start + n) * HOP], self.seed, n)
                out.append(self._emit(aa, last=True))
            elif self.pending is not None and n > 0:
                out.append((self.start, self.pending[:n]))
                self.start += min(n, len(self.pending))
                self.pending = None
            self.closed = True
        return out


streams = {}


def stream_request(request):
    op, sid = request['op'], request.get('session')
    if not isinstance(sid, str) or not 8 <= len(sid) <= 64:
        raise ValueError('invalid session')
    if op == 'open':
        while len(streams) >= 8:                     # sessions whose client went away are dropped oldest first
            streams.pop(next(iter(streams)))
        streams[sid] = Stream()
        return dict(window=WIN, seed_frames=PRE)
    if op == 'close':
        streams.pop(sid, None)
        return {}
    if op != 'push':
        raise ValueError('unknown op')
    stream = streams.get(sid)
    if stream is None:
        raise KeyError('unknown session')
    pcm = np.frombuffer(base64.b64decode(request['pcm']), dtype='<i2')
    channels, rate = int(request['channels']), int(request['rate'])
    if channels not in (1, 2) or not 8000 <= rate <= 48000 or len(pcm) % channels or len(pcm) > rate * channels * 60:
        raise ValueError('unsupported pcm')
    audio = pcm.reshape(-1, channels).mean(1).astype(np.float32) / 32768
    if len(audio) and rate != sr:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=sr).astype(np.float32)
    started = time.time()
    chunks = stream.push(audio, bool(request.get('final')))
    if stream.closed:
        streams.pop(sid, None)
    return dict(chunks=[dict(start=int(s0), frames=int(len(f)),
                             poses=base64.b64encode(np.ascontiguousarray(f, dtype='<f4').tobytes()).decode('ascii')) for s0, f in chunks],
                compute_ms=round((time.time() - started) * 1000, 3))


run(np.zeros(sr * 2, np.float32))  # warm-up
librosa.resample(np.zeros(2400, np.float32), orig_sr=24000, target_sr=sr)  # first resample call loads its backend (~2 s)
reply(ready=True, fps=30, window=WIN, device=str(dev), load_s=round(time.time() - t0, 3))

for line in sys.stdin:
    if not line.strip():
        continue
    request_id = None
    try:
        request = json.loads(line)
        request_id = request['id']
        if 'op' in request:
            reply(id=request_id, ok=True, **stream_request(request))
            continue
        audio, _ = librosa.load(request['wav'], sr=sr)
        started = time.time()
        poses = run(audio.astype(np.float32))
        compute_ms = (time.time() - started) * 1000
        assert poses.ndim == 2 and poses.shape[1] == 165, poses.shape
        np.ascontiguousarray(poses, dtype='<f4').tofile(request['out'])
        reply(id=request_id, ok=True, frames=int(poses.shape[0]), fps=30, compute_ms=round(compute_ms, 3))
    except Exception as exc:  # report and keep serving; the API decides whether to restart
        reply(id=request_id, ok=False, error=f'{type(exc).__name__}: {exc}'[:500])
