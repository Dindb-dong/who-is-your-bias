"""2D whole-body keypoints (COCO-WholeBody, 133 points) for every timestep of a VHAP NeRF export.

    python keypoints.py <export_dir> <out.npz> [--device cpu|cuda] [--mode balanced|performance|lightweight]

Uses rtmlib (RTMPose whole-body, ONNX). Reads transforms_{train,val,test}.json for the image of each timestep
(camera 0). Output: kp (T,133,3) = x, y, score in export-image pixels; timesteps without a person get score 0.
When several people are detected, keeps the one whose body box centre is closest to the image centre.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from rtmlib import Wholebody

ap = argparse.ArgumentParser()
ap.add_argument('export'); ap.add_argument('out')
ap.add_argument('--device', default='cpu'); ap.add_argument('--mode', default='balanced')
ap.add_argument('--stride', type=int, default=1, help='every Nth timestep only (quick checks); others keep score 0')
ap.add_argument('--allow-cpu', action='store_true', help='with --device cuda, continue if onnxruntime falls back to CPU')
a = ap.parse_args()

files = {}
for split in ('train', 'val', 'test'):
    p = os.path.join(a.export, f'transforms_{split}.json')
    if os.path.exists(p):
        for fr in json.load(open(p))['frames']:
            if fr.get('camera_index', 0) == 0:
                files[int(fr['timestep_index'])] = fr['file_path']
T = max(files) + 1
print(f'{len(files)} timesteps (max index {T - 1})')
if a.device == 'cuda':
    # onnxruntime-gpu silently runs on CPU when it cannot load CUDA/cuDNN (seen: ~7 s per 1024 frame). It needs the
    # CUDA 12 runtime wheels (nvidia-*-cu12); torch here ships CUDA 13, so do not import torch first. preload_dlls()
    # loads the cu12 libraries from site-packages; then refuse to continue on CPU unless asked.
    import onnxruntime as ort
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
    print('onnxruntime', ort.__version__, 'available providers', ort.get_available_providers(), flush=True)
model = Wholebody(mode=a.mode, backend='onnxruntime', device=a.device)
if a.device == 'cuda':
    used = [getattr(getattr(model, part, None), 'session', None) for part in ('det_model', 'pose_model')]
    used = [s.get_providers() for s in used if s is not None]
    print('session providers', used, flush=True)
    if not used or any(p[0] != 'CUDAExecutionProvider' for p in used):
        if not a.allow_cpu:
            raise SystemExit('onnxruntime is not using CUDA for keypoints; fix the CUDA/cuDNN libraries or pass --allow-cpu')
kp = np.zeros((T, 133, 3), np.float32)
done = np.zeros(T, bool)
partial = a.out + '.partial.npz'
if os.path.exists(partial):                                       # resume an interrupted run
    z = np.load(partial); kp[:len(z['kp'])] = z['kp']; done[:len(z['done'])] = z['done']
    print(f'resuming: {int(done.sum())} timesteps already done')
w = h = 0
t0 = time.time()
for n, (t, rel) in enumerate(sorted(files.items())):
    if n % a.stride or done[t]:
        continue
    if n % 100 == 0:
        np.savez(partial, kp=kp, done=done)
        print(f'  {n}/{len(files)}  {time.time() - t0:.0f}s', flush=True)
    done[t] = True
    img = cv2.imread(os.path.join(a.export, rel))
    if img is None:
        continue
    pts, sc = model(img)
    if len(pts) == 0:
        continue
    h, w = img.shape[:2]
    body = pts[:, :17]
    centre = np.abs(body.mean(1) - [w / 2, h / 2]).sum(1) - 1000 * sc[:, :17].mean(1)
    i = int(np.argmin(centre))
    kp[t, :, :2] = pts[i]; kp[t, :, 2] = sc[i]
if not w:
    w, h = cv2.imread(os.path.join(a.export, next(iter(files.values())))).shape[1::-1]
np.savez_compressed(a.out, kp=kp, timesteps=np.array(sorted(files)), image_size=np.array([w, h]))
good = kp[:, :, 2] > 0.3
print(f'wrote {a.out} | body score>0.3: shoulders {good[:, 5:7].mean():.2f} elbows {good[:, 7:9].mean():.2f} '
      f'wrists {good[:, 9:11].mean():.2f} hands {good[:, 91:133].mean():.2f} | {time.time() - t0:.0f}s')
