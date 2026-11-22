"""Convert a BEAT2-format SMPL-X + FLAME motion .npz (EMAGE / StreamTalk / GestureLSM output) for the SMPL-X head preview.

    python npz_to_motion.py <motion.npz> <out_stem> [--audio <speech.wav>] [--max-seconds 60]

Input keys (BEAT2 format, 30 fps): poses (T,165) SMPL-X axis-angle, expressions (T,100) FLAME, trans (T,3).
Writes <out_stem>.json + <out_stem>.bin (schema smplx-motion.v1) and copies the audio to <out_stem>.wav.
Root translation is dropped (the preview keeps the speaker in place); root orientation is kept.
"""
import argparse
import json
import shutil

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('npz')
ap.add_argument('out')
ap.add_argument('--audio')
ap.add_argument('--max-seconds', type=float, default=0)
a = ap.parse_args()

d = np.load(a.npz, allow_pickle=False)
poses = d['poses'].astype(np.float32)
expr = d['expressions'].astype(np.float32) if 'expressions' in d.files else np.zeros((len(poses), 100), np.float32)
fps = int(d['mocap_frame_rate']) if 'mocap_frame_rate' in d.files else 30
assert poses.ndim == 2 and poses.shape[1] == 165, poses.shape
assert expr.shape == (len(poses), 100), expr.shape
if a.max_seconds > 0:
    n = int(a.max_seconds * fps)
    poses, expr = poses[:n], expr[:n]

blobs, buffers, off = [], {}, 0
for name, arr in (('poses', poses), ('expressions', expr)):
    b = np.ascontiguousarray(arr).tobytes()
    buffers[name] = dict(dtype='f32', shape=list(arr.shape), offset=off, bytes=len(b))
    blobs.append(b); off += len(b)
header = dict(schema='smplx-motion.v1', fps=fps, n_frames=len(poses), buffers=buffers, source=a.npz.replace('\\', '/').rsplit('/', 1)[-1],
              note='generated or captured motion; check the source licence before sharing (BEAT2 data is CC BY-NC 4.0)')
open(a.out + '.bin', 'wb').write(b''.join(blobs))
json.dump(header, open(a.out + '.json', 'w'), indent=1)
if a.audio:
    shutil.copyfile(a.audio, a.out + '.wav')
print(f'wrote {a.out}.json/.bin  frames {len(poses)} @ {fps} fps ({len(poses)/fps:.1f} s)  '
      f'expr |mean| {np.abs(expr).mean():.3f}  root0 {poses[0, :3].round(2).tolist()}')
