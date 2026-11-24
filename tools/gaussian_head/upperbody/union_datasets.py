"""Merge prepared upper-body training folders (same subject, same outfit, separately tracked) into one source folder.

    python union_datasets.py <out_dir> <dataset_a> <dataset_b> [...]

Each input is a prepare_dataset.py output (optionally filtered by frame_audit.py). Frames keep their own images,
masks and FLAME parameter files (referenced by relative path, nothing is copied); later sequences get their timestep
indices shifted past the earlier ones, and smplx_body_param.npz is concatenated on the same shifted timesteps.
SmplxGaussianModel takes the head shape from the transplant, so per-sequence FLAME shape differences do not split the
identity; expressions, head pose and translation stay per frame.
"""
import json
import os
import sys

import numpy as np

out, sources = sys.argv[1], sys.argv[2:]
if len(sources) < 2:
    raise SystemExit(__doc__)
os.makedirs(out, exist_ok=True)
offset = 0
merged = {split: None for split in ('train', 'val', 'test')}
bodies = []
for k, src in enumerate(sources):
    fit = np.load(os.path.join(src, 'smplx_body_param.npz'))
    for split in merged:
        p = os.path.join(src, f'transforms_{split}.json')
        if not os.path.exists(p):
            continue
        meta = json.load(open(p))
        for fr in meta['frames']:
            fr['timestep_index'] = int(fr['timestep_index']) + offset
            fr['sequence'] = os.path.basename(os.path.normpath(src))
            for key in ('file_path', 'fg_mask_path', 'flame_param_path'):
                if key in fr:
                    fr[key] = os.path.relpath(os.path.realpath(os.path.join(src, fr[key])), out)
        if merged[split] is None:
            merged[split] = meta
        else:
            merged[split]['frames'] += meta['frames']
    bodies.append((fit, offset))
    n = int(fit['timesteps'].max()) + 1
    print(f'{src}: timesteps {offset}..{offset + n - 1}')
    offset += n

for split, meta in merged.items():
    if meta is not None:
        with open(os.path.join(out, f'transforms_{split}.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(split, len(meta['frames']), 'frames')
keys = {}
for name in ('body_pose', 'reproj_body_px', 'reproj_hand_px'):
    keys[name] = np.concatenate([fit[name] for fit, _ in bodies])
keys['timesteps'] = np.concatenate([fit['timesteps'] + off for fit, off in bodies])
keys['settings'] = bodies[0][0]['settings']
np.savez_compressed(os.path.join(out, 'smplx_body_param.npz'), **keys)
# other per-dataset files (canonical FLAME parameters etc.) come from the first sequence
for name in os.listdir(sources[0]):
    if name.startswith('transforms_') or name in ('images', 'fg_masks', 'smplx_body_param.npz', 'audit', 'flame_param'):
        continue
    dst = os.path.join(out, name)
    if not os.path.lexists(dst):
        os.symlink(os.path.realpath(os.path.join(sources[0], name)), dst)
print('wrote', out, 'timesteps', len(keys['timesteps']))
