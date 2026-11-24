#!/bin/bash
# After train_upperbody.sbatch: render test views and export browser bundles for chosen iterations.
#   AVATAR_ROOT=<root> sbatch -w <node> -o $AVATAR_ROOT/logs/%x_%j.log --export=ALL export_upperbody.sbatch [iters...]
# Needs $AVATAR_ROOT/upperbody/SMPL-X__FLAME_vertex_ids.npy (SMPL-X download; licensed, never committed).
# Outputs (research-licensed, keep off Git):
#   $U/export/jensen_upperbody_<iter>.{json,bin}       smplx-head-preview.v1 bundles for tools/gaussian_head/smplx/preview.html
#   $U/export/test_compare_<iter>.jpg                   ground truth | render for 6 test frames
#SBATCH -J avatar-upperbody-export
#SBATCH -p batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -eo pipefail
ITERS=${@:-120000 240000}
R=${AVATAR_ROOT:?set AVATAR_ROOT}; U=$R/upperbody
source $R/env.sh
export GA_MIN_EXTENT=1.0 GA_UPPERBODY=$U/jensen_c1c3_smplx_head_b0.06.npz
NAME=${UB_NAME:-jensen_a_upperbody}                     # e.g. jensen_a_upperbody_hr, jensen_ab_upperbody_hr
MODEL=$R/GaussianAvatars/output/$NAME
IDS=$U/SMPL-X__FLAME_vertex_ids.npy
[ -f $IDS ] || { echo "missing $IDS"; exit 2; }
mkdir -p $U/export
cd $R/GaussianAvatars
python $U/tools/ga_patch/apply_patch.py $R/GaussianAvatars
for IT in $ITERS; do
  echo "== export $IT $(date)"
  GA_ITER=$IT python $U/tools/export_upperbody.py $MODEL $GA_UPPERBODY $IDS $U/export/${NAME}_$IT
  ls -la $U/export/${NAME}_$IT.*
  echo "== render test $IT $(date)"
  python render.py -m $MODEL --iteration $IT --skip_train --skip_val
  python - $MODEL $IT $U/export/test_compare_${NAME}_$IT.jpg <<'PY'
import glob, sys
from PIL import Image
model, it, out = sys.argv[1:4]
renders = sorted(glob.glob(f'{model}/test/ours_{it}/renders/*.png'))
gts = sorted(glob.glob(f'{model}/test/ours_{it}/gt/*.png'))
assert renders and len(renders) == len(gts), (len(renders), len(gts))
pick = [round(i * (len(renders) - 1) / 5) for i in range(6)]
tiles = []
for i in pick:
    g, r = Image.open(gts[i]).convert('RGB'), Image.open(renders[i]).convert('RGB')
    w, h = 256, round(256 * g.height / g.width)
    pair = Image.new('RGB', (2 * w, h)); pair.paste(g.resize((w, h)), (0, 0)); pair.paste(r.resize((w, h)), (w, 0))
    tiles.append(pair)
sheet = Image.new('RGB', (tiles[0].width * 3, tiles[0].height * 2), 'white')
for k, t in enumerate(tiles):
    sheet.paste(t, ((k % 3) * t.width, (k // 3) * t.height))
sheet.save(out, quality=88)
print('wrote', out, 'frames', pick, 'of', len(renders))
PY
done
echo "== EXPORT DONE $(date)"
