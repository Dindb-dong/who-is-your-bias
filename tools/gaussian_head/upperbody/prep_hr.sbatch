#!/bin/bash
# Full-resolution (1024) upper-body data prep for one tracked sequence, reusing its VHAP tracking.
#   AVATAR_ROOT=<root> sbatch -w <node> -J ub-prep-<seq> -o $AVATAR_ROOT/logs/%x_%j.log --export=ALL prep_hr.sbatch <seq>
#   e.g. <seq> = jensen_a | jensen_b   (run both at once on two GPUs)
# VHAP tracked at n_downsample_rgb=2 (512). Its monocular camera focal length is relative to the image size, so the
# same tracking exports at 1024 by pointing a config copy at the original images (n_downsample_rgb: null).
# Steps: export (no neck mask) -> whole-body keypoints -> SMPL-X body fit -> masked folder -> drop frames whose head and
# nose disagree (frame_audit.py). Outputs:
#   $R/VHAP/export/monocular/<seq>_hr_noMeshMask       full-res export
#   $R/upperbody/hr/<seq>/{kp_wholebody,body_fit}.npz  keypoints / fit (+ fit_check overlay)
#   $R/VHAP/export/monocular/<seq>_upperbody_hr        training folder (GaussianAvatars source)
#SBATCH -p batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
set -eo pipefail
SEQ=${1:?usage: prep_hr.sbatch <sequence>}
R=${AVATAR_ROOT:?set AVATAR_ROOT}; U=$R/upperbody; TOOLS=$U/tools; H=$U/hr/$SEQ
source $R/env.sh
PY=$U/kpenv/bin/python
DEV=$($PY -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
mkdir -p $H
cd $R/VHAP

# 1) a tracking-run copy whose config reads the original 1024 images (symlinks to the tracked results)
SRC=$(ls -d output/monocular/${SEQ}_whiteBg_staticOffset/*/ | tail -1); SRC=${SRC%/}
HR=output/monocular/${SEQ}_whiteBg_staticOffset_hr/$(basename $SRC)
if [ ! -f $HR/config.yml ]; then
  mkdir -p $HR
  for f in $SRC/*; do [ "$(basename $f)" = config.yml ] || ln -sfn $(realpath $f) $HR/$(basename $f); done
  sed 's/^  n_downsample_rgb: 2$/  n_downsample_rgb: null/' $SRC/config.yml > $HR/config.yml
  grep -q '^  n_downsample_rgb: null$' $HR/config.yml || { echo "could not switch n_downsample_rgb"; exit 2; }
fi
EXP=$R/VHAP/export/monocular/${SEQ}_hr_noMeshMask
OUT=$R/VHAP/export/monocular/${SEQ}_upperbody_hr
echo "== export $(date)"
[ -f $EXP/transforms_train.json ] || python vhap/export_as_nerf_dataset.py --src_folder $HR --tgt_folder $EXP \
    --background-color white --no-create-mask-from-mesh
python -c "import json; f=json.load(open('$EXP/transforms_train.json'))['frames'][0]; print('export size', f['w'], f['h'], 'fl', round(f['fl_x'], 1))"

echo "== keypoints $(date)"
# REUSE_KP=<keypoints.npz from a lower-resolution export of the same sequence>: scale its pixel coordinates instead of
# detecting again (detection ran at ~7 s/frame on CPU for 1024 images).
if [ ! -f $H/kp_wholebody.npz ] && [ -n "$REUSE_KP" ]; then
  $PY - "$REUSE_KP" "$EXP" $H/kp_wholebody.npz <<'PY'
import json, sys
import numpy as np
src, export, out = sys.argv[1:4]
z = np.load(src)
w = json.load(open(export + '/transforms_train.json'))['frames'][0]['w']
s = w / float(z['image_size'][0])
kp = z['kp'].copy(); kp[..., :2] *= s
np.savez_compressed(out, kp=kp, timesteps=z['timesteps'], image_size=(z['image_size'] * s).round().astype(z['image_size'].dtype))
print(f'reused {src}: x{s:g} -> {out}', kp.shape)
PY
fi
[ -f $H/kp_wholebody.npz ] || $PY $TOOLS/keypoints.py $EXP $H/kp_wholebody.npz --device $DEV --mode balanced
echo "== fit $(date)"
$PY $TOOLS/fit_body.py $EXP $U/jensen_c1c3_smplx_head_b0.06.npz $U/SMPLX_NEUTRAL_2020.npz $H/kp_wholebody.npz $H/body_fit.npz \
    --device $DEV --overlay_dir $H/fit_check
echo "== prepare $(date)"
$PY $TOOLS/prepare_dataset.py $EXP $H/body_fit.npz $U/jensen_c1c3_smplx_head_b0.06.npz $OUT --margin_px 24   # 12 px at 512
echo "== audit $(date)"
$PY $TOOLS/frame_audit.py $EXP $U/jensen_c1c3_smplx_head_b0.06.npz $H/kp_wholebody.npz $H/body_fit.npz $OUT --max_arm_px 100000 --max_head_px 160 --apply   # 80 px at 512
echo "== PREP HR DONE $SEQ $(date)"
