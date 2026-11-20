#!/bin/bash
# Upper-body data prep: VHAP re-export without the neck mask -> 2D whole-body keypoints -> SMPL-X body fit -> masked folder.
#   AVATAR_ROOT=<root> sbatch -w <node> -o $AVATAR_ROOT/logs/%x_%j.log --export=ALL upperbody_prep.sbatch
# Needs in $AVATAR_ROOT/upperbody/: jensen_c1c3_smplx_head_b0.06.npz (smplx_head.py) and SMPLX_NEUTRAL_2020.npz (licensed),
# and this folder's scripts in $AVATAR_ROOT/upperbody/tools/.
#SBATCH -J upperbody-prep
#SBATCH -p batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
set -eo pipefail
R=${AVATAR_ROOT:?set AVATAR_ROOT}; U=$R/upperbody; TOOLS=$U/tools
source $R/env.sh
EXP=$R/VHAP/export/monocular/jensen_a_whiteBg_staticOffset_noMeshMask
OUT=$R/VHAP/export/monocular/jensen_a_upperbody
# keypoints / fit / prepare run in their own env so the training env is not touched
if [ ! -f $U/kpenv/.installed ]; then
  conda create -y -q -p $U/kpenv python=3.10
  $U/kpenv/bin/pip install -q "numpy<2" torch opencv-python-headless rtmlib onnxruntime-gpu
  touch $U/kpenv/.installed
fi
PY=$U/kpenv/bin/python
DEV=$($PY -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
echo "kpenv device $DEV"
cd $R/VHAP
[ -f $EXP/transforms_train.json ] || python vhap/export_as_nerf_dataset.py --src_folder output/monocular/jensen_a_whiteBg_staticOffset \
    --tgt_folder export/monocular/jensen_a_whiteBg_staticOffset_noMeshMask --background-color white --no-create-mask-from-mesh
echo "== export OK $(date)"
[ -f $U/kp_wholebody.npz ] || $PY $TOOLS/keypoints.py $EXP $U/kp_wholebody.npz --device $DEV --mode balanced
echo "== keypoints OK $(date)"
$PY $TOOLS/fit_body.py $EXP $U/jensen_c1c3_smplx_head_b0.06.npz $U/SMPLX_NEUTRAL_2020.npz $U/kp_wholebody.npz $U/body_fit.npz \
    --device $DEV --overlay_dir $U/fit_check
echo "== fit OK $(date)"
$PY $TOOLS/prepare_dataset.py $EXP $U/body_fit.npz $U/jensen_c1c3_smplx_head_b0.06.npz $OUT
echo "== PREP DONE $(date)"
