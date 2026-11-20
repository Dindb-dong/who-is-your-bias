#!/bin/bash
# Upper-body GaussianAvatars training (head + torso + arms + hands bound to SMPL-X with the transplanted FLAME head).
#   AVATAR_ROOT=<root> sbatch -w <node> -o $AVATAR_ROOT/logs/%x_%j.log --export=ALL train_upperbody.sbatch [iterations]
#SBATCH -J avatar-upperbody
#SBATCH -p batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
set -eo pipefail
ITERS=${1:-240000}
R=${AVATAR_ROOT:?set AVATAR_ROOT}; U=$R/upperbody
source $R/env.sh
export GA_MIN_EXTENT=1.0 GA_UPPERBODY=$U/jensen_c1c3_smplx_head_b0.06.npz
SRC=$R/VHAP/export/monocular/jensen_a_upperbody
OUT=$R/GaussianAvatars/output/jensen_a_upperbody
PORT=$((20000 + SLURM_JOB_ID % 20000))
cd $R/GaussianAvatars
python $U/tools/ga_patch/apply_patch.py $R/GaussianAvatars
echo "== smoke $(date)"
rm -rf $OUT-smoke
python train.py -s $SRC -m $OUT-smoke --eval --white_background --bind_to_mesh --port $PORT --iterations 300 --save_iterations 300 --test_iterations 300 --flame_expr_lr 1e-4
ls $OUT-smoke/point_cloud/iteration_300/point_cloud.ply
echo "== SMOKE OK $(date)"
python train.py -s $SRC -m $OUT --eval --white_background --bind_to_mesh --port $PORT --iterations $ITERS --flame_expr_lr 1e-4
echo "== TRAIN DONE $(date)"
