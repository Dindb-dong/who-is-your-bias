#!/bin/bash
# Data-quantity experiment: c1c3 settings on the first P% of train frames, stopped at 240k.
# GaussianAvatars fixes position_lr_max_steps and densify_until_iter at 600k, so this run follows the same
# schedule as the full-data c1c3 run up to 240k and compares directly against its 60k-240k checkpoints.
#   make_train_subset.sh <export> 50 && sbatch -w <node> -o $AVATAR_ROOT/logs/%x_%j.log train_scale.sbatch 50
#SBATCH -J avatar-scale
#SBATCH -p batch
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -eo pipefail
P=$1
R=${AVATAR_ROOT:?set AVATAR_ROOT to the avatar work root}
source $R/env.sh
export GA_MIN_EXTENT=1.0
EXP=$R/VHAP/export/monocular/jensen_a_whiteBg_staticOffset_maskBelowLine_train$P
OUT=$R/GaussianAvatars/output/jensen_a_c1c3_train$P
PORT=$((20000 + SLURM_JOB_ID % 20000))
cd $R/GaussianAvatars
echo "== train$P smoke $(date)"
rm -rf $OUT-smoke
python train.py -s $EXP -m $OUT-smoke --eval --white_background --bind_to_mesh --port $PORT --iterations 300 --save_iterations 300 --test_iterations 300 --flame_expr_lr 1e-4
ls $OUT-smoke/point_cloud/iteration_300/point_cloud.ply
echo "== SMOKE OK, training to 240k $(date)"
python train.py -s $EXP -m $OUT --eval --white_background --bind_to_mesh --port $PORT --iterations 240000 --flame_expr_lr 1e-4
echo "== TRAIN DONE $(date)"
