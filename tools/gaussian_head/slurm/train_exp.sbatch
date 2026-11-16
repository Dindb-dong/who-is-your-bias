#!/bin/bash
# GaussianAvatars experiment on the full VHAP export (baseline / c1_ext1 / c1c3).
#   sbatch -w <node> -o $AVATAR_ROOT/logs/%x_%j.log train_exp.sbatch <name> <min_extent> "<extra train.py args>"
#   c1c3: sbatch ... train_exp.sbatch c1c3 1.0 "--flame_expr_lr 1e-4"
# Measured MaxRSS ~4.7 GB, ~11.6 h to 600k on one RTX 4090.
#SBATCH -J avatar-exp
#SBATCH -p batch
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
set -eo pipefail
NAME=$1; EXTENT=$2; EXTRA=$3
R=${AVATAR_ROOT:?set AVATAR_ROOT to the avatar work root}
source $R/env.sh
export GA_MIN_EXTENT=$EXTENT
EXP=$R/VHAP/export/monocular/jensen_a_whiteBg_staticOffset_maskBelowLine
OUT=$R/GaussianAvatars/output/jensen_a_$NAME
PORT=$((20000 + SLURM_JOB_ID % 20000))
cd $R/GaussianAvatars
echo "== $NAME smoke $(date)"
rm -rf $OUT-smoke
python train.py -s $EXP -m $OUT-smoke --eval --white_background --bind_to_mesh --port $PORT --iterations 300 --save_iterations 300 --test_iterations 300 $EXTRA
ls $OUT-smoke/point_cloud/iteration_300/point_cloud.ply
echo "== SMOKE OK $(date)"
python train.py -s $EXP -m $OUT --eval --white_background --bind_to_mesh --port $PORT $EXTRA
echo "== TRAIN DONE $(date)"
