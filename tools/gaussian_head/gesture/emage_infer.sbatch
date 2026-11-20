#!/bin/bash
# Speech -> SMPL-X body/hands + FLAME face motion with pretrained EMAGE (character-policy repo), inference only.
#   AVATAR_ROOT=<work root> CONDA_SH=<miniconda>/etc/profile.d/conda.sh \
#     sbatch -w <node> -o $AVATAR_ROOT/logs/%x_%j.log --export=ALL emage_infer.sbatch
# Input:  $AVATAR_ROOT/gesture/audio/*.wav (any rate, mono; resampled to 16 kHz)
# Needs: $AVATAR_ROOT/gesture/tools/emage_run.py (copy of this folder).
# Output: $AVATAR_ROOT/gesture/out/<name>_output.npz (BEAT2 format, 30 fps) -> npz_to_motion.py on the laptop.
# Code and weights are research/education only (not for commercial use or redistribution).
#SBATCH -J gesture-emage
#SBATCH -p batch
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -eo pipefail
R=${AVATAR_ROOT:?set AVATAR_ROOT}; G=$R/gesture
source ${CONDA_SH:?set CONDA_SH to conda.sh}
export HF_HOME=$G/hf_cache PIP_CACHE_DIR=$G/pip_cache PYOPENGL_PLATFORM=egl
mkdir -p $G/out
cd $G
[ -d character-policy ] || git clone -q https://github.com/haiyangliu0/character-policy.git
cd character-policy && git checkout -q c4b7d0d34ad9a64b7d36274990d338e9f1fb4aee
[ -d $G/env ] || conda create -y -q -p $G/env -c conda-forge --override-channels python=3.10
conda activate $G/env
if [ ! -f $G/env/.installed ]; then
  echo "== installing $(date)"
  pip install -q "setuptools<70" pip wheel "numpy<1.24"
  pip install -q torch torchvision torchaudio librosa==0.9.2 soundfile scipy scikit-learn tqdm pyyaml wandb
  pip install -q transformers==4.46.3 huggingface_hub accelerate einops timm
  pip install -q smplx==0.1.28 trimesh pyrender pyopengl imageio moviepy==1.0.3 opencv-python-headless wget
  pip install -q --no-build-isolation chumpy
  touch $G/env/.installed
fi
python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__); assert torch.cuda.is_available()"
ls $G/audio/*.wav
echo "== smoke $(date)"
rm -rf $G/smoke && mkdir -p $G/smoke/audio && cp "$(ls $G/audio/*.wav | head -1)" $G/smoke/audio/
python $R/gesture/tools/emage_run.py . $G/smoke/audio $G/smoke/out --device cuda
ls $G/smoke/out/*_output.npz
echo "== SMOKE OK $(date)"
python $R/gesture/tools/emage_run.py . $G/audio $G/out --device cuda
python - $G/out <<'PY'
import glob, sys, numpy as np
for f in sorted(glob.glob(sys.argv[1] + '/*_output.npz')):
    d = np.load(f); print(f.rsplit('/', 1)[1], {k: d[k].shape for k in ('poses', 'expressions', 'trans')})
PY
echo "== EMAGE DONE $(date)"
