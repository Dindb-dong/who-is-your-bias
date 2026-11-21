#!/bin/bash
# Trial of Sophon-Realtime (Apache-2.0, full-duplex speech model) on one 24 GB GPU: 4B-Tag-W4 build.
#   AVATAR_ROOT=<root> sbatch -w <node> -o $AVATAR_ROOT/logs/%x_%j.log --export=ALL sophon_trial.sbatch
# Needs: $AVATAR_ROOT/realtime/questions/*.wav (spoken English questions). Optional: $AVATAR_ROOT/realtime/voices/<name>.wav
# (3-10 s, one speaker, permission to use) -> also answered in that cloned voice.
# Everything (venv, uv, caches, weights ~15 GB) stays in $AVATAR_ROOT/realtime/sophon. Writes replies to realtime/out_<job>/.
# Measures first-audio latency of POST /v1/chat/completions (half duplex); the full-duplex websocket is not exercised here.
#SBATCH -J sophon-trial
#SBATCH -p batch
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -eo pipefail
R=${AVATAR_ROOT:?set AVATAR_ROOT}; T=$R/realtime; S=$T/sophon
mkdir -p $S $T/out_$SLURM_JOB_ID
echo "== node $(hostname) $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRV" -lt 580 ]; then echo "STOP: driver $DRV < 580; vLLM 0.30 wheels need CUDA 13 (driver >= 580)."; exit 2; fi

export UV_CACHE_DIR=$S/cache/uv UV_PYTHON_INSTALL_DIR=$S/uv-python XDG_CACHE_HOME=$S/cache MODELSCOPE_CACHE=$S/cache/modelscope \
       HF_HOME=$S/cache/hf PIP_CACHE_DIR=$S/cache/pip TMPDIR=$S/tmp
mkdir -p $TMPDIR
command -v python3 >/dev/null || source $R/env.sh
if [ ! -x $S/uvboot/bin/uv ]; then python3 -m venv $S/uvboot && $S/uvboot/bin/pip install -q uv; fi
export PATH=$S/uvboot/bin:$PATH

cd $S
[ -d repo ] || git clone -q https://github.com/adventists-ai/sophon-realtime repo
cd repo && git checkout -q ace091b1841adb2b0588cd609ac06e3dfc261643 && cd tag
if [ ! -f .installed ]; then
  echo "== install $(date)"
  SIZE=4B-W4 bash install.sh
  touch .installed
fi
PORT=$((20000 + SLURM_JOB_ID % 20000))
trap 'PORT=$PORT bash stop.sh >/dev/null 2>&1 || true' EXIT
for v in $T/voices/*.wav; do if [ -f "$v" ]; then mkdir -p model/voices; cp "$v" model/voices/; fi; done
echo "== start $(date)"
MPS=0 GPU=0 PORT=$PORT bash start.sh
nvidia-smi --query-gpu=memory.used --format=csv
echo "== SERVER READY $(date)"

voices="ryan_en"
for v in model/voices/*.wav; do if [ -f "$v" ]; then voices="$voices $(basename "$v" .wav)"; fi; done
for q in $T/questions/*.wav; do
  for voice in $voices; do
    for run in 1 2; do
      out=$T/out_$SLURM_JOB_ID/$(basename "$q" .wav)__${voice}__$run.wav
      echo "-- $(basename "$q") voice=$voice run=$run"
      .venv/bin/python client_example.py --url http://127.0.0.1:$PORT --wav "$q" --lang en --voice $voice --out "$out"
    done
  done
done
echo "== TRIAL DONE $(date)  replies: $T/out_$SLURM_JOB_ID"
