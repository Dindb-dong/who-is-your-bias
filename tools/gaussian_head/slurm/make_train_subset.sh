#!/bin/bash
# Data-quantity experiment: a copy of the VHAP export whose transforms_train.json keeps only the first P% of
# train frames (contiguous, by timestep). Everything else is symlinked; the test split is unchanged.
#   make_train_subset.sh <export_dir> <percent>   ->  <export_dir>_train<percent>
set -eo pipefail
EXP=${1:?export dir}; P=${2:?percent}
D=${EXP}_train$P; mkdir -p $D
for f in $EXP/*; do b=$(basename $f); [ "$b" = transforms_train.json ] || ln -sfn $f $D/$b; done
python3 - $EXP $D $P <<'PY'
import json, sys
src, dst, p = sys.argv[1], sys.argv[2], int(sys.argv[3])
tj = json.load(open(src + '/transforms_train.json'))
fr = sorted(tj['frames'], key=lambda f: f['timestep_index'])
n = len(fr) * p // 100; tj['frames'] = fr[:n]
json.dump(tj, open(dst + '/transforms_train.json', 'w'), indent=1)
print(dst.rsplit('/', 1)[1], 'train frames', n, 'of', len(fr), '| timesteps', fr[0]['timestep_index'], '-', fr[n - 1]['timestep_index'])
PY
