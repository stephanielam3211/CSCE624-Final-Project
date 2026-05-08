#!/bin/bash
#Submit using flux batch train_stroke.sh

# flux: --exclusive
# flux: -N 1
# flux: --setattr=gpumode=SPX
# flux: --conf=resource.rediscover=true
# flux: -t 30m
# flux: -q pdebug
# flux: --output=output_train_stroke2.out
# flux: --error=output_train_stroke2.err

set -euo pipefail

cd /usr/workspace/wsa/lam44/csce624/scripts
module load rocm/6.3.1
source /usr/workspace/wsa/lam44/mytorchenv/bin/activate

python train.py \
  --mode stroke \
  --data_dir ../quickdraw_npy \
  --num_classes 10 \
  --seq_len 150 \
  --batch_size 32 \
  --epochs 5 \
  --device cuda