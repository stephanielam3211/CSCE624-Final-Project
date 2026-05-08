#!/bin/bash
#Submit using flux batch benchmark_amp.sh

# flux: --exclusive
# flux: -N 1
# flux: --setattr=gpumode=SPX
# flux: --conf=resource.rediscover=true
# flux: -t 60m
# flux: -q pdebug
# flux: --output=output_benchmark_amp2.out
# flux: --error=output_benchmark_amp2.err

set -euo pipefail

cd /usr/workspace/wsa/lam44/csce624/scripts
module load rocm/6.3.1
source /usr/workspace/wsa/lam44/mytorchenv/bin/activate

echo "=== Raster no AMP ==="
python benchmark.py \
  --mode raster \
  --data_dir ../quickdraw_npy \
  --num_classes 10 \
  --batch_size 64 \
  --device cuda \
  --steps 50

echo ""
echo "=== Raster AMP ==="
python benchmark.py \
  --mode raster \
  --data_dir ../quickdraw_npy \
  --num_classes 10 \
  --batch_size 64 \
  --device cuda \
  --steps 50 \
  --amp

echo ""
echo "=== Stroke no AMP ==="
python benchmark.py \
  --mode stroke \
  --data_dir ../quickdraw_npy \
  --num_classes 10 \
  --seq_len 150 \
  --batch_size 64 \
  --device cuda \
  --steps 50

echo ""
echo "=== Stroke AMP ==="
python benchmark.py \
  --mode stroke \
  --data_dir ../quickdraw_npy \
  --num_classes 10 \
  --seq_len 150 \
  --batch_size 64 \
  --device cuda \
  --steps 50 \
  --amp

echo ""
echo "=== Stroke RNN no AMP ==="
python benchmark.py \
  --mode stroke_rnn \
  --data_dir ../quickdraw_npy \
  --num_classes 10 \
  --seq_len 150 \
  --batch_size 64 \
  --device cuda \
  --steps 50

echo ""
echo "=== Stroke RNN AMP ==="
python benchmark.py \
  --mode stroke_rnn \
  --data_dir ../quickdraw_npy \
  --num_classes 10 \
  --seq_len 150 \
  --batch_size 64 \
  --device cuda \
  --steps 50 \
  --amp