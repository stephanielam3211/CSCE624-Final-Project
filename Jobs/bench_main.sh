#!/bin/bash
#Submit using flux batch benchmark_main.sh

# flux: --exclusive
# flux: -N 1
# flux: --setattr=gpumode=SPX
# flux: --conf=resource.rediscover=true
# flux: -t 30m
# flux: -q pdebug
# flux: --output=output_benchmark_main2.out
# flux: --error=output_benchmark_main2.err

set -euo pipefail

cd /usr/workspace/wsa/lam44/csce624/scripts
module load rocm/6.3.1
source /usr/workspace/wsa/lam44/mytorchenv/bin/activate

echo "=== Raster runs ==="
for BS in 32 64 128; do
  echo ""
  echo ">>> Raster batch size ${BS}"
  python benchmark.py \
    --mode raster \
    --data_dir ../quickdraw_npy \
    --num_classes 10 \
    --batch_size ${BS} \
    --device cuda \
    --steps 50
done

echo ""
echo "=== Stroke runs ==="
for BS in 32 64 128; do
  echo ""
  echo ">>> Stroke batch size ${BS}"
  python benchmark.py \
    --mode stroke \
    --data_dir ../quickdraw_npy \
    --num_classes 10 \
    --seq_len 150 \
    --batch_size ${BS} \
    --device cuda \
    --steps 50
done

echo ""
echo "=== Stroke runs RNN ==="
for BS in 32 64 128; do
  echo ""
  echo ">>> Stroke batch size ${BS}"
  python benchmark.py \
    --mode stroke_rnn \
    --data_dir ../quickdraw_npy \
    --num_classes 10 \
    --seq_len 150 \
    --batch_size ${BS} \
    --device cuda \
    --steps 50
done