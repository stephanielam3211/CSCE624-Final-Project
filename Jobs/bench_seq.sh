#!/bin/bash
#Submit using flux batch benchmark_seq.sh

# flux: --exclusive
# flux: -N 1
# flux: --setattr=gpumode=SPX
# flux: --conf=resource.rediscover=true
# flux: -t 30m
# flux: -q pdebug
# flux: --output=output_benchmark_seq2.out
# flux: --error=output_benchmark_seq2.err

set -euo pipefail

cd /usr/workspace/wsa/lam44/csce624/scripts
module load rocm/6.3.1
source /usr/workspace/wsa/lam44/mytorchenv/bin/activate

echo "=== Stroke sequence length runs ==="

for L in 50 100 150 200; do
  echo ""
  echo ">>> Stroke seq_len ${L}"
  python benchmark.py \
    --mode stroke \
    --data_dir ../quickdraw_npy \
    --num_classes 10 \
    --seq_len ${L} \
    --batch_size 64 \
    --device cuda \
    --steps 50
done

echo "=== RNN Stroke sequence length runs ==="

for L in 50 100 150 200; do
  echo ""
  echo ">>> Stroke seq_len ${L}"
  python benchmark.py \
    --mode stroke_rnn \
    --data_dir ../quickdraw_npy \
    --num_classes 10 \
    --seq_len ${L} \
    --batch_size 64 \
    --device cuda \
    --steps 50
done