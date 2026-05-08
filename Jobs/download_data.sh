#!/bin/bash
#Submit using flux batch download_data.sh

# flux: --exclusive
# flux: -N 1
# flux: --setattr=gpumode=SPX
# flux: --conf=resource.rediscover=true
# flux: -t 20m
# flux: -q pdebug
# flux: --output=output_download_data3.out
# flux: --error=output_download_data3.err

set -euo pipefail

cd /usr/workspace/wsa/lam44/csce624/scripts
source /usr/workspace/wsa/lam44/mytorchenv/bin/activate

python download_quickdraw.py
python convert_quickdraw_to_npy.py

ls -lh ../quickdraw_npy