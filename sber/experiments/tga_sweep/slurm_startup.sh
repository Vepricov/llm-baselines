#!/bin/bash
# tga_sweep - inside the container on one node: 8 sweep lines in parallel, one per GPU.
set -uo pipefail

cd "${TASK_DIR:?}"
export PYTHONPATH="$TASK_DIR/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export TIKTOKEN_CACHE_DIR="$TASK_DIR/tiktoken_cache"
export OMP_NUM_THREADS=8
export LAB_RUN_OFFLINE=1

# the interpreter that has torch, whatever it is called in the image
PY=""
for c in python python3; do
    if command -v "$c" >/dev/null && "$c" -c "import torch" 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "error: no python with torch in the container"; exit 1; }
export PY

ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
GPUS="${GPUS_PER_NODE:-8}"

# shakespeare is tokenized per array task, so parallel nodes never write the same file
if [ "$DATASET" = "shakespeare-char" ]; then
    export DATA_DIR="$TASK_DIR/data_task$ARRAY_ID"
    mkdir -p "$DATA_DIR/shakespeare"
    cp sber/data/shakespeare/raw.txt "$DATA_DIR/shakespeare/raw.txt"
    "$PY" -c "import sys; sys.path.insert(0, 'src'); from data.shakespeare import get_shakespeare_data; get_shakespeare_data('$DATA_DIR')"
else
    export DATA_DIR="$TASK_DIR/data"
fi

echo "node $(hostname), array task $ARRAY_ID, $(date '+%d-%m-%Y %H:%M:%S')"
for g in $(seq 0 $((GPUS - 1))); do
    idx=$((ARRAY_ID * GPUS + g))
    CUDA_VISIBLE_DEVICES=$g bash sber/sweep_worker.sh "$idx" &
done
wait
echo "array task $ARRAY_ID finished $(date '+%d-%m-%Y %H:%M:%S')"
