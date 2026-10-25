#!/bin/bash
# mlmc_pretrain - торчран внутри контейнера.
# Параметры можно менять переменными окружения при запуске:
#   DATASET=shakespeare-char ITERATIONS=500 bash run.sh mlmc_pretrain
set -euo pipefail

cd "${TASK_DIR:-$(pwd)}"

export PYTHONPATH="$TASK_DIR/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export TIKTOKEN_CACHE_DIR="$TASK_DIR/tiktoken_cache"

export MASTER_ADDR="${SLURM_LAUNCH_NODE_IPADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-8}"

DATASET="${DATASET:-fineweb}"
OPTIMIZER="${OPTIMIZER:-adamw}"
LR="${LR:-3e-3}"
ITERATIONS="${ITERATIONS:-20000}"
WARMUP="${WARMUP:-2000}"
BATCH="${BATCH:-256}"
SEQ="${SEQ:-512}"
N_EMBD="${N_EMBD:-768}"
N_HEAD="${N_HEAD:-12}"
N_LAYER="${N_LAYER:-12}"

torchrun \
    --nnodes="${SLURM_NNODES:-1}" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --node-rank="${SLURM_NODEID:-0}" \
    --rdzv-id="${SLURM_JOB_ID:-$RANDOM}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    src/main.py \
    --run_prefix mlmc_pretrain \
    --distributed_backend nccl \
    --model llama \
    --dataset "$DATASET" \
    --opt "$OPTIMIZER" \
    --lr "$LR" \
    --iterations "$ITERATIONS" \
    --warmup_steps "$WARMUP" \
    --batch_size "$BATCH" --sequence_length "$SEQ" --acc_steps 1 \
    --n_embd "$N_EMBD" --n_head "$N_HEAD" --n_layer "$N_LAYER" \
    --eval_interval 500 --latest_ckpt_interval 1000 --log_interval 20 \
    --datasets_dir "$TASK_DIR/data" \
    --results_base_folder "$TASK_DIR/exps"

if [ "${SLURM_NODEID:-0}" = "0" ]; then
    python3 "$TASK_DIR/plot_metrics.py" "${LOG_FILE:-$TASK_DIR/logs/log_${SLURM_JOB_ID}.log}" "$TASK_DIR" \
        || echo "plot_metrics skipped/failed"
fi
