#!/bin/bash
# mlmc_pretrain - претрейн llama на FineWeb через llm-baselines.
# Плейсхолдеры @LOG_DIR@ и @IMAGE@ подставляет sber/cluster_run.sh.

#SBATCH --job-name=mlmc_pretrain
#SBATCH --partition=HGX
#SBATCH --output=@LOG_DIR@/log_%j.log
#SBATCH --error=@LOG_DIR@/log_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

set -euo pipefail

TASK_DIR="$(pwd)"
export TASK_DIR
export LOG_FILE="@LOG_DIR@/log_${SLURM_JOB_ID}.log"

srun --gpus=8 \
     --container-image="@IMAGE@" \
     --container-mounts=/external/nfs:/external/nfs \
     --container-workdir="$TASK_DIR" \
     --export=ALL \
     bash "$TASK_DIR/slurm_startup.sh"
