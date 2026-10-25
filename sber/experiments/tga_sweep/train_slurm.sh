#!/bin/bash
# tga_sweep - Slurm array: one array task = @GPUS@ GPUs on one node = up to @GPUS@ sweep lines.
# Placeholders are filled by sber/cluster_sweep.sh; --array and --job-name come from sbatch.

#SBATCH --partition=@PARTITION@
#SBATCH --output=@LOG_DIR@/node_%A_%a.log
#SBATCH --error=@LOG_DIR@/node_%A_%a.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=@GPUS@
#SBATCH --time=@TIME_LIMIT@

set -euo pipefail

TASK_DIR="$(pwd)"
export TASK_DIR
export GPUS_PER_NODE=@GPUS@

srun --gpus=@GPUS@ \
     --container-image="@IMAGE@" \
     --container-mounts=/external/nfs:/external/nfs \
     --container-workdir="$TASK_DIR" \
     --export=ALL \
     bash "$TASK_DIR/slurm_startup.sh"
