#!/bin/bash
# tga_sweep - Slurm array: one array task = one node with 8 GPUs = up to 8 sweep lines.
# Placeholders are filled by sber/cluster_sweep.sh; --array and --job-name come from sbatch.

#SBATCH --partition=HGX
#SBATCH --output=@LOG_DIR@/node_%A_%a.log
#SBATCH --error=@LOG_DIR@/node_%A_%a.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

set -euo pipefail

TASK_DIR="$(pwd)"
export TASK_DIR

srun --gpus=8 \
     --container-image="@IMAGE@" \
     --container-mounts=/external/nfs:/external/nfs \
     --container-workdir="$TASK_DIR" \
     --export=ALL \
     bash "$TASK_DIR/slurm_startup.sh"
