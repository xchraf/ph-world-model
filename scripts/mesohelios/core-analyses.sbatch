#!/bin/bash
#SBATCH --job-name=wm-core-analyses
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.err

set -euo pipefail

REPO=/Home/Users/aelmessaoudi/world-model
WORK=/Work/Users/aelmessaoudi/world-model
IMAGE="$WORK/containers/pytorch-2.6.0-cuda12.4.sif"
APPTAINER=/Softs/helios/apptainer/bin/apptainer

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=/workspace

"$APPTAINER" exec --nv \
  --bind "$REPO:/workspace,$WORK:/work" \
  "$IMAGE" \
  python /workspace/scripts/mesohelios/run_core_analyses.py \
    /work/outputs/main-recovery-12000/checkpoint.pt \
    /work/outputs/core-analyses
