#!/bin/bash
#SBATCH --job-name=wm-ph-audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
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

mkdir -p "$WORK/outputs/port-hamiltonian-audit"

"$APPTAINER" exec --nv \
  --bind "$REPO:/workspace,$WORK:/work" \
  "$IMAGE" \
  python -m blocket_league.port_hamiltonian_audit \
    /work/outputs/main-recovery-12000/checkpoint.pt \
    /work/outputs/port-hamiltonian-audit/audit.json \
    --trajectories 1024 \
    --transitions-per-trajectory 8 \
    --horizons 1,2,4,8 \
    --fit-fraction 0.75 \
    --goal-centered-fraction 0.20 \
    --batch-size 32 \
    --mlp-steps 300 \
    --device cuda
