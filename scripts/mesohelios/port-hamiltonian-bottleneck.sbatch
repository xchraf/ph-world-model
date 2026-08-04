#!/bin/bash
#SBATCH --job-name=wm-ph-bottleneck
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

B_SEED="${B_SEED:-91410731}"
B_OUTPUT="${B_OUTPUT:-port-hamiltonian-bottleneck-seed-$B_SEED}"
B_TRAJECTORIES="${B_TRAJECTORIES:-8192}"
B_STEPS="${B_STEPS:-6000}"
B_BATCH_SIZE="${B_BATCH_SIZE:-256}"
B_FEATURE_BATCH_SIZE="${B_FEATURE_BATCH_SIZE:-64}"
B_LOG_EVERY="${B_LOG_EVERY:-100}"

"$APPTAINER" exec --nv \
  --bind "$REPO:/workspace,$WORK:/work" \
  "$IMAGE" \
  python -m blocket_league.port_hamiltonian_bottleneck \
    /work/outputs/main-recovery-12000/checkpoint.pt \
    "/work/outputs/$B_OUTPUT" \
    --trajectories "$B_TRAJECTORIES" \
    --transitions 8 \
    --steps "$B_STEPS" \
    --batch-size "$B_BATCH_SIZE" \
    --feature-batch-size "$B_FEATURE_BATCH_SIZE" \
    --seed "$B_SEED" \
    --log-every "$B_LOG_EVERY" \
    --device cuda
