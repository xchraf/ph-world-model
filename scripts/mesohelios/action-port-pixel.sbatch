#!/bin/bash
#SBATCH --job-name=wm-action-port-pixel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.err

set -euo pipefail

PHWM_REPO_ROOT="${PHWM_REPO_ROOT:-/Home/Users/aelmessaoudi/world-model}"
PHWM_WORK_ROOT="${PHWM_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
IMAGE="$PHWM_WORK_ROOT/containers/pytorch-2.6.0-cuda12.4.sif"
APPTAINER=/Softs/helios/apptainer/bin/apptainer

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=/workspace

C_SEED="${C_SEED:-101510731}"
C_OUTPUT="${C_OUTPUT:-action-port-pixel-seed-$C_SEED}"
C_TRAJECTORIES="${C_TRAJECTORIES:-8192}"
C_DYNAMICS_STEPS="${C_DYNAMICS_STEPS:-6000}"
C_RENDERER_STEPS="${C_RENDERER_STEPS:-5000}"
C_DYNAMICS_BATCH_SIZE="${C_DYNAMICS_BATCH_SIZE:-256}"
C_RENDERER_BATCH_SIZE="${C_RENDERER_BATCH_SIZE:-64}"
C_FEATURE_BATCH_SIZE="${C_FEATURE_BATCH_SIZE:-64}"
C_LOG_EVERY="${C_LOG_EVERY:-100}"

"$APPTAINER" exec --nv \
  --bind "$PHWM_REPO_ROOT:/workspace,$PHWM_WORK_ROOT:/work" \
  "$IMAGE" \
  python -m blocket_league.action_port_pixel_experiment \
    /work/outputs/main-recovery-12000/checkpoint.pt \
    "/work/outputs/$C_OUTPUT" \
    --trajectories "$C_TRAJECTORIES" \
    --dynamics-steps "$C_DYNAMICS_STEPS" \
    --renderer-steps "$C_RENDERER_STEPS" \
    --dynamics-batch-size "$C_DYNAMICS_BATCH_SIZE" \
    --renderer-batch-size "$C_RENDERER_BATCH_SIZE" \
    --feature-batch-size "$C_FEATURE_BATCH_SIZE" \
    --seed "$C_SEED" \
    --log-every "$C_LOG_EVERY" \
    --device cuda
