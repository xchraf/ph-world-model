#!/bin/bash
#SBATCH --job-name=wm-pixel-only-ph
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.err

set -euo pipefail

PHWM_REPO_ROOT="${PHWM_REPO_ROOT:-/Home/Users/aelmessaoudi/world-model}"
PHWM_WORK_ROOT="${PHWM_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
IMAGE="$PHWM_WORK_ROOT/containers/pytorch-2.6.0-cuda12.4.sif"
APPTAINER=/Softs/helios/apptainer/bin/apptainer

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=/workspace

POPH_SEED="${POPH_SEED:-131610731}"
POPH_OUTPUT="${POPH_OUTPUT:-pixel-only-ph-seed-$POPH_SEED}"
POPH_FIT_POLICY="${POPH_FIT_POLICY:-3072}"
POPH_FIT_CARDINAL="${POPH_FIT_CARDINAL:-3072}"
POPH_TEST_POLICY="${POPH_TEST_POLICY:-512}"
POPH_TEST_DIAGONAL="${POPH_TEST_DIAGONAL:-256}"
POPH_TEST_REVERSAL="${POPH_TEST_REVERSAL:-256}"
POPH_AUDIT="${POPH_AUDIT:-1024}"
POPH_AUTOENCODER_STEPS="${POPH_AUTOENCODER_STEPS:-3000}"
POPH_AUTOENCODER_BATCH="${POPH_AUTOENCODER_BATCH:-256}"
POPH_DYNAMICS_STEPS="${POPH_DYNAMICS_STEPS:-5000}"
POPH_DYNAMICS_BATCH="${POPH_DYNAMICS_BATCH:-64}"
POPH_FEATURE_BATCH="${POPH_FEATURE_BATCH:-64}"
POPH_PLANNER_SAMPLES="${POPH_PLANNER_SAMPLES:-64}"
POPH_PLANNER_STEPS="${POPH_PLANNER_STEPS:-40}"
POPH_LOG_EVERY="${POPH_LOG_EVERY:-50}"

mkdir -p "$PHWM_WORK_ROOT/outputs/$POPH_OUTPUT"

"$APPTAINER" exec --nv \
  --bind "$PHWM_REPO_ROOT:/workspace,$PHWM_WORK_ROOT:/work" \
  "$IMAGE" \
  python -m blocket_league.pixel_only_ph_experiment \
    /work/outputs/main-recovery-12000/checkpoint.pt \
    "/work/outputs/$POPH_OUTPUT" \
    --fit-policy-trajectories "$POPH_FIT_POLICY" \
    --fit-cardinal-trajectories "$POPH_FIT_CARDINAL" \
    --test-policy-trajectories "$POPH_TEST_POLICY" \
    --test-diagonal-trajectories "$POPH_TEST_DIAGONAL" \
    --test-reversal-trajectories "$POPH_TEST_REVERSAL" \
    --audit-trajectories "$POPH_AUDIT" \
    --autoencoder-steps "$POPH_AUTOENCODER_STEPS" \
    --autoencoder-batch-size "$POPH_AUTOENCODER_BATCH" \
    --dynamics-steps "$POPH_DYNAMICS_STEPS" \
    --dynamics-batch-size "$POPH_DYNAMICS_BATCH" \
    --feature-batch-size "$POPH_FEATURE_BATCH" \
    --planner-samples "$POPH_PLANNER_SAMPLES" \
    --planner-steps "$POPH_PLANNER_STEPS" \
    --seed "$POPH_SEED" \
    --log-every "$POPH_LOG_EVERY" \
    --device cuda
