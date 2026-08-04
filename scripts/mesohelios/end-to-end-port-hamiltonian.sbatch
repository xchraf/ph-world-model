#!/bin/bash
#SBATCH --job-name=wm-e2e-ph
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.err

set -euo pipefail

PHWM_REPO_ROOT="${PHWM_REPO_ROOT:-/Home/Users/aelmessaoudi/world-model}"
PHWM_WORK_ROOT="${PHWM_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
IMAGE="$PHWM_WORK_ROOT/containers/pytorch-2.6.0-cuda12.4.sif"
APPTAINER=/Softs/helios/apptainer/bin/apptainer

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=/workspace

E2EPH_SEED="${E2EPH_SEED:-131610731}"
E2EPH_OUTPUT="${E2EPH_OUTPUT:-end-to-end-ph-seed-$E2EPH_SEED}"
E2EPH_FIT_POLICY="${E2EPH_FIT_POLICY:-3072}"
E2EPH_FIT_CARDINAL="${E2EPH_FIT_CARDINAL:-3072}"
E2EPH_TEST_POLICY="${E2EPH_TEST_POLICY:-512}"
E2EPH_TEST_DIAGONAL="${E2EPH_TEST_DIAGONAL:-256}"
E2EPH_TEST_REVERSAL="${E2EPH_TEST_REVERSAL:-256}"
E2EPH_AUDIT="${E2EPH_AUDIT:-1024}"
E2EPH_VISUAL_STEPS="${E2EPH_VISUAL_STEPS:-800}"
E2EPH_VISUAL_BATCH="${E2EPH_VISUAL_BATCH:-32}"
E2EPH_DYNAMICS_STEPS="${E2EPH_DYNAMICS_STEPS:-5000}"
E2EPH_DYNAMICS_BATCH="${E2EPH_DYNAMICS_BATCH:-8}"
E2EPH_PLANNER_SAMPLES="${E2EPH_PLANNER_SAMPLES:-64}"
E2EPH_PLANNER_STEPS="${E2EPH_PLANNER_STEPS:-40}"
E2EPH_LOG_EVERY="${E2EPH_LOG_EVERY:-50}"

mkdir -p "$PHWM_WORK_ROOT/outputs/$E2EPH_OUTPUT"

"$APPTAINER" exec --nv \
  --bind "$PHWM_REPO_ROOT:/workspace,$PHWM_WORK_ROOT:/work" \
  "$IMAGE" \
  python -m blocket_league.end_to_end_ph_experiment \
    /work/outputs/main-recovery-12000/checkpoint.pt \
    "/work/outputs/$E2EPH_OUTPUT" \
    --fit-policy-trajectories "$E2EPH_FIT_POLICY" \
    --fit-cardinal-trajectories "$E2EPH_FIT_CARDINAL" \
    --test-policy-trajectories "$E2EPH_TEST_POLICY" \
    --test-diagonal-trajectories "$E2EPH_TEST_DIAGONAL" \
    --test-reversal-trajectories "$E2EPH_TEST_REVERSAL" \
    --audit-trajectories "$E2EPH_AUDIT" \
    --visual-warmup-steps "$E2EPH_VISUAL_STEPS" \
    --visual-warmup-batch-size "$E2EPH_VISUAL_BATCH" \
    --dynamics-steps "$E2EPH_DYNAMICS_STEPS" \
    --dynamics-batch-size "$E2EPH_DYNAMICS_BATCH" \
    --planner-samples "$E2EPH_PLANNER_SAMPLES" \
    --planner-steps "$E2EPH_PLANNER_STEPS" \
    --seed "$E2EPH_SEED" \
    --log-every "$E2EPH_LOG_EVERY" \
    --device cuda
