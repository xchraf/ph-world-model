#!/bin/bash
#SBATCH --job-name=wm-neural-ph
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
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

NPH_SEED="${NPH_SEED:-111610731}"
NPH_OUTPUT="${NPH_OUTPUT:-neural-ph-seed-$NPH_SEED}"
NPH_FIT_POLICY="${NPH_FIT_POLICY:-3072}"
NPH_FIT_CARDINAL="${NPH_FIT_CARDINAL:-3072}"
NPH_TEST_POLICY="${NPH_TEST_POLICY:-1024}"
NPH_TEST_DIAGONAL="${NPH_TEST_DIAGONAL:-512}"
NPH_TEST_REVERSAL="${NPH_TEST_REVERSAL:-512}"
NPH_STEPS="${NPH_STEPS:-4000}"
NPH_BATCH_SIZE="${NPH_BATCH_SIZE:-128}"
NPH_FEATURE_BATCH_SIZE="${NPH_FEATURE_BATCH_SIZE:-64}"
NPH_HIDDEN_SIZE="${NPH_HIDDEN_SIZE:-64}"
NPH_INTEGRATION_METHOD="${NPH_INTEGRATION_METHOD:-midpoint}"
NPH_INTEGRATION_SUBSTEPS="${NPH_INTEGRATION_SUBSTEPS:-1}"
NPH_LOG_EVERY="${NPH_LOG_EVERY:-50}"
NPH_RENDERER_CHECKPOINT="${NPH_RENDERER_CHECKPOINT:-/workspace/results/mesohelios-action-port-pixel/artifacts/action-port-pixel-seed-101510731/checkpoint.pt}"

"$APPTAINER" exec --nv \
  --bind "$PHWM_REPO_ROOT:/workspace,$PHWM_WORK_ROOT:/work" \
  "$IMAGE" \
  python -m blocket_league.neural_ph_experiment \
    /work/outputs/main-recovery-12000/checkpoint.pt \
    "$NPH_RENDERER_CHECKPOINT" \
    "/work/outputs/$NPH_OUTPUT" \
    --fit-policy-trajectories "$NPH_FIT_POLICY" \
    --fit-cardinal-trajectories "$NPH_FIT_CARDINAL" \
    --test-policy-trajectories "$NPH_TEST_POLICY" \
    --test-diagonal-trajectories "$NPH_TEST_DIAGONAL" \
    --test-reversal-trajectories "$NPH_TEST_REVERSAL" \
    --dynamics-steps "$NPH_STEPS" \
    --dynamics-batch-size "$NPH_BATCH_SIZE" \
    --feature-batch-size "$NPH_FEATURE_BATCH_SIZE" \
    --hidden-size "$NPH_HIDDEN_SIZE" \
    --integration-method "$NPH_INTEGRATION_METHOD" \
    --integration-substeps "$NPH_INTEGRATION_SUBSTEPS" \
    --seed "$NPH_SEED" \
    --log-every "$NPH_LOG_EVERY" \
    --device cuda
