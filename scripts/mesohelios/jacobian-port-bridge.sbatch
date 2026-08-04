#!/bin/bash
#SBATCH --job-name=wm-jac-port
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.err

set -euo pipefail

PHWM_REPO_ROOT="${PHWM_REPO_ROOT:-/Home/Users/aelmessaoudi/world-model}"
PHWM_WORK_ROOT="${PHWM_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
IMAGE="$PHWM_WORK_ROOT/containers/pytorch-2.6.0-cuda12.4.sif"
APPTAINER=/Softs/helios/apptainer/bin/apptainer

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=/workspace

D1_SEED="${D1_SEED:-121610731}"
D1_OUTPUT="${D1_OUTPUT:-jacobian-port-bridge-seed-$D1_SEED}"
D1_PH_CHECKPOINT="${D1_PH_CHECKPOINT:-/workspace/results/mesohelios-neural-port-hamiltonian/artifacts/neural-ph-seed-111610731/checkpoint.pt}"
D1_LENS_FIT_SAMPLES="${D1_LENS_FIT_SAMPLES:-512}"
D1_TEST_POLICY_SAMPLES="${D1_TEST_POLICY_SAMPLES:-256}"
D1_TEST_CARDINAL_SAMPLES="${D1_TEST_CARDINAL_SAMPLES:-256}"
D1_BATCH_SIZE="${D1_BATCH_SIZE:-32}"
D1_STRENGTHS="${D1_STRENGTHS:-1,4,8}"
D1_PRIMARY_STRENGTH="${D1_PRIMARY_STRENGTH:-8}"
D1_RANDOM_DIRECTION_PAIRS="${D1_RANDOM_DIRECTION_PAIRS:-16}"

mkdir -p "$PHWM_WORK_ROOT/outputs/$D1_OUTPUT"

"$APPTAINER" exec --nv \
  --bind "$PHWM_REPO_ROOT:/workspace,$PHWM_WORK_ROOT:/work" \
  "$IMAGE" \
  python -m blocket_league.jacobian_port_bridge_experiment \
    /work/outputs/main-recovery-12000/checkpoint.pt \
    "$D1_PH_CHECKPOINT" \
    "/work/outputs/$D1_OUTPUT/summary.json" \
    --lens-fit-samples "$D1_LENS_FIT_SAMPLES" \
    --test-policy-samples "$D1_TEST_POLICY_SAMPLES" \
    --test-cardinal-samples "$D1_TEST_CARDINAL_SAMPLES" \
    --batch-size "$D1_BATCH_SIZE" \
    --strengths "$D1_STRENGTHS" \
    --primary-strength "$D1_PRIMARY_STRENGTH" \
    --random-direction-pairs "$D1_RANDOM_DIRECTION_PAIRS" \
    --seed "$D1_SEED" \
    --device cuda
