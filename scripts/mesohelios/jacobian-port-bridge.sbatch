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

mkdir -p "$PHWM_WORK_ROOT/outputs/$D1_OUTPUT"

"$APPTAINER" exec --nv \
  --bind "$PHWM_REPO_ROOT:/workspace,$PHWM_WORK_ROOT:/work" \
  "$IMAGE" \
  python -m blocket_league.jacobian_port_bridge_experiment \
    /work/outputs/main-recovery-12000/checkpoint.pt \
    "$D1_PH_CHECKPOINT" \
    "/work/outputs/$D1_OUTPUT/summary.json" \
    --lens-fit-samples 512 \
    --test-policy-samples 256 \
    --test-cardinal-samples 256 \
    --batch-size 32 \
    --strengths 1,4,8 \
    --primary-strength 8 \
    --random-direction-pairs 16 \
    --seed "$D1_SEED" \
    --device cuda
