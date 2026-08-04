#!/bin/bash
#SBATCH --job-name=wm-passive-jac-ph
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.err

set -euo pipefail

PHWM_REPO_ROOT="${PHWM_REPO_ROOT:-/Home/Users/aelmessaoudi/world-model}"
PHWM_WORK_ROOT="${PHWM_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
IMAGE="$PHWM_WORK_ROOT/containers/pytorch-2.6.0-cuda12.4.sif"
APPTAINER=/Softs/helios/apptainer/bin/apptainer

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=/workspace

E_SEED="${E_SEED:-141810733}"
E_OUTPUT="${E_OUTPUT:-passive-jacobian-ph-control-seed-$E_SEED}"
E_BLOCKET_CHECKPOINT="${E_BLOCKET_CHECKPOINT:-/work/outputs/main-recovery-12000/checkpoint.pt}"
E_FIT_TRAJECTORIES="${E_FIT_TRAJECTORIES:-2048}"
E_TEST_TRAJECTORIES="${E_TEST_TRAJECTORIES:-256}"
E_VISUAL_STEPS="${E_VISUAL_STEPS:-1200}"
E_PASSIVE_DYNAMICS_STEPS="${E_PASSIVE_DYNAMICS_STEPS:-4000}"
E_BASELINE_STEPS="${E_BASELINE_STEPS:-3000}"
E_PORT_STEPS="${E_PORT_STEPS:-1500}"
E_PENDULUM_BACKBONE_STEPS="${E_PENDULUM_BACKBONE_STEPS:-12000}"
E_CONTROL_EPISODES="${E_CONTROL_EPISODES:-32}"
E_PLANNER_ITERATIONS="${E_PLANNER_ITERATIONS:-30}"
E_LOG_EVERY="${E_LOG_EVERY:-100}"

mkdir -p "$PHWM_WORK_ROOT/outputs/$E_OUTPUT"

"$APPTAINER" exec --nv \
  --bind "$PHWM_REPO_ROOT:/workspace,$PHWM_WORK_ROOT:/work" \
  "$IMAGE" \
  python -m blocket_league.passive_jacobian_ph_experiment \
    "$E_BLOCKET_CHECKPOINT" \
    "/work/outputs/$E_OUTPUT" \
    --seed "$E_SEED" \
    --fit-trajectories "$E_FIT_TRAJECTORIES" \
    --test-trajectories "$E_TEST_TRAJECTORIES" \
    --visual-steps "$E_VISUAL_STEPS" \
    --passive-dynamics-steps "$E_PASSIVE_DYNAMICS_STEPS" \
    --baseline-steps "$E_BASELINE_STEPS" \
    --port-steps "$E_PORT_STEPS" \
    --pendulum-backbone-steps "$E_PENDULUM_BACKBONE_STEPS" \
    --control-episodes "$E_CONTROL_EPISODES" \
    --planner-iterations "$E_PLANNER_ITERATIONS" \
    --log-every "$E_LOG_EVERY" \
    --device cuda
