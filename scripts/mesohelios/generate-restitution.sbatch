#!/bin/bash
#SBATCH --job-name=wm-restitution
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.err

set -euo pipefail

REPO=/Home/Users/aelmessaoudi/world-model
WORK=/Work/Users/aelmessaoudi/world-model
IMAGE="$WORK/containers/pytorch-2.6.0-cuda12.4.sif"
APPTAINER=/Softs/helios/apptainer/bin/apptainer
CHECKPOINT=/work/outputs/main-recovery-12000/checkpoint.pt
OUTPUT=/work/outputs/restitution

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=/workspace

run_python() {
  "$APPTAINER" exec --nv \
    --bind "$REPO:/workspace,$WORK:/work" \
    "$IMAGE" \
    python "$@"
}

mkdir -p "$WORK/outputs/restitution"

run_python -m blocket_league.experiment_assets \
  "$CHECKPOINT" "$OUTPUT/truth-vs-prediction" \
  --rollout-frames 64 \
  --playback-fps 8 \
  --scenario-set default \
  --lane-label "Recovery checkpoint"

run_python -m blocket_league.hallucination_assets \
  "$CHECKPOINT" "$OUTPUT/free-rollouts" \
  --input-frames 12 \
  --output-frames 36 \
  --device cuda

run_python -m blocket_league.position_rollout_assets \
  "$CHECKPOINT" "$OUTPUT/decoded-position-rollouts" \
  --fit-samples 2048 \
  --batch-size 32 \
  --block 5 \
  --input-frames 8 \
  --output-frames 20 \
  --device cuda
