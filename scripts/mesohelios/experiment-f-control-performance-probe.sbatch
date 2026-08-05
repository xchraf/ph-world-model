#!/bin/bash
#SBATCH --job-name=wm-f-cem-perf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12-00:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%A_%a.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%A_%a.err

set -euo pipefail
usage() { printf '%s\n' "Usage: sbatch experiment-f-control-performance-probe.sbatch" "       bash experiment-f-control-performance-probe.sbatch --dry-run [--system pendulum|blocket]" "Synthetic exact four-episode closed-loop shard timing only; never scientific evidence."; }
dry_run="${F_DRY_RUN:-0}"; requested_system=""
while (($#)); do case "$1" in --dry-run) dry_run=1;; --system) shift; requested_system="${1:?missing system}";; --help|-h) usage; exit 0;; *) usage >&2; exit 2;; esac; shift; done
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -n "$requested_system" ]]; then system="$requested_system"; else systems=(pendulum blocket); [[ "$task_id" =~ ^[01]$ ]] || exit 2; system="${systems[$task_id]}"; fi
[[ "$system" == pendulum || "$system" == blocket ]] || exit 2
repo_root="${F_REPO_ROOT:-/Work/Users/aelmessaoudi/phwm-direct-dev}"; work_root="${F_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
dependency_root="${F_DEPENDENCY_ROOT:-$work_root/benchmarks/experiment-f-control/dependencies}"
probe_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-dry-run}}"; output_root="$work_root/benchmarks/experiment-f-control/$probe_id-$system"
image="$work_root/containers/pytorch-2.6.0-cuda12.4.sif"; apptainer=/Softs/helios/apptainer/bin/apptainer
for path in "$repo_root" "$work_root" "$dependency_root" "$output_root"; do [[ "$path" == /* && "$path" != *[:,]* ]] || exit 2; done
command=(
  "$apptainer" exec --nv --containall --cleanenv --no-mount home,cwd,hostfs
  --bind "$repo_root:/workspace:ro,$output_root:/output:rw,$dependency_root:/deps:rw"
  "$image" env PYTHONPATH=/workspace XDG_CACHE_HOME=/deps/cache TORCH_HOME=/deps/torch OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
  python -m blocket_league.direct_control_performance_probe /output/control-performance.json
  --system "$system" --device cuda --image-size "${F_IMAGE_SIZE:-64}" --patch-size "${F_PATCH_SIZE:-4}"
  --backbone-preset "${F_BACKBONE_PRESET:-tiny}" --implicit-iterations "${F_IMPLICIT_ITERATIONS:-32}"
  --episodes-per-shard 4
  --float32-matmul-precision "${F_FLOAT32_MATMUL_PRECISION:-highest}"
)
if [[ "$dry_run" == 1 ]]; then printf 'kind=synthetic-exact-full-control-shard-timing system=%s command=' "$system"; printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
mkdir -p "$output_root" "$dependency_root/cache" "$dependency_root/torch"
exec "${command[@]}"
