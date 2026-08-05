#!/bin/bash
#SBATCH --job-name=wm-f-perf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.err

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: sbatch --export=F_SYSTEM=blocket,F_VARIANT=full experiment-f-performance-probe.sbatch" \
    "       bash experiment-f-performance-probe.sbatch --dry-run --system blocket --variant full" \
    "Synthetic shape-exact timing only; the result is never scientific evidence."
}

dry_run="${F_DRY_RUN:-0}"
system="${F_SYSTEM:-blocket}"
variant="${F_VARIANT:-full}"
while (($#)); do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --system) shift; system="${1:?--system needs a value}" ;;
    --variant) shift; variant="${1:?--variant needs a value}" ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done
[[ "$system" == pendulum || "$system" == blocket ]] || { printf 'Invalid system\n' >&2; exit 2; }
case "$variant" in
  full|no_jacobian|single_horizon|shuffled_lens|skew_only|constant_port) ;;
  *) printf 'Invalid variant\n' >&2; exit 2 ;;
esac

repo_root="${F_REPO_ROOT:-/Work/Users/aelmessaoudi/phwm-direct-dev}"
work_root="${F_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
dependency_root="${F_DEPENDENCY_ROOT:-$work_root/benchmarks/experiment-f-exact/dependencies}"
probe_id="${SLURM_JOB_ID:-dry-run}"
output_root="$work_root/benchmarks/experiment-f-exact/$probe_id-$system-$variant"
image="$work_root/containers/pytorch-2.6.0-cuda12.4.sif"
apptainer=/Softs/helios/apptainer/bin/apptainer
for path in "$repo_root" "$work_root" "$dependency_root" "$output_root"; do
  [[ "$path" == /* && "$path" != *[:,]* ]] || { printf 'Unsafe path: %s\n' "$path" >&2; exit 2; }
done

command=(
  "$apptainer" exec --nv --containall --cleanenv --no-mount home,cwd,hostfs
  --bind "$repo_root:/workspace:ro,$output_root:/output:rw,$dependency_root:/deps:rw"
  "$image" env PYTHONPATH=/workspace XDG_CACHE_HOME=/deps/cache TORCH_HOME=/deps/torch
  OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
  python -m blocket_league.direct_performance_probe /output
  --system "$system" --variant "$variant" --device cuda
  --micro-batch-size "${F_MICRO_BATCH_SIZE:-16}"
  --gradient-accumulation "${F_GRADIENT_ACCUMULATION:-1}"
  --lens-batch-size "${F_LENS_BATCH_SIZE:-4}"
  --implicit-iterations "${F_IMPLICIT_ITERATIONS:-32}"
  --image-size "${F_IMAGE_SIZE:-64}" --patch-size "${F_PATCH_SIZE:-4}"
  --backbone-preset "${F_BACKBONE_PRESET:-tiny}"
  --float32-matmul-precision "${F_FLOAT32_MATMUL_PRECISION:-highest}"
  --energy-gradient-implementation "${F_ENERGY_GRADIENT_IMPLEMENTATION:-exact}"
)
if [[ "$dry_run" == 1 ]]; then
  printf 'kind=synthetic-shape-exact-performance-probe output=%s command=' "$output_root"
  printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
mkdir -p "$output_root" "$dependency_root/cache" "$dependency_root/torch"
exec "${command[@]}"
