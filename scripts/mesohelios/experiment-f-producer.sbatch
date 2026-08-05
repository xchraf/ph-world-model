#!/bin/bash
#SBATCH --job-name=wm-f-producer
#SBATCH --partition=smp
#SBATCH --array=0-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%A_%a.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%A_%a.err

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: sbatch experiment-f-producer.sbatch" \
    "       bash experiment-f-producer.sbatch --dry-run [--system pendulum|blocket]" \
    "Environment: F_RUN_ROOT, F_REPO_ROOT, F_WORK_ROOT." \
    "This CPU-only producer writes sanitized pixels and never writes its seed."
}

dry_run="${F_DRY_RUN:-0}"
requested_system=""
while (($#)); do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --system) shift; requested_system="${1:?--system needs a value}" ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -n "$requested_system" ]]; then
  system="$requested_system"
  [[ "$system" == pendulum ]] && task_id=0 || task_id=1
else
  systems=(pendulum blocket)
  [[ "$task_id" =~ ^[01]$ ]] || { printf 'Array task must be 0 or 1\n' >&2; exit 2; }
  system="${systems[$task_id]}"
fi
[[ "$system" == pendulum || "$system" == blocket ]] || { printf 'Invalid system\n' >&2; exit 2; }

repo_root="${F_REPO_ROOT:-/Home/Users/aelmessaoudi/world-model}"
work_root="${F_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
run_root="${F_RUN_ROOT:-$work_root/outputs/direct-experiment-f-seed-151910737}"
image="$work_root/containers/pytorch-2.6.0-cuda12.4.sif"
apptainer=/Softs/helios/apptainer/bin/apptainer
sanitized_root="$run_root/sanitized"
dependency_root="$run_root/producer-cache/$system"
producer_private="$run_root/producer-private"
producer_seed_file="$producer_private/producer-seed.hex"
source_manifest="$run_root/source-manifest.json"

for path in "$repo_root" "$work_root" "$run_root" "$sanitized_root" "$dependency_root" "$producer_private" "$source_manifest"; do
  [[ "$path" == /* && "$path" != *[:,]* ]] || { printf 'Unsafe bind path: %s\n' "$path" >&2; exit 2; }
done

command=(
  "$apptainer" exec --containall --cleanenv --no-mount home,cwd,hostfs
  --bind "$repo_root:/workspace:ro,$sanitized_root:/sanitized:rw,$dependency_root:/deps:rw,$producer_private:/producer-private:ro,$source_manifest:/source-manifest.json:ro"
  "$image" env
  PYTHONPATH=/workspace
  XDG_CACHE_HOME=/deps/cache
  TORCH_HOME=/deps/torch
  OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
  python -m blocket_league.direct_experiment_f_producer /sanitized
  --system "$system"
  --producer-seed-file /producer-private/producer-seed.hex
  --source-manifest /source-manifest.json
  --fit-trajectories 4096
  --validation-trajectories 512
  --test-trajectories 512
  --transitions 8
  --cache-frames 24
  --image-size 64
  --patch-size 4
  --backbone-preset tiny
)

if [[ "$dry_run" == 1 ]]; then
  printf 'stage=producer system=%s command=' "$system"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
mkdir -p "$sanitized_root" "$dependency_root/cache" "$dependency_root/torch"
[[ -f "$producer_seed_file" && ! -L "$producer_seed_file" ]] || {
  printf 'Missing private producer seed file. Launch through submit-experiment-f.sh.\n' >&2
  exit 1
}
[[ -f "$source_manifest" && ! -L "$source_manifest" ]] || {
  printf 'Missing sealed source manifest. Launch through submit-experiment-f.sh.\n' >&2
  exit 1
}
exec "${command[@]}"
