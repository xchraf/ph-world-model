#!/bin/bash
#SBATCH --job-name=wm-f-pf-prepare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --array=0-1%1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12-00:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%A_%a.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%A_%a.err

# Collect Gates 1--5 and the sole shared calibration/realizability banks once
# per system.  Existing authenticated artifacts are validated and resumed.
set -euo pipefail

usage() { printf '%s\n' "Usage: sbatch --dependency=afterok:<training-finalize> experiment-f-postfreeze-prepare.sbatch" "       bash experiment-f-postfreeze-prepare.sbatch --dry-run [--system pendulum|blocket]"; }
dry_run="${F_DRY_RUN:-0}"; requested_system=""
while (($#)); do case "$1" in --dry-run) dry_run=1;; --system) shift; requested_system="${1:?missing system}";; --help|-h) usage; exit 0;; *) usage >&2; exit 2;; esac; shift; done
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -n "$requested_system" ]]; then system="$requested_system"; else systems=(pendulum blocket); [[ "$task_id" =~ ^[01]$ ]] || exit 2; system="${systems[$task_id]}"; fi
[[ "$system" == pendulum || "$system" == blocket ]] || exit 2

repo_root="${F_REPO_ROOT:-/Home/Users/aelmessaoudi/world-model}"
work_root="${F_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
run_root="${F_RUN_ROOT:-$work_root/outputs/direct-experiment-f-seed-151910737}"
sanitized_root="$run_root/sanitized"; training_root="$run_root/training"
result_root="$run_root/postfreeze"; dependency_root="$run_root/dependencies"
image="$work_root/containers/pytorch-2.6.0-cuda12.4.sif"; apptainer=/Softs/helios/apptainer/bin/apptainer
for path in "$repo_root" "$work_root" "$run_root" "$sanitized_root" "$training_root" "$result_root" "$dependency_root"; do [[ "$path" == /* && "$path" != *[:,]* ]] || { printf 'Unsafe bind path: %s\n' "$path" >&2; exit 2; }; done

command=(
  "$apptainer" exec --nv --containall --cleanenv --no-mount home,cwd,hostfs
  --bind "$repo_root:/workspace:ro,$sanitized_root:/sanitized:ro,$training_root:/training:ro,$result_root:/results:rw,$dependency_root:/deps:rw"
  "$image" env PYTHONPATH=/workspace XDG_CACHE_HOME=/deps/cache TORCH_HOME=/deps/torch OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
  python -m blocket_league.direct_postfreeze_complete prepare-system
  /sanitized /training /results --system "$system" --device cuda
)
if [[ "$dry_run" == 1 ]]; then printf 'stage=postfreeze-prepare system=%s command=' "$system"; printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
[[ -f "$training_root/$system/training-complete.json" ]] || { printf 'Strict training aggregate missing for %s\n' "$system" >&2; exit 1; }
mkdir -p "$result_root" "$dependency_root/cache" "$dependency_root/torch"
exec "${command[@]}"
