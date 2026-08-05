#!/bin/bash
#SBATCH --job-name=wm-f-postfreeze
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12-00:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.err

# Compatibility finalizer only.  The launcher now schedules prepare[2] and
# control[64] first; this script refuses partial evaluation and never fills in
# missing physical or CEM evidence.
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage:" \
    "  sbatch --dependency=afterok:<control-array> experiment-f-postfreeze-parent.sbatch" \
    "  bash experiment-f-postfreeze-parent.sbatch --dry-run" \
    "The job mounts all sanitized data read-only only after training finalization."
}
dry_run="${F_DRY_RUN:-0}"
while (($#)); do
  case "$1" in --dry-run) dry_run=1;; --help|-h) usage; exit 0;; *) usage >&2; exit 2;; esac
  shift
done

repo_root="${F_REPO_ROOT:-/Home/Users/aelmessaoudi/world-model}"
work_root="${F_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
run_root="${F_RUN_ROOT:-$work_root/outputs/direct-experiment-f-seed-151910737}"
sanitized_root="$run_root/sanitized"; training_root="$run_root/training"
result_root="$run_root/postfreeze"; dependency_root="$run_root/dependencies"
image="$work_root/containers/pytorch-2.6.0-cuda12.4.sif"; apptainer=/Softs/helios/apptainer/bin/apptainer
for path in "$repo_root" "$work_root" "$run_root" "$sanitized_root" "$training_root" "$result_root" "$dependency_root"; do
  [[ "$path" == /* && "$path" != *[:,]* ]] || { printf 'Unsafe bind path: %s\n' "$path" >&2; exit 2; }
done

run_command=(
  "$apptainer" exec --nv --containall --cleanenv --no-mount home,cwd,hostfs
  --bind "$repo_root:/workspace:ro,$sanitized_root:/sanitized:ro,$training_root:/training:ro,$result_root:/results:rw,$dependency_root:/deps:rw"
  "$image" env PYTHONPATH=/workspace XDG_CACHE_HOME=/deps/cache TORCH_HOME=/deps/torch
  OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
  python -m blocket_league.direct_postfreeze_complete finalize
  /sanitized /training /results --device cuda --require-gates 1,2,3,4,5,6,7,8,9
)
if [[ "$dry_run" == 1 ]]; then
  printf 'stage=postfreeze-parent status=compatibility-finalizer refusing partial evaluation command='; printf '%q ' "${run_command[@]}"; printf '\n'
  exit 0
fi
[[ -f "$repo_root/blocket_league/direct_postfreeze_complete.py" ]] || {
  printf 'Typed complete post-freeze CLI is absent; refusing evaluation.\n' >&2
  exit 1
}
[[ -f "$training_root/pendulum/training-complete.json" && -f "$training_root/blocket/training-complete.json" ]] || {
  printf 'Both strict training aggregates are required.\n' >&2
  exit 1
}
mkdir -p "$result_root" "$dependency_root/cache" "$dependency_root/torch"
"${run_command[@]}"
