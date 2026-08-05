#!/bin/bash
#SBATCH --job-name=wm-f-pf-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12-00:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.err

# Re-audit cached structural tensors, validate the one shared physical bank,
# replay every stored command trace, and derive Gates 1--9.  No CEM is run.
set -euo pipefail

usage() { printf '%s\n' "Usage: sbatch --dependency=afterok:<control-array> experiment-f-postfreeze-finalize.sbatch" "       bash experiment-f-postfreeze-finalize.sbatch --dry-run"; }
dry_run="${F_DRY_RUN:-0}"
while (($#)); do case "$1" in --dry-run) dry_run=1;; --help|-h) usage; exit 0;; *) usage >&2; exit 2;; esac; shift; done
repo_root="${F_REPO_ROOT:-/Home/Users/aelmessaoudi/world-model}"; work_root="${F_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
run_root="${F_RUN_ROOT:-$work_root/outputs/direct-experiment-f-seed-151910737}"
sanitized_root="$run_root/sanitized"; training_root="$run_root/training"; result_root="$run_root/postfreeze"; dependency_root="$run_root/dependencies"
image="$work_root/containers/pytorch-2.6.0-cuda12.4.sif"; apptainer=/Softs/helios/apptainer/bin/apptainer
for path in "$repo_root" "$work_root" "$run_root" "$sanitized_root" "$training_root" "$result_root" "$dependency_root"; do [[ "$path" == /* && "$path" != *[:,]* ]] || exit 2; done
command=(
  "$apptainer" exec --nv --containall --cleanenv --no-mount home,cwd,hostfs
  --bind "$repo_root:/workspace:ro,$sanitized_root:/sanitized:ro,$training_root:/training:ro,$result_root:/results:rw,$dependency_root:/deps:rw"
  "$image" env PYTHONPATH=/workspace XDG_CACHE_HOME=/deps/cache TORCH_HOME=/deps/torch OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
  python -m blocket_league.direct_postfreeze_complete finalize
  /sanitized /training /results --device cuda --require-gates 1,2,3,4,5,6,7,8,9
)
if [[ "$dry_run" == 1 ]]; then printf 'stage=postfreeze-finalize no-cem=1 command='; printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
for system in pendulum blocket; do
  [[ -f "$result_root/$system/prepared-system.pt" && -f "$result_root/$system/physical-shared.pt" ]] || { printf 'Prepared evidence missing for %s\n' "$system" >&2; exit 1; }
  for interface in native unseen; do
    for start in $(seq 0 4 60); do printf -v stop '%02d' $((start + 4)); printf -v begin '%02d' "$start"; [[ -f "$result_root/$system/control/$interface/$begin-$stop.pt" ]] || { printf 'Missing control shard %s/%s/%s-%s\n' "$system" "$interface" "$begin" "$stop" >&2; exit 1; }; done
  done
done
mkdir -p "$dependency_root/cache" "$dependency_root/torch"
exec "${command[@]}"
