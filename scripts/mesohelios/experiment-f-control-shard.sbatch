#!/bin/bash
#SBATCH --job-name=wm-f-control
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --array=0-63%1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12-00:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%A_%a.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%A_%a.err

# Exact mapping: 2 systems x 2 interfaces x 16 four-episode ranges.  Every
# array element writes one authenticated file and is safe to resubmit.
set -euo pipefail

usage() { printf '%s\n' "Usage: sbatch --dependency=afterok:<prepare-array> experiment-f-control-shard.sbatch" "       bash experiment-f-control-shard.sbatch --dry-run [--task-id 0..63]"; }
dry_run="${F_DRY_RUN:-0}"; task_id="${SLURM_ARRAY_TASK_ID:-0}"
while (($#)); do case "$1" in --dry-run) dry_run=1;; --task-id) shift; task_id="${1:?missing task id}";; --help|-h) usage; exit 0;; *) usage >&2; exit 2;; esac; shift; done
[[ "$task_id" =~ ^([0-9]|[1-5][0-9]|6[0-3])$ ]] || { printf 'Task id must be 0..63\n' >&2; exit 2; }
systems=(pendulum blocket); interfaces=(native unseen)
system_index=$((task_id / 32)); within_system=$((task_id % 32))
interface_index=$((within_system / 16)); range_index=$((within_system % 16))
system="${systems[$system_index]}"; interface="${interfaces[$interface_index]}"
start=$((range_index * 4)); stop=$((start + 4))

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
  python -m blocket_league.direct_postfreeze_complete control-shard
  /sanitized /training /results --system "$system" --interface "$interface"
  --start "$start" --stop "$stop" --device cuda
)
if [[ "$dry_run" == 1 ]]; then printf 'stage=control-shard task=%s system=%s interface=%s start=%s stop=%s command=' "$task_id" "$system" "$interface" "$start" "$stop"; printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
[[ -f "$result_root/$system/prepared-system.pt" && -f "$result_root/$system/physical-shared.pt" ]] || { printf 'Prepared/shared evidence missing for %s\n' "$system" >&2; exit 1; }
mkdir -p "$result_root/$system/control/$interface" "$dependency_root/cache" "$dependency_root/torch"
exec "${command[@]}"
