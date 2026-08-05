#!/bin/bash
#SBATCH --job-name=wm-f-full-real
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%j.err

set -euo pipefail

repo_root="${F_EVALUATOR_REPO_ROOT:?missing F_EVALUATOR_REPO_ROOT}"
work_root="${F_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
run_root="${F_RUN_ROOT:?missing F_RUN_ROOT}"
training_system="$run_root/training/blocket"
result_root="$run_root/postfreeze-full-only/blocket"
dependency_root="$run_root/dependencies-full-only"
manifest="$result_root/evaluator-source-manifest.json"
image="$work_root/containers/pytorch-2.6.0-cuda12.4.sif"
apptainer=/Softs/helios/apptainer/bin/apptainer
for path in "$repo_root" "$work_root" "$run_root" "$training_system" "$result_root" "$dependency_root"; do
  [[ "$path" == /* && "$path" != *[:,]* ]] || { printf 'Unsafe path: %s\n' "$path" >&2; exit 2; }
done
[[ -d "$repo_root/.git" ]] || { printf 'Evaluator Git workspace is missing\n' >&2; exit 1; }
[[ -z "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" ]] || { printf 'Evaluator Git workspace is dirty\n' >&2; exit 1; }
[[ -f "$training_system/direct/full/best.pt" ]] || { printf 'Full checkpoint is missing\n' >&2; exit 1; }
mkdir -p "$result_root" "$dependency_root/cache" "$dependency_root/torch"
python "$repo_root/blocket_league/source_provenance.py" create "$manifest" --repo-root "$repo_root" >/dev/null
exec "$apptainer" exec --nv --containall --cleanenv --no-mount home,cwd,hostfs \
  --bind "$repo_root:/workspace:ro,$training_system:/training:ro,$result_root:/results:rw,$dependency_root:/deps:rw" \
  "$image" env PYTHONPATH=/workspace PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 \
  XDG_CACHE_HOME=/deps/cache TORCH_HOME=/deps/torch OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}" \
  python -I -m blocket_league.direct_full_only_realizability \
  /training /results --system blocket --device cuda \
  --evaluator-source-manifest /results/evaluator-source-manifest.json
