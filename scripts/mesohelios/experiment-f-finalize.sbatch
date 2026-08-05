#!/bin/bash
#SBATCH --job-name=wm-f-finalize
#SBATCH --partition=smp
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/Work/Users/aelmessaoudi/world-model/logs/%x-%A_%a.out
#SBATCH --error=/Work/Users/aelmessaoudi/world-model/logs/%x-%A_%a.err

set -euo pipefail
usage() { printf '%s\n' "Usage: sbatch --dependency=afterok:<baseline> experiment-f-finalize.sbatch" "       bash experiment-f-finalize.sbatch --dry-run [--system pendulum|blocket]"; }
dry_run="${F_DRY_RUN:-0}"; requested_system=""
while (($#)); do case "$1" in --dry-run) dry_run=1;; --system) shift; requested_system="${1:?missing system}";; --help|-h) usage; exit 0;; *) usage >&2; exit 2;; esac; shift; done
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -n "$requested_system" ]]; then system="$requested_system"; else systems=(pendulum blocket); [[ "$task_id" =~ ^[01]$ ]] || exit 2; system="${systems[$task_id]}"; fi
[[ "$system" == pendulum || "$system" == blocket ]] || exit 2
work_root="${F_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
run_root="${F_RUN_ROOT:-$work_root/outputs/direct-experiment-f-seed-151910737}"; trainer_system="$run_root/sanitized/trainer-mount/$system"
learner_root="${F_LEARNER_BUNDLE_ROOT:-}"; learner_sha256="${F_LEARNER_SOURCE_TREE_SHA256:-}"; output_system="$run_root/training/$system"; cache_root="$run_root/runtime-cache/$system/finalize"; image="$work_root/containers/pytorch-2.6.0-cuda12.4.sif"
apptainer=/Softs/helios/apptainer/bin/apptainer
for path in "$learner_root" "$work_root" "$run_root" "$trainer_system" "$output_system" "$cache_root"; do [[ "$path" == /* && "$path" != *[:,]* ]] || exit 2; done
[[ "$learner_sha256" =~ ^[0-9a-f]{64}$ ]] || { printf 'Missing learner source SHA-256\n' >&2; exit 2; }
bootstrap_code='import hashlib, json, runpy, stat, sys
from pathlib import Path
root = Path(sys.argv[2])
manifest_path = root / "learner-source-manifest.json"
if root.is_symlink() or manifest_path.is_symlink():
    raise SystemExit("symbolic learner source boundary")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
unsigned = {key: value for key, value in manifest.items() if key != "treeSha256"}
digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
if digest != manifest.get("treeSha256") or digest != sys.argv[1]:
    raise SystemExit("learner manifest digest mismatch")
expected = {item["path"]: (item["bytes"], item["sha256"]) for item in manifest["files"]}
observed_files = set()
observed_directories = set()
for path in root.rglob("*"):
    relative = path.relative_to(root).as_posix()
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise SystemExit("symbolic learner bundle entry")
    if stat.S_ISREG(mode):
        observed_files.add(relative)
        if relative != "learner-source-manifest.json":
            value = path.read_bytes()
            if relative not in expected or (len(value), hashlib.sha256(value).hexdigest()) != expected[relative]:
                raise SystemExit("learner source byte mismatch")
    elif stat.S_ISDIR(mode):
        observed_directories.add(relative)
    else:
        raise SystemExit("special learner bundle entry")
expected_directories = {parent.as_posix() for relative in expected for parent in Path(relative).parents if parent.as_posix() != "."}
expected_directories.add("blocket_league")
if observed_files != set(expected) | {"learner-source-manifest.json"} or observed_directories != expected_directories:
    raise SystemExit("learner bundle contents mismatch")
sys.path.insert(0, str(root))
module = sys.argv[3]
sys.argv = [module, *sys.argv[4:]]
runpy.run_module(module, run_name="__main__")'
command=("$apptainer" exec --containall --cleanenv --no-mount home,cwd,hostfs --bind "$learner_root:/workspace:ro,$trainer_system:/trainer-mount/$system:ro,$output_system:/output:rw,$cache_root:/learner-cache:rw" "$image" env PYTHONPATH=/workspace PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/learner-cache/xdg TORCH_HOME=/learner-cache/torch OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" python -I -c "$bootstrap_code" "$learner_sha256" /workspace blocket_league.direct_distributed_training finalize /trainer-mount /output --system "$system" --device cpu --learner-bundle-root /workspace --learner-cache /learner-cache)
if [[ "$dry_run" == 1 ]]; then printf 'stage=finalize system=%s command=' "$system"; printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
[[ -f "$output_system/baseline-independent/best.pt" && -f "$output_system/baseline-independent/last.pt" ]] || { printf 'Independent baseline best/last prerequisite missing\n' >&2; exit 1; }
[[ -f "$learner_root/learner-source-manifest.json" ]] || { printf 'Sealed learner bundle is missing\n' >&2; exit 1; }
mkdir -p "$cache_root/xdg" "$cache_root/torch"
exec "${command[@]}"
