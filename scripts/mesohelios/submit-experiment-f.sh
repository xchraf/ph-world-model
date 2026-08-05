#!/bin/bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: submit-experiment-f.sh [--dry-run] [--include-postfreeze]" \
    "Submits producer -> backbone -> exact Jacobian-port extraction -> {12 variants, independent baseline} -> finalize." \
    "Post-freeze is opt-in: prepare[2] -> control[64] -> authenticated finalizer." \
    "All job IDs are persisted under F_RUN_ROOT/submissions/."
}
dry_run=0; include_postfreeze=0
while (($#)); do
  case "$1" in
    --dry-run) dry_run=1;;
    --include-postfreeze) include_postfreeze=1;;
    --help|-h) usage; exit 0;;
    *) usage >&2; exit 2;;
  esac
  shift
done

# Scientific values are registered below in the stage scripts, not tunable
# launch parameters.  Refuse stale shell exports instead of silently producing
# a differently configured run under the Experiment F result path.  Synthetic
# timing probes have their own scripts and remain explicitly configurable.
registered_scientific_environment=(
  F_FIT_TRAJECTORIES F_VALIDATION_TRAJECTORIES F_TEST_TRAJECTORIES
  F_TRANSITIONS F_CACHE_FRAMES F_IMAGE_SIZE F_PATCH_SIZE F_BACKBONE_PRESET
  F_BACKBONE_STEPS F_DIRECT_STEPS F_BASELINE_STEPS F_MICRO_BATCH_SIZE
  F_GRADIENT_ACCUMULATION F_LENS_BATCH_SIZE F_IMPLICIT_ITERATIONS
  F_ENERGY_GRADIENT_IMPLEMENTATION F_FLOAT32_MATMUL_PRECISION
)
for variable_name in "${registered_scientific_environment[@]}"; do
  if [[ "${!variable_name+x}" == x ]]; then
    printf 'Refusing scientific override %s. Experiment F uses registered literals; use a performance-probe script for timing changes.\n' \
      "$variable_name" >&2
    exit 2
  fi
done

script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export F_REPO_ROOT="${F_REPO_ROOT:-/Home/Users/aelmessaoudi/world-model}"
export F_WORK_ROOT="${F_WORK_ROOT:-/Work/Users/aelmessaoudi/world-model}"
export F_RUN_ROOT="${F_RUN_ROOT:-$F_WORK_ROOT/outputs/direct-experiment-f-seed-151910737}"
for path in "$F_REPO_ROOT" "$F_WORK_ROOT" "$F_RUN_ROOT"; do
  [[ "$path" == /* ]] && [[ "$path" != *:* && "$path" != *,* && "$path" != *$'\n'* && "$path" != *$'\r'* ]] || {
    printf 'Unsafe root: %s\n' "$path" >&2
    exit 2
  }
done

producer_script="$script_root/experiment-f-producer.sbatch"
backbone_script="$script_root/experiment-f-backbone.sbatch"
port_script="$script_root/experiment-f-port.sbatch"
variant_script="$script_root/experiment-f-variant.sbatch"
baseline_script="$script_root/experiment-f-baseline.sbatch"
finalize_script="$script_root/experiment-f-finalize.sbatch"
postfreeze_prepare_script="$script_root/experiment-f-postfreeze-prepare.sbatch"
postfreeze_control_script="$script_root/experiment-f-control-shard.sbatch"
postfreeze_finalize_script="$script_root/experiment-f-postfreeze-finalize.sbatch"

if [[ "$dry_run" == 1 ]]; then
  printf 'sbatch --parsable %q\n' "$producer_script"
  printf 'sbatch --parsable --dependency=afterok:<producer_job_id> %q\n' "$backbone_script"
  printf 'sbatch --parsable --dependency=afterok:<backbone_job_id> %q\n' "$port_script"
  printf 'sbatch --parsable --dependency=afterok:<port_job_id> %q\n' "$variant_script"
  printf 'sbatch --parsable --dependency=afterok:<port_job_id> %q\n' "$baseline_script"
  printf 'sbatch --parsable --dependency=afterok:<variant_array_job_id>:<baseline_job_id> %q\n' "$finalize_script"
  if [[ "$include_postfreeze" == 1 ]]; then
    printf 'sbatch --parsable --dependency=afterok:<finalize_job_id> %q\n' "$postfreeze_prepare_script"
    printf 'sbatch --parsable --dependency=afterok:<postfreeze_prepare_array_job_id> %q\n' "$postfreeze_control_script"
    printf 'sbatch --parsable --dependency=afterok:<postfreeze_control_array_job_id> %q\n' "$postfreeze_finalize_script"
  fi
  exit 0
fi

[[ -d "$F_REPO_ROOT/.git" ]] || {
  printf 'Launch requires a clean Git workspace for hygiene; .git is missing.\n' >&2
  exit 1
}
source_status="$(git -C "$F_REPO_ROOT" status --porcelain=v1 --untracked-files=all -- \
  blocket_league scripts/mesohelios tests \
  docs/direct_jacobian_poisson_ph_experiment.md pyproject.toml uv.lock)"
[[ -z "$source_status" ]] || {
  printf 'Registered Experiment F source workspace is not clean:\n%s\n' "$source_status" >&2
  exit 1
}

mkdir -p "$F_WORK_ROOT/logs" "$F_RUN_ROOT/submissions" \
  "$F_RUN_ROOT/runtime-cache" "$F_RUN_ROOT/producer-cache" \
  "$F_RUN_ROOT/training/pendulum" "$F_RUN_ROOT/training/blocket" \
  "$F_RUN_ROOT/producer-private"
chmod 700 "$F_RUN_ROOT/producer-private"
producer_seed_file="$F_RUN_ROOT/producer-private/producer-seed.hex"
if [[ -e "$producer_seed_file" ]]; then
  [[ -f "$producer_seed_file" && ! -L "$producer_seed_file" ]] || {
    printf 'Existing producer seed source is not a regular nonsymbolic file.\n' >&2
    exit 1
  }
  [[ "$(LC_ALL=C tr -d '\n' < "$producer_seed_file")" =~ ^[0-9a-f]{32}$ ]] || {
    printf 'Existing producer seed source is not 128-bit lowercase hex.\n' >&2
    exit 1
  }
  chmod 600 "$producer_seed_file"
else
  seed_temporary="$F_RUN_ROOT/producer-private/.producer-seed.$$.tmp"
  (umask 077; "${F_PYTHON:-python}" -c 'import secrets; print(secrets.token_hex(16))' > "$seed_temporary")
  chmod 600 "$seed_temporary"
  mv -- "$seed_temporary" "$producer_seed_file"
fi
source_manifest="$F_RUN_ROOT/source-manifest.json"
source_tree_sha256="$(
  "${F_PYTHON:-python}" "$F_REPO_ROOT/blocket_league/source_provenance.py" \
    create "$source_manifest" \
    --repo-root "$F_REPO_ROOT"
)"
export F_LEARNER_BUNDLE_ROOT="$F_RUN_ROOT/learner-source/$source_tree_sha256"
learner_source_tree_sha256="$(
  PYTHONPATH="$F_REPO_ROOT" "${F_PYTHON:-python}" \
    -m blocket_league.learner_source_bundle build \
    "$F_LEARNER_BUNDLE_ROOT" \
    --repo-root "$F_REPO_ROOT" \
    --source-manifest "$source_manifest"
)"
export F_LEARNER_SOURCE_TREE_SHA256="$learner_source_tree_sha256"
for system in pendulum blocket; do
  installed_manifest="$F_RUN_ROOT/training/$system/source-manifest.json"
  if [[ -f "$installed_manifest" ]]; then
    cmp --silent "$source_manifest" "$installed_manifest" || {
      printf 'Existing %s source manifest differs from launch seal.\n' "$system" >&2
      exit 1
    }
  else
    cp -- "$source_manifest" "$installed_manifest"
  fi
done
producer_id="$(sbatch --parsable "$producer_script")"
backbone_id="$(sbatch --parsable --dependency="afterok:$producer_id" "$backbone_script")"
port_id="$(sbatch --parsable --dependency="afterok:$backbone_id" "$port_script")"
variant_id="$(sbatch --parsable --dependency="afterok:$port_id" "$variant_script")"
baseline_id="$(sbatch --parsable --dependency="afterok:$port_id" "$baseline_script")"
finalize_id="$(sbatch --parsable --dependency="afterok:$variant_id:$baseline_id" "$finalize_script")"
postfreeze_prepare_id=""; postfreeze_control_id=""; postfreeze_finalize_id=""
if [[ "$include_postfreeze" == 1 ]]; then
  [[ -f "$F_REPO_ROOT/blocket_league/direct_postfreeze_complete.py" ]] || {
    printf 'Typed post-freeze CLI missing; training DAG was submitted, post-freeze was not.\n' >&2
    exit 1
  }
  postfreeze_prepare_id="$(sbatch --parsable --dependency="afterok:$finalize_id" "$postfreeze_prepare_script")"
  postfreeze_control_id="$(sbatch --parsable --dependency="afterok:$postfreeze_prepare_id" "$postfreeze_control_script")"
  postfreeze_finalize_id="$(sbatch --parsable --dependency="afterok:$postfreeze_control_id" "$postfreeze_finalize_script")"
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
manifest="$F_RUN_ROOT/submissions/launch-$stamp.json"
manifest_temporary="$F_RUN_ROOT/submissions/.launch-$stamp.$$.tmp"
printf '{\n  "sourceTreeSha256": "%s",\n  "learnerSourceTreeSha256": "%s",\n  "learnerBundleRoot": "%s",\n  "producer": "%s",\n  "backbone": "%s",\n  "jacobianPortPrecompute": "%s",\n  "variants": "%s",\n  "baseline": "%s",\n  "finalize": "%s",\n  "postfreezePrepare": "%s",\n  "postfreezeControl": "%s",\n  "postfreezeFinalize": "%s"\n}\n' \
  "$source_tree_sha256" "$learner_source_tree_sha256" "$F_LEARNER_BUNDLE_ROOT" "$producer_id" "$backbone_id" "$port_id" "$variant_id" "$baseline_id" "$finalize_id" "$postfreeze_prepare_id" "$postfreeze_control_id" "$postfreeze_finalize_id" > "$manifest_temporary"
mv -- "$manifest_temporary" "$manifest"
printf 'Persistent launch manifest: %s\n' "$manifest"
