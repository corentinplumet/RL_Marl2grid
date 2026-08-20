#!/usr/bin/env bash
# Submit the matched seed-0 NLS WCCI mk32 scratch/fine-tuning campaign.
# CLUSTER selects the job script: jed (academic CPU partition, the default
# here because these configs set cuda = false and n_envs = 72) or izar.
# Generated from the mk64 campaign by
# tools/generate_shared_wcci_capped_seed0_configs.py.

set -euo pipefail
shopt -s nullglob

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../../../.." && pwd)"
task_dir="$repo_dir/Topology_Task"
task_config_dir="configs/no_leakage_config/W_wcci_cas_hl_shared_mk32_seed0"
action_space_rel="outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk32.json"
action_space_abs="$task_dir/$action_space_rel"

dry_run="${DRY_RUN:-false}"
force_launch="${FORCE_LAUNCH:-false}"
skip_active="${SKIP_ACTIVE:-true}"
exclude_nodes="${EXCLUDE_NODES:-${SBATCH_EXCLUDE:-}}"

cluster="${CLUSTER:-jed}"
case "$cluster" in
  jed)  job_script="job_jed.sh" ;;
  izar) job_script="job_izar.sh" ;;
  *) echo "CLUSTER must be 'jed' or 'izar', got: $cluster" >&2; exit 1 ;;
esac

is_true() {
  case "${1:-false}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

matches_filters() {
  local label="$1"
  shift
  local filter
  for filter in "$@"; do
    if [[ "$label" != *"$filter"* ]]; then
      return 1
    fi
  done
  return 0
}

if [[ ! -f "$action_space_abs" ]]; then
  if is_true "$dry_run"; then
    echo "Dry-run warning: missing local mk32 artifact $action_space_abs" >&2
  else
    echo "Missing mk32 reduced action space: $action_space_abs" >&2
    exit 1
  fi
else
  python - "$action_space_abs" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    reduction = json.load(handle)
sizes = {
    agent: int(payload.get("selected_action_size", -1))
    for agent, payload in sorted(reduction.get("agents", {}).items())
}
if not sizes or any(size != 32 for size in sizes.values()):
    raise SystemExit(f"Expected exactly 32 actions for every agent in {path}; got {sizes}")
print("Validated mk32 action space: " + ", ".join(
    f"{agent}={size}" for agent, size in sizes.items()
))
PY
fi

active_job_names=""
if is_true "$skip_active" && command -v squeue >/dev/null 2>&1; then
  active_job_names="$(squeue -h -u "${USER:?USER is not set}" -o '%j')"
fi

configs=("$script_dir"/*.toml)
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No TOML configs found in $script_dir" >&2
  exit 1
fi

selected=0
submitted=0
skipped=0
cd "$repo_dir"

echo "========== NLS WCCI mk32 seed-0 campaign =========="
echo "Cluster:         $cluster ($job_script)"
echo "Configs:         $script_dir"
echo "Action space:    $action_space_rel"
echo "Skip active:     $skip_active"
echo "Excluded nodes:  ${exclude_nodes:-none}"
echo "DRY_RUN:         $dry_run"
echo "Filters:         ${*:-none}"
echo "====================================================="

for config_path in "${configs[@]}"; do
  label="$(basename "$config_path" .toml)"
  if ! matches_filters "$label" "$@"; then
    continue
  fi
  selected=$((selected + 1))

  if [[ "$label" == ft32c_* ]]; then
    source_checkpoint="$(sed -n 's/^transfer_encoder_checkpoint = "\(.*\)"/\1/p' "$config_path")"
    if [[ -z "$source_checkpoint" ]]; then
      echo "Missing transfer_encoder_checkpoint in $config_path" >&2
      exit 1
    fi
    if [[ ! -f "$task_dir/$source_checkpoint" ]] && ! is_true "$dry_run"; then
      echo "Missing source checkpoint: $task_dir/$source_checkpoint" >&2
      exit 1
    fi
  fi

  if ! is_true "$force_launch"; then
    if [[ -f "$task_dir/checkpoint/$label.tar" ||
          -f "$task_dir/checkpoint/best_test_$label.tar" ||
          -f "$task_dir/checkpoint/final_$label.tar" ]]; then
      echo "Skip checkpointed run: $label"
      skipped=$((skipped + 1))
      continue
    fi
    if [[ -n "$active_job_names" ]] && grep -Fxq "$label" <<<"$active_job_names"; then
      echo "Skip active Slurm job: $label"
      skipped=$((skipped + 1))
      continue
    fi
  fi

  command=(sbatch)
  if [[ -n "$exclude_nodes" ]]; then
    command+=("--exclude=$exclude_nodes")
  fi
  command+=(
    "--job-name=$label"
    "$job_script"
    "$task_config_dir/$label.toml"
  )

  if is_true "$dry_run"; then
    printf 'Would submit:'
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
  submitted=$((submitted + 1))
done

if [[ $selected -eq 0 ]]; then
  echo "No configs matched filters: $*" >&2
  echo "Examples: ft32c  sc32c  mean_f0_a0h0  tmean" >&2
  exit 1
fi

echo "Selected=$selected submitted_or_planned=$submitted skipped=$skipped"
