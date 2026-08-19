#!/usr/bin/env bash
# Submit the bus14 shared-candidate runs with the adaptive intervention budget.
# Usage:  bash configs/no_leakage_config/Y_nl_cas_hl_aib/launch_izar.sh [name filters...]
#   DRY_RUN=true    print the sbatch commands instead of submitting
#   SKIP_ACTIVE=false  submit even if a job with the same name is queued

set -euo pipefail
shopt -s nullglob

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../../../.." && pwd)"
task_config_dir="configs/no_leakage_config/Y_nl_cas_hl_aib"

dry_run="${DRY_RUN:-false}"
skip_active="${SKIP_ACTIVE:-true}"
exclude_nodes="${EXCLUDE_NODES:-${SBATCH_EXCLUDE:-}}"

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

echo "========== bus14 cas_hl adaptive-intervention-budget campaign =========="
echo "Configs:        $script_dir"
echo "Skip active:    $skip_active"
echo "Excluded nodes: ${exclude_nodes:-none}"
echo "DRY_RUN:        $dry_run"
echo "Filters:        ${*:-none}"
echo "========================================================================"

for config_path in "${configs[@]}"; do
  label="$(basename "$config_path" .toml)"
  if ! matches_filters "$label" "$@"; then
    continue
  fi
  selected=$((selected + 1))

  if is_true "$skip_active" && [[ -n "$active_job_names" ]] \
      && grep -Fxq "$label" <<<"$active_job_names"; then
    echo "Skip active Slurm job: $label"
    skipped=$((skipped + 1))
    continue
  fi

  command=(sbatch)
  if [[ -n "$exclude_nodes" ]]; then
    command+=("--exclude=$exclude_nodes")
  fi
  command+=("--job-name=$label" job_izar.sh "$task_config_dir/$label.toml")

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
  exit 1
fi
echo "Selected $selected, submitted $submitted, skipped $skipped."
