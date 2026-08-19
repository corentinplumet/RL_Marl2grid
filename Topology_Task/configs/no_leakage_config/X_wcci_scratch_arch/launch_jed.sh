#!/usr/bin/env bash
# Submit the WCCI-from-scratch architecture screen on JED.
# Usage: bash configs/no_leakage_config/X_wcci_scratch_arch/launch_jed.sh [filters...]
#   DRY_RUN=true       print sbatch commands instead of submitting
#   SKIP_ACTIVE=false  submit even if a job with the same name is queued
#   TIME_LIMIT=4980    in-run wall-clock budget in minutes (JED allows 3-12:00:00)
# Filters match on the config name, e.g. "mk64", "bus_", "het_".

set -euo pipefail
shopt -s nullglob

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../../../.." && pwd)"
task_dir="$repo_dir/Topology_Task"
# job_jed.sh resolves config paths relative to Topology_Task.
task_config_dir="configs/no_leakage_config/X_wcci_scratch_arch"

dry_run="${DRY_RUN:-false}"
skip_active="${SKIP_ACTIVE:-true}"
jed_time_limit="${TIME_LIMIT:-4980}"
exclude_nodes="${EXCLUDE_NODES:-${SBATCH_EXCLUDE:-}}"

is_true() {
  case "${1:-false}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

matches_filters() {
  local label="$1"; shift
  local filter
  for filter in "$@"; do
    if [[ "$label" != *"$filter"* ]]; then return 1; fi
  done
  return 0
}

# Every config must point at an action space that exists locally.
for config_path in "$script_dir"/*.toml; do
  rel="$(sed -n 's/^reduced_action_space = "\(.*\)"/\1/p' "$config_path")"
  if [[ -z "$rel" ]]; then
    echo "Missing reduced_action_space in $config_path" >&2
    exit 1
  fi
  if [[ ! -f "$task_dir/$rel" ]]; then
    echo "Missing action space artifact: $task_dir/$rel" >&2
    exit 1
  fi
done

active_job_names=""
if is_true "$skip_active" && command -v squeue >/dev/null 2>&1; then
  active_job_names="$(squeue -h -u "${USER:?USER is not set}" -o '%j')"
fi

configs=("$script_dir"/*.toml)
selected=0; submitted=0; skipped=0
cd "$repo_dir"

echo "========== WCCI from-scratch architecture screen (JED) =========="
echo "Configs:        $script_dir"
echo "TIME_LIMIT:     $jed_time_limit min"
echo "Skip active:    $skip_active"
echo "DRY_RUN:        $dry_run"
echo "Filters:        ${*:-none}"
echo "================================================================="

for config_path in "${configs[@]}"; do
  label="$(basename "$config_path" .toml)"
  matches_filters "$label" "$@" || continue
  selected=$((selected + 1))

  if is_true "$skip_active" && [[ -n "$active_job_names" ]] \
      && grep -Fxq "$label" <<<"$active_job_names"; then
    echo "Skip active Slurm job: $label"; skipped=$((skipped + 1)); continue
  fi

  command=(sbatch)
  [[ -n "$exclude_nodes" ]] && command+=("--exclude=$exclude_nodes")
  command+=(
    "--job-name=$label"
    "--export=ALL,TIME_LIMIT=$jed_time_limit"
    job_jed.sh
    "$task_config_dir/$label.toml"
  )

  if is_true "$dry_run"; then
    printf 'Would submit:'; printf ' %q' "${command[@]}"; printf '\n'
  else
    "${command[@]}"
  fi
  submitted=$((submitted + 1))
done

if [[ $selected -eq 0 ]]; then
  echo "No configs matched filters: $*" >&2; exit 1
fi
echo "Selected $selected, submitted $submitted, skipped $skipped."
