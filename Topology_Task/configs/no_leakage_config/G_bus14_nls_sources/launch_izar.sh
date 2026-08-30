#!/usr/bin/env bash
# bus14 NLS source retrains (Chapter 9)
# Usage: bash configs/no_leakage_config/G_bus14_nls_sources/launch_izar.sh [filters...]
#   DRY_RUN=true     print the sbatch commands instead of submitting
#   FORCE_LAUNCH=true  submit even when a checkpoint already exists
#   SKIP_ACTIVE=false  submit even when a job of the same name is queued

set -euo pipefail
shopt -s nullglob

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../../../.." && pwd)"
task_dir="$repo_dir/Topology_Task"
task_config_dir="configs/no_leakage_config/G_bus14_nls_sources"

dry_run="${DRY_RUN:-false}"
force_launch="${FORCE_LAUNCH:-false}"
skip_active="${SKIP_ACTIVE:-true}"
exclude_nodes="${EXCLUDE_NODES:-${SBATCH_EXCLUDE:-}}"

is_true() { case "${1:-false}" in 1|true|TRUE|yes|YES|on|ON) return 0 ;; *) return 1 ;; esac; }
matches_filters() {
  local label="$1"; shift
  for filter in "$@"; do [[ "$label" != *"$filter"* ]] && return 1; done
  return 0
}

active_job_names=""
if is_true "$skip_active" && command -v squeue >/dev/null 2>&1; then
  active_job_names="$(squeue -h -u "${USER:?USER is not set}" -o '%j')"
fi

configs=("$script_dir"/*.toml)
[[ ${#configs[@]} -eq 0 ]] && { echo "No TOML configs in $script_dir" >&2; exit 1; }

selected=0; submitted=0; skipped=0; missing=0
cd "$repo_dir"

echo "========== bus14 NLS source retrains (Chapter 9) =========="
echo "Configs:         $task_config_dir"
echo "Launcher:        job_izar.sh"
echo "Skip active:     $skip_active"
echo "DRY_RUN:         $dry_run"
echo "Filters:         ${*:-none}"
echo "==============================================================="

for config_path in "${configs[@]}"; do
  label="$(basename "$config_path" .toml)"
  matches_filters "$label" "$@" || continue
  selected=$((selected + 1))

  # A frozen-encoder route is meaningless if its source weights are absent, so
  # the checkpoint is verified here rather than failing 30 seconds into the job.
  source_checkpoint="$(sed -n 's/^transfer_encoder_checkpoint = "\(.*\)"/\1/p' "$config_path")"
  if [[ -n "$source_checkpoint" && ! -f "$task_dir/$source_checkpoint" ]]; then
    echo "MISSING source checkpoint for $label: $source_checkpoint" >&2
    missing=$((missing + 1))
    is_true "$dry_run" || continue
  fi

  if ! is_true "$force_launch"; then
    if [[ -f "$task_dir/checkpoint/$label.tar" ||
          -f "$task_dir/checkpoint/best_test_$label.tar" ||
          -f "$task_dir/checkpoint/final_$label.tar" ]]; then
      echo "Skip checkpointed run: $label"; skipped=$((skipped + 1)); continue
    fi
    if [[ -n "$active_job_names" ]] && grep -Fxq "$label" <<<"$active_job_names"; then
      echo "Skip active Slurm job: $label"; skipped=$((skipped + 1)); continue
    fi
  fi

  command=(sbatch)
  [[ -n "$exclude_nodes" ]] && command+=("--exclude=$exclude_nodes")
  command+=("--job-name=$label" job_izar.sh "$task_config_dir/$label.toml")

  if is_true "$dry_run"; then
    printf 'Would submit:'; printf ' %q' "${command[@]}"; printf '\n'
  else
    "${command[@]}"
  fi
  submitted=$((submitted + 1))
done

[[ $selected -eq 0 ]] && { echo "No configs matched filters: $*" >&2; exit 1; }
echo "Selected=$selected submitted_or_planned=$submitted skipped=$skipped missing_source=$missing"
[[ $missing -gt 0 ]] && echo "WARNING: $missing config(s) reference a source checkpoint that does not exist yet." >&2
exit 0
