#!/usr/bin/env bash
# Submit conservative NLS Gmax-delta fine-tuning on JED or Izar.
# Usage: bash launch.sh <jed|izar> [filename filters...]

set -euo pipefail
shopt -s nullglob

if [[ $# -lt 1 || ( "$1" != "jed" && "$1" != "izar" ) ]]; then
  echo "Usage: bash $0 <jed|izar> [filename filters...]" >&2
  exit 1
fi

cluster="$1"
shift
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../../../.." && pwd)"
task_dir="$repo_dir/Topology_Task"
task_config_dir="configs/no_leakage_config/Y_wcci_gmax_delta_finetune"
dry_run="${DRY_RUN:-false}"
skip_active="${SKIP_ACTIVE:-true}"
force_launch="${FORCE_LAUNCH:-false}"
jed_time_limit="${TIME_LIMIT:-4980}"
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
    [[ "$label" == *"$filter"* ]] || return 1
  done
  return 0
}

active_job_names=""
if is_true "$skip_active" && command -v squeue >/dev/null 2>&1; then
  active_job_names="$(squeue -h -u "${USER:?USER is not set}" -o '%j')"
fi

configs=("$script_dir"/ftgd*.toml)
selected=0
submitted=0
skipped=0
cd "$repo_dir"

for config_path in "${configs[@]}"; do
  label="$(basename "$config_path" .toml)"
  matches_filters "$label" "$@" || continue
  selected=$((selected + 1))

  action_space="$(sed -n 's/^reduced_action_space = "\(.*\)"/\1/p' "$config_path")"
  source_checkpoint="$(sed -n 's/^transfer_encoder_checkpoint = "\(.*\)"/\1/p' "$config_path")"
  if [[ -z "$action_space" || -z "$source_checkpoint" ]]; then
    echo "Incomplete transfer metadata in $config_path" >&2
    exit 1
  fi
  if ! is_true "$dry_run"; then
    [[ -f "$task_dir/$action_space" ]] || {
      echo "Missing action space: $task_dir/$action_space" >&2; exit 1;
    }
    [[ -f "$task_dir/$source_checkpoint" ]] || {
      echo "Missing source checkpoint: $task_dir/$source_checkpoint" >&2; exit 1;
    }
  fi

  if ! is_true "$force_launch"; then
    if [[ -f "$task_dir/checkpoint/$label.tar" ||
          -f "$task_dir/checkpoint/final_$label.tar" ||
          -f "$task_dir/checkpoint/best_test_$label.tar" ]]; then
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
  [[ -n "$exclude_nodes" ]] && command+=("--exclude=$exclude_nodes")
  command+=("--job-name=$label")
  if [[ "$cluster" == "jed" ]]; then
    command+=(
      "--export=ALL,TIME_LIMIT=$jed_time_limit"
      job_jed.sh
      "$task_config_dir/$label.toml"
    )
  else
    command+=(job_izar.sh "$task_config_dir/$label.toml")
  fi

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
echo "Selected=$selected submitted_or_planned=$submitted skipped=$skipped"
