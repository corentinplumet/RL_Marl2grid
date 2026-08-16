#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run from anywhere. Optional arguments are ANDed filename substrings, e.g.:
#   DRY_RUN=true bash launch_izar.sh NLS finetune
#   bash launch_izar.sh NL mean_f0_a0h0

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../../../.." && pwd)"
task_dir="$repo_dir/Topology_Task"
config_dir="$task_dir/configs/no_leakage_config/W_wcci_cas_hl_shared_transfer"
task_config_dir="configs/no_leakage_config/W_wcci_cas_hl_shared_transfer"
reduced_action_space="$task_dir/outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk256.json"
dry_run="${DRY_RUN:-false}"

cd "$repo_dir"

selected=()
for config_path in "$config_dir"/trcas_shared_*.toml; do
  config_name="$(basename "$config_path" .toml)"
  matches=true
  for filter in "$@"; do
    if [[ "$config_name" != *"$filter"* ]]; then
      matches=false
      break
    fi
  done
  if [[ "$matches" == true ]]; then
    selected+=("$config_path")
  fi
done

if [[ ${#selected[@]} -eq 0 ]]; then
  echo "No configs matched filters: $*" >&2
  exit 1
fi

if [[ "$dry_run" != true && ! -f "$reduced_action_space" ]]; then
  echo "Missing WCCI reduced action space: $reduced_action_space" >&2
  exit 1
fi

echo "Selected ${#selected[@]} config(s). DRY_RUN=$dry_run"
for config_path in "${selected[@]}"; do
  config_name="$(basename "$config_path" .toml)"
  if [[ "$config_name" == *"_finetune_"* ]]; then
    checkpoint="$(sed -n 's/^transfer_encoder_checkpoint = "\(.*\)"/\1/p' "$config_path")"
    if [[ -z "$checkpoint" ]]; then
      echo "Missing transfer checkpoint field: $config_path" >&2
      exit 1
    fi
    if [[ "$dry_run" != true && ! -f "$task_dir/$checkpoint" ]]; then
      echo "Missing source checkpoint: $task_dir/$checkpoint" >&2
      exit 1
    fi
  fi

  task_config="$task_config_dir/$config_name.toml"
  if [[ "$dry_run" == true ]]; then
    echo "sbatch --job-name=$config_name job_izar.sh $task_config"
  else
    sbatch --job-name="$config_name" job_izar.sh "$task_config"
  fi
done
