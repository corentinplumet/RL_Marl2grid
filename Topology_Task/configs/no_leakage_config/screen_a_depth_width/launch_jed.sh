#!/usr/bin/env bash
# Corrected-construction rerun of this screen, seed 0, on JED.
# Pass filters to launch a subset; every filter must match.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

config_dir="Topology_Task/configs/no_leakage_config/screen_a_depth_width"
filters=("$@")
submitted=0
jed_time_limit="${TIME_LIMIT:-4980}"

is_true() { case "${1:-false}" in 1|true|TRUE|yes|YES|on|ON) return 0;; *) return 1;; esac; }

for config_path in "$config_dir"/nl_*.toml; do
  config_name="$(basename "$config_path" .toml)"
  skip=false
  for f in ${filters+"${filters[@]}"}; do
    [[ "$config_name" != *"$f"* ]] && { skip=true; break; }
  done
  [[ "$skip" == true ]] && continue
  submit=(sbatch --job-name="$config_name" --export="ALL,TIME_LIMIT=$jed_time_limit" job_jed.sh "${config_path#Topology_Task/}")
  if is_true "${DRY_RUN:-false}"; then
    printf 'Would submit:'; printf ' %q' "${submit[@]}"; printf '\n'
  else
    "${submit[@]}"
  fi
  submitted=$((submitted + 1))
done
echo "$( is_true "${DRY_RUN:-false}" && echo Validated || echo Submitted) $submitted runs for screen_a_depth_width on JED."
