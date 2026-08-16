#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

config_dir="Topology_Task/configs/gnn_action_scoring/no_leak"
walltime="${JED_WALLTIME:-7-00:00:00}"
submitted=0

is_true() {
  case "${1:-false}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

for config_path in "$config_dir"/cas_hl_NL_*.toml; do
  config_name="$(basename "$config_path" .toml)"
  config_arg="${config_path#Topology_Task/}"
  submit=(
    sbatch
    --job-name="$config_name"
    --time="$walltime"
    job_jed.sh
    "$config_arg"
  )

  if is_true "${DRY_RUN:-false}"; then
    printf 'Would submit:'
    printf ' %q' "${submit[@]}"
    printf '\n'
  else
    "${submit[@]}"
  fi
  submitted=$((submitted + 1))
done

if is_true "${DRY_RUN:-false}"; then
  echo "Validated $submitted no-leak CAS JED submissions."
else
  echo "Submitted $submitted no-leak CAS runs on JED."
fi
