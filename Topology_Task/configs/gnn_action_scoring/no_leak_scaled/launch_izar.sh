#!/usr/bin/env bash
# No-leak candidate-action scoring with corrected encoder inputs, on IZAR.
# Pass filters to launch a subset. Every filter must match, so
#   ./launch_izar.sh _s0         -> the 11 seed-0 runs
#   ./launch_izar.sh tattn _s0   -> the 3 adaptive seed-0 runs
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

config_dir="Topology_Task/configs/gnn_action_scoring/no_leak_scaled"
filters=("$@")
walltime="${IZAR_WALLTIME:-}"
submitted=0

is_true() {
  case "${1:-false}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

for config_path in "$config_dir"/cas_hl_NLS_*.toml; do
  config_name="$(basename "$config_path" .toml)"
  skip=false
  for filter in ${filters+"${filters[@]}"}; do
    if [[ "$config_name" != *"$filter"* ]]; then
      skip=true
      break
    fi
  done
  if [[ "$skip" == true ]]; then
    continue
  fi

  submit=(sbatch --job-name="$config_name")
  # job_izar.sh already requests 3-00:12:00, which the configs' time_limit is
  # sized against. Only override deliberately.
  if [[ -n "$walltime" ]]; then
    submit+=(--time="$walltime")
  fi
  submit+=(job_izar.sh "${config_path#Topology_Task/}")

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
  echo "Validated $submitted scaled no-leak CAS IZAR submissions."
else
  echo "Submitted $submitted scaled no-leak CAS runs on IZAR."
fi
