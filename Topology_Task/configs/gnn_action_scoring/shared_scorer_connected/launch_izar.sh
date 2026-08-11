#!/usr/bin/env bash
# Launch shared-scorer bus14 runs on IZAR, optionally filtered by name.
set -euo pipefail

task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$task_dir"

config_dir="configs/gnn_action_scoring/shared_scorer_connected"
filters=("$@")
submitted=0

is_true() {
  case "${1:-false}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

for config_path in "$config_dir"/cas_hl_NL*_shared_*.toml; do
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

  command=(sbatch --job-name="$config_name" job_izar.sh "$config_path")
  if is_true "${DRY_RUN:-false}"; then
    printf 'Would submit:'
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
  submitted=$((submitted + 1))
done

if [[ "$submitted" -eq 0 ]]; then
  echo "No configurations matched the supplied filters." >&2
  exit 1
fi
if is_true "${DRY_RUN:-false}"; then
  echo "Validated $submitted IZAR submissions."
else
  echo "Submitted $submitted shared-scorer runs on IZAR."
fi
