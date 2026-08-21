#!/usr/bin/env bash
# Submit the NL max-readout architecture ablation on Izar.

set -euo pipefail
shopt -s nullglob

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_dir="$(cd "$script_dir/../../../.." && pwd)"
repo_dir="$(cd "$task_dir/.." && pwd)"
task_config_dir="configs/no_leakage_config/F_nl_cas_hl_max_arch/NL"
filters=("$@")
submitted=0

is_true() {
  case "${1:-false}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

cd "$repo_dir"
for config_path in "$script_dir"/*.toml; do
  config_name="$(basename "$config_path" .toml)"
  for filter in ${filters+"${filters[@]}"}; do
    [[ "$config_name" == *"$filter"* ]] || continue 2
  done
  command=(
    sbatch
    "--job-name=$config_name"
    job_izar.sh
    "$task_config_dir/$config_name.toml"
  )
  if is_true "${DRY_RUN:-false}"; then
    printf 'Would submit:'; printf ' %q' "${command[@]}"; printf '\n'
  else
    "${command[@]}"
  fi
  submitted=$((submitted + 1))
done

if [[ $submitted -eq 0 ]]; then
  echo "No NL configurations matched: ${*:-<none>}." >&2
  exit 1
fi
echo "NL configs selected: $submitted."
