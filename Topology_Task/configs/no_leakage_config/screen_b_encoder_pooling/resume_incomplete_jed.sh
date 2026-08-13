#!/usr/bin/env bash
# Resume only unfinished seed-0 runs from the encoder/pooling screen on JED.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

submit=false
if [[ "${1:-}" == "--submit" ]]; then
  submit=true
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--submit]" >&2
  exit 2
fi

target_timesteps="${TARGET_TIMESTEPS:-15000000}"
time_limit_minutes="${TIME_LIMIT_MINUTES:-1300}"
launcher="${LAUNCHER:-job_jed.sh}"
checkpoint_dir="${CHECKPOINT_DIR:-checkpoint}"
reset_environments="${RESET_ENVIRONMENTS:-true}"
config_dir="Topology_Task/configs/no_leakage_config/screen_b_encoder_pooling"

is_true() {
  case "${1:-false}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

active_commands=""
if command -v squeue >/dev/null 2>&1; then
  active_commands="$(squeue -u "${USER:-$(whoami)}" -h -o "%o" 2>/dev/null || true)"
fi

checked=0
skipped_running=0
skipped_complete=0
prepared=0
failed=0

for config_path in "$config_dir"/nl_s3ep_*_s0.toml; do
  checked=$((checked + 1))
  config_rel="${config_path#Topology_Task/}"
  run_name="$(basename "${config_path%.toml}")"

  if [[ -n "$active_commands" ]] && {
    grep -Fq -- "$config_rel" <<<"$active_commands" ||
      grep -Fq -- "$config_path" <<<"$active_commands"
  }; then
    echo "Still running on Slurm, skipping: $run_name"
    skipped_running=$((skipped_running + 1))
    continue
  fi

  command=(
    python Topology_Task/scripts/resume_from_config.py
    "$config_rel"
    --checkpoint-dir "$checkpoint_dir"
    --target-timesteps "$target_timesteps"
    --time-limit "$time_limit_minutes"
    --launcher "$launcher"
    --allow-device-migration
  )
  if is_true "$reset_environments"; then
    command+=(--reset-environments)
  fi
  if [[ "$submit" == true ]]; then
    command+=(--submit)
  fi

  echo
  echo "Resume candidate: $run_name"
  set +e
  output="$("${command[@]}" 2>&1)"
  status=$?
  set -e
  printf '%s\n' "$output"

  if [[ $status -eq 0 ]]; then
    prepared=$((prepared + 1))
  elif grep -Fq "already at or beyond the requested target" <<<"$output"; then
    skipped_complete=$((skipped_complete + 1))
  else
    failed=$((failed + 1))
    echo "Could not prepare resume for: $run_name" >&2
  fi
done

echo
echo "Checked $checked nl_s3ep seed-0 configurations."
echo "Skipped $skipped_running active Slurm job(s)."
echo "Skipped $skipped_complete completed run(s)."
if [[ "$submit" == true ]]; then
  echo "Submitted $prepared continuation job(s)."
else
  echo "Prepared $prepared continuation job(s); this was a dry run."
fi
echo "Failed $failed candidate(s)."

if [[ $failed -gt 0 ]]; then
  exit 1
fi
