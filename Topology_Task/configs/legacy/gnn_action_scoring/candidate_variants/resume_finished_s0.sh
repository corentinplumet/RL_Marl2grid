#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

submit=false
if [[ "${1:-}" == "--submit" ]]; then
  submit=true
  shift
fi

target_timesteps="${TARGET_TIMESTEPS:-15000000}"
time_limit_minutes="${TIME_LIMIT_MINUTES:-10080}"
launcher="${LAUNCHER:-job_jed.sh}"
checkpoint_dir="${CHECKPOINT_DIR:-checkpoint}"
config_dir="Topology_Task/configs/gnn_action_scoring/candidate_variants"

active_commands=""
if command -v squeue >/dev/null 2>&1; then
  active_commands="$(squeue -u "${USER:-$(whoami)}" -h -o "%o" 2>/dev/null || true)"
fi

checked=0
skipped_running=0
skipped_complete=0
attempted=0
failed=0

for config_path in "$config_dir"/*_s0.toml; do
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
  )
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
  if [[ "$status" -eq 0 ]]; then
    attempted=$((attempted + 1))
  elif grep -Fq "already at or beyond the requested target" <<<"$output"; then
    skipped_complete=$((skipped_complete + 1))
  else
    failed=$((failed + 1))
    echo "Could not prepare resume for: $run_name" >&2
  fi
done

echo
echo "Checked $checked seed-0 candidate configs."
echo "Skipped $skipped_running still-running Slurm job(s)."
echo "Skipped $skipped_complete already-complete checkpoint(s)."
if [[ "$submit" == true ]]; then
  echo "Submitted/resolved $attempted resume candidate(s)."
else
  echo "Printed $attempted dry-run resume candidate(s)."
fi
echo "Failed $failed candidate(s)."
