#!/usr/bin/env bash
# Re-evaluate every NL/NLS shared-candidate checkpoint zero-shot on WCCI with
# exactly 64 target actions per agent. The original mk256 results are untouched.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_dir="$(cd "$script_dir/.." && pwd)"
repo_dir="$(cd "$task_dir/.." && pwd)"

action_space_rel="outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk64.json"
action_space_abs="$task_dir/$action_space_rel"
dry_run="${DRY_RUN:-false}"
force_results="${FORCE_RESULTS:-false}"
save_action_trace="${SAVE_ACTION_TRACE:-false}"

variants=(
  mean_f0_a0h0
  mean_f0_a0h1
  mean_f1_a0h0
  mean_f1_a0h1
  tmean_f0_a0h0
  tmean_f0_a0h1
  tmean_f1_a0h0
  tmean_f1_a0h1
)

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
    if [[ "$label" != *"$filter"* ]]; then
      return 1
    fi
  done
  return 0
}

plan_labels=()
plan_checkpoints=()
plan_outputs=()
plan_action_dirs=()
matched=0
skipped_existing=0

for family in NL NLS; do
  if [[ "$family" == NL ]]; then
    checkpoint_dir="checkpoint/no_leak/shared/NL_cas_hl_shared"
    source_prefix="cas_hl_NL_shared"
    result_group="NL_cas_hl_shared_wcci36_mk64"
  else
    # This is the scaled Izar family used by the existing mk256 zero-shot run.
    checkpoint_dir="checkpoint/no_leak/shared/NL_cas_hl_scaled_shared_izar"
    source_prefix="cas_hl_NLS_izar_shared"
    result_group="NLS_cas_hl_izar_shared_wcci36_mk64"
  fi

  for variant in "${variants[@]}"; do
    label="zs64_${family}_${variant}"
    if ! matches_filters "$label" "$@"; then
      continue
    fi
    matched=$((matched + 1))

    source_name="${source_prefix}_${variant}_s0"
    checkpoint="${checkpoint_dir}/best_test_${source_name}.tar"
    output="outputs/full_test_eval/shared/${result_group}/best_test_${source_name}_mk64.json"
    action_dir="outputs/full_test_eval_actions/shared/${result_group}/best_test_${source_name}_mk64"

    if [[ -f "$task_dir/$output" ]] && ! is_true "$force_results"; then
      echo "Skip existing result: $output (FORCE_RESULTS=true to rerun)"
      skipped_existing=$((skipped_existing + 1))
      continue
    fi

    plan_labels+=("$label")
    plan_checkpoints+=("$checkpoint")
    plan_outputs+=("$output")
    plan_action_dirs+=("$action_dir")
  done
done

if [[ $matched -eq 0 ]]; then
  echo "No zero-shot evaluations matched filters: $*" >&2
  echo "Examples: NL_  NLS_  mean_f0_a0h0" >&2
  exit 1
fi

if [[ ${#plan_labels[@]} -eq 0 ]]; then
  echo "Nothing to submit: all $matched matching results already exist."
  exit 0
fi

if ! is_true "$dry_run"; then
  if [[ ! -f "$action_space_abs" ]]; then
    echo "Missing mk64 reduced action space: $action_space_abs" >&2
    echo "Generate it from the repository root with:" >&2
    echo "  TOP_KS=64 Topology_Task/teacher_student/reduce_wcci_action_space_sizes.sh" >&2
    exit 1
  fi

  # Do not launch any jobs unless the artifact really contains 64 actions for
  # every target agent; a filename alone is not sufficient evidence.
  python - "$action_space_abs" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    reduction = json.load(handle)
agents = reduction.get("agents", {})
if not agents:
    raise SystemExit(f"No agent reductions found in {path}")
sizes = {
    agent: int(payload.get("selected_action_size", -1))
    for agent, payload in sorted(agents.items())
}
invalid = {agent: size for agent, size in sizes.items() if size != 64}
if invalid:
    raise SystemExit(
        f"Expected exactly 64 actions per agent in {path}; got {sizes}"
    )
print("Validated mk64 target action space: " + ", ".join(
    f"{agent}={size}" for agent, size in sizes.items()
))
PY

  for checkpoint in "${plan_checkpoints[@]}"; do
    if [[ ! -f "$task_dir/$checkpoint" ]]; then
      echo "Missing source checkpoint: $task_dir/$checkpoint" >&2
      exit 1
    fi
  done
fi

echo "========== Shared zero-shot WCCI mk64 =========="
echo "Matched:             $matched"
echo "Skipped existing:    $skipped_existing"
echo "Planned submissions: ${#plan_labels[@]}"
echo "Save action trace:   $save_action_trace"
echo "DRY_RUN:             $dry_run"
echo "================================================"

cd "$repo_dir"
for index in "${!plan_labels[@]}"; do
  label="${plan_labels[$index]}"
  checkpoint="${plan_checkpoints[$index]}"
  output="${plan_outputs[$index]}"
  action_dir="${plan_action_dirs[$index]}"

  command=(
    sbatch
    "--job-name=$label"
    Topology_Task/full_test_eval/job_full_test_eval_izar.sh
    --checkpoint "$checkpoint"
    --target-env-id bus36_wcci_nomaint
    --target-reduced-action-space "$action_space_rel"
    --split test
    --split-chronics true
    --eval-all-split-chronics false
    --eval-episodes 50
    --deterministic-eval true
    --eval-action-heuristic none
    --obs-normalization disable
    --save-action-summary true
    --save-action-trace "$save_action_trace"
    --decode-action-distribution true
    --action-log-dir "$action_dir"
    --output-json "$output"
  )

  if is_true "$dry_run"; then
    printf 'Would submit:'
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
done

if is_true "$dry_run"; then
  echo "Validated ${#plan_labels[@]} mk64 submission command(s)."
else
  echo "Submitted ${#plan_labels[@]} mk64 zero-shot evaluation job(s)."
fi
