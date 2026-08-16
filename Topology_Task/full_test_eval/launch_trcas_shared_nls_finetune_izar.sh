#!/usr/bin/env bash
# Evaluate all eight WCCI-fine-tuned shared NLS candidate-action checkpoints.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_dir="$(cd "$script_dir/.." && pwd)"
repo_dir="$(cd "$task_dir/.." && pwd)"

dry_run="${DRY_RUN:-false}"
force_results="${FORCE_RESULTS:-false}"
save_action_trace="${SAVE_ACTION_TRACE:-false}"
obs_normalization="${OBS_NORMALIZATION:-require}"
exclude_nodes="${EXCLUDE_NODES:-${SBATCH_EXCLUDE:-}}"

case "$obs_normalization" in
  auto|disable|require) ;;
  *)
    echo "OBS_NORMALIZATION must be auto, disable, or require; got: $obs_normalization" >&2
    exit 1
    ;;
esac

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

result_group="trcas_shared_NLS_finetune_wcci36_mk256"
plan_labels=()
plan_checkpoints=()
plan_outputs=()
plan_action_dirs=()
matched=0
skipped_existing=0

for variant in "${variants[@]}"; do
  run_name="trcas_shared_NLS_${variant}_finetune_s0"
  label="fte_NLS_${variant}"
  if ! matches_filters "$label" "$@"; then
    continue
  fi
  matched=$((matched + 1))

  checkpoint="checkpoint/best_test_${run_name}.tar"
  output="outputs/full_test_eval/shared/${result_group}/best_test_${run_name}.json"
  action_dir="outputs/full_test_eval_actions/shared/${result_group}/best_test_${run_name}"

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

if [[ $matched -eq 0 ]]; then
  echo "No NLS fine-tuning evaluations matched filters: $*" >&2
  echo "Examples: mean_f0_a0h0  tmean  a0h1" >&2
  exit 1
fi

if [[ ${#plan_labels[@]} -eq 0 ]]; then
  echo "Nothing to submit: all $matched matching results already exist."
  exit 0
fi

if ! is_true "$dry_run"; then
  for checkpoint in "${plan_checkpoints[@]}"; do
    if [[ ! -f "$task_dir/$checkpoint" ]]; then
      echo "Missing fine-tuned checkpoint: $task_dir/$checkpoint" >&2
      exit 1
    fi
  done
fi

echo "========== NLS trcas fine-tune full-test evaluation =========="
echo "Dataset:             bus36_wcci_nomaint test split"
echo "Episodes per model:  50"
echo "Reduced action cap:  mk256 (stored in each checkpoint)"
echo "Matched:             $matched"
echo "Skipped existing:    $skipped_existing"
echo "Planned submissions: ${#plan_labels[@]}"
echo "Obs normalization:   $obs_normalization"
echo "Save action trace:   $save_action_trace"
echo "Excluded nodes:      ${exclude_nodes:-none}"
echo "DRY_RUN:             $dry_run"
echo "=============================================================="

cd "$repo_dir"
for index in "${!plan_labels[@]}"; do
  label="${plan_labels[$index]}"
  checkpoint="${plan_checkpoints[$index]}"
  output="${plan_outputs[$index]}"
  action_dir="${plan_action_dirs[$index]}"

  command=(sbatch)
  if [[ -n "$exclude_nodes" ]]; then
    command+=("--exclude=$exclude_nodes")
  fi
  command+=(
    "--job-name=$label"
    Topology_Task/full_test_eval/job_full_test_eval_izar.sh
    --checkpoint "$checkpoint"
    --split test
    --split-chronics true
    --eval-all-split-chronics false
    --eval-episodes 50
    --deterministic-eval true
    --eval-action-heuristic none
    --obs-normalization "$obs_normalization"
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
  echo "Validated ${#plan_labels[@]} NLS fine-tuning submission command(s)."
else
  echo "Submitted ${#plan_labels[@]} NLS fine-tuning evaluation job(s)."
fi
