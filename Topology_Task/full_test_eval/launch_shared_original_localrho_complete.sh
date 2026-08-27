#!/usr/bin/env bash
# Close the local-rho 0.95 gate panel of the ungated zero-shot grid.
#
# The ungated `cas_hl` block is a complete 16 architectures x 4 caps. The gated
# panel beside it is not: mk32 is complete (16/16), but mk64, mk128 and mk256
# were only ever screened opportunistically, so 32 of the 64 cells have no
# rho-0.95 evaluation at all. This launcher submits exactly those 32.
#
# Nothing here is training. Every cell is a zero-shot evaluation of a checkpoint
# that already exists — the same bus14 checkpoint its ungated twin used — with
# the local-rho gate switched on at evaluation time.
#
# The plan below was generated from the data, not typed:
#
#     ungated (variant, cap) cells  MINUS  cells with a local_rho_threshold
#     evaluation at rho == 0.95, over caps 64/128/256
#
# Re-derive it from `wcci_full_matrix_report.ipynb` after any new results land.
#
# Results are written straight into `outputs/full_test_eval/shared/wcci/...`,
# which is where `wcci_transfer_data.load()` reads from. The older
# `launch_shared_original_localrho_*.sh` scripts write one level up and their
# output was moved into `wcci/` afterwards; do not copy that path.
#
#   DRY_RUN=true bash Topology_Task/full_test_eval/launch_shared_original_localrho_complete.sh
#   DRY_RUN=true CLUSTER=jed bash Topology_Task/full_test_eval/launch_shared_original_localrho_complete.sh
#
# Remove DRY_RUN=true to submit. Positional arguments filter by substring, e.g.
# `mk64`, `NLS_`, `tmean_f1_a0h1`.
#
# CLUSTER=izar (default) submits job_full_test_eval_izar.sh: gpu partition, qos
# long, one GPU, --device auto.
# CLUSTER=jed submits job_full_test_eval.sh: academic partition and qos, no GPU.
# Two things have to change for that job script. It declares --mem-per-cpu,
# which at its default 72 cpus asks for far more memory than an evaluation
# needs, so this script overrides --cpus-per-task and --mem the way
# launch_jed_bus14_nl_evals.sh does. And the checkpoint config carries
# cuda=True, so --device auto would try CUDA and abort on a cpu-only node:
# --device cpu is passed explicitly.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_dir="$(cd "$script_dir/.." && pwd)"
repo_dir="$(cd "$task_dir/.." && pwd)"

rho_threshold="${RHO_THRESHOLD:-0.95}"
dry_run="${DRY_RUN:-false}"
force_results="${FORCE_RESULTS:-false}"
save_action_trace="${SAVE_ACTION_TRACE:-false}"
exclude_nodes="${EXCLUDE_NODES:-${SBATCH_EXCLUDE:-}}"
group="${GROUP:-origin_localrho_complete}"
cluster="${CLUSTER:-izar}"
eval_cpus="${EVAL_CPUS:-8}"
eval_mem="${EVAL_MEM:-128G}"

case "$cluster" in
  izar)
    job_script="Topology_Task/full_test_eval/job_full_test_eval_izar.sh"
    device="${DEVICE:-auto}"
    ;;
  jed)
    job_script="Topology_Task/full_test_eval/job_full_test_eval.sh"
    device="${DEVICE:-cpu}"
    ;;
  *)
    echo "CLUSTER must be izar or jed, got: $cluster" >&2
    exit 1
    ;;
esac

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

rho_tag="$(python - "$rho_threshold" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value < 0.0:
    raise SystemExit(f"RHO_THRESHOLD must be a finite non-negative number, got {value!r}")
print(f"{round(100.0 * value):03d}")
PY
)"

# family|variant|action-cap  — the 32 cells missing from the rho-0.95 panel.
screen_plan=(
  "NLS|mean_f0_a0h0|128"
  "NLS|mean_f0_a0h1|64"
  "NLS|mean_f0_a0h1|128"
  "NLS|mean_f0_a0h1|256"
  "NLS|mean_f1_a0h1|64"
  "NLS|mean_f1_a0h1|128"
  "NLS|mean_f1_a0h1|256"
  "NLS|tmean_f0_a0h1|64"
  "NLS|tmean_f0_a0h1|128"
  "NLS|tmean_f0_a0h1|256"
  "NLS|tmean_f1_a0h0|128"
  "NLS|tmean_f1_a0h0|256"
  "NLS|tmean_f1_a0h1|128"
  "NLS|tmean_f1_a0h1|256"
  "NL|mean_f0_a0h0|64"
  "NL|mean_f0_a0h0|128"
  "NL|mean_f0_a0h0|256"
  "NL|mean_f0_a0h1|64"
  "NL|mean_f0_a0h1|128"
  "NL|mean_f0_a0h1|256"
  "NL|mean_f1_a0h0|64"
  "NL|mean_f1_a0h0|128"
  "NL|mean_f1_a0h0|256"
  "NL|mean_f1_a0h1|64"
  "NL|mean_f1_a0h1|128"
  "NL|mean_f1_a0h1|256"
  "NL|tmean_f0_a0h0|256"
  "NL|tmean_f1_a0h0|64"
  "NL|tmean_f1_a0h0|128"
  "NL|tmean_f1_a0h0|256"
  "NL|tmean_f1_a0h1|128"
  "NL|tmean_f1_a0h1|256"
)

plan_labels=()
plan_checkpoints=()
plan_action_spaces=()
plan_outputs=()
plan_action_dirs=()
matched=0
skipped_existing=0

for specification in "${screen_plan[@]}"; do
  IFS='|' read -r family variant action_size <<<"$specification"

  if [[ "$family" == "NLS" ]]; then
    checkpoint_dir="checkpoint/no_leak/shared/NL_cas_hl_scaled_shared_izar"
    source_prefix="cas_hl_NLS_izar_shared"
  else
    checkpoint_dir="checkpoint/no_leak/shared/NL_cas_hl_shared"
    source_prefix="cas_hl_NL_shared"
  fi

  label="orig_${family}_${variant}_mk${action_size}_lr${rho_tag}"
  if ! matches_filters "$label" "$@"; then
    continue
  fi
  matched=$((matched + 1))

  checkpoint="${checkpoint_dir}/best_test_${source_prefix}_${variant}_s0.tar"
  action_space="outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk${action_size}.json"
  output="outputs/full_test_eval/shared/wcci/${group}/${label}.json"
  action_dir="outputs/full_test_eval_actions/shared/${group}/${label}"

  if [[ -f "$task_dir/$output" ]] && ! is_true "$force_results"; then
    echo "Skip existing result: $output (FORCE_RESULTS=true to rerun)"
    skipped_existing=$((skipped_existing + 1))
    continue
  fi

  plan_labels+=("$label")
  plan_checkpoints+=("$checkpoint")
  plan_action_spaces+=("$action_space")
  plan_outputs+=("$output")
  plan_action_dirs+=("$action_dir")
done

if [[ $matched -eq 0 ]]; then
  echo "No evaluations matched filters: $*" >&2
  echo "Examples: NLS_  NL_  mean_f1_a0h1  mk64  mk128  mk256" >&2
  exit 1
fi

if [[ ${#plan_labels[@]} -eq 0 ]]; then
  echo "Nothing to submit: all $matched matching results already exist."
  exit 0
fi

# Every checkpoint must exist before anything is submitted: a missing one is a
# silently absent cell in the finished panel, which is the exact problem this
# script is closing.
for checkpoint in "${plan_checkpoints[@]}"; do
  if [[ ! -f "$task_dir/$checkpoint" ]]; then
    echo "Missing source checkpoint: $task_dir/$checkpoint" >&2
    exit 1
  fi
done

if [[ ! -f "$repo_dir/$job_script" ]]; then
  echo "Missing job script: $repo_dir/$job_script" >&2
  exit 1
fi

for action_size in 64 128 256; do
  action_space_abs="$task_dir/outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk${action_size}.json"
  if [[ ! -f "$action_space_abs" ]]; then
    echo "Missing reduced action space: $action_space_abs" >&2
    exit 1
  fi
done

echo "========== Complete the rho-${rho_threshold} gate panel =========="
echo "Cluster:             $cluster"
echo "Job script:          $job_script"
echo "Device:              $device"
echo "Group:               $group"
echo "Rho threshold:       $rho_threshold"
echo "Action caps:         64, 128, 256 (mk32 is already complete)"
echo "Matched:             $matched"
echo "Skipped existing:    $skipped_existing"
echo "Planned submissions: ${#plan_labels[@]}"
if [[ "$cluster" == "jed" ]]; then
  echo "CPU / memory:        $eval_cpus / $eval_mem"
fi
echo "Excluded nodes:      ${exclude_nodes:-none}"
echo "DRY_RUN:             $dry_run"
echo "=================================================================="

cd "$repo_dir"
for index in "${!plan_labels[@]}"; do
  label="${plan_labels[$index]}"

  command=(sbatch)
  if [[ -n "$exclude_nodes" ]]; then
    command+=("--exclude=$exclude_nodes")
  fi
  command+=("--job-name=$label")
  if [[ "$cluster" == "jed" ]]; then
    command+=("--cpus-per-task=$eval_cpus" "--mem=$eval_mem")
  fi
  command+=(
    "$job_script"
    --checkpoint "${plan_checkpoints[$index]}"
    --target-env-id bus36_wcci_nomaint
    --target-reduced-action-space "${plan_action_spaces[$index]}"
    --split test
    --split-chronics true
    --eval-all-split-chronics false
    --eval-episodes 50
    --deterministic-eval true
    --eval-action-heuristic local_rho_threshold
    --eval-action-rho-threshold "$rho_threshold"
    --device "$device"
    --obs-normalization disable
    --save-action-summary true
    --save-action-trace "$save_action_trace"
    --decode-action-distribution true
    --action-log-dir "${plan_action_dirs[$index]}"
    --output-json "${plan_outputs[$index]}"
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
  echo "Validated ${#plan_labels[@]} submission command(s)."
else
  echo "Submitted ${#plan_labels[@]} local-rho evaluation job(s)."
fi
