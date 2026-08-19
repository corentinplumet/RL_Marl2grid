#!/usr/bin/env bash
# Full bus14 test-split evaluation for corrected Screens C and D on JED.
# Uses best_test checkpoints and deliberately excludes:
#   nl_s2em_bus_n0_none_e0n1v0_s0
#   nl_hmdem_hetero_n0_none_gb2a_lbi_s0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_dir="$(cd "${script_dir}/.." && pwd)"
repo_dir="$(cd "${task_dir}/.." && pwd)"

dry_run="${DRY_RUN:-false}"
force_results="${FORCE_RESULTS:-false}"
save_action_trace="${SAVE_ACTION_TRACE:-false}"
eval_cpus="${EVAL_CPUS:-8}"
eval_mem="${EVAL_MEM:-128G}"
result_group="bus14_nl_s2em_nl_hmdem"

is_true() {
  case "${1:-false}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

runs=(
  # nl_s2em: 7/8 cells; e0n1v0 is intentionally excluded.
  nl_s2em_bus_n0_none_e0n0v0_s0
  nl_s2em_bus_n0_none_e0n0v1_s0
  nl_s2em_bus_n0_none_e0n1v1_s0
  nl_s2em_bus_n0_none_e1n0v0_s0
  nl_s2em_bus_n0_none_e1n0v1_s0
  nl_s2em_bus_n0_none_e1n1v0_s0
  nl_s2em_bus_n0_none_e1n1v1_s0

  # nl_hmdem: 8/9 cells; gb2a_lbi is intentionally excluded.
  nl_hmdem_hetero_n0_none_ga2b_la2b_s0
  nl_hmdem_hetero_n0_none_ga2b_lb2a_s0
  nl_hmdem_hetero_n0_none_ga2b_lbi_s0
  nl_hmdem_hetero_n0_none_gb2a_la2b_s0
  nl_hmdem_hetero_n0_none_gb2a_lb2a_s0
  nl_hmdem_hetero_n0_none_gbi_la2b_s0
  nl_hmdem_hetero_n0_none_gbi_lb2a_s0
  nl_hmdem_hetero_n0_none_gbi_lbi_s0
)

if [[ ! -f "${task_dir}/full_test_eval/job_full_test_eval.sh" ]]; then
  echo "Could not find the JED full-test job wrapper." >&2
  exit 1
fi

plan_runs=()
plan_checkpoints=()
plan_outputs=()
plan_action_dirs=()
missing=()
skipped_existing=0

for run in "${runs[@]}"; do
  checkpoint="checkpoint/best_test_${run}.tar"
  output="outputs/full_test_eval/${result_group}/best_test_${run}.json"
  action_dir="outputs/full_test_eval_actions/${result_group}/best_test_${run}"

  if [[ ! -f "${task_dir}/${checkpoint}" ]]; then
    missing+=("${checkpoint}")
    continue
  fi
  if [[ -f "${task_dir}/${output}" ]] && ! is_true "${force_results}"; then
    echo "Skip existing result: ${output} (FORCE_RESULTS=true to rerun)"
    skipped_existing=$((skipped_existing + 1))
    continue
  fi

  plan_runs+=("${run}")
  plan_checkpoints+=("${checkpoint}")
  plan_outputs+=("${output}")
  plan_action_dirs+=("${action_dir}")
done

if (( ${#missing[@]} > 0 )); then
  echo "Missing ${#missing[@]} required best-test checkpoint(s); no jobs submitted:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

echo "========== JED bus14 full-test evaluation =========="
echo "Requested runs:      ${#runs[@]}"
echo "Excluded runs:       2"
echo "Skipped existing:    ${skipped_existing}"
echo "Planned submissions: ${#plan_runs[@]}"
echo "Result group:        ${result_group}"
echo "CPU / memory:        ${eval_cpus} / ${eval_mem}"
echo "Action trace:        ${save_action_trace}"
echo "DRY_RUN:             ${dry_run}"
echo "====================================================="

cd "${repo_dir}"
for index in "${!plan_runs[@]}"; do
  run="${plan_runs[$index]}"
  checkpoint="${plan_checkpoints[$index]}"
  output="${plan_outputs[$index]}"
  action_dir="${plan_action_dirs[$index]}"

  command=(
    sbatch
    "--job-name=b14_${run}"
    "--cpus-per-task=${eval_cpus}"
    "--mem=${eval_mem}"
    Topology_Task/full_test_eval/job_full_test_eval.sh
    --checkpoint "${checkpoint}"
    --split test
    --split-chronics true
    --eval-all-split-chronics true
    --deterministic-eval true
    --eval-action-heuristic none
    --obs-normalization require
    --device cpu
    --progress true
    --save-action-summary true
    --save-action-trace "${save_action_trace}"
    --decode-action-distribution true
    --action-log-dir "${action_dir}"
    --output-json "${output}"
  )

  if is_true "${dry_run}"; then
    printf 'Would submit:'
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
done

if is_true "${dry_run}"; then
  echo "Validated ${#plan_runs[@]} JED submission command(s)."
else
  echo "Submitted ${#plan_runs[@]} JED evaluation job(s)."
fi
