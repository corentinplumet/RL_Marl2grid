#!/usr/bin/env bash
# Stage-2 original bus14 -> WCCI local-rho transfer study.
#
# This is a selective, matched design rather than a full factorial sweep:
# - complete threshold curves for the two contrasting leading candidates;
# - complete the matched NL typed-mean f0 h0/h1 comparison;
# - screen the still-uncovered typed-mean f1 family at rho=.90/.95.
#
# mk512 and mk1024 are intentionally excluded.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_dir="$(cd "$script_dir/.." && pwd)"
repo_dir="$(cd "$task_dir/.." && pwd)"

dry_run="${DRY_RUN:-false}"
force_results="${FORCE_RESULTS:-false}"
save_action_trace="${SAVE_ACTION_TRACE:-false}"
exclude_nodes="${EXCLUDE_NODES:-${SBATCH_EXCLUDE:-}}"

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

# family|variant|action-cap|local-rho threshold
stage2_plan=(
  # Full threshold curves for the current winner and early-intervention model.
  "NL|tmean_f0_a0h1|128|0.85"
  "NL|tmean_f0_a0h1|128|0.90"
  "NL|tmean_f0_a0h1|128|1.00"
  "NLS|mean_f1_a0h0|128|0.85"
  "NLS|mean_f1_a0h0|128|0.90"
  "NLS|mean_f1_a0h0|128|1.00"

  # Matched h0/h1 comparison; h1=.90 and both=.95 are already covered/pending.
  "NL|tmean_f0_a0h0|128|0.90"

  # Missing typed-mean f1 architecture coverage at its viable action cap.
  "NL|tmean_f1_a0h1|64|0.90"
  "NL|tmean_f1_a0h1|64|0.95"
  "NLS|tmean_f1_a0h1|64|0.90"
  "NLS|tmean_f1_a0h1|64|0.95"
  "NLS|tmean_f1_a0h0|64|0.90"
  "NLS|tmean_f1_a0h0|64|0.95"
)

plan_labels=()
plan_checkpoints=()
plan_action_spaces=()
plan_thresholds=()
plan_outputs=()
plan_action_dirs=()
matched=0
skipped_existing=0

for specification in "${stage2_plan[@]}"; do
  IFS='|' read -r family variant action_size rho_threshold <<<"$specification"

  if [[ "$family" == "NLS" ]]; then
    checkpoint_dir="checkpoint/no_leak/shared/NL_cas_hl_scaled_shared_izar"
    source_prefix="cas_hl_NLS_izar_shared"
  else
    checkpoint_dir="checkpoint/no_leak/shared/NL_cas_hl_shared"
    source_prefix="cas_hl_NL_shared"
  fi

  rho_tag="$(python - "$rho_threshold" <<'PY'
import sys
print(f"{round(100.0 * float(sys.argv[1])):03d}")
PY
)"
  label="origlr${rho_tag}_${family}_${variant}_mk${action_size}"
  if ! matches_filters "$label" "$@"; then
    continue
  fi
  matched=$((matched + 1))

  source_name="${source_prefix}_${variant}_s0"
  checkpoint="${checkpoint_dir}/best_test_${source_name}.tar"
  action_space="outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk${action_size}.json"
  output="outputs/full_test_eval/shared/origin_localrho_stage2/orig_${family}_${variant}_mk${action_size}_lr${rho_tag}.json"
  action_dir="outputs/full_test_eval_actions/shared/origin_localrho_stage2/orig_${family}_${variant}_mk${action_size}_lr${rho_tag}"

  if [[ -f "$task_dir/$output" ]] && ! is_true "$force_results"; then
    echo "Skip existing result: $output (FORCE_RESULTS=true to rerun)"
    skipped_existing=$((skipped_existing + 1))
    continue
  fi

  plan_labels+=("$label")
  plan_checkpoints+=("$checkpoint")
  plan_action_spaces+=("$action_space")
  plan_thresholds+=("$rho_threshold")
  plan_outputs+=("$output")
  plan_action_dirs+=("$action_dir")
done

if [[ $matched -eq 0 ]]; then
  echo "No stage-2 evaluations matched filters: $*" >&2
  echo "Examples: NL_  NLS_  tmean_f1  lr090  mk128" >&2
  exit 1
fi

if [[ ${#plan_labels[@]} -eq 0 ]]; then
  echo "Nothing to submit: all $matched matching results already exist."
  exit 0
fi

# Validate each distinct target action space selected by the filtered plan.
checked_action_spaces=()
for action_space in "${plan_action_spaces[@]}"; do
  already_checked=false
  for checked_action_space in "${checked_action_spaces[@]:-}"; do
    if [[ "$checked_action_space" == "$action_space" ]]; then
      already_checked=true
      break
    fi
  done
  if is_true "$already_checked"; then
    continue
  fi
  checked_action_spaces+=("$action_space")

  action_space_abs="$task_dir/$action_space"
  action_size="${action_space##*_mk}"
  action_size="${action_size%.json}"
  if [[ ! -f "$action_space_abs" ]]; then
    if is_true "$dry_run"; then
      echo "Dry-run warning: missing local artifact $action_space_abs" >&2
      continue
    fi
    echo "Missing reduced action space: $action_space_abs" >&2
    exit 1
  fi

  python - "$action_space_abs" "$action_size" <<'PY'
import json
import sys

path = sys.argv[1]
expected = int(sys.argv[2])
with open(path, encoding="utf-8") as handle:
    reduction = json.load(handle)

agents = reduction.get("agents", {})
if not agents:
    raise SystemExit(f"No agent reductions found in {path}")
sizes = {
    agent: int(payload.get("selected_action_size", -1))
    for agent, payload in sorted(agents.items())
}
original = {
    agent: int(payload.get("original_action_size", -1))
    for agent, payload in sorted(agents.items())
}
invalid = {
    agent: size
    for agent, size in sizes.items()
    if size <= 0 or size > min(expected, original[agent])
}
if invalid:
    raise SystemExit(
        f"Invalid mk{expected} selected sizes in {path}: "
        f"selected={sizes}, original={original}"
    )

configured = {
    agent: int(value)
    for agent, value in reduction.get("top_k_by_agent", {}).items()
}
if configured:
    wrong = {
        agent: {
            "configured": configured.get(agent),
            "accepted": [expected, min(expected, original[agent])],
        }
        for agent in sizes
        if configured.get(agent) not in {expected, min(expected, original[agent])}
    }
    if wrong:
        raise SystemExit(f"Incorrect mk{expected} per-agent caps in {path}: {wrong}")
    over_cap = {
        agent: {"selected": sizes[agent], "configured": configured[agent]}
        for agent in sizes
        if sizes[agent] > configured[agent]
    }
    if over_cap:
        raise SystemExit(f"Selected sizes exceed configured caps in {path}: {over_cap}")
elif int(reduction.get("top_k", -1)) != expected:
    raise SystemExit(
        f"Expected top_k={expected} in {path}; got "
        f"{reduction.get('top_k')!r} and no top_k_by_agent override"
    )

print(
    f"Validated mk{expected}: "
    + ", ".join(f"{agent}={size}" for agent, size in sizes.items())
)
PY
done

if ! is_true "$dry_run"; then
  for checkpoint in "${plan_checkpoints[@]}"; do
    if [[ ! -f "$task_dir/$checkpoint" ]]; then
      echo "Missing original source checkpoint: $task_dir/$checkpoint" >&2
      exit 1
    fi
  done
fi

echo "========== Original local-rho WCCI stage 2 =========="
echo "Matched:             $matched"
echo "Skipped existing:    $skipped_existing"
echo "Planned submissions: ${#plan_labels[@]}"
echo "Action caps:         64 and 128 only"
echo "Save action trace:   $save_action_trace"
echo "Excluded nodes:      ${exclude_nodes:-none}"
echo "DRY_RUN:             $dry_run"
echo "======================================================"

cd "$repo_dir"
for index in "${!plan_labels[@]}"; do
  label="${plan_labels[$index]}"
  checkpoint="${plan_checkpoints[$index]}"
  action_space="${plan_action_spaces[$index]}"
  rho_threshold="${plan_thresholds[$index]}"
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
    --target-env-id bus36_wcci_nomaint
    --target-reduced-action-space "$action_space"
    --split test
    --split-chronics true
    --eval-all-split-chronics false
    --eval-episodes 50
    --deterministic-eval true
    --eval-action-heuristic local_rho_threshold
    --eval-action-rho-threshold "$rho_threshold"
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
  echo "Validated ${#plan_labels[@]} stage-2 submission command(s)."
else
  echo "Submitted ${#plan_labels[@]} original stage-2 evaluation job(s)."
fi
