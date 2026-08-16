#!/usr/bin/env bash
# Re-evaluate every NL/NLS shared-candidate checkpoint zero-shot on WCCI.
# ACTION_SIZE defaults to 64; result directories are separated by mk size.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_dir="$(cd "$script_dir/.." && pwd)"
repo_dir="$(cd "$task_dir/.." && pwd)"

action_size="${ACTION_SIZE:-64}"
if [[ ! "$action_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACTION_SIZE must be a positive integer, got: $action_size" >&2
  exit 1
fi
action_space_rel="outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk${action_size}.json"
action_space_abs="$task_dir/$action_space_rel"
dry_run="${DRY_RUN:-false}"
force_results="${FORCE_RESULTS:-false}"
save_action_trace="${SAVE_ACTION_TRACE:-false}"
# Convert either launcher variable into an explicit sbatch command-line option.
# Some Slurm installations do not honor SBATCH_EXCLUDE from the environment.
exclude_nodes="${EXCLUDE_NODES:-${SBATCH_EXCLUDE:-}}"

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
    result_group="NL_cas_hl_shared_wcci36_mk${action_size}"
  else
    # This is the scaled Izar family used by the existing mk256 zero-shot run.
    checkpoint_dir="checkpoint/no_leak/shared/NL_cas_hl_scaled_shared_izar"
    source_prefix="cas_hl_NLS_izar_shared"
    result_group="NLS_cas_hl_izar_shared_wcci36_mk${action_size}"
  fi

  for variant in "${variants[@]}"; do
    label="zs${action_size}_${family}_${variant}"
    if ! matches_filters "$label" "$@"; then
      continue
    fi
    matched=$((matched + 1))

    source_name="${source_prefix}_${variant}_s0"
    checkpoint="${checkpoint_dir}/best_test_${source_name}.tar"
    output="outputs/full_test_eval/shared/${result_group}/best_test_${source_name}_mk${action_size}.json"
    action_dir="outputs/full_test_eval_actions/shared/${result_group}/best_test_${source_name}_mk${action_size}"

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

if [[ ! -f "$action_space_abs" ]]; then
  if ! is_true "$dry_run"; then
    echo "Missing mk${action_size} reduced action space: $action_space_abs" >&2
    echo "Generate it from the repository root with:" >&2
    echo "  TOP_KS=${action_size} Topology_Task/teacher_student/reduce_wcci_action_space_sizes.sh" >&2
    exit 1
  else
    echo "Dry-run warning: mk${action_size} artifact is not available locally; skipping its content validation." >&2
  fi
else
  # The reducer interprets top-k as a cap: an agent can contain fewer actions
  # when fewer candidates satisfy the ranking filters. Reject empty or
  # oversized sets, but accept and report valid undersized sets.
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
original_sizes = {
    agent: int(payload.get("original_action_size", -1))
    for agent, payload in sorted(agents.items())
}
invalid = {
    agent: size
    for agent, size in sizes.items()
    if size <= 0
    or size > expected
    or original_sizes[agent] <= 0
    or size > original_sizes[agent]
}
if invalid:
    raise SystemExit(
        f"Invalid selected/original action sizes in {path}: "
        f"selected={sizes}, original={original_sizes}, requested_cap={expected}"
    )
declared_top_k = int(reduction.get("top_k", -1))
configured_caps = {
    agent: int(cap)
    for agent, cap in reduction.get("top_k_by_agent", {}).items()
}
if configured_caps:
    missing_caps = sorted(set(sizes) - set(configured_caps))
    if missing_caps:
        raise SystemExit(f"Missing per-agent caps in {path}: {missing_caps}")
    expected_caps = {
        agent: min(expected, original_sizes[agent])
        for agent in sizes
    }
    wrong_caps = {
        agent: {"configured": configured_caps[agent], "expected": expected_caps[agent]}
        for agent in sizes
        if configured_caps[agent] != expected_caps[agent]
    }
    if wrong_caps:
        raise SystemExit(
            f"Per-agent caps do not represent mk{expected} in {path}: {wrong_caps}"
        )
    if declared_top_k != expected:
        print(
            f"Warning: global top_k={declared_top_k} is stale; validated the "
            f"explicit mk{expected} per-agent caps instead."
        )
else:
    if declared_top_k != expected:
        raise SystemExit(
            f"Expected top_k={expected} in {path}; got top_k={declared_top_k} "
            "and no explicit top_k_by_agent override."
        )
    configured_caps = {
        agent: min(expected, original_sizes[agent])
        for agent in sizes
    }
over_cap = {
    agent: {"selected": size, "cap": configured_caps[agent]}
    for agent, size in sizes.items()
    if size > configured_caps[agent]
}
if over_cap:
    raise SystemExit(f"Selected action sizes exceed per-agent caps in {path}: {over_cap}")
print(f"Validated mk{expected} target action space: " + ", ".join(
    f"{agent}={size}" for agent, size in sizes.items()
))
undersized = {agent: size for agent, size in sizes.items() if size < expected}
if undersized:
    print(
        f"Warning: mk{expected} is a maximum; this artifact retains fewer for "
        + ", ".join(f"{agent} ({size})" for agent, size in undersized.items())
    )
PY
fi

if ! is_true "$dry_run"; then
  for checkpoint in "${plan_checkpoints[@]}"; do
    if [[ ! -f "$task_dir/$checkpoint" ]]; then
      echo "Missing source checkpoint: $task_dir/$checkpoint" >&2
      exit 1
    fi
  done
fi

echo "========== Shared zero-shot WCCI mk${action_size} =========="
echo "Matched:             $matched"
echo "Skipped existing:    $skipped_existing"
echo "Planned submissions: ${#plan_labels[@]}"
echo "Save action trace:   $save_action_trace"
echo "Excluded nodes:      ${exclude_nodes:-none}"
echo "DRY_RUN:             $dry_run"
echo "================================================"

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
  echo "Validated ${#plan_labels[@]} mk${action_size} submission command(s)."
else
  echo "Submitted ${#plan_labels[@]} mk${action_size} zero-shot evaluation job(s)."
fi
