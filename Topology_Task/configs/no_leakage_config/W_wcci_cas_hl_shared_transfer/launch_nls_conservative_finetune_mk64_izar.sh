#!/usr/bin/env bash
# Conservative WCCI fine-tuning of the two strongest original NLS mk64 actors.
#
# The existing matching NLS fine-tune TOMLs provide the architecture and source
# checkpoint.  This launcher overrides only the target action space and the
# conservative optimization protocol, keeping the comparison easy to audit.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../../../.." && pwd)"
task_dir="$repo_dir/Topology_Task"
task_config_dir="configs/no_leakage_config/W_wcci_cas_hl_shared_transfer"

action_space_rel="outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk64.json"
action_space_abs="$task_dir/$action_space_rel"
dry_run="${DRY_RUN:-false}"
force_launch="${FORCE_LAUNCH:-false}"
exclude_nodes="${EXCLUDE_NODES:-${SBATCH_EXCLUDE:-}}"
run_suffix="${RUN_SUFFIX:-}"

actor_lr="${ACTOR_LR:-0.00003}"
critic_lr="${CRITIC_LR:-0.0001}"
total_timesteps="${TOTAL_TIMESTEPS:-5000000}"
update_epochs="${UPDATE_EPOCHS:-5}"
clip_coef="${CLIP_COEF:-0.1}"
target_kl="${TARGET_KL:-0.01}"
entropy_coef="${ENTROPY_COEF:-0.001}"
entropy_coef_final="${ENTROPY_COEF_FINAL:-0.0001}"
lr_final_frac="${LR_FINAL_FRAC:-0.1}"

if [[ ! "$run_suffix" =~ ^[A-Za-z0-9_.-]*$ ]]; then
  echo "RUN_SUFFIX may contain only letters, digits, dots, underscores, and hyphens." >&2
  exit 1
fi

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

# These answer complementary questions:
# - mean_f1_a0h0: strongest original difficult-chronic robustness at mk64.
# - tmean_f0_a0h0: strongest original overall/local-rho result at mk64.
variants=(
  mean_f1_a0h0
  tmean_f0_a0h0
)

if [[ ! -f "$action_space_abs" ]]; then
  if is_true "$dry_run"; then
    echo "Dry-run warning: missing local mk64 artifact $action_space_abs" >&2
  else
    echo "Missing mk64 reduced action space: $action_space_abs" >&2
    exit 1
  fi
else
  python - "$action_space_abs" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    reduction = json.load(handle)
sizes = {
    agent: int(payload.get("selected_action_size", -1))
    for agent, payload in sorted(reduction.get("agents", {}).items())
}
if not sizes or any(size != 64 for size in sizes.values()):
    raise SystemExit(f"Expected exactly 64 actions for every agent in {path}; got {sizes}")
print("Validated mk64 target action space: " + ", ".join(
    f"{agent}={size}" for agent, size in sizes.items()
))
PY
fi

selected=0
cd "$repo_dir"

echo "========== Conservative NLS WCCI mk64 fine-tuning =========="
echo "Actor LR:            $actor_lr"
echo "Critic LR:           $critic_lr"
echo "LR final fraction:   $lr_final_frac"
echo "Total timesteps:      $total_timesteps"
echo "PPO update epochs:    $update_epochs"
echo "Clip / target KL:     $clip_coef / $target_kl"
echo "Entropy start/final:  $entropy_coef / $entropy_coef_final"
echo "Encoder frozen:       false"
echo "Action head frozen:   false"
echo "Local-rho gate:       none during training"
echo "Excluded nodes:       ${exclude_nodes:-none}"
echo "DRY_RUN:              $dry_run"
echo "============================================================="

for variant in "${variants[@]}"; do
  label="ft64c_NLS_${variant}_s0${run_suffix}"
  if ! matches_filters "$label" "$@"; then
    continue
  fi
  selected=$((selected + 1))

  config_rel="${task_config_dir}/trcas_shared_NLS_${variant}_finetune_s0.toml"
  config_abs="$task_dir/$config_rel"
  if [[ ! -f "$config_abs" ]]; then
    echo "Missing base fine-tuning config: $config_abs" >&2
    exit 1
  fi

  checkpoint="$(sed -n 's/^transfer_encoder_checkpoint = "\(.*\)"/\1/p' "$config_abs")"
  if [[ -z "$checkpoint" ]]; then
    echo "Missing transfer_encoder_checkpoint in $config_abs" >&2
    exit 1
  fi
  if [[ ! -f "$task_dir/$checkpoint" ]] && ! is_true "$dry_run"; then
    echo "Missing source checkpoint: $task_dir/$checkpoint" >&2
    exit 1
  fi

  # Refuse an accidental duplicate checkpoint name unless explicitly asked.
  checkpoint_output="$task_dir/checkpoint/${label}.tar"
  best_checkpoint_output="$task_dir/checkpoint/best_test_${label}.tar"
  final_checkpoint_output="$task_dir/checkpoint/final_${label}.tar"
  if {
    [[ -f "$checkpoint_output" ]] ||
      [[ -f "$best_checkpoint_output" ]] ||
      [[ -f "$final_checkpoint_output" ]]
  } && ! is_true "$force_launch"; then
    echo "Skip existing run checkpoint for $label (FORCE_LAUNCH=true to relaunch)"
    continue
  fi

  command=(sbatch)
  if [[ -n "$exclude_nodes" ]]; then
    command+=("--exclude=$exclude_nodes")
  fi
  command+=(
    "--job-name=$label"
    job_izar.sh
    "$config_rel"
    --exp-tag "$label"
    --reduced-action-space "$action_space_rel"
    --total-timesteps "$total_timesteps"
    --actor-lr "$actor_lr"
    --critic-lr "$critic_lr"
    --anneal-lr true
    --lr-final-frac "$lr_final_frac"
    --update-epochs "$update_epochs"
    --clip-coef "$clip_coef"
    --target-kl "$target_kl"
    --entropy-coef "$entropy_coef"
    --entropy-coef-final "$entropy_coef_final"
    --transfer-freeze-encoder false
    --transfer-action-head true
    --transfer-freeze-action-head false
    --intervention-gate false
    --eval-action-heuristic none
  )

  if is_true "$dry_run"; then
    printf 'Would submit:'
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
done

if [[ $selected -eq 0 ]]; then
  echo "No conservative fine-tuning run matched filters: $*" >&2
  echo "Examples: mean_f1_a0h0  tmean_f0_a0h0" >&2
  exit 1
fi

if is_true "$dry_run"; then
  echo "Validated $selected conservative fine-tuning command(s)."
else
  echo "Processed $selected conservative fine-tuning run(s)."
fi
