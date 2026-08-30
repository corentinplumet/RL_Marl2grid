#!/usr/bin/env bash
# Compare four WCCI actor families with policy-guided top-k solver look-ahead.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TASK_DIR="${REPO_DIR}/Topology_Task"
JOB_SCRIPT="${TASK_DIR}/full_test_eval/job_full_test_eval.sh"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
TOP_K="${TOP_K:-3}"
RHO_THRESHOLD="${RHO_THRESHOLD:-0.95}"
RHO_TAG="${RHO_THRESHOLD//./}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_RESULTS="${FORCE_RESULTS:-false}"

MK64_REL="outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk64.json"
MK64_ABS="${TASK_DIR}/${MK64_REL}"
RESULT_ROOT="${TASK_DIR}/outputs/full_test_eval/solver_topk${TOP_K}_rho${RHO_TAG}_ep${EVAL_EPISODES}"
ACTION_ROOT="${TASK_DIR}/outputs/full_test_eval_actions/solver_topk${TOP_K}_rho${RHO_TAG}_ep${EVAL_EPISODES}"

if [[ ! -f "${JOB_SCRIPT}" ]]; then
    echo "Missing job script: ${JOB_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${MK64_ABS}" ]]; then
    echo "Missing mk64 reduced action space: ${MK64_ABS}" >&2
    exit 1
fi

submit_one() {
    local label="$1"
    local checkpoint="$2"
    local obs_normalization="$3"
    shift 3

    local checkpoint_abs="${TASK_DIR}/${checkpoint}"
    local output_json="${RESULT_ROOT}/${label}.json"
    local action_dir="${ACTION_ROOT}/${label}"
    if [[ ! -f "${checkpoint_abs}" ]]; then
        echo "Missing checkpoint: ${checkpoint_abs}" >&2
        return 1
    fi
    if [[ -f "${output_json}" && "${FORCE_RESULTS}" != "true" ]]; then
        echo "SKIP existing result: ${output_json}"
        return 0
    fi

    local cmd=(
        sbatch
        --job-name="slv_${label}"
        "${JOB_SCRIPT}"
        --checkpoint "${checkpoint}"
        --split test
        --split-chronics true
        --eval-all-split-chronics false
        --eval-episodes "${EVAL_EPISODES}"
        --deterministic-eval true
        --eval-action-heuristic none
        --eval-solver-lookahead true
        --eval-solver-top-k "${TOP_K}"
        --eval-solver-rho-threshold "${RHO_THRESHOLD}"
        --eval-solver-include-noop true
        --obs-normalization "${obs_normalization}"
        --device cpu
        --n-threads 4
        --progress true
        --save-action-summary true
        --save-action-trace false
        --action-log-dir "${action_dir}"
        --output-json "${output_json}"
        "$@"
    )

    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" != "true" ]]; then
        "${cmd[@]}"
    fi
}

submit_one \
    "wmlp_mk64_s0" \
    "checkpoint/no_leak/wmlp/wmlp_mk64_s0.tar" \
    "require"

submit_one \
    "wsc_het_gbi_lbi_mk64_s0" \
    "checkpoint/no_leak/wsc/wsc_het_gbi_lbi_mk64_s0.tar" \
    "require"

submit_one \
    "zero_shot_NLS_mean_f1_a0h0_mk64" \
    "checkpoint/no_leak/shared/NL_cas_hl_scaled_shared_izar/best_test_cas_hl_NLS_izar_shared_mean_f1_a0h0_s0.tar" \
    "disable" \
    --target-env-id bus36_wcci_nomaint \
    --target-reduced-action-space "${MK64_REL}"

submit_one \
    "finetune_NLS_tmean_f0_a0h1_mk64" \
    "checkpoint/no_leak/shared/NL_cas_hl_scaled_izar_finetune/ft64c_NLS_tmean_f0_a0h1_s0.tar" \
    "require"
