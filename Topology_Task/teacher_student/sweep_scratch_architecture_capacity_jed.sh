#!/usr/bin/env bash
# Submit controlled 300-epoch, one-shard scratch-capacity ablations on JED.
# Run from the RL_Marl2grid repository root.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
TASK_DIR="${REPO_DIR}/Topology_Task"
JOB_SCRIPT="Topology_Task/teacher_student/job_train_dangerous_graph_bc_jed.sh"
DRY_RUN="${DRY_RUN:-false}"

if [[ ! -d "${TASK_DIR}" || ! -f "${REPO_DIR}/${JOB_SCRIPT}" ]]; then
    echo "Run from the RL_Marl2grid repository root, or set REPO_DIR." >&2
    exit 1
fi

COMMON_ARGS=(
    --dataset outputs/teacher_student_datasets/wcci_nomaint_danger090_mk32_rank_full
    --checkpoint checkpoint/no_leak/shared/NL_cas_hl_scaled_shared_izar/best_test_cas_hl_NLS_izar_shared_mean_f1_a0h0_s0.tar
    --initialization scratch
    --objective utility_regression
    --max-shards 1
    --epochs 300
    --batch-size 64
    --lr 0.00001
    --weight-decay 0
    --max-grad-norm 1.0
    --utility-weight 1.0
    --utility-huber-beta 0.5
    --unsafe-weight 0.5
    --unsafe-margin 1.0
    --min-trainable-candidates 2
    --aux-intervention-loss true
    --aux-weight 0.25
    --distill-weight 0
    --validation-chronic-frac 0
    --eval-batches 0
    --agent-update-mode mixed
    --unshare-candidate-scorer false
    --freeze-encoder false
    --seed 0
    --device cpu
    --n-threads 8
)

submit_variant() {
    local job_name="$1"
    local output_stem="$2"
    shift 2
    local command=(
        sbatch
        --job-name="${job_name}"
        "${JOB_SCRIPT}"
        "${COMMON_ARGS[@]}"
        --output "checkpoint/dangerous_graph_bc/${output_stem}.tar"
        --exp-tag "${output_stem}"
        "$@"
    )
    printf 'Submitting %-18s -> %s.tar\n' "${job_name}" "${output_stem}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        printf '  %q' "${command[@]}"
        printf '\n'
    else
        (cd "${REPO_DIR}" && "${command[@]}")
    fi
}

# 1. Exact object/source/target-busbar action-delta tokens, keeping mp2.
submit_variant \
    rank32_delta \
    rankcap32_1shard_scratch_delta_shared_NLS_mean_f1_a0h0_s0 \
    --scratch-action-delta-encoder true

# 2. Four-hop message passing protected by residual and multi-scale JK paths.
submit_variant \
    rank32_resjk4 \
    rankcap32_1shard_scratch_resjk4_shared_NLS_mean_f1_a0h0_s0 \
    --scratch-gnn-layers 4 \
    --scratch-gnn-residual true \
    --scratch-gnn-jumping-knowledge concat

# 3. Complementary typical-state (mean) and bottleneck-state (max) pooling.
submit_variant \
    rank32_dualread \
    rankcap32_1shard_scratch_dualread_shared_NLS_mean_f1_a0h0_s0 \
    --scratch-gnn-readout-aggr energized_mean_max

# Combined architecture. This is the candidate to carry to MAPPO if it wins.
submit_variant \
    rank32_all3 \
    rankcap32_1shard_scratch_delta_resjk4_dual_shared_NLS_mean_f1_a0h0_s0 \
    --scratch-action-delta-encoder true \
    --scratch-gnn-layers 4 \
    --scratch-gnn-residual true \
    --scratch-gnn-jumping-knowledge concat \
    --scratch-gnn-readout-aggr energized_mean_max
