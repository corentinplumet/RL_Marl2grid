#!/usr/bin/env bash
# EPFL JED SLURM launcher for reducing sharded brute-force action outcomes.
# Submit after the collection array completes:
#   DATASET_ROOT=outputs/teacher_student_datasets/bus36_bruteforce_rho090 \
#   sbatch Topology_Task/teacher_student/job_reduce_action_space_jed.sh
# Or use a config file:
#   sbatch Topology_Task/teacher_student/job_reduce_action_space_jed.sh \
#     Topology_Task/teacher_student/reduction_configs/bus36_improvement_rate_k208.env
#SBATCH --job-name=reduce_actions
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=Topology_Task/teacher_student/slurm-%x-%j.out
#SBATCH --error=Topology_Task/teacher_student/slurm-%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TASK_DIR="${REPO_DIR}/Topology_Task"

if [ ! -d "${TASK_DIR}" ]; then
    echo "Could not find ${TASK_DIR}." >&2
    echo "Submit from the repository root, or set REPO_DIR=/path/to/RL_Marl2grid." >&2
    exit 1
fi

CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-marl2grid}}"

if [ -n "${CONDA_BASE:-}" ]; then
    :
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="${HOME}/miniconda3"
elif [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="${HOME}/miniforge3"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="${HOME}/anaconda3"
else
    echo "Could not find conda. Load your conda module or set CONDA_ENV/CONDA_BASE before submitting." >&2
    exit 1
fi

if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "Could not find ${CONDA_BASE}/etc/profile.d/conda.sh." >&2
    exit 1
fi

REDUCTION_CONFIG="${1:-}"
if [ -n "${REDUCTION_CONFIG}" ]; then
    if [[ "${REDUCTION_CONFIG}" == --* ]]; then
        echo "Unexpected option '${REDUCTION_CONFIG}'. Pass a config file path or set environment variables." >&2
        exit 1
    fi
    if [ -f "${REDUCTION_CONFIG}" ]; then
        REDUCTION_CONFIG_PATH="${REDUCTION_CONFIG}"
    elif [ -f "${REPO_DIR}/${REDUCTION_CONFIG}" ]; then
        REDUCTION_CONFIG_PATH="${REPO_DIR}/${REDUCTION_CONFIG}"
    elif [ -f "${TASK_DIR}/${REDUCTION_CONFIG}" ]; then
        REDUCTION_CONFIG_PATH="${TASK_DIR}/${REDUCTION_CONFIG}"
    else
        echo "Reduction config file does not exist: ${REDUCTION_CONFIG}" >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "${REDUCTION_CONFIG_PATH}"
    shift
else
    REDUCTION_CONFIG_PATH=""
fi

DATASET_ROOT="${DATASET_ROOT:-outputs/teacher_student_datasets/bus36_bruteforce_rho090}"
ACTION_REDUCTION_TOP_K="${ACTION_REDUCTION_TOP_K:-208}"
ACTION_REDUCTION_TOP_K_BY_AGENT="${ACTION_REDUCTION_TOP_K_BY_AGENT:-}"
ACTION_REDUCTION_MIN_COUNT="${ACTION_REDUCTION_MIN_COUNT:-1}"
ACTION_REDUCTION_METRIC="${ACTION_REDUCTION_METRIC:-delta_vs_do_nothing}"
ACTION_REDUCTION_METHOD="${ACTION_REDUCTION_METHOD:-best_per_state}"
ACTION_REDUCTION_REQUIRE_IMPROVEMENT="${ACTION_REDUCTION_REQUIRE_IMPROVEMENT:-true}"
ACTION_REDUCTION_IMPROVEMENT_TOLERANCE="${ACTION_REDUCTION_IMPROVEMENT_TOLERANCE:-1e-3}"
ACTION_REDUCTION_PROGRESS_EVERY_SHARDS="${ACTION_REDUCTION_PROGRESS_EVERY_SHARDS:-25}"
ACTION_REDUCTION_NAME="${ACTION_REDUCTION_NAME:-}"
ACTION_REDUCTION_OUTPUT="${ACTION_REDUCTION_OUTPUT:-}"

if [ -z "${ACTION_REDUCTION_OUTPUT}" ]; then
    if [ -n "${ACTION_REDUCTION_NAME}" ]; then
        ACTION_REDUCTION_OUTPUT="${DATASET_ROOT}/metadata/reduced_action_space_${ACTION_REDUCTION_NAME}.json"
    elif [ -n "${ACTION_REDUCTION_TOP_K_BY_AGENT}" ]; then
        ACTION_REDUCTION_OUTPUT="${DATASET_ROOT}/metadata/reduced_action_space_${ACTION_REDUCTION_METHOD}_${ACTION_REDUCTION_METRIC}_custom_topk_min${ACTION_REDUCTION_MIN_COUNT}.json"
    else
        ACTION_REDUCTION_OUTPUT="${DATASET_ROOT}/metadata/reduced_action_space_${ACTION_REDUCTION_METHOD}_${ACTION_REDUCTION_METRIC}_k${ACTION_REDUCTION_TOP_K}_min${ACTION_REDUCTION_MIN_COUNT}.json"
    fi
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${TASK_DIR}"

shopt -s nullglob
part_dirs=("${DATASET_ROOT}"/parts/part_*)
shopt -u nullglob

if [ "${#part_dirs[@]}" -eq 0 ]; then
    echo "No part datasets found under ${DATASET_ROOT}/parts/part_*." >&2
    exit 1
fi

echo "Reducing ${#part_dirs[@]} part datasets from ${DATASET_ROOT}"
echo "Config: ${REDUCTION_CONFIG_PATH:-none}"
echo "Method: ${ACTION_REDUCTION_METHOD}"
echo "Metric: ${ACTION_REDUCTION_METRIC}"
echo "Top-k: ${ACTION_REDUCTION_TOP_K}"
echo "Top-k by agent: ${ACTION_REDUCTION_TOP_K_BY_AGENT:-global top-k}"
echo "Min count: ${ACTION_REDUCTION_MIN_COUNT}"
echo "Require improvement: ${ACTION_REDUCTION_REQUIRE_IMPROVEMENT}"
echo "Improvement tolerance: ${ACTION_REDUCTION_IMPROVEMENT_TOLERANCE}"
echo "Progress every shards: ${ACTION_REDUCTION_PROGRESS_EVERY_SHARDS}"
echo "Output: ${ACTION_REDUCTION_OUTPUT}"
printf "  %s\n" "${part_dirs[@]}"

reducer_args=(
    --dataset "${part_dirs[@]}"
    --output "${ACTION_REDUCTION_OUTPUT}"
    --top-k "${ACTION_REDUCTION_TOP_K}"
    --min-count "${ACTION_REDUCTION_MIN_COUNT}"
    --metric "${ACTION_REDUCTION_METRIC}"
    --selection-method "${ACTION_REDUCTION_METHOD}"
    --require-improvement "${ACTION_REDUCTION_REQUIRE_IMPROVEMENT}"
    --improvement-tolerance "${ACTION_REDUCTION_IMPROVEMENT_TOLERANCE}"
    --progress-every-shards "${ACTION_REDUCTION_PROGRESS_EVERY_SHARDS}"
)

if [ -n "${ACTION_REDUCTION_TOP_K_BY_AGENT}" ]; then
    reducer_args+=(--top-k-by-agent "${ACTION_REDUCTION_TOP_K_BY_AGENT}")
fi

python -u teacher_student/reduce_action_space_from_outcomes.py "${reducer_args[@]}"
