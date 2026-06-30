#!/usr/bin/env bash
# EPFL JED SLURM launcher for reducing sharded brute-force action outcomes.
# Submit after the collection array completes:
#   DATASET_ROOT=outputs/teacher_student_datasets/bus36_bruteforce_rho090 \
#   sbatch Topology_Task/teacher_student/job_reduce_action_space_jed.sh
#SBATCH --job-name=reduce_actions
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

DATASET_ROOT="${DATASET_ROOT:-outputs/teacher_student_datasets/bus36_bruteforce_rho090}"
ACTION_REDUCTION_TOP_K="${ACTION_REDUCTION_TOP_K:-208}"
ACTION_REDUCTION_MIN_COUNT="${ACTION_REDUCTION_MIN_COUNT:-1}"
ACTION_REDUCTION_METRIC="${ACTION_REDUCTION_METRIC:-delta_vs_do_nothing}"
ACTION_REDUCTION_METHOD="${ACTION_REDUCTION_METHOD:-best_per_state}"
ACTION_REDUCTION_REQUIRE_IMPROVEMENT="${ACTION_REDUCTION_REQUIRE_IMPROVEMENT:-true}"
ACTION_REDUCTION_IMPROVEMENT_TOLERANCE="${ACTION_REDUCTION_IMPROVEMENT_TOLERANCE:-1e-3}"

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
printf "  %s\n" "${part_dirs[@]}"

python -u teacher_student/reduce_action_space_from_outcomes.py \
    --dataset "${part_dirs[@]}" \
    --output "${DATASET_ROOT}/metadata/reduced_action_space.json" \
    --top-k "${ACTION_REDUCTION_TOP_K}" \
    --min-count "${ACTION_REDUCTION_MIN_COUNT}" \
    --metric "${ACTION_REDUCTION_METRIC}" \
    --selection-method "${ACTION_REDUCTION_METHOD}" \
    --require-improvement "${ACTION_REDUCTION_REQUIRE_IMPROVEMENT}" \
    --improvement-tolerance "${ACTION_REDUCTION_IMPROVEMENT_TOLERANCE}"
