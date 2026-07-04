#!/usr/bin/env bash
# EPFL JED SLURM launcher for filtering action-outcome datasets by max rho.
#
# Example:
#   SOURCE_DATASET=outputs/teacher_student_datasets/wcci_full2048a_90_v3 \
#   OUTPUT_DIR=outputs/teacher_student_datasets/wcci_full2048a_95_v3 \
#   RHO_THRESHOLD=0.95 \
#   OVERWRITE=false \
#   sbatch Topology_Task/teacher_student/job_filter_action_outcomes_jed.sh
#SBATCH --job-name=filter_outcomes
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=Topology_Task/teacher_student/slurm-%x-%j.out
#SBATCH --error=Topology_Task/teacher_student/slurm-%x-%j.err

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  SOURCE_DATASET=outputs/teacher_student_datasets/wcci_full2048a_90_v3 \
  OUTPUT_DIR=outputs/teacher_student_datasets/wcci_full2048a_95_v3 \
  RHO_THRESHOLD=0.95 \
  sbatch Topology_Task/teacher_student/job_filter_action_outcomes_jed.sh

This does not rerun Grid2Op simulations. It copies the action-outcome dataset
layout and keeps only rows whose original risky state satisfies:

  global_max_rho_before > RHO_THRESHOLD

Common overrides:
  SOURCE_DATASET=outputs/teacher_student_datasets/wcci_full2048a_90_v3
  OUTPUT_DIR=outputs/teacher_student_datasets/wcci_full2048a_95_v3
  RHO_THRESHOLD=0.95
  INCLUSIVE=false       # true means >= threshold instead of > threshold
  OVERWRITE=false
  COMPRESS=true
  MAX_PARTS=
  MAX_SHARDS=
  PROGRESS_EVERY_SHARDS=25

Additional arguments passed to this script are forwarded to
filter_action_outcome_dataset.py.
EOF
    exit 0
fi

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

SOURCE_DATASET="${SOURCE_DATASET:-outputs/teacher_student_datasets/wcci_full2048a_90_v3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/teacher_student_datasets/wcci_full2048a_95_v3}"
RHO_THRESHOLD="${RHO_THRESHOLD:-0.95}"
INCLUSIVE="${INCLUSIVE:-false}"
OVERWRITE="${OVERWRITE:-false}"
COMPRESS="${COMPRESS:-true}"
MAX_PARTS="${MAX_PARTS:-}"
MAX_SHARDS="${MAX_SHARDS:-}"
PROGRESS_EVERY_SHARDS="${PROGRESS_EVERY_SHARDS:-25}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${TASK_DIR}"

echo "Launching action-outcome dataset filter on JED"
echo "Conda env: ${CONDA_ENV}"
echo "Source dataset: ${SOURCE_DATASET}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Rho threshold: ${RHO_THRESHOLD}"
echo "Inclusive: ${INCLUSIVE}"
echo "Overwrite: ${OVERWRITE}"
echo "Compress: ${COMPRESS}"
echo "Max parts: ${MAX_PARTS:-all}"
echo "Max shards per part: ${MAX_SHARDS:-all}"
echo "Progress every shards: ${PROGRESS_EVERY_SHARDS}"
echo "Extra args: $*"

filter_args=(
    --source-dataset "${SOURCE_DATASET}"
    --output-dir "${OUTPUT_DIR}"
    --rho-threshold "${RHO_THRESHOLD}"
    --inclusive "${INCLUSIVE}"
    --overwrite "${OVERWRITE}"
    --compress "${COMPRESS}"
    --progress-every-shards "${PROGRESS_EVERY_SHARDS}"
)

if [ -n "${MAX_PARTS}" ]; then
    filter_args+=(--max-parts "${MAX_PARTS}")
fi

if [ -n "${MAX_SHARDS}" ]; then
    filter_args+=(--max-shards "${MAX_SHARDS}")
fi

python -u teacher_student/filter_action_outcome_dataset.py "${filter_args[@]}" "$@"
