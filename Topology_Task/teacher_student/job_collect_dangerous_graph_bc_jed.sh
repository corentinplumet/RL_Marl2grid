#!/usr/bin/env bash
# EPFL JED launcher for dangerous-state graph behavior-cloning collection.
# Submit from the repository root.
#SBATCH --job-name=collect_danger_bc
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-12:00:00
#SBATCH --output=Topology_Task/teacher_student/slurm-%x-%j.out
#SBATCH --error=Topology_Task/teacher_student/slurm-%x-%j.err

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  sbatch Topology_Task/teacher_student/job_collect_dangerous_graph_bc_jed.sh \
    --checkpoint checkpoint/final_ft64c_NLS_mean_f1_a0h0_s0.tar \
    --label-action-space mk64=outputs/.../reduced_action_space_..._mk64.json \
    --label-action-space mk128=outputs/.../reduced_action_space_..._mk128.json \
    --label-action-space mk256=outputs/.../reduced_action_space_..._mk256.json \
    --output-dir outputs/teacher_student_datasets/wcci_danger090_multi

Environment:
  CONDA_ENV   Conda environment name. Default: marl2grid
  CONDA_BASE  Conda installation root if conda is not on PATH.
  REPO_DIR    Repository root. Default: SLURM_SUBMIT_DIR or current directory.
EOF
    exit 0
fi

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TASK_DIR="${REPO_DIR}/Topology_Task"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-marl2grid}}"

if [[ ! -d "${TASK_DIR}" ]]; then
    echo "Could not find ${TASK_DIR}." >&2
    echo "Submit from the repository root, or set REPO_DIR." >&2
    exit 1
fi

if [[ -n "${CONDA_BASE:-}" ]]; then
    :
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="${HOME}/miniconda3"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="${HOME}/miniforge3"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="${HOME}/anaconda3"
else
    echo "Could not find conda; set CONDA_BASE." >&2
    exit 1
fi

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "Could not find ${CONDA_BASE}/etc/profile.d/conda.sh." >&2
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${TASK_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Dangerous graph-BC collection job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Conda env: ${CONDA_ENV}"
echo "Arguments: $*"
python -u teacher_student/collect_dangerous_graph_bc_dataset.py "$@"

