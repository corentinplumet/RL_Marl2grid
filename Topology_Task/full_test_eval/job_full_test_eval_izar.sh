#!/usr/bin/env bash
#SBATCH --job-name=full_test_eval
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=Topology_Task/full_test_eval/slurm-%x-%j.out
#SBATCH --error=Topology_Task/full_test_eval/slurm-%x-%j.err

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  python Topology_Task/full_test_eval/evaluate_checkpoint.py --list-checkpoints
  sbatch Topology_Task/full_test_eval/job_full_test_eval_izar.sh --checkpoint checkpoint/best_test_run.tar
  sbatch Topology_Task/full_test_eval/job_full_test_eval_izar.sh --model run_name --step 15000000

Environment:
  CONDA_ENV       Conda env to activate. Default: marl2grid
  CONDA_BASE      Conda installation path, if conda is not on PATH.
  REPO_DIR        Repository root. Default: SLURM_SUBMIT_DIR or current directory.

Notes:
  This Izar wrapper intentionally avoids hardcoding a partition/QOS. If your
  Izar account requires one, pass it at submission time, for example:
    sbatch --partition=<partition> --qos=<qos> Topology_Task/full_test_eval/job_full_test_eval_izar.sh ...
EOF
    exit 0
fi

echo "Full-test evaluation job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"

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

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${TASK_DIR}"

# Avoid accidental user-site packages shadowing the conda environment on Izar.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

echo "Using conda env: ${CONDA_ENV}"
echo "Python: $(command -v python)"
echo "Arguments: $*"
python -u full_test_eval/evaluate_checkpoint.py "$@"

echo "Full-test evaluation job ${SLURM_JOB_ID:-local} finished at $(date)"
