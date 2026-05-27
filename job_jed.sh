#!/usr/bin/env bash
#SBATCH --job-name=marl2grid_jed
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=7000M
#SBATCH --time=24:00:00
#SBATCH --output=Topology_Task/slurm-%x-%j.out
#SBATCH --error=Topology_Task/slurm-%x-%j.err

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  sbatch job_jed.sh
  sbatch job_jed.sh configs/old_training.toml
  sbatch job_jed.sh configs/old_training.toml --deterministic-eval false
  sbatch --array=0-7 job_jed.sh configs/old_training.toml

Environment:
  JED_CONFIG      Default config path. Default: configs/old_training.toml
  CONDA_ENV       Conda env to activate. Default: marl2grid
  CONDA_BASE      Conda installation path, if conda is not on PATH.
  DRY_RUN=true    Print the resolved command without launching training.

Environment variables can override [args] values from the TOML by using the
uppercase argument name, for example N_ENVS, SEED, TOTAL_TIMESTEPS, or
TORCH_THREADS for n_threads.

Config paths are resolved relative to Topology_Task.
EOF
    exit 0
fi

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TASK_DIR="${REPO_DIR}/Topology_Task"

if [ ! -d "${TASK_DIR}" ]; then
    echo "Could not find ${TASK_DIR}." >&2
    echo "Submit from the repository root, or set REPO_DIR=/path/to/RL_Marl2grid." >&2
    exit 1
fi

# The conda env in Topology_Task/conda_env.yml is named "marl2grid".
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-marl2grid}}"

CONFIG="${JED_CONFIG:-configs/old_training.toml}"
if [[ $# -gt 0 && "${1:0:2}" != "--" ]]; then
    CONFIG="$1"
    shift
fi

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

run_config_args=()
if [[ "${DRY_RUN:-false}" == "true" ]]; then
    run_config_args+=(--dry-run)
fi

echo "Using conda env: ${CONDA_ENV}"
echo "Using config: ${CONFIG}"
python -u run_from_config.py "${run_config_args[@]}" "${CONFIG}" "$@"

echo "Job ${SLURM_JOB_ID:-local} finished at $(date)"
