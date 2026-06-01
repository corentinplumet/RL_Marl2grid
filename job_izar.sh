#!/bin/bash
# IZAR / EPFL SLURM launcher.
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --job-name=marl2grid_izar
#SBATCH --output=routput_jobs_izar/job_out_%j.log
#SBATCH --error=routput_jobs_izar/job_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=23:10:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  sbatch job_izar.sh
  sbatch job_izar.sh configs/gine/shared_gine_a1_no_entropy_decay_s0_det.toml
  sbatch job_izar.sh configs/gine/shared_gine_a1_no_entropy_decay_s0_det.toml --deterministic-eval false
  sbatch --array=0-7 job_izar.sh configs/gine/shared_gine_a1_no_entropy_decay_s0_det.toml

Environment:
  IZAR_CONFIG     Default config path. Default: configs/gine/shared_gine_a1_no_entropy_decay_s0_det.toml
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

CONFIG="${IZAR_CONFIG:-configs/gine/shared_gine_a1_no_entropy_decay_s0_det.toml}"
if [[ $# -gt 0 && "${1:0:2}" != "--" ]]; then
    CONFIG="$1"
    shift
fi

# Izar's gpu partition reserves a GPU, so default the training config to CUDA.
export CUDA="${CUDA:-true}"

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
