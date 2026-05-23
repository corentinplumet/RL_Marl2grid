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

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="${REPO_DIR}/Topology_Task"

# The conda env in Topology_Task/conda_env.yml is named "marl2grid".
CONDA_ENV="${CONDA_ENV:-marl2grid}"

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
mkdir -p checkpoint

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export WANDB_DIR="${WANDB_DIR:-${TASK_DIR}/wandb}"

ALG="${ALG:-MAPPO}"
ENV_ID="${ENV_ID:-bus14}"
ACTION_TYPE="${ACTION_TYPE:-topology}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-25000000}"
CHECKPOINT="${CHECKPOINT:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-Grid2Op}"
WANDB_ENTITY="${WANDB_ENTITY:-ismail-nejjar}"
WANDB_MODE="${WANDB_MODE:-online}"
ACTION_TIME_LIMIT="${ACTION_TIME_LIMIT:-1300}"
TORCH_THREADS="${TORCH_THREADS:-4}"

if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    DEFAULT_N_ENVS=$((SLURM_CPUS_PER_TASK - TORCH_THREADS))
    if [ "${DEFAULT_N_ENVS}" -lt 1 ]; then
        DEFAULT_N_ENVS=1
    fi
else
    DEFAULT_N_ENVS=20
fi

N_ENVS="${N_ENVS:-${DEFAULT_N_ENVS}}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Training configuration:"
echo "  conda env       : ${CONDA_ENV}"
echo "  algorithm       : ${ALG}"
echo "  env             : ${ENV_ID}"
echo "  seed            : ${SEED}"
echo "  total timesteps : ${TOTAL_TIMESTEPS}"
echo "  n envs          : ${N_ENVS}"
echo "  torch threads   : ${TORCH_THREADS}"
echo "  wandb mode      : ${WANDB_MODE}"

cmd=(
    python -u main.py
    --alg "${ALG}"
    --env-id "${ENV_ID}"
    --action-type "${ACTION_TYPE}"
    --seed "${SEED}"
    --total-timesteps "${TOTAL_TIMESTEPS}"
    --n-envs "${N_ENVS}"
    --n-threads "${TORCH_THREADS}"
    --cuda False
    --checkpoint "${CHECKPOINT}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-entity "${WANDB_ENTITY}"
    --wandb-mode "${WANDB_MODE}"
    --time-limit "${ACTION_TIME_LIMIT}"
)

if [ -n "${EXTRA_ARGS}" ]; then
    # Deliberately split simple flag strings such as: --difficulty 1 --track False
    # shellcheck disable=SC2206
    extra_argv=(${EXTRA_ARGS})
else
    extra_argv=()
fi

echo "Running: ${cmd[*]} ${extra_argv[*]} $*"
srun "${cmd[@]}" "${extra_argv[@]}" "$@"

echo "Job ${SLURM_JOB_ID:-local} finished at $(date)"
