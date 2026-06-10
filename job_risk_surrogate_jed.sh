#!/usr/bin/env bash
#SBATCH --job-name=risk_surrogate
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=24:00:00
#SBATCH --output=Topology_Task/slurm-%x-%j.out
#SBATCH --error=Topology_Task/slurm-%x-%j.err

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  sbatch job_risk_surrogate_jed.sh --dataset /path/to/phase1_dataset.npz
  sbatch job_risk_surrogate_jed.sh --dataset /path/to/phase1_dataset.npz --epochs 80 --batch-size 1024

Environment:
  REPO_DIR        Repository root. Default: SLURM_SUBMIT_DIR or pwd.
  CONDA_ENV       Conda env to activate. Default: marl2grid.
  CONDA_BASE      Conda installation path, if conda is not on PATH.
  DATASET         Optional Phase 1 .npz path. Command-line --dataset also works.
  RUN_NAME        Optional output/W&B run name.
  OUTPUT_DIR      Output model directory. Default: <repo>/outputs/risk_prior_models.
  EPOCHS          Default: 50.
  BATCH_SIZE      Default: 512.
  LR              Default: 3e-4.
  DEVICE          Default: auto.
  WANDB           true/false. Default: false.
  WANDB_PROJECT   Default: risk_prior.
  WANDB_MODE      Optional W&B mode, e.g. offline.

Any command-line args after the script name are forwarded to
python -m risk_prior.train_surrogate and override duplicate defaults.
EOF
    exit 0
fi

echo "Risk-surrogate training job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"

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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

DATASET_ARGS=()
if [ -n "${DATASET:-}" ]; then
    DATASET_ARGS=(--dataset "${DATASET}")
fi

RUN_NAME_ARGS=()
if [ -n "${RUN_NAME:-}" ]; then
    RUN_NAME_ARGS=(--run-name "${RUN_NAME}")
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/risk_prior_models}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-3e-4}"
DEVICE="${DEVICE:-auto}"
WANDB="${WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-risk_prior}"
WANDB_MODE="${WANDB_MODE:-}"

echo "Using conda env: ${CONDA_ENV}"
echo "Using output dir: ${OUTPUT_DIR}"
echo "Using epochs: ${EPOCHS}"
echo "Using batch size: ${BATCH_SIZE}"
echo "Using lr: ${LR}"
echo "Using device: ${DEVICE}"

python -u -m risk_prior.train_surrogate \
    "${DATASET_ARGS[@]}" \
    "${RUN_NAME_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --device "${DEVICE}" \
    --wandb "${WANDB}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-mode "${WANDB_MODE}" \
    "$@"

echo "Risk-surrogate training job ${SLURM_JOB_ID:-local} finished at $(date)"
