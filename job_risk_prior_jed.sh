#!/usr/bin/env bash
#SBATCH --job-name=risk_prior_collect
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=24:00:00
#SBATCH --output=Topology_Task/slurm-%x-%j.out
#SBATCH --error=Topology_Task/slurm-%x-%j.err

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  sbatch job_risk_prior_jed.sh
  sbatch job_risk_prior_jed.sh --max-examples 200 --max-hazard-states 5 --actions-per-agent 8
  sbatch --array=0-2 job_risk_prior_jed.sh --max-examples 20000 --actions-per-agent 32

Environment:
  REPO_DIR              Repository root. Default: SLURM_SUBMIT_DIR or pwd.
  CONDA_ENV            Conda env to activate. Default: marl2grid.
  CONDA_BASE           Conda installation path, if conda is not on PATH.
  ENV_ID               Grid2Op env id. Default: bus14.
  SEED                 Collector seed. Default: SLURM_ARRAY_TASK_ID or 0.
  MAX_EXAMPLES         Labelled examples to collect. Default: 20000.
  MAX_HAZARD_STATES    Hazard states to label. Default: 1000.
  ACTIONS_PER_AGENT    Local actions sampled per agent. Default: 32.
  RHO_THRESHOLD        Hazard threshold. Default: 0.90.
  OUTPUT               Output .npz path. Default includes env, seed, and job id.

Any command-line args after the script name are forwarded to
python -m risk_prior.collect_dataset and override duplicate defaults.
EOF
    exit 0
fi

echo "Risk-prior collection job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"

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

ENV_ID="${ENV_ID:-bus14}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
MAX_EXAMPLES="${MAX_EXAMPLES:-20000}"
MAX_HAZARD_STATES="${MAX_HAZARD_STATES:-1000}"
ACTIONS_PER_AGENT="${ACTIONS_PER_AGENT:-32}"
RHO_THRESHOLD="${RHO_THRESHOLD:-0.90}"
JOB_ID="${SLURM_JOB_ID:-local}"
OUTPUT="${OUTPUT:-${REPO_DIR}/outputs/risk_prior/${ENV_ID}_seed${SEED}_job${JOB_ID}.npz}"

mkdir -p "$(dirname "${OUTPUT}")"

echo "Using conda env: ${CONDA_ENV}"
echo "Using env id: ${ENV_ID}"
echo "Using seed: ${SEED}"
echo "Writing dataset to: ${OUTPUT}"

python -u -m risk_prior.collect_dataset \
    --env-id "${ENV_ID}" \
    --seed "${SEED}" \
    --rho-threshold "${RHO_THRESHOLD}" \
    --max-examples "${MAX_EXAMPLES}" \
    --max-hazard-states "${MAX_HAZARD_STATES}" \
    --actions-per-agent "${ACTIONS_PER_AGENT}" \
    --output "${OUTPUT}" \
    "$@"

echo "Risk-prior collection job ${SLURM_JOB_ID:-local} finished at $(date)"
