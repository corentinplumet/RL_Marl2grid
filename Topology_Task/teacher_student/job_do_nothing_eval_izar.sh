#!/bin/bash
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --job-name=do_nothing_eval
#SBATCH --output=Topology_Task/teacher_student/slurm-%x-%j.out
#SBATCH --error=Topology_Task/teacher_student/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --qos=long
#SBATCH --gres=gpu:1


set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  ENV_ID=bus36_wcci SPLIT=all MAX_EPISODES=2880 \
  sbatch Topology_Task/teacher_student/job_do_nothing_eval_izar.sh

Common overrides:
  ENV_ID=bus36_wcci
  SPLIT=test                 # train, test, or all
  SPLIT_CHRONICS=true
  TEST_CHRONICS_PCT=0.2
  CHRONIC_SPLIT_SEED=        # leave empty to use SEED
  CHRONIC_SHARD_COUNT=1
  CHRONIC_SHARD_INDEX=0
  SEED=0
  MAX_EPISODES=              # leave empty for train/test split size; use 2880 for SPLIT=all on WCCI
  MAX_ENV_STEPS=             # optional per-episode cap for smoke tests
  OUTPUT_DIR=outputs/do_nothing_eval/bus36_wcci_all
  PROGRESS=true

Extra arguments are forwarded to evaluate_do_nothing.py.

Notes:
  This Izar wrapper intentionally avoids hardcoding a partition/QOS. If your
  Izar account requires one, pass it at submission time, for example:
    sbatch --partition=<partition> --qos=<qos> Topology_Task/teacher_student/job_do_nothing_eval_izar.sh
EOF
    exit 0
fi

echo "Do-nothing evaluation job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"

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

ENV_ID="${ENV_ID:-bus36_wcci}"
ACTION_TYPE="${ACTION_TYPE:-topology}"
ENV_CONFIG_PATH="${ENV_CONFIG_PATH:-scenario.json}"
SPLIT="${SPLIT:-test}"
SPLIT_CHRONICS="${SPLIT_CHRONICS:-true}"
TEST_CHRONICS_PCT="${TEST_CHRONICS_PCT:-0.2}"
CHRONIC_SPLIT_SEED="${CHRONIC_SPLIT_SEED:-}"
CHRONIC_SHARD_COUNT="${CHRONIC_SHARD_COUNT:-1}"
CHRONIC_SHARD_INDEX="${CHRONIC_SHARD_INDEX:-0}"
SEED="${SEED:-0}"
DIFFICULTY="${DIFFICULTY:-0}"
DECENTRALIZED="${DECENTRALIZED:-true}"
OPTIMIZE_MEM="${OPTIMIZE_MEM:-true}"
MAX_EPISODES="${MAX_EPISODES:-}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/do_nothing_eval/${ENV_ID}_${SPLIT}}"
PROGRESS="${PROGRESS:-true}"

args=(
    --env-id "${ENV_ID}"
    --action-type "${ACTION_TYPE}"
    --env-config-path "${ENV_CONFIG_PATH}"
    --split "${SPLIT}"
    --split-chronics "${SPLIT_CHRONICS}"
    --test-chronics-pct "${TEST_CHRONICS_PCT}"
    --chronic-shard-count "${CHRONIC_SHARD_COUNT}"
    --chronic-shard-index "${CHRONIC_SHARD_INDEX}"
    --seed "${SEED}"
    --difficulty "${DIFFICULTY}"
    --decentralized "${DECENTRALIZED}"
    --optimize-mem "${OPTIMIZE_MEM}"
    --output-dir "${OUTPUT_DIR}"
    --progress "${PROGRESS}"
)

if [ -n "${CHRONIC_SPLIT_SEED}" ]; then
    args+=(--chronic-split-seed "${CHRONIC_SPLIT_SEED}")
fi

if [ -n "${MAX_EPISODES}" ]; then
    args+=(--max-episodes "${MAX_EPISODES}")
fi

if [ -n "${MAX_ENV_STEPS}" ]; then
    args+=(--max-env-steps "${MAX_ENV_STEPS}")
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${TASK_DIR}"

# Avoid accidental user-site packages shadowing the conda environment on Izar.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Launching do-nothing evaluation on Izar"
echo "Conda env: ${CONDA_ENV}"
echo "Python: $(command -v python)"
echo "Env id: ${ENV_ID}"
echo "Split: ${SPLIT}"
echo "Split chronics: ${SPLIT_CHRONICS}"
echo "Test chronics pct: ${TEST_CHRONICS_PCT}"
echo "Chronic split seed: ${CHRONIC_SPLIT_SEED:-SEED}"
echo "Chronic shard: ${CHRONIC_SHARD_INDEX}/${CHRONIC_SHARD_COUNT}"
echo "Max episodes: ${MAX_EPISODES:-split size}"
echo "Max env steps: ${MAX_ENV_STEPS:-none}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Extra args: $*"

python -u teacher_student/evaluate_do_nothing.py "${args[@]}" "$@"

echo "Do-nothing evaluation job ${SLURM_JOB_ID:-local} finished at $(date)"

