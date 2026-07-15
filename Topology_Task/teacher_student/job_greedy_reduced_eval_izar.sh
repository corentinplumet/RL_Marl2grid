#!/bin/bash
# IZAR / EPFL launcher for simulator-greedy reduced-action evaluation.
# Submit from the repository root:
#   REDUCED_ACTION_SPACE=outputs/.../reduced_action_space.json \
#   sbatch Topology_Task/teacher_student/job_greedy_reduced_eval_izar.sh
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --job-name=greedy_reduce_eval
#SBATCH --output=Topology_Task/teacher_student/slurm-%x-%j.out
#SBATCH --error=Topology_Task/teacher_student/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  REDUCED_ACTION_SPACE=outputs/.../metadata/reduced_action_space.json \
  sbatch Topology_Task/teacher_student/job_greedy_reduced_eval_izar.sh

This compares two policies on the same chronic split:
  1. do nothing
  2. simulator-greedy over the reduced unilateral action set

Common overrides:
  ENV_ID=bus36
  REDUCED_ACTION_SPACE=outputs/.../reduced_action_space.json
  SPLIT=test
  MAX_EPISODES=              # leave empty for all chronics in split
  CHRONIC_SPLIT_SEED=        # set to match the training split, e.g. 0
  CHRONIC_SAMPLE_MODE=sequential
  CHRONIC_SAMPLE_SEED=
  CHRONIC_SAMPLE_REPLACEMENT=false
  DECISION_RHO_THRESHOLD=0.90
  REQUIRE_IMPROVEMENT=true
  IMPROVEMENT_TOLERANCE=1e-3
  CANDIDATE_SAMPLE_SIZE=     # optional fixed random subset of unilateral actions
  CANDIDATE_SAMPLE_SEED=     # leave empty to use SEED
  COMPARE_DO_NOTHING=true
  SIM_WORKERS=39             # defaults to SLURM_CPUS_PER_TASK - 1
  OUTPUT_DIR=outputs/teacher_student_greedy_eval

Extra arguments are forwarded to evaluate_greedy_reduced_actions.py.

Notes:
  This Izar wrapper uses the gpu/long defaults used by the other Izar jobs in
  this repo. Override them at submission time if needed, for example:
    sbatch --time=12:00:00 Topology_Task/teacher_student/job_greedy_reduced_eval_izar.sh
EOF
    exit 0
fi

echo "Greedy reduced-action evaluation job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"

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

ENV_ID="${ENV_ID:-bus36}"
REDUCED_ACTION_SPACE="${REDUCED_ACTION_SPACE:-}"
SPLIT="${SPLIT:-test}"
SPLIT_CHRONICS="${SPLIT_CHRONICS:-true}"
TEST_CHRONICS_PCT="${TEST_CHRONICS_PCT:-0.2}"
CHRONIC_SPLIT_SEED="${CHRONIC_SPLIT_SEED:-}"
SEED="${SEED:-0}"
DIFFICULTY="${DIFFICULTY:-0}"
DECENTRALIZED="${DECENTRALIZED:-true}"
OPTIMIZE_MEM="${OPTIMIZE_MEM:-true}"
MAX_EPISODES="${MAX_EPISODES:-}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-}"
CHRONIC_SAMPLE_MODE="${CHRONIC_SAMPLE_MODE:-sequential}"
CHRONIC_SAMPLE_SEED="${CHRONIC_SAMPLE_SEED:-}"
CHRONIC_SAMPLE_REPLACEMENT="${CHRONIC_SAMPLE_REPLACEMENT:-false}"
DECISION_RHO_THRESHOLD="${DECISION_RHO_THRESHOLD:-0.90}"
REQUIRE_IMPROVEMENT="${REQUIRE_IMPROVEMENT:-true}"
IMPROVEMENT_TOLERANCE="${IMPROVEMENT_TOLERANCE:-1e-3}"
CANDIDATE_SAMPLE_SIZE="${CANDIDATE_SAMPLE_SIZE:-}"
CANDIDATE_SAMPLE_SEED="${CANDIDATE_SAMPLE_SEED:-}"
COMPARE_DO_NOTHING="${COMPARE_DO_NOTHING:-true}"
TIME_STEP="${TIME_STEP:-1}"
DEFAULT_SIM_WORKERS="${SLURM_CPUS_PER_TASK:-1}"
if [ "${DEFAULT_SIM_WORKERS}" -gt 1 ]; then
    DEFAULT_SIM_WORKERS="$((DEFAULT_SIM_WORKERS - 1))"
fi
SIM_WORKERS="${SIM_WORKERS:-${DEFAULT_SIM_WORKERS}}"
SIM_START_METHOD="${SIM_START_METHOD:-spawn}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/teacher_student_greedy_eval}"
PROGRESS="${PROGRESS:-true}"

if [ -z "${REDUCED_ACTION_SPACE}" ]; then
    echo "REDUCED_ACTION_SPACE is required." >&2
    exit 1
fi

args=(
    --env-id "${ENV_ID}"
    --reduced-action-space "${REDUCED_ACTION_SPACE}"
    --split "${SPLIT}"
    --split-chronics "${SPLIT_CHRONICS}"
    --test-chronics-pct "${TEST_CHRONICS_PCT}"
    --seed "${SEED}"
    --difficulty "${DIFFICULTY}"
    --decentralized "${DECENTRALIZED}"
    --optimize-mem "${OPTIMIZE_MEM}"
    --chronic-sample-mode "${CHRONIC_SAMPLE_MODE}"
    --chronic-sample-replacement "${CHRONIC_SAMPLE_REPLACEMENT}"
    --decision-rho-threshold "${DECISION_RHO_THRESHOLD}"
    --require-improvement "${REQUIRE_IMPROVEMENT}"
    --improvement-tolerance "${IMPROVEMENT_TOLERANCE}"
    --compare-do-nothing "${COMPARE_DO_NOTHING}"
    --time-step "${TIME_STEP}"
    --sim-workers "${SIM_WORKERS}"
    --sim-start-method "${SIM_START_METHOD}"
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

if [ -n "${CHRONIC_SAMPLE_SEED}" ]; then
    args+=(--chronic-sample-seed "${CHRONIC_SAMPLE_SEED}")
fi

if [ -n "${CANDIDATE_SAMPLE_SIZE}" ]; then
    args+=(--candidate-sample-size "${CANDIDATE_SAMPLE_SIZE}")
fi

if [ -n "${CANDIDATE_SAMPLE_SEED}" ]; then
    args+=(--candidate-sample-seed "${CANDIDATE_SAMPLE_SEED}")
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

echo "Launching greedy reduced-action evaluation on Izar"
echo "Conda env: ${CONDA_ENV}"
echo "Python: $(command -v python)"
echo "Env id: ${ENV_ID}"
echo "Reduced action space: ${REDUCED_ACTION_SPACE}"
echo "Split: ${SPLIT}"
echo "Chronic split seed: ${CHRONIC_SPLIT_SEED:-SEED}"
echo "Max episodes: ${MAX_EPISODES:-split size}"
echo "Chronic sample mode: ${CHRONIC_SAMPLE_MODE}"
echo "Chronic sample seed: ${CHRONIC_SAMPLE_SEED:-SEED}"
echo "Chronic sample replacement: ${CHRONIC_SAMPLE_REPLACEMENT}"
echo "Decision rho threshold: ${DECISION_RHO_THRESHOLD}"
echo "Require improvement: ${REQUIRE_IMPROVEMENT}"
echo "Improvement tolerance: ${IMPROVEMENT_TOLERANCE}"
echo "Candidate sample size: ${CANDIDATE_SAMPLE_SIZE:-all}"
echo "Candidate sample seed: ${CANDIDATE_SAMPLE_SEED:-SEED}"
echo "Compare do-nothing replay: ${COMPARE_DO_NOTHING}"
echo "Simulation workers: ${SIM_WORKERS}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Extra args: $*"

python -u teacher_student/evaluate_greedy_reduced_actions.py "${args[@]}" "$@"

echo "Greedy reduced-action evaluation job ${SLURM_JOB_ID:-local} finished at $(date)"
