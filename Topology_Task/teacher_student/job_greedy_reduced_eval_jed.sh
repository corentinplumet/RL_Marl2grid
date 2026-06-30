#!/usr/bin/env bash
# EPFL JED launcher for simulator-greedy reduced-action evaluation.
# Submit from the repository root:
#   REDUCED_ACTION_SPACE=outputs/.../reduced_action_space.json \
#   sbatch Topology_Task/teacher_student/job_greedy_reduced_eval_jed.sh
#SBATCH --job-name=greedy_reduce_eval
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=Topology_Task/teacher_student/slurm-%x-%j.out
#SBATCH --error=Topology_Task/teacher_student/slurm-%x-%j.err

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  REDUCED_ACTION_SPACE=outputs/.../metadata/reduced_action_space.json \
  sbatch Topology_Task/teacher_student/job_greedy_reduced_eval_jed.sh

This compares two policies on the same chronic split:
  1. do nothing
  2. simulator-greedy over the reduced unilateral action set

Common overrides:
  ENV_ID=bus36
  REDUCED_ACTION_SPACE=outputs/.../reduced_action_space.json
  SPLIT=test
  MAX_EPISODES=              # leave empty for all chronics in split
  DECISION_RHO_THRESHOLD=0.90
  REQUIRE_IMPROVEMENT=true
  IMPROVEMENT_TOLERANCE=1e-3
  SIM_WORKERS=71
  OUTPUT_DIR=outputs/teacher_student_greedy_eval

Extra arguments are forwarded to evaluate_greedy_reduced_actions.py.
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

ENV_ID="${ENV_ID:-bus36}"
REDUCED_ACTION_SPACE="${REDUCED_ACTION_SPACE:-}"
SPLIT="${SPLIT:-test}"
SPLIT_CHRONICS="${SPLIT_CHRONICS:-true}"
TEST_CHRONICS_PCT="${TEST_CHRONICS_PCT:-0.2}"
SEED="${SEED:-0}"
DIFFICULTY="${DIFFICULTY:-0}"
DECENTRALIZED="${DECENTRALIZED:-true}"
OPTIMIZE_MEM="${OPTIMIZE_MEM:-true}"
MAX_EPISODES="${MAX_EPISODES:-}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-}"
DECISION_RHO_THRESHOLD="${DECISION_RHO_THRESHOLD:-0.90}"
REQUIRE_IMPROVEMENT="${REQUIRE_IMPROVEMENT:-true}"
IMPROVEMENT_TOLERANCE="${IMPROVEMENT_TOLERANCE:-1e-3}"
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
    --decision-rho-threshold "${DECISION_RHO_THRESHOLD}"
    --require-improvement "${REQUIRE_IMPROVEMENT}"
    --improvement-tolerance "${IMPROVEMENT_TOLERANCE}"
    --time-step "${TIME_STEP}"
    --sim-workers "${SIM_WORKERS}"
    --sim-start-method "${SIM_START_METHOD}"
    --output-dir "${OUTPUT_DIR}"
    --progress "${PROGRESS}"
)

if [ -n "${MAX_EPISODES}" ]; then
    args+=(--max-episodes "${MAX_EPISODES}")
fi

if [ -n "${MAX_ENV_STEPS}" ]; then
    args+=(--max-env-steps "${MAX_ENV_STEPS}")
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${TASK_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Launching greedy reduced-action evaluation on JED"
echo "Conda env: ${CONDA_ENV}"
echo "Env id: ${ENV_ID}"
echo "Reduced action space: ${REDUCED_ACTION_SPACE}"
echo "Split: ${SPLIT}"
echo "Max episodes: ${MAX_EPISODES:-split size}"
echo "Decision rho threshold: ${DECISION_RHO_THRESHOLD}"
echo "Require improvement: ${REQUIRE_IMPROVEMENT}"
echo "Improvement tolerance: ${IMPROVEMENT_TOLERANCE}"
echo "Simulation workers: ${SIM_WORKERS}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Extra args: $*"

python -u teacher_student/evaluate_greedy_reduced_actions.py "${args[@]}" "$@"

