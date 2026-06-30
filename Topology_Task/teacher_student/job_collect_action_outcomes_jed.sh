#!/usr/bin/env bash
# EPFL JED SLURM launcher for brute-force action-space reduction.
# Submit from the repository root:
#   sbatch Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh
#SBATCH --job-name=bf_action_reduce
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=1-06:00:00
#SBATCH --output=Topology_Task/teacher_student/slurm-%x-%j.out
#SBATCH --error=Topology_Task/teacher_student/slurm-%x-%j.err

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  sbatch Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh

This job does not need a checkpoint. It runs a brute-force Teacher on Grid2Op:
when rho_max crosses COLLECTION_RHO_THRESHOLD, it simulates candidate topology
actions, logs rho before/after/delta, executes the best simulated action, and
then writes metadata/reduced_action_space.json.

Common overrides:
  ENV_ID=bus36
  OUTPUT_DIR=outputs/teacher_student_datasets/bus36_bruteforce_rho090
  COLLECTION_RHO_THRESHOLD=0.90
  CHRONIC_SHARD_COUNT=16      # or submit with: sbatch --array=0-15 ...
  MAX_EPISODES=               # leave empty for one pass over this shard
  MAX_ENV_STEPS=
  OUTCOME_ACTION_SAMPLE_SIZE=64   # set to all to evaluate every action
  OUTCOME_SIM_WORKERS=71          # parallel sims; defaults to cpus-per-task minus 1
  OUTCOME_SIM_START_METHOD=spawn
  TIMING_EVERY_ENV_STEPS=1
  ACTION_REDUCTION_TOP_K=208
  OVERWRITE=false

Examples:
  # Short calibration job.
  OUTPUT_DIR=outputs/teacher_student_datasets/smoke_bus36_bruteforce_rho090 \
  MAX_EPISODES=2 \
  OUTCOME_ACTION_SAMPLE_SIZE=32 \
  TIMING_EVERY_ENV_STEPS=1 \
  ACTION_REDUCTION_TOP_K=64 \
  sbatch Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh

  # Full bus36 brute-force reduction over every action.
  sbatch --array=0-15 Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh

  # Exhaustive action evaluation is usually infeasible, but still available.
  OUTPUT_DIR=outputs/teacher_student_datasets/bus36_bruteforce_rho090 \
  OUTCOME_ACTION_SAMPLE_SIZE=all \
  ACTION_REDUCTION_TOP_K=208 \
  sbatch Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh

Additional arguments passed to this script are forwarded to
collect_bruteforce_action_outcomes.py.
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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/teacher_student_datasets/bus36_bruteforce_rho090}"
SPLIT="${SPLIT:-train}"
SPLIT_CHRONICS="${SPLIT_CHRONICS:-true}"
TEST_CHRONICS_PCT="${TEST_CHRONICS_PCT:-0.2}"
SEED="${SEED:-0}"
DIFFICULTY="${DIFFICULTY:-0}"
DECENTRALIZED="${DECENTRALIZED:-true}"
COLLECTION_RHO_THRESHOLD="${COLLECTION_RHO_THRESHOLD:-0.90}"
MAX_EPISODES="${MAX_EPISODES:-}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-}"
OUTCOME_ACTION_SAMPLE_SIZE="${OUTCOME_ACTION_SAMPLE_SIZE:-64}"
DEFAULT_OUTCOME_SIM_WORKERS="${SLURM_CPUS_PER_TASK:-1}"
if [ "${DEFAULT_OUTCOME_SIM_WORKERS}" -gt 1 ]; then
    DEFAULT_OUTCOME_SIM_WORKERS="$((DEFAULT_OUTCOME_SIM_WORKERS - 1))"
fi
OUTCOME_SIM_WORKERS="${OUTCOME_SIM_WORKERS:-${DEFAULT_OUTCOME_SIM_WORKERS}}"
OUTCOME_SIM_START_METHOD="${OUTCOME_SIM_START_METHOD:-spawn}"
OUTCOME_ROLLOUT_POLICY="${OUTCOME_ROLLOUT_POLICY:-best_simulated}"
OUTCOME_DELTA_TOLERANCE="${OUTCOME_DELTA_TOLERANCE:-1e-3}"
TIMING_EVERY_ENV_STEPS="${TIMING_EVERY_ENV_STEPS:-100}"
SHARD_SIZE="${SHARD_SIZE:-50000}"
COMPRESS="${COMPRESS:-true}"
OVERWRITE="${OVERWRITE:-false}"
REDUCE_AFTER_WAS_SET="${REDUCE_AFTER+x}"
ACTION_REDUCTION_TOP_K="${ACTION_REDUCTION_TOP_K:-208}"
ACTION_REDUCTION_MIN_COUNT="${ACTION_REDUCTION_MIN_COUNT:-1}"
ACTION_REDUCTION_METRIC="${ACTION_REDUCTION_METRIC:-delta_vs_do_nothing}"
ACTION_REDUCTION_METHOD="${ACTION_REDUCTION_METHOD:-best_per_state}"
ACTION_REDUCTION_REQUIRE_IMPROVEMENT="${ACTION_REDUCTION_REQUIRE_IMPROVEMENT:-true}"

if [ -n "${CHRONIC_SHARD_INDEX:-}" ]; then
    CHRONIC_SHARD_INDEX="${CHRONIC_SHARD_INDEX}"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    CHRONIC_SHARD_INDEX="$((SLURM_ARRAY_TASK_ID - ${SLURM_ARRAY_TASK_MIN:-0}))"
else
    CHRONIC_SHARD_INDEX=0
fi
CHRONIC_SHARD_COUNT="${CHRONIC_SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"

if [ "${CHRONIC_SHARD_COUNT}" -gt 1 ]; then
    if [ -z "${REDUCE_AFTER_WAS_SET}" ]; then
        REDUCE_AFTER=false
    fi
    BASE_OUTPUT_DIR="${OUTPUT_DIR}"
    if [ "${OUTPUT_DIR_IS_PART:-false}" != "true" ]; then
        PART_NAME="$(printf "part_%03d" "${CHRONIC_SHARD_INDEX}")"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/parts/${PART_NAME}"
    fi
else
    BASE_OUTPUT_DIR="${OUTPUT_DIR}"
fi
REDUCE_AFTER="${REDUCE_AFTER:-true}"

collector_args=(
    --env-id "${ENV_ID}"
    --split "${SPLIT}"
    --split-chronics "${SPLIT_CHRONICS}"
    --test-chronics-pct "${TEST_CHRONICS_PCT}"
    --chronic-shard-count "${CHRONIC_SHARD_COUNT}"
    --chronic-shard-index "${CHRONIC_SHARD_INDEX}"
    --seed "${SEED}"
    --difficulty "${DIFFICULTY}"
    --decentralized "${DECENTRALIZED}"
    --collection-rho-threshold "${COLLECTION_RHO_THRESHOLD}"
    --outcome-rollout-policy "${OUTCOME_ROLLOUT_POLICY}"
    --outcome-delta-tolerance "${OUTCOME_DELTA_TOLERANCE}"
    --outcome-sim-workers "${OUTCOME_SIM_WORKERS}"
    --outcome-sim-start-method "${OUTCOME_SIM_START_METHOD}"
    --timing-every-env-steps "${TIMING_EVERY_ENV_STEPS}"
    --shard-size "${SHARD_SIZE}"
    --compress "${COMPRESS}"
    --overwrite "${OVERWRITE}"
    --reduce-after "${REDUCE_AFTER}"
    --action-reduction-top-k "${ACTION_REDUCTION_TOP_K}"
    --action-reduction-min-count "${ACTION_REDUCTION_MIN_COUNT}"
    --action-reduction-metric "${ACTION_REDUCTION_METRIC}"
    --action-reduction-method "${ACTION_REDUCTION_METHOD}"
    --action-reduction-require-improvement "${ACTION_REDUCTION_REQUIRE_IMPROVEMENT}"
    --output-dir "${OUTPUT_DIR}"
)

if [ -n "${MAX_EPISODES}" ]; then
    collector_args+=(--max-episodes "${MAX_EPISODES}")
fi

if [ -n "${MAX_ENV_STEPS}" ]; then
    collector_args+=(--max-env-steps "${MAX_ENV_STEPS}")
fi

if [ -n "${OUTCOME_ACTION_SAMPLE_SIZE}" ] && [ "${OUTCOME_ACTION_SAMPLE_SIZE}" != "all" ]; then
    collector_args+=(--outcome-action-sample-size "${OUTCOME_ACTION_SAMPLE_SIZE}")
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${TASK_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Launching brute-force action-space reduction on JED"
echo "Conda env: ${CONDA_ENV}"
echo "Env id: ${ENV_ID}"
echo "Chronic shard: ${CHRONIC_SHARD_INDEX}/${CHRONIC_SHARD_COUNT}"
echo "Base output dir: ${BASE_OUTPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Collection rho threshold: ${COLLECTION_RHO_THRESHOLD}"
echo "Max episodes: ${MAX_EPISODES:-collector default}"
echo "Max env steps: ${MAX_ENV_STEPS:-none}"
echo "Action sample size: ${OUTCOME_ACTION_SAMPLE_SIZE:-all}"
echo "Simulation workers: ${OUTCOME_SIM_WORKERS}"
echo "Simulation start method: ${OUTCOME_SIM_START_METHOD}"
echo "Action reduction top-k: ${ACTION_REDUCTION_TOP_K}"
echo "Reduce after: ${REDUCE_AFTER}"
echo "Extra args: $*"

python -u teacher_student/collect_bruteforce_action_outcomes.py "${collector_args[@]}" "$@"
