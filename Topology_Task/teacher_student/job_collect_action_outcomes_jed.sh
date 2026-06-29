#!/usr/bin/env bash
# EPFL JED SLURM launcher for action-outcome teacher datasets.
# Submit from the repository root:
#   sbatch Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh
#
# Override defaults with environment variables, for example:
#   CHECKPOINT=checkpoint/with_obs_stats/best_test_run.tar \
#   OUTPUT_DIR=outputs/teacher_student_datasets/action_outcomes_rho090_s0 \
#   sbatch Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh
#SBATCH --job-name=action_outcomes
#SBATCH --mail-user=corentin.plumet@epfl.ch
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=7000M
#SBATCH --time=1-06:00:00
#SBATCH --output=Topology_Task/teacher_student/slurm-%x-%j.out
#SBATCH --error=Topology_Task/teacher_student/slurm-%x-%j.err

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  sbatch Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh

Common overrides:
  CHECKPOINT=checkpoint/with_obs_stats/best_test_run.tar
  OUTPUT_DIR=outputs/teacher_student_datasets/action_outcomes_rho090_s0
  COLLECTION_RHO_THRESHOLD=0.90
  MAX_EPISODES=803
  OUTCOME_ACTION_SAMPLE_SIZE=32   # omit / leave empty to evaluate every action
  TIMING_EVERY_ENV_STEPS=1
  OVERWRITE=false

Examples:
  # Short calibration job.
  CHECKPOINT=checkpoint/with_obs_stats/best_test_run.tar \
  OUTPUT_DIR=outputs/teacher_student_datasets/smoke_action_outcomes_rho090 \
  MAX_EPISODES=2 \
  OUTCOME_ACTION_SAMPLE_SIZE=32 \
  TIMING_EVERY_ENV_STEPS=1 \
  sbatch Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh

  # Full collection over every action.
  CHECKPOINT=checkpoint/with_obs_stats/best_test_run.tar \
  OUTPUT_DIR=outputs/teacher_student_datasets/action_outcomes_rho090_s0 \
  MAX_EPISODES=803 \
  sbatch Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh

Additional arguments passed to this script are forwarded to collect_teacher_dataset.py.
EOF
    exit 0
fi

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TASK_DIR="${REPO_DIR}/Topology_Task"
COLLECTOR_JOB="${TASK_DIR}/teacher_student/job_collect_teacher_dataset.sh"

if [ ! -d "${TASK_DIR}" ]; then
    echo "Could not find ${TASK_DIR}." >&2
    echo "Submit from the repository root, or set REPO_DIR=/path/to/RL_Marl2grid." >&2
    exit 1
fi

if [ ! -f "${COLLECTOR_JOB}" ]; then
    echo "Could not find ${COLLECTOR_JOB}." >&2
    exit 1
fi

CHECKPOINT="${CHECKPOINT:-checkpoint/with_obs_stats/best_test_run.tar}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/teacher_student_datasets/action_outcomes_rho090}"
SPLIT="${SPLIT:-train}"
COLLECTION_RHO_THRESHOLD="${COLLECTION_RHO_THRESHOLD:-0.90}"
EVAL_ACTION_RHO_THRESHOLD="${EVAL_ACTION_RHO_THRESHOLD:-${COLLECTION_RHO_THRESHOLD}}"
OBS_NORMALIZATION="${OBS_NORMALIZATION:-require}"
MAX_EPISODES="${MAX_EPISODES:-}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-}"
OUTCOME_ACTION_SAMPLE_SIZE="${OUTCOME_ACTION_SAMPLE_SIZE:-}"
OUTCOME_ROLLOUT_POLICY="${OUTCOME_ROLLOUT_POLICY:-best_simulated}"
OUTCOME_DELTA_TOLERANCE="${OUTCOME_DELTA_TOLERANCE:-1e-3}"
TIMING_EVERY_ENV_STEPS="${TIMING_EVERY_ENV_STEPS:-100}"
SHARD_SIZE="${SHARD_SIZE:-50000}"
COMPRESS="${COMPRESS:-true}"
OVERWRITE="${OVERWRITE:-false}"

collector_args=(
    --checkpoint "${CHECKPOINT}"
    --split "${SPLIT}"
    --eval-all-split-chronics true
    --dataset-mode action_outcomes
    --collection-rho-threshold "${COLLECTION_RHO_THRESHOLD}"
    --eval-action-rho-threshold "${EVAL_ACTION_RHO_THRESHOLD}"
    --outcome-rollout-policy "${OUTCOME_ROLLOUT_POLICY}"
    --outcome-delta-tolerance "${OUTCOME_DELTA_TOLERANCE}"
    --obs-normalization "${OBS_NORMALIZATION}"
    --timing-every-env-steps "${TIMING_EVERY_ENV_STEPS}"
    --shard-size "${SHARD_SIZE}"
    --compress "${COMPRESS}"
    --overwrite "${OVERWRITE}"
    --output-dir "${OUTPUT_DIR}"
)

if [ -n "${MAX_EPISODES}" ]; then
    collector_args+=(--max-episodes "${MAX_EPISODES}")
fi

if [ -n "${MAX_ENV_STEPS}" ]; then
    collector_args+=(--max-env-steps "${MAX_ENV_STEPS}")
fi

if [ -n "${OUTCOME_ACTION_SAMPLE_SIZE}" ]; then
    collector_args+=(--outcome-action-sample-size "${OUTCOME_ACTION_SAMPLE_SIZE}")
fi

echo "Launching action-outcome collection on JED"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Collection rho threshold: ${COLLECTION_RHO_THRESHOLD}"
echo "Max episodes: ${MAX_EPISODES:-collector default}"
echo "Max env steps: ${MAX_ENV_STEPS:-none}"
echo "Action sample size: ${OUTCOME_ACTION_SAMPLE_SIZE:-all}"
echo "Extra args: $*"

bash "${COLLECTOR_JOB}" "${collector_args[@]}" "$@"
