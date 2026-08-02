#!/usr/bin/env bash
# Evaluate the simulator-greedy policy on bus36_wcci_nomaint across several
# action-space reduction sizes, on exactly the chronics of an earlier
# do-nothing run so every size is measured on the same episodes.
#
# Run from the repository root:
#   Topology_Task/teacher_student/sweep_greedy_reduced_sizes_izar.sh
#
# Overrides:
#   TOP_KS="16 32 64 128 256 512"   sizes to evaluate
#   SUMMARY=<do_nothing_summary.json whose chronics define the episode set>
#   ENV_ID=bus36_wcci_nomaint
#   DATASET_ROOT / NAME_PREFIX      where the reduced-action JSONs live
#   OUTPUT_ROOT=outputs/greedy_wcci_nomaint_50
#   DRY_RUN=true                    print the submissions without sbatch

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
TASK_DIR="${REPO_DIR}/Topology_Task"
if [ ! -d "${TASK_DIR}" ]; then
    echo "Could not find ${TASK_DIR}. Run from the repository root, or set REPO_DIR." >&2
    exit 1
fi

ENV_ID="${ENV_ID:-bus36_wcci_nomaint}"
DATASET_ROOT="${DATASET_ROOT:-outputs/teacher_student_datasets/wcci_full2048a_90_v3}"
NAME_PREFIX="${NAME_PREFIX:-wcci_full2048a_90_v3}"
SUMMARY="${SUMMARY:-outputs/do_nothing_eval/wcci_test_nomaint_50/do_nothing_summary.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/greedy_wcci_nomaint_50}"
TOP_KS="${TOP_KS:-16 32 64 128 256 512}"
SPLIT="${SPLIT:-test}"
CHRONIC_SPLIT_SEED="${CHRONIC_SPLIT_SEED:-0}"
TEST_CHRONICS_PCT="${TEST_CHRONICS_PCT:-0.2}"
DRY_RUN="${DRY_RUN:-false}"

if [ ! -f "${TASK_DIR}/${SUMMARY}" ]; then
    echo "Chronic set not found: ${TASK_DIR}/${SUMMARY}" >&2
    echo "Point SUMMARY at the do-nothing summary whose episodes you want to reuse." >&2
    exit 1
fi

# Pin the exact chronics rather than re-deriving them from the split, so the
# greedy runs land on the same episodes as the do-nothing baseline.
NAMES="$(python - "${TASK_DIR}/${SUMMARY}" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    episodes = json.load(handle)["episodes"]
print(",".join(episode["chronic_name"] for episode in episodes))
PY
)"
N_CHRONICS="$(awk -F, '{print NF}' <<<"${NAMES}")"

echo "========== Greedy reduction-size sweep =========="
echo "Env:       ${ENV_ID}"
echo "Chronics:  ${N_CHRONICS} from ${SUMMARY}"
echo "Sizes:     ${TOP_KS}"
echo "Output:    ${OUTPUT_ROOT}/mk<k>"
echo "================================================"

for top_k in ${TOP_KS}; do
    reduced="${DATASET_ROOT}/metadata/reduced_action_space_${NAME_PREFIX}_mk${top_k}.json"
    if [ ! -f "${TASK_DIR}/${reduced}" ]; then
        echo "missing ${reduced}" >&2
        echo "Generate it first: Topology_Task/teacher_student/reduce_wcci_action_space_sizes.sh" >&2
        exit 1
    fi
    echo "--- mk${top_k}: ${reduced}"
    if [ "${DRY_RUN}" = "true" ]; then
        continue
    fi
    ENV_ID="${ENV_ID}" \
    REDUCED_ACTION_SPACE="${reduced}" \
    SPLIT="${SPLIT}" \
    CHRONIC_SPLIT_SEED="${CHRONIC_SPLIT_SEED}" \
    TEST_CHRONICS_PCT="${TEST_CHRONICS_PCT}" \
    TARGET_CHRONIC_NAMES="${NAMES}" \
    COMPARE_DO_NOTHING=true \
    OUTPUT_DIR="${OUTPUT_ROOT}/mk${top_k}" \
    sbatch --job-name="greedy_mk${top_k}" \
        Topology_Task/teacher_student/job_greedy_reduced_eval_izar.sh
done
