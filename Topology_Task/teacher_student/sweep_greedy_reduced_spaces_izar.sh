#!/usr/bin/env bash
# Evaluate the simulator-greedy policy on bus36_wcci_nomaint for each reduced
# action space you already have, on exactly the chronics of an earlier
# do-nothing run so every reduction is measured on the same episodes.
#
# Run from the repository root. With no arguments it evaluates every
# reduced_action_space_*.json under outputs/teacher_student_datasets/*/metadata:
#
#   Topology_Task/teacher_student/sweep_greedy_reduced_spaces_izar.sh
#
# Or name the ones you want (paths relative to Topology_Task/ or absolute):
#
#   Topology_Task/teacher_student/sweep_greedy_reduced_spaces_izar.sh \
#     outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk256.json
#
# Overrides:
#   SUMMARY=<do_nothing_summary.json whose chronics define the episode set>
#   ENV_ID=bus36_wcci_nomaint
#   OUTPUT_ROOT=outputs/greedy_wcci_nomaint_50
#   DRY_RUN=true      list what would be submitted, submit nothing

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
TASK_DIR="${REPO_DIR}/Topology_Task"
if [ ! -d "${TASK_DIR}" ]; then
    echo "Could not find ${TASK_DIR}. Run from the repository root, or set REPO_DIR." >&2
    exit 1
fi

ENV_ID="${ENV_ID:-bus36_wcci_nomaint}"
SUMMARY="${SUMMARY:-outputs/do_nothing_eval/wcci_test_nomaint_50/do_nothing_summary.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/greedy_wcci_nomaint_50}"
SPLIT="${SPLIT:-test}"
CHRONIC_SPLIT_SEED="${CHRONIC_SPLIT_SEED:-0}"
TEST_CHRONICS_PCT="${TEST_CHRONICS_PCT:-0.2}"
DRY_RUN="${DRY_RUN:-false}"

REDUCTIONS=("$@")
if [ ${#REDUCTIONS[@]} -eq 0 ]; then
    while IFS= read -r path; do
        REDUCTIONS+=("${path#"${TASK_DIR}/"}")
    done < <(find "${TASK_DIR}/outputs/teacher_student_datasets" \
        -path '*/metadata/reduced_action_space_*.json' 2>/dev/null | sort)
fi
if [ ${#REDUCTIONS[@]} -eq 0 ]; then
    echo "No reduced_action_space_*.json found under" >&2
    echo "  ${TASK_DIR}/outputs/teacher_student_datasets/*/metadata/" >&2
    echo "Pass the paths explicitly if they live elsewhere." >&2
    exit 1
fi

if [ ! -f "${TASK_DIR}/${SUMMARY}" ] && [ ! -f "${SUMMARY}" ]; then
    echo "Chronic set not found: ${SUMMARY}" >&2
    echo "Point SUMMARY at the do-nothing summary whose episodes you want to reuse." >&2
    exit 1
fi
SUMMARY_PATH="${TASK_DIR}/${SUMMARY}"
[ -f "${SUMMARY_PATH}" ] || SUMMARY_PATH="${SUMMARY}"

# Pin the exact chronics rather than re-deriving them from the split, so every
# reduction lands on the same episodes as the do-nothing baseline.
NAMES="$(python - "${SUMMARY_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    episodes = json.load(handle)["episodes"]
print(",".join(episode["chronic_name"] for episode in episodes))
PY
)"
N_CHRONICS="$(awk -F, '{print NF}' <<<"${NAMES}")"

echo "========== Greedy reduced-action sweep =========="
echo "Env:      ${ENV_ID}"
echo "Chronics: ${N_CHRONICS} from ${SUMMARY}"
echo "Spaces:   ${#REDUCTIONS[@]}"
echo "Output:   ${OUTPUT_ROOT}/<label>"
echo "================================================"

for reduced in "${REDUCTIONS[@]}"; do
    reduced_path="${TASK_DIR}/${reduced}"
    [ -f "${reduced_path}" ] || reduced_path="${reduced}"
    if [ ! -f "${reduced_path}" ]; then
        echo "missing reduced action space: ${reduced}" >&2
        exit 1
    fi

    label="$(basename "${reduced_path}" .json)"
    label="${label#reduced_action_space_}"

    # Report what each reduction actually contains, so the sweep is readable
    # without opening the JSONs.
    python - "${reduced_path}" "${label}" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    reduction = json.load(handle)
agents = reduction.get("agents", {})
sizes = ", ".join(
    f"{agent}={payload.get('selected_action_size')}/{payload.get('original_action_size')}"
    for agent, payload in sorted(agents.items())
)
print(
    f"--- {sys.argv[2]}: total="
    f"{reduction.get('total_selected_action_size')}/"
    f"{reduction.get('total_original_action_size')} "
    f"top_k={reduction.get('top_k')} method={reduction.get('selection_method')}"
)
print(f"    {sizes}")
PY

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
    OUTPUT_DIR="${OUTPUT_ROOT}/${label}" \
    sbatch --job-name="greedy_${label}" \
        Topology_Task/teacher_student/job_greedy_reduced_eval_izar.sh
done
