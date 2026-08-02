#!/usr/bin/env bash
# Generate WCCI reduced action spaces at several sizes.
#
# Every setting except --top-k is copied from an existing reduction (mk256 by
# default), so the resulting JSONs differ only in how many actions survive.
#
# Run from the repository root:
#   Topology_Task/teacher_student/reduce_wcci_action_space_sizes.sh
#
# Overrides:
#   TOP_KS="16 32 64 128 256 512"   sizes to produce
#   DATASET_ROOT=outputs/teacher_student_datasets/wcci_full2048a_90_v3
#   REFERENCE=<path to an existing reduced_action_space json>
#   NAME_PREFIX=wcci_full2048a_90_v3
#   FORCE=true                      regenerate JSONs that already exist

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
TASK_DIR="${REPO_DIR}/Topology_Task"
if [ ! -d "${TASK_DIR}" ]; then
    echo "Could not find ${TASK_DIR}. Run from the repository root, or set REPO_DIR." >&2
    exit 1
fi
cd "${TASK_DIR}"

DATASET_ROOT="${DATASET_ROOT:-outputs/teacher_student_datasets/wcci_full2048a_90_v3}"
NAME_PREFIX="${NAME_PREFIX:-wcci_full2048a_90_v3}"
REFERENCE="${REFERENCE:-${DATASET_ROOT}/metadata/reduced_action_space_${NAME_PREFIX}_mk256.json}"
TOP_KS="${TOP_KS:-16 32 64 128 256 512}"
FORCE="${FORCE:-false}"

if [ ! -d "${DATASET_ROOT}" ]; then
    echo "Dataset root not found: ${TASK_DIR}/${DATASET_ROOT}" >&2
    exit 1
fi
if [ ! -f "${REFERENCE}" ]; then
    echo "Reference reduction not found: ${TASK_DIR}/${REFERENCE}" >&2
    echo "Set REFERENCE to an existing reduced_action_space_*.json." >&2
    exit 1
fi

# The reducer wants every shard directory; older datasets keep the parts at the
# root instead.
DATASET_DIRS=()
while IFS= read -r part; do
    DATASET_DIRS+=("${part}")
done < <(find "${DATASET_ROOT}/parts" -maxdepth 1 -mindepth 1 -type d -name 'part_*' 2>/dev/null | sort)
if [ ${#DATASET_DIRS[@]} -eq 0 ]; then
    DATASET_DIRS=("${DATASET_ROOT}")
fi

# Reuse the reference reduction's settings so only top-k varies.
read -r METRIC METHOD MIN_COUNT REQUIRE_IMPROVEMENT IMPROVEMENT_TOLERANCE INCLUDE_ZERO <<EOF
$(python - "${REFERENCE}" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    reduction = json.load(handle)
print(
    reduction.get("metric", "delta_vs_do_nothing"),
    reduction.get("selection_method", "best_per_state"),
    int(reduction.get("min_count", 1)),
    str(reduction.get("require_improvement", True)).lower(),
    reduction.get("improvement_tolerance", 1e-3),
    str(reduction.get("include_action_zero", True)).lower(),
)
PY
)
EOF

echo "========== WCCI action-space reduction sweep =========="
echo "Dataset dirs: ${#DATASET_DIRS[@]} (first: ${DATASET_DIRS[0]})"
echo "Reference:    ${REFERENCE}"
echo "Settings:     metric=${METRIC} method=${METHOD} min_count=${MIN_COUNT}"
echo "              require_improvement=${REQUIRE_IMPROVEMENT} tolerance=${IMPROVEMENT_TOLERANCE}"
echo "Sizes:        ${TOP_KS}"
echo "======================================================="

for top_k in ${TOP_KS}; do
    output="${DATASET_ROOT}/metadata/reduced_action_space_${NAME_PREFIX}_mk${top_k}.json"
    if [ -f "${output}" ] && [ "${FORCE}" != "true" ]; then
        echo "skip mk${top_k}: ${output} already exists (FORCE=true to regenerate)"
        continue
    fi
    echo "--- mk${top_k} -> ${output}"
    python -u teacher_student/reduce_action_space_from_outcomes.py \
        --dataset "${DATASET_DIRS[@]}" \
        --output "${output}" \
        --metric "${METRIC}" \
        --selection-method "${METHOD}" \
        --top-k "${top_k}" \
        --min-count "${MIN_COUNT}" \
        --include-action-zero "${INCLUDE_ZERO}" \
        --require-improvement "${REQUIRE_IMPROVEMENT}" \
        --improvement-tolerance "${IMPROVEMENT_TOLERANCE}"
done
