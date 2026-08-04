#!/usr/bin/env bash
# Report which stage-1 source checkpoints exist, so stage 2 is only launched
# for pairs that can actually load an encoder.
#
# Run from the repository root:
#   Topology_Task/configs/transfer_scaled/check_sources.sh

set -uo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
TASK_DIR="${REPO_DIR}/Topology_Task"
if [ ! -d "${TASK_DIR}" ]; then
    echo "Could not find ${TASK_DIR}. Run from the repository root, or set REPO_DIR." >&2
    exit 1
fi
cd "${TASK_DIR}"

missing=0
found=0
for config in configs/transfer_scaled/wcci/*_frozen_*.toml; do
    checkpoint="$(sed -n 's/^transfer_encoder_checkpoint = "\(.*\)"$/\1/p' "${config}")"
    if [ -z "${checkpoint}" ]; then
        echo "no checkpoint declared in ${config}" >&2
        missing=$((missing + 1))
        continue
    fi
    if [ -f "${checkpoint}" ]; then
        printf 'ok      %-34s %s\n' "$(basename "${config}" .toml)" "${checkpoint}"
        found=$((found + 1))
    else
        printf 'MISSING %-34s %s\n' "$(basename "${config}" .toml)" "${checkpoint}"
        missing=$((missing + 1))
    fi
done

echo
echo "${found} ready, ${missing} missing"
if [ "${missing}" -gt 0 ]; then
    echo "Launch or finish the corresponding stage 1 runs first:"
    echo "  Topology_Task/configs/transfer_scaled/sources/launch_izar.sh"
    exit 1
fi
