#!/usr/bin/env bash
# Launch the bus14 -> WCCI(no-maintenance) transfer study.
# Pass a filter to launch a subset, e.g. ./launch_izar.sh _s0
set -euo pipefail

FILTER="${1:-}"

if [[ -z "$FILTER" || "tr_hmd_gb2a_la2b_frozen_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_hmd_gb2a_la2b_frozen_s0.toml
fi
if [[ -z "$FILTER" || "tr_hmd_gb2a_la2b_scratch_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_hmd_gb2a_la2b_scratch_s0.toml
fi
if [[ -z "$FILTER" || "tr_hmd_gb2a_la2b_frozen_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_hmd_gb2a_la2b_frozen_s1.toml
fi
if [[ -z "$FILTER" || "tr_hmd_gb2a_la2b_scratch_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_hmd_gb2a_la2b_scratch_s1.toml
fi
if [[ -z "$FILTER" || "tr_hmd_gb2a_la2b_frozen_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_hmd_gb2a_la2b_frozen_s2.toml
fi
if [[ -z "$FILTER" || "tr_hmd_gb2a_la2b_scratch_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_hmd_gb2a_la2b_scratch_s2.toml
fi
if [[ -z "$FILTER" || "tr_hmd_gb2a_lb2a_frozen_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_hmd_gb2a_lb2a_frozen_s0.toml
fi
if [[ -z "$FILTER" || "tr_hmd_gb2a_lb2a_scratch_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_hmd_gb2a_lb2a_scratch_s0.toml
fi
if [[ -z "$FILTER" || "tr_hmd_gb2a_lb2a_frozen_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_hmd_gb2a_lb2a_frozen_s1.toml
fi
if [[ -z "$FILTER" || "tr_hmd_gb2a_lb2a_scratch_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_hmd_gb2a_lb2a_scratch_s1.toml
fi
if [[ -z "$FILTER" || "tr_hmd_gb2a_lb2a_frozen_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_hmd_gb2a_lb2a_frozen_s2.toml
fi
if [[ -z "$FILTER" || "tr_hmd_gb2a_lb2a_scratch_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_hmd_gb2a_lb2a_scratch_s2.toml
fi
if [[ -z "$FILTER" || "tr_s2_e0n0v0_frozen_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_s2_e0n0v0_frozen_s0.toml
fi
if [[ -z "$FILTER" || "tr_s2_e0n0v0_scratch_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_s2_e0n0v0_scratch_s0.toml
fi
if [[ -z "$FILTER" || "tr_s2_e0n0v0_frozen_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_s2_e0n0v0_frozen_s1.toml
fi
if [[ -z "$FILTER" || "tr_s2_e0n0v0_scratch_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_s2_e0n0v0_scratch_s1.toml
fi
if [[ -z "$FILTER" || "tr_s2_e0n0v0_frozen_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_s2_e0n0v0_frozen_s2.toml
fi
if [[ -z "$FILTER" || "tr_s2_e0n0v0_scratch_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_s2_e0n0v0_scratch_s2.toml
fi
if [[ -z "$FILTER" || "tr_s2_e1n0v0_frozen_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_s2_e1n0v0_frozen_s0.toml
fi
if [[ -z "$FILTER" || "tr_s2_e1n0v0_scratch_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_s2_e1n0v0_scratch_s0.toml
fi
if [[ -z "$FILTER" || "tr_s2_e1n0v0_frozen_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_s2_e1n0v0_frozen_s1.toml
fi
if [[ -z "$FILTER" || "tr_s2_e1n0v0_scratch_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_s2_e1n0v0_scratch_s1.toml
fi
if [[ -z "$FILTER" || "tr_s2_e1n0v0_frozen_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_s2_e1n0v0_frozen_s2.toml
fi
if [[ -z "$FILTER" || "tr_s2_e1n0v0_scratch_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_bus14_to_wcci_nomaint/tr_s2_e1n0v0_scratch_s2.toml
fi
