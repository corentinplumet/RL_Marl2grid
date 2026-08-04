#!/usr/bin/env bash
# Stage 2: WCCI transfer arms with corrected inputs.
# Requires the stage 1 checkpoints; run check_sources.sh first.
# Pass a filter to launch a subset, e.g. ./launch_izar.sh _s0
set -euo pipefail

FILTER="${1:-}"

if [[ -z "$FILTER" || "trs_hmd_gb2a_la2b_frozen_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_hmd_gb2a_la2b_frozen_s0.toml
fi
if [[ -z "$FILTER" || "trs_hmd_gb2a_la2b_scratch_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_hmd_gb2a_la2b_scratch_s0.toml
fi
if [[ -z "$FILTER" || "trs_hmd_gb2a_la2b_frozen_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_hmd_gb2a_la2b_frozen_s1.toml
fi
if [[ -z "$FILTER" || "trs_hmd_gb2a_la2b_scratch_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_hmd_gb2a_la2b_scratch_s1.toml
fi
if [[ -z "$FILTER" || "trs_hmd_gb2a_la2b_frozen_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_hmd_gb2a_la2b_frozen_s2.toml
fi
if [[ -z "$FILTER" || "trs_hmd_gb2a_la2b_scratch_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_hmd_gb2a_la2b_scratch_s2.toml
fi
if [[ -z "$FILTER" || "trs_hmd_gb2a_lb2a_frozen_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_hmd_gb2a_lb2a_frozen_s0.toml
fi
if [[ -z "$FILTER" || "trs_hmd_gb2a_lb2a_scratch_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_hmd_gb2a_lb2a_scratch_s0.toml
fi
if [[ -z "$FILTER" || "trs_hmd_gb2a_lb2a_frozen_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_hmd_gb2a_lb2a_frozen_s1.toml
fi
if [[ -z "$FILTER" || "trs_hmd_gb2a_lb2a_scratch_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_hmd_gb2a_lb2a_scratch_s1.toml
fi
if [[ -z "$FILTER" || "trs_hmd_gb2a_lb2a_frozen_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_hmd_gb2a_lb2a_frozen_s2.toml
fi
if [[ -z "$FILTER" || "trs_hmd_gb2a_lb2a_scratch_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_hmd_gb2a_lb2a_scratch_s2.toml
fi
if [[ -z "$FILTER" || "trs_s2_e0n0v0_frozen_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_s2_e0n0v0_frozen_s0.toml
fi
if [[ -z "$FILTER" || "trs_s2_e0n0v0_scratch_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_s2_e0n0v0_scratch_s0.toml
fi
if [[ -z "$FILTER" || "trs_s2_e0n0v0_frozen_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_s2_e0n0v0_frozen_s1.toml
fi
if [[ -z "$FILTER" || "trs_s2_e0n0v0_scratch_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_s2_e0n0v0_scratch_s1.toml
fi
if [[ -z "$FILTER" || "trs_s2_e0n0v0_frozen_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_s2_e0n0v0_frozen_s2.toml
fi
if [[ -z "$FILTER" || "trs_s2_e0n0v0_scratch_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_s2_e0n0v0_scratch_s2.toml
fi
if [[ -z "$FILTER" || "trs_s2_e1n0v0_frozen_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_s2_e1n0v0_frozen_s0.toml
fi
if [[ -z "$FILTER" || "trs_s2_e1n0v0_scratch_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_s2_e1n0v0_scratch_s0.toml
fi
if [[ -z "$FILTER" || "trs_s2_e1n0v0_frozen_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_s2_e1n0v0_frozen_s1.toml
fi
if [[ -z "$FILTER" || "trs_s2_e1n0v0_scratch_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_s2_e1n0v0_scratch_s1.toml
fi
if [[ -z "$FILTER" || "trs_s2_e1n0v0_frozen_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_s2_e1n0v0_frozen_s2.toml
fi
if [[ -z "$FILTER" || "trs_s2_e1n0v0_scratch_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/wcci/trs_s2_e1n0v0_scratch_s2.toml
fi
