#!/usr/bin/env bash
# Stage 1: bus14 sources with corrected inputs.
# Pass a filter to launch a subset, e.g. ./launch_izar.sh _s0
set -euo pipefail

FILTER="${1:-}"

if [[ -z "$FILTER" || "trsrc_hmd_gb2a_la2b_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/sources/trsrc_hmd_gb2a_la2b_s0.toml
fi
if [[ -z "$FILTER" || "trsrc_hmd_gb2a_la2b_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/sources/trsrc_hmd_gb2a_la2b_s1.toml
fi
if [[ -z "$FILTER" || "trsrc_hmd_gb2a_la2b_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/sources/trsrc_hmd_gb2a_la2b_s2.toml
fi
if [[ -z "$FILTER" || "trsrc_hmd_gb2a_lb2a_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/sources/trsrc_hmd_gb2a_lb2a_s0.toml
fi
if [[ -z "$FILTER" || "trsrc_hmd_gb2a_lb2a_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/sources/trsrc_hmd_gb2a_lb2a_s1.toml
fi
if [[ -z "$FILTER" || "trsrc_hmd_gb2a_lb2a_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/sources/trsrc_hmd_gb2a_lb2a_s2.toml
fi
if [[ -z "$FILTER" || "trsrc_s2_e0n0v0_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/sources/trsrc_s2_e0n0v0_s0.toml
fi
if [[ -z "$FILTER" || "trsrc_s2_e0n0v0_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/sources/trsrc_s2_e0n0v0_s1.toml
fi
if [[ -z "$FILTER" || "trsrc_s2_e0n0v0_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/sources/trsrc_s2_e0n0v0_s2.toml
fi
if [[ -z "$FILTER" || "trsrc_s2_e1n0v0_s0" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/sources/trsrc_s2_e1n0v0_s0.toml
fi
if [[ -z "$FILTER" || "trsrc_s2_e1n0v0_s1" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/sources/trsrc_s2_e1n0v0_s1.toml
fi
if [[ -z "$FILTER" || "trsrc_s2_e1n0v0_s2" == *"$FILTER"* ]]; then
    sbatch job_izar.sh configs/transfer_scaled/sources/trsrc_s2_e1n0v0_s2.toml
fi
