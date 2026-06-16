#!/usr/bin/env bash
set -euo pipefail

sbatch job_jed.sh configs/phase3_flat_vs_gated_s0/phase3_00_flat_no_penalty_s0.toml
sbatch job_jed.sh configs/phase3_flat_vs_gated_s0/phase3_01_flat_penalty001_s0.toml
sbatch job_jed.sh configs/phase3_flat_vs_gated_s0/phase3_02_gated_no_penalty_s0.toml
sbatch job_jed.sh configs/phase3_flat_vs_gated_s0/phase3_03_gated_penalty001_s0.toml
