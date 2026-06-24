#!/usr/bin/env bash
set -euo pipefail

sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_flat_p000_s0.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_flat_p000_s1.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_flat_p000_s2.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_flat_p001_s0.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_flat_p001_s1.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_flat_p001_s2.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_flat_p003_s0.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_flat_p003_s1.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_flat_p003_s2.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_flat_p010_s0.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_flat_p010_s1.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_flat_p010_s2.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_gated_p000_s0.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_gated_p000_s1.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_gated_p000_s2.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_gated_p001_s0.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_gated_p001_s1.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_gated_p001_s2.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_gated_p003_s0.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_gated_p003_s1.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_gated_p003_s2.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_gated_p010_s0.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_gated_p010_s1.toml
sbatch job_jed.sh configs/a0_sparse16/a0_sparse16_gated_p010_s2.toml
