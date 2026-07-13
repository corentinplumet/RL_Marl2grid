#!/usr/bin/env bash
set -euo pipefail

sbatch job_jed.sh configs/wcci_sparse16/wcci_sparse16_flat_p003_72x576_s0.toml
sbatch job_jed.sh configs/wcci_sparse16/wcci_sparse16_flat_p003_72x576_s1.toml
sbatch job_jed.sh configs/wcci_sparse16/wcci_sparse16_flat_p003_72x576_s2.toml
