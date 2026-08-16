#!/usr/bin/env bash
set -euo pipefail

sbatch job_jed.sh configs/a0_hvg/a0_hvg_00_baseline_s0.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_00_baseline_s1.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_00_baseline_s2.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_01_eval_rho090_s0.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_01_eval_rho090_s1.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_01_eval_rho090_s2.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_02_gate_final_map_s0.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_02_gate_final_map_s1.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_02_gate_final_map_s2.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_03_gate_hierarchical_s0.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_03_gate_hierarchical_s1.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_03_gate_hierarchical_s2.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_04_eval_local_rho090_s0.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_04_eval_local_rho090_s1.toml
sbatch job_jed.sh configs/a0_hvg/a0_hvg_04_eval_local_rho090_s2.toml
