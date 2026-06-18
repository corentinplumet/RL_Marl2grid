#!/usr/bin/env bash
set -euo pipefail

sbatch job_jed.sh configs/heuristic_vs_gate_s0_s1/hvg_00_baseline_s0.toml
sbatch job_jed.sh configs/heuristic_vs_gate_s0_s1/hvg_01_eval_rho090_s0.toml
sbatch job_jed.sh configs/heuristic_vs_gate_s0_s1/hvg_02_gate_final_map_s0.toml
sbatch job_jed.sh configs/heuristic_vs_gate_s0_s1/hvg_03_gate_hierarchical_s0.toml
sbatch job_jed.sh configs/heuristic_vs_gate_s0_s1/hvg_00_baseline_s1.toml
sbatch job_jed.sh configs/heuristic_vs_gate_s0_s1/hvg_01_eval_rho090_s1.toml
sbatch job_jed.sh configs/heuristic_vs_gate_s0_s1/hvg_02_gate_final_map_s1.toml
sbatch job_jed.sh configs/heuristic_vs_gate_s0_s1/hvg_03_gate_hierarchical_s1.toml
