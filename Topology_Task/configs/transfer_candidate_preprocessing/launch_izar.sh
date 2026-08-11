#!/usr/bin/env bash
set -euo pipefail

# Launch from Topology_Task: bash configs/transfer_candidate_preprocessing/launch_izar.sh

required_checkpoints=(
  "checkpoint/no_leak/cas_hl_NL/best_test_cas_hl_NL_mean_f1_a0h0_s0.tar"
  "checkpoint/no_leak/cas_hl_NL/best_test_cas_hl_NL_mean_f0_a0h1_s0.tar"
  "checkpoint/no_leak/cas_hl_NL/best_test_cas_hl_NL_mean_f0_a0h0_s0.tar"
  "checkpoint/no_leak/NL_cas_hl_scaled/best_test_cas_hl_NLS_mean_f1_a0h0_s0.tar"
  "checkpoint/no_leak/NL_cas_hl_scaled/best_test_cas_hl_NLS_mean_f0_a0h1_s0.tar"
  "checkpoint/no_leak/NL_cas_hl_scaled/best_test_cas_hl_NLS_mean_f0_a0h0_s0.tar"
)

for checkpoint in "${required_checkpoints[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing source checkpoint: $checkpoint" >&2
    exit 1
  fi
done

sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NL_mean_f1_a0h0_frozen_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NL_mean_f1_a0h0_scratch_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NL_mean_f1_a0h0_finetune_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NL_mean_f0_a0h1_frozen_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NL_mean_f0_a0h1_scratch_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NL_mean_f0_a0h1_finetune_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NL_mean_f0_a0h0_frozen_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NL_mean_f0_a0h0_scratch_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NL_mean_f0_a0h0_finetune_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NLS_mean_f1_a0h0_frozen_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NLS_mean_f1_a0h0_scratch_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NLS_mean_f1_a0h0_finetune_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NLS_mean_f0_a0h1_frozen_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NLS_mean_f0_a0h1_scratch_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NLS_mean_f0_a0h1_finetune_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NLS_mean_f0_a0h0_frozen_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NLS_mean_f0_a0h0_scratch_s0.toml
sbatch job_izar.sh configs/transfer_candidate_preprocessing/trcas_NLS_mean_f0_a0h0_finetune_s0.toml
