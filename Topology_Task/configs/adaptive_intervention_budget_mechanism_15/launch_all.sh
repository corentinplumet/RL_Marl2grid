#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

configs=(
  "configs/adaptive_intervention_budget_mechanism_15/aibm_00_fixed_nonidle_p006_s0.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_00_fixed_nonidle_p006_s1.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_00_fixed_nonidle_p006_s2.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_01_fixed_global_safe_p006_s0.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_01_fixed_global_safe_p006_s1.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_01_fixed_global_safe_p006_s2.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_02_adaptive_global_t020_s0.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_02_adaptive_global_t020_s1.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_02_adaptive_global_t020_s2.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_03_adaptive_local_t020_r085_s0.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_03_adaptive_local_t020_r085_s1.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_03_adaptive_local_t020_r085_s2.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_04_adaptive_local_t020_r095_s0.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_04_adaptive_local_t020_r095_s1.toml"
  "configs/adaptive_intervention_budget_mechanism_15/aibm_04_adaptive_local_t020_r095_s2.toml"
)

for config in "${configs[@]}"; do
  sbatch job_jed.sh "${config}"
done
