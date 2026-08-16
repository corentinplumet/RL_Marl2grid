#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

configs=(
  "configs/adaptive_intervention_budget_7/aib_01_flat_local_t010_s1.toml"
  "configs/adaptive_intervention_budget_7/aib_01_flat_local_t010_s2.toml"
  "configs/adaptive_intervention_budget_7/aib_02_flat_local_t035_s2.toml"
  "configs/adaptive_intervention_budget_7/aib_03_gate_hgreedy_sep_local_t020_s1.toml"
  "configs/adaptive_intervention_budget_7/aib_03_gate_hgreedy_sep_local_t020_s2.toml"
  "configs/adaptive_intervention_budget_7/aib_04_flat_nonidle_t020_s0.toml"
  "configs/adaptive_intervention_budget_7/aib_04_flat_nonidle_t020_s1.toml"
  "configs/adaptive_intervention_budget_7/aib_04_flat_nonidle_t020_s2.toml"
)

for config in "${configs[@]}"; do
  sbatch job_jed.sh "${config}"
done
