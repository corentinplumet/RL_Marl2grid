#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

config_dir="Topology_Task/configs/gnn_action_scoring/candidate_variants"
submitted=0
for config_path in "$config_dir"/*.toml; do
  sbatch --time=7-00:00:00 job_jed.sh "${config_path#Topology_Task/}"
  submitted=$((submitted + 1))
done

echo "Submitted $submitted candidate-action scoring runs on JED."
