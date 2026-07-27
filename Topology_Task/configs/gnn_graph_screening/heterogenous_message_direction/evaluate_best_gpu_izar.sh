#!/usr/bin/env bash
set -euo pipefail

task_dir="Topology_Task"
checkpoint_dir="${task_dir}/checkpoint"
evaluator="${task_dir}/full_test_eval/job_full_test_eval_izar.sh"

checkpoints=(
  "best_test_gs_hmd_hetero_n0_none_ga2b_la2b_s0.tar"
  "best_test_gs_hmd_hetero_n0_none_ga2b_la2b_s2.tar"
  "best_test_gs_hmd_hetero_n0_none_ga2b_lb2a_s0.tar"
  "best_test_gs_hmd_hetero_n0_none_ga2b_lbi_s2.tar"
  "best_test_gs_hmd_hetero_n0_none_gb2a_la2b_s2.tar"
  "best_test_gs_hmd_hetero_n0_none_gb2a_lb2a_s0.tar"
  "best_test_gs_hmd_hetero_n0_none_gb2a_lb2a_s1.tar"
  "best_test_gs_hmd_hetero_n0_none_gb2a_lbi_s1.tar"
  "best_test_gs_hmd_hetero_n0_none_gbi_la2b_s0.tar"
  "best_test_gs_hmd_hetero_n0_none_gbi_lb2a_s1.tar"
  "best_test_gs_hmd_hetero_n0_none_gbi_lbi_s1.tar"
  "best_test_gs_hmd_hetero_n0_none_gbi_lbi_s2.tar"
)

if [[ ! -f "${evaluator}" ]]; then
  echo "Run this script from the RL_Marl2grid repository root." >&2
  exit 1
fi

missing=()
for checkpoint in "${checkpoints[@]}"; do
  if [[ ! -f "${checkpoint_dir}/${checkpoint}" ]]; then
    missing+=("${checkpoint}")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "Missing ${#missing[@]} Izar best checkpoint(s); no jobs were submitted:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
  sbatch "${evaluator}" \
    --checkpoint "checkpoint/${checkpoint}" \
    --split test \
    --split-chronics true \
    --eval-all-split-chronics true \
    --deterministic-eval true \
    --eval-action-heuristic none \
    --obs-normalization require \
    --device cuda \
    --progress true
done
