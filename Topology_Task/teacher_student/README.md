# Teacher-Student Dataset Collection

This folder contains Phase 1 of the heuristic teacher-student roadmap:
collecting behavior-cloning datasets from a trained MAPPO policy with an
evaluation-time rho heuristic.

The teacher is:

```text
trained MAPPO policy + eval_action_heuristic
```

The label saved for supervised learning is the final executed teacher action:

```text
teacher_action = action after the heuristic override
```

The policy proposal before override is also saved:

```text
policy_action = action proposed by the actor before the heuristic
```

## Collect A Dataset

Run from the repository root:

```bash
sbatch Topology_Task/teacher_student/job_collect_teacher_dataset.sh \
  --checkpoint checkpoint/with_obs_stats/best_test_a0_hvg_04_eval_local_rho090_s0.tar \
  --split train \
  --eval-all-split-chronics true \
  --eval-action-heuristic local_rho_threshold \
  --eval-action-rho-threshold 0.90 \
  --obs-normalization require \
  --max-episodes 803 \
  --output-dir outputs/teacher_student_datasets/local_rho090_s0
```

Useful smoke test:

```bash
python Topology_Task/teacher_student/collect_teacher_dataset.py \
  --checkpoint checkpoint/with_obs_stats/best_test_a0_hvg_04_eval_local_rho090_s0.tar \
  --split train \
  --eval-action-heuristic local_rho_threshold \
  --eval-action-rho-threshold 0.90 \
  --obs-normalization require \
  --max-episodes 2 \
  --output-dir outputs/teacher_student_datasets/smoke_local_rho090_s0
```

By default, the collector refuses to write into a non-empty output directory.
Use a unique `--output-dir` per checkpoint/seed.

## Summarize A Dataset

```bash
python Topology_Task/teacher_student/summarize_dataset.py \
  --dataset outputs/teacher_student_datasets/local_rho090_s0
```

The summary reports:

```text
teacher action-0 fraction
teacher non-idle fraction
base policy non-idle fraction
heuristic overwrite fraction
unique chronic fingerprints
episode length statistics
shard readability
```

## Output Format

Each dataset directory contains:

```text
metadata.json
summary.json
shard_00000.npz
shard_00001.npz
...
```

Each shard stores one row per environment step. Per-agent arrays are stored with
the agent id in the key:

```text
obs_agent_0
policy_action_agent_0
teacher_action_agent_0
force_noop_agent_0
was_overwritten_agent_0
local_max_rho_agent_0
```

The observation is the normalized actor input after the checkpoint's saved
observation statistics are applied.

## Current Scope

This Phase 1 implementation supports flat MLP observations. If a checkpoint
uses graph observations, the collector fails clearly instead of saving an
ambiguous graph object format. Graph student datasets can be added after the MLP
teacher-student path is validated.
