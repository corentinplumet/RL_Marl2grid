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

New collections use a split layout:

```text
outputs/teacher_student_datasets/local_rho090_s0/
  shards/
    shard_00000.npz
    shard_00001.npz
    ...
  metadata/
    metadata.json
```

They also write uniquely named lightweight copies next to all datasets:

```text
outputs/teacher_student_datasets/metadata_exports/
  local_rho090_s0_metadata.json
  local_rho090_s0_summary.json
  local_rho090_s1_metadata.json
  local_rho090_s1_summary.json
  ...
```

This keeps the heavy `.npz` files separate from the lightweight files you might
want to copy locally for plotting. If you only want dataset summary plots on
your laptop, download `outputs/teacher_student_datasets/metadata_exports/` and
skip the per-dataset `shards/` folders.

## Summarize A Dataset

```bash
python Topology_Task/teacher_student/summarize_dataset.py \
  --dataset outputs/teacher_student_datasets/local_rho090_s0
```

By default the summary is written to:

```text
outputs/teacher_student_datasets/local_rho090_s0/metadata/summary.json
```

and copied to:

```text
outputs/teacher_student_datasets/metadata_exports/local_rho090_s0_summary.json
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
../metadata_exports/
  local_rho090_s0_metadata.json
  local_rho090_s0_summary.json
metadata/
  metadata.json
  summary.json
shards/
  shard_00000.npz
  shard_00001.npz
  ...
```

The loader remains backward compatible with the previous flat layout where
`metadata.json`, `summary.json`, and `shard_*.npz` were all directly under the
dataset directory.

## Organize Existing Datasets

For datasets already collected with the old flat layout, run this on the
cluster. It moves files in place, so it does not duplicate the heavy shards,
and it creates/refreshes the uniquely named `metadata_exports/*.json` files:

```bash
python Topology_Task/teacher_student/organize_dataset_layout.py \
  --root outputs/teacher_student_datasets
```

Dry run first if you want to inspect the moves:

```bash
python Topology_Task/teacher_student/organize_dataset_layout.py \
  --root outputs/teacher_student_datasets \
  --dry-run true
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

## Train A Behavior-Cloned Student

Smoke test one shard first:

```bash
python Topology_Task/teacher_student/train_student_bc.py \
  --dataset outputs/teacher_student_datasets/local_rho090_s0 \
  --teacher-checkpoint checkpoint/with_obs_stats/a0_hvg_04_eval_local_rho090_s0.tar \
  --output checkpoint/teacher_student/smoke_local_bc_s0.tar \
  --epochs 1 \
  --max-shards 1 \
  --batch-size 4096 \
  --balanced-nonidle-frac 0.5 \
  --nonidle-weight 5.0 \
  --aux-intervention-loss true \
  --aux-weight 0.5
```

Full cluster run:

```bash
sbatch Topology_Task/teacher_student/job_train_student_bc.sh \
  --dataset outputs/teacher_student_datasets/local_rho090_s0 \
  --teacher-checkpoint checkpoint/with_obs_stats/a0_hvg_04_eval_local_rho090_s0.tar \
  --output checkpoint/teacher_student/local_bc_s0.tar \
  --epochs 3 \
  --batch-size 4096 \
  --balanced-nonidle-frac 0.5 \
  --nonidle-weight 5.0 \
  --aux-intervention-loss true \
  --aux-weight 0.5
```

The trainer initializes the student from the teacher actor weights by default
and then trains each decentralized actor on its own agent labels.

The saved checkpoint is intended to run without a heuristic:

```bash
python Topology_Task/full_test_eval/evaluate_checkpoint.py \
  --checkpoint checkpoint/teacher_student/local_bc_s0.tar \
  --split test \
  --eval-all-split-chronics true \
  --eval-action-heuristic none \
  --obs-normalization require \
  --progress true
```
