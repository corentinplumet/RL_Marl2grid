# Topology Task Tools

Small utility scripts for inspecting models and checkpoint folders.

Run commands from the repository root:

```bash
cd /path/to/RL_Marl2grid
```

Use the same Python environment as training/evaluation on the cluster, because
the checkpoint tools need PyTorch and the model-count script may need Grid2Op,
LightSim, and PyTorch Geometric.

## Count Model Parameters

Script:

```bash
Topology_Task/tools/count_model_parameters.py
```

Use it to build the actor/critic defined by a TOML config and print the number
of parameters.

Human-readable output:

```bash
python Topology_Task/tools/count_model_parameters.py \
  Topology_Task/configs/gine_s0_s1_s2/best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update_s0.toml
```

JSON output:

```bash
python Topology_Task/tools/count_model_parameters.py \
  --json \
  Topology_Task/configs/gine_s0_s1_s2/best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update_s0.toml
```

You can pass command-line overrides after the config path:

```bash
python Topology_Task/tools/count_model_parameters.py \
  Topology_Task/configs/gine_s0_s1_s2/best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update_s0.toml \
  --intervention-gate true
```

If a GNN config fails with a PyTorch Geometric import error, run the command in
the same conda environment used for GNN training.

## Organize Checkpoints

Script:

```bash
Topology_Task/tools/organize_checkpoints.py
```

Use it after downloading or finishing trainings to split checkpoint files into:

```text
checkpoint/with_obs_stats
checkpoint/without_obs_stats
```

It then renames checkpoints with observation stats using the saved `exp_tag`:

```text
regular checkpoint:   {exp_tag}.tar
best test checkpoint: best_test_{exp_tag}.tar
final checkpoint:     final_{exp_tag}.tar
```

Dry run first:

```bash
python Topology_Task/tools/organize_checkpoints.py \
  --checkpoint-dir Topology_Task/checkpoint \
  --dry-run
```

Apply the organization:

```bash
python Topology_Task/tools/organize_checkpoints.py \
  --checkpoint-dir Topology_Task/checkpoint
```

Write a JSON report:

```bash
python Topology_Task/tools/organize_checkpoints.py \
  --checkpoint-dir Topology_Task/checkpoint \
  --report-json checkpoint_organize_report.json
```

Do not run this on a live checkpoint directory while training jobs are writing
to it. Run it after jobs finish, or run it on a copy of the checkpoint folder.

## Check Checkpoint Health

Script:

```bash
Topology_Task/tools/check_checkpoint_health.py
```

Use it to list which runs are complete, incomplete, or corrupted. A run is
considered complete when it has a readable final checkpoint or a readable
checkpoint whose `global_step` reaches the saved `total_timesteps`.

Full report:

```bash
python Topology_Task/tools/check_checkpoint_health.py \
  --checkpoint-dir Topology_Task/checkpoint
```

Show only corrupted or incomplete runs:

```bash
python Topology_Task/tools/check_checkpoint_health.py \
  --checkpoint-dir Topology_Task/checkpoint \
  --only-problems
```

Inspect one run, including regular, `best_test`, `final`, old `MAPPO_...`
filenames, and readable aliases recovered from checkpoint metadata:

```bash
python Topology_Task/tools/check_checkpoint_health.py \
  --checkpoint-dir Topology_Task/checkpoint \
  --run rerun_a0_opt_s0
```

The `--run` filter accepts an `exp_tag`, a checkpoint filename substring, a W&B
run path substring, or an old opaque `MAPPO_...` run id.

Safer check while jobs might still be running:

```bash
python Topology_Task/tools/check_checkpoint_health.py \
  --checkpoint-dir Topology_Task/checkpoint \
  --only-problems \
  --min-age-seconds 120
```

The `--min-age-seconds 120` option skips `.tar` files modified in the last two
minutes, which avoids reading a file while a training job is still writing it.

Print the files under each run:

```bash
python Topology_Task/tools/check_checkpoint_health.py \
  --checkpoint-dir Topology_Task/checkpoint \
  --show-files
```

Export reports:

```bash
python Topology_Task/tools/check_checkpoint_health.py \
  --checkpoint-dir Topology_Task/checkpoint \
  --csv checkpoint_health.csv \
  --json checkpoint_health.json
```

Common status labels:

```text
complete                    readable final/completed run, no corrupted files
incomplete                  readable checkpoint exists, but no completed final
corrupted                   no readable checkpoint for that inferred run
complete_with_corruption    completed run exists, but at least one file is bad
incomplete_with_corruption  incomplete run with at least one bad file
```
