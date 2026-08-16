# No-Leak Candidate-Action Scoring Runs

This folder recreates every historical `cas_hl` experiment on the corrected
local-graph information boundary. The new family uses the `cas_hl_NL` prefix
so its W&B metadata, checkpoints, scheduler names, and output directories do
not collide with the historical runs.

## Inventory

There are 33 paired configurations:

- 24 uniform candidate-pooling runs: `mean` and `typed_mean`, with both action
  feature settings, both action-0 head settings, and seeds 0, 1, and 2;
- 9 adaptive typed-attention runs: `affected`, `soft_prior`, and `all`, with
  seeds 0, 1, and 2.

The optimizer, GINE encoder, critic, reward, rollout, evaluation, and 15M-step
training settings are unchanged. Each recreated config changes only:

```toml
name = "cas_hl_NL_<variant>"
exp_tag = "cas_hl_NL_<variant>"
gnn_include_neighbors = false
gnn_context_requires_connection = true
```

## Information boundary

All runs use `gnn_graph_type = "heterogeneous_line"`. In the corrected graph
builder, an actor's local graph contains only:

- busbars, generators, and loads in substations controlled by that actor;
- every transmission line touching the controlled domain as an explicit line
  node;
- line-node attachment edges only at endpoints controlled by that actor.

The far-end neighboring busbar, generator, and load are not constructed. They
therefore cannot enter either the two GINE message-passing layers or any of the
candidate-pooling modes. For `soft_prior` and `all`, "all nodes" now means all
allowed nodes in this leakage-free local graph.

These configs must be run from the fixed `pooling` branch (or a descendant
containing commit `8ba467a`). Do not run them from the historical
`codex/gnn-candidate-action-scoring` branch.

## Validate submissions on JED

From the repository root:

```bash
DRY_RUN=true bash \
  Topology_Task/configs/gnn_action_scoring/no_leak/launch_all.sh
```

The dry run must report 33 submissions.

## Optional one-rollout smoke test

```bash
N_ENVS=12 \
TOTAL_TIMESTEPS=6912 \
TRACK=false \
CHECKPOINT=false \
sbatch \
  --job-name=cas_hl_NL_smoke \
  --cpus-per-task=12 \
  --time=01:00:00 \
  job_jed.sh \
  configs/gnn_action_scoring/no_leak/cas_hl_NL_mean_f1_a0h0_s0.toml
```

## Launch all runs on JED

```bash
bash Topology_Task/configs/gnn_action_scoring/no_leak/launch_all.sh
```

Override the seven-day request if needed:

```bash
JED_WALLTIME=5-00:00:00 \
bash Topology_Task/configs/gnn_action_scoring/no_leak/launch_all.sh
```

Monitor the jobs and logs with:

```bash
squeue -u "$USER" -o "%.18i %.55j %.2t %.10M %.6D %R"
tail -F Topology_Task/slurm-cas_hl_NL_*.out
```
