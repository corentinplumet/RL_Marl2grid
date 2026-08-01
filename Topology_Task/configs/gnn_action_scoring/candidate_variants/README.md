# Candidate-Action Scoring Variants

This folder contains the complete candidate-head factorial on bus14:

```text
2 pooling modes x 2 action-feature settings x 2 action-0 head settings
x 3 seeds = 24 runs
```

All configurations are copied from:

```text
configs/gnn_graph_screening/stage1e_heterogeneous_structure/
gs_s1e_hetero_line_n0_none_e0n0v0_s0.toml
```

The heterogeneous-line encoder, message directions, critic, PPO settings,
reward, chronic split, normalization, and 8M-step budget are unchanged.
Only the candidate-head flags and training seed vary. The chronic split seed
stays fixed at 0 for paired comparisons.

## Naming

```text
cas_hl_<pool>_<features>_<action0-head>_s<seed>
```

- `mean`: one mean over all affected node types;
- `tmean`: separate typed means for busbars, lines, loads, and generators;
- `f0` / `f1`: static action features disabled / enabled;
- `a0h0` / `a0h1`: dedicated state-dependent action-0 head disabled / enabled;
- `s0`, `s1`, `s2`: training seeds 0, 1, and 2.

Every run uses:

```toml
actor_action_head = "candidate_pool"
intervention_gate = false
gnn_graph_type = "heterogeneous_line"
gnn_include_neighbors = true
```

## Launch On JED

From `/home/plumet/RL_Marl2grid`:

```bash
bash Topology_Task/configs/gnn_action_scoring/candidate_variants/launch_all.sh
```

To launch only seed 0:

```bash
for cfg in Topology_Task/configs/gnn_action_scoring/candidate_variants/*_s0.toml; do
  sbatch --time=7-00:00:00 job_jed.sh "${cfg#Topology_Task/}"
done
```

The complete factor table is in `manifest.csv`.
