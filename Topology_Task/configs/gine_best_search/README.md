# GINE Best-Search Configs

This folder is a focused follow-up to the numbered GINE batch and the historical best run
`shared_gine_a4_concat_flat_no_entropy_decay_s1_det`.

Known-good ingredients kept fixed in the search configs unless stated otherwise:

- A4 exploration: `entropy_coef = 0.02`, `init_do_nothing_prob = 0.7`
- `gnn_type = "gine"`
- bus graph: `gnn_graph_type = "bus"`
- no neighbor expansion: `gnn_include_neighbors = false`
- mean readout: `gnn_readout_aggr = "mean"`
- learned node IDs: `gnn_node_id_embeddings = true`
- shared actor GNN: `share_actor_gnn = true`
- deterministic evaluation

Most configs use robust-but-not-crazy evaluation:

- `eval_freq = 80000`
- `eval_episodes = 10`
- `eval_train_chronics = false`

The exception is `best_00_rerun_usual_shared_gine_a4_concat_flat_s1`, which intentionally keeps the
historical best config's evaluation behavior and adds `optimize_critic_updates = false` so it preserves
legacy critic-update behavior on the optimization branch.

| Config | Purpose |
|---|---|
| `best_00_rerun_usual_shared_gine_a4_concat_flat_s1` | Rerun the historical best setup with legacy critic updates. |
| `best_01_full_concat_gnn_optcritic_s1` | Same full concat-flat GNN-critic architecture, but optimized critic updates. |
| `best_02_full_concat_mlp_optcritic_s1` | Full concat-flat actor with MLP critic. Tests whether GNN critic is needed. |
| `best_03_full_concat_no_edge_gnn_optcritic_s1` | Full concat-flat GNN critic without edge pre-encoder. |
| `best_04_full_concat_no_edge_mlp_optcritic_s1` | Full concat-flat MLP critic without edge pre-encoder. Cheap full-size candidate. |
| `best_05_light_mlp_s1` | Light seed-1 version of the strongest light A4 MLP-critic setup. |
| `best_06_light_concat_mlp_s1` | Light MLP-critic setup with concat-flat. Tests whether concat-flat helps light GNN. |
| `best_07_light_gnn_s1` | Light setup with GNN critic. Tests MLP vs GNN critic at seed 1. |
| `best_08_light_concat_gnn_s1` | Light setup with both concat-flat and GNN critic. |
| `best_09_light_no_node_pre_mlp_s1` | Light MLP-critic setup without node pre-encoder. Tests whether node pre-encoder is worth keeping. |
| `best_10_rerun_light_mlp_s0` | Robust-eval rerun of the 100% survival light A4 MLP-critic seed-0 run. |

Launch from the repository root with:

```bash
sbatch job_jed.sh configs/gine_best_search/<config>.toml
```
