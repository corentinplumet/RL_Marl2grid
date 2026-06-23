# GINE Best-Search 2 Configs

This folder contains a controlled follow-up around the strongest graph-encoder runs.
It is organized as twelve experiment families, each repeated with seeds `s0`, `s1`,
and `s2`.

Common settings kept fixed unless the family name says otherwise:

- A4 exploration: `entropy_coef = 0.02`, `entropy_coef_final = 0.02`, `init_do_nothing_prob = 0.7`
- GNN size: `gnn_layers = 2`, `gnn_hidden_dim = 128`, `gnn_out_dim = 128`
- actor encoder: `actor_encoder = "gnn"`
- bus graph: `gnn_graph_type = "bus"`
- no neighbor expansion: `gnn_include_neighbors = false`
- mean readout: `gnn_readout_aggr = "mean"`
- node pre-encoder: `gnn_node_pre_encoder = true`
- edge pre-encoder: `gnn_edge_pre_encoder = true`
- learned node IDs: `gnn_node_id_embeddings = true`
- deterministic evaluation
- `eval_freq = 80000`
- `eval_episodes = 5`
- `eval_train_chronics = true`
- `n_threads = 20`

The `best_09` and `best_10` run-8 rerun families keep the old light run-8 runtime profile with
`n_threads = 1`; they otherwise use the same evaluation cadence and evaluation split as the newer
baseline.

| Family | Seeds | Main purpose |
|---|---:|---|
| `best_000_shared_actor_gnn_gine_a4_no_concat_flat_critic_gnn_legacy_update_s*` | 0, 1, 2 | Shared A4 GINE without flat-observation concatenation. |
| `best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update_s*` | 0, 1, 2 | Historical best-style shared concat-flat GINE with legacy critic updates. |
| `best_01_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_optcritic_s*` | 0, 1, 2 | Same shared concat-flat GNN-critic setup, but with optimized critic updates. |
| `best_02_shared_actor_gnn_gine_a4_concat_flat_critic_mlp_legacy_update_s*` | 0, 1, 2 | Same shared concat-flat actor setup with an MLP critic and legacy critic updates. |
| `best_03_nonshared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update_s*` | 0, 1, 2 | Same concat-flat GNN setup but without shared actor GNN weights. |
| `best_04_shared_actor_gnn_light_gine_a4_concat_flat_critic_gnn_legacy_update_s*` | 0, 1, 2 | Same setup with a lighter 1-layer, 64-dimensional GINE encoder. |
| `best_05_shared_actor_gnn_gine_a4_entropy_decay_concat_flat_critic_gnn_legacy_update_s*` | 0, 1, 2 | Same setup with entropy decayed from `0.02` to `0.0`. |
| `best_06_shared_actor_gnn_gine_a4_no_node_id_concat_flat_critic_gnn_legacy_update_s*` | 0, 1, 2 | Same setup without learned node ID embeddings. |
| `best_07_shared_actor_gnn_gat_a4_concat_flat_critic_gnn_legacy_update_s*` | 0, 1, 2 | Same setup with GAT message passing instead of GINE. |
| `best_08_shared_actor_gnn_weighted_gcn_a4_concat_flat_critic_gnn_legacy_update_s*` | 0, 1, 2 | Same setup with GCN message passing weighted directly by raw `rho`. |
| `best_09_shared_actor_gnn_light_gine_a4_no_concat_flat_critic_mlp_optcritic_s*` | 0, 1, 2 | Old numbered run-8 light A4 GINE setup with MLP critic and no concat-flat, using the newer baseline evaluation protocol. |
| `best_10_shared_actor_gnn_light_gine_a4_concat_flat_critic_mlp_optcritic_s*` | 0, 1, 2 | Same old run-8 light A4 MLP-critic setup, but with flat-observation concatenation enabled. |
| `best_11_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update_initbias0_s*` | 0, 1, 2 | Same as `best_00`, but with `init_do_nothing_prob = 0.0` instead of `0.7`. |

Useful controlled comparisons:

- `best_000` vs `best_00`: effect of `gnn_concat_flat`.
- `best_00` vs `best_01`: legacy critic update vs optimized critic update.
- `best_00` vs `best_02`: GNN critic vs MLP critic, with legacy critic updates fixed.
- `best_00` vs `best_03`: shared actor GNN weights vs separate actor GNN weights.
- `best_00` vs `best_04`: standard GINE encoder vs light GINE encoder.
- `best_00` vs `best_05`: no entropy decay vs entropy decay.
- `best_00` vs `best_06`: learned node ID embeddings vs no node ID embeddings.
- `best_00` vs `best_07`: GINE message passing vs GAT message passing.
- `best_00` vs `best_08`: GINE message passing vs raw-`rho` weighted GCN message passing.
- `best_09` vs `best_10`: old run-8 light MLP setup without vs with flat-observation concatenation.
- `best_00` vs `best_11`: effect of removing the initial action-0 / do-nothing bias in the same GINE setup.

Launch from the repository root with:

```bash
sbatch job_jed.sh configs/gine_s0_s1_s2/<config>.toml
```
