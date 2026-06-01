# GINE No-Entropy Deterministic Configs

These configs replace the old `gine` and `gine_light` sweeps. They are copied
from the deterministic no-entropy-decay A1/A4 MLP recipes, then switched to the
new busbar GINE architecture and a few targeted diagnostic variants:

- `actor_encoder = "gnn"`
- `gnn_type = "gine"`
- `gnn_node_pre_encoder = true`
- `gnn_edge_pre_encoder = true`
- `gnn_node_id_embeddings = true`
- `gnn_include_neighbors = false`

The A1 configs keep `entropy_coef = 0.01` and `init_do_nothing_prob = 0.3`.
The A4 configs keep `entropy_coef = 0.02` and `init_do_nothing_prob = 0.7`.
All configs use deterministic evaluation.

Config families:

- `shared_gine_a1_no_entropy_decay_*`: shared actor GINE, GNN critic, pure graph.
- `shared_gine_a4_no_entropy_decay_*`: shared actor GINE, GNN critic, pure graph.
- `shared_gine_a4_concat_flat_no_entropy_decay_*`: shared actor GINE, GNN critic, graph plus flat observation.
- `shared_gine_a4_mlp_critic_no_entropy_decay_*`: shared actor GINE, MLP critic.
- `gine_a4_nonshared_actor_no_entropy_decay_*`: separate actor GINE encoders, GNN critic.

Launch examples:

```bash
sbatch job_izar.sh configs/gine/shared_gine_a1_no_entropy_decay_s0_det.toml
sbatch job_izar.sh configs/gine/shared_gine_a4_no_entropy_decay_s0_det.toml
sbatch job_izar.sh configs/gine/shared_gine_a4_concat_flat_no_entropy_decay_s0_det.toml
sbatch job_izar.sh configs/gine/shared_gine_a4_mlp_critic_no_entropy_decay_s0_det.toml
sbatch job_izar.sh configs/gine/gine_a4_nonshared_actor_no_entropy_decay_s0_det.toml
```

Available seeds: `s0`, `s1`, `s2`.
