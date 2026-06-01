# Shared GINE No-Entropy Deterministic Configs

These configs replace the old `gine` and `gine_light` sweeps. They are copied
from the deterministic no-entropy-decay A1/A4 MLP recipes, then switched to the
new busbar GINE architecture:

- `actor_encoder = "gnn"`
- `critic_encoder = "gnn"`
- `gnn_type = "gine"`
- `share_actor_gnn = true`
- separate actor MLP heads per agent
- `gnn_node_pre_encoder = true`
- `gnn_edge_pre_encoder = true`
- `gnn_node_id_embeddings = true`
- `gnn_include_neighbors = false`

The A1 configs keep `entropy_coef = 0.01` and `init_do_nothing_prob = 0.3`.
The A4 configs keep `entropy_coef = 0.02` and `init_do_nothing_prob = 0.7`.
All configs use deterministic evaluation.

Launch examples:

```bash
sbatch job_jed.sh configs/gine/shared_gine_a1_no_entropy_decay_s0_det.toml
sbatch job_jed.sh configs/gine/shared_gine_a4_no_entropy_decay_s0_det.toml
```

Available seeds: `s0`, `s1`, `s2`.
