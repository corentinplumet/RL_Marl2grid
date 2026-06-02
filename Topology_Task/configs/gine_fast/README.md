# Fast GINE Training Configs

These configs are faster diagnostic GNN runs for rented GPU/RunPod-style training.
They keep deterministic evaluation and seeds `s0`, `s1`, `s2`, but reduce the
largest avoidable runtime costs:

- `eval_freq = 200000`
- `eval_episodes = 3`
- `eval_train_chronics = false`
- `n_threads = 1`
- `gnn_include_neighbors = false`

Families:

- `fast_shared_gine_a1_mlp_critic_no_entropy_decay_*`: full actor GINE, MLP critic, A1 exploration settings.
- `fast_shared_gine_a4_mlp_critic_no_entropy_decay_*`: full actor GINE, MLP critic, A4 conservative settings.
- `fast_light_shared_gine_a4_no_entropy_decay_*`: light actor GINE, light GNN critic.
- `fast_light_shared_gine_a4_mlp_critic_no_entropy_decay_*`: light actor GINE, MLP critic.

The light variants use:

- `gnn_hidden_dim = 64`
- `gnn_out_dim = 64`
- `gnn_layers = 1`
- `gnn_edge_pre_encoder = false`
- `actor_layers = [128, 128]`
- `critic_layers = [128, 128]`

Launch example from `Topology_Task`:

```bash
python -u run_from_config.py \
  configs/gine_fast/fast_light_shared_gine_a4_mlp_critic_no_entropy_decay_s0_det.toml \
  --cuda true
```
