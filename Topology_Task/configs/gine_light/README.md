# Light GINE Run Configs

These configs are the same 8 GINE training recipes as `configs/gine/`, but with
a lighter graph encoder to improve SPS:

- `gnn_hidden_dim = 64` instead of `128`
- `gnn_out_dim = 64` instead of `128`
- `gnn_layers = 1` instead of `2`

Everything else is intentionally kept the same, including PPO settings,
train/test split, and `gnn_include_neighbors = true`.

Launch all 8 light GINE runs:

```bash
sbatch job_jed.sh configs/gine_light/gine_light_a1_decay_det.toml
sbatch job_jed.sh configs/gine_light/gine_light_a1_decay_stoch.toml
sbatch job_jed.sh configs/gine_light/gine_light_p999_decay_det.toml
sbatch job_jed.sh configs/gine_light/gine_light_p999_decay_stoch.toml
sbatch job_jed.sh configs/gine_light/gine_light_a3_fixed_det.toml
sbatch job_jed.sh configs/gine_light/gine_light_a3_fixed_stoch.toml
sbatch job_jed.sh configs/gine_light/gine_light_a3_logit_decay_det.toml
sbatch job_jed.sh configs/gine_light/gine_light_a3_logit_decay_stoch.toml
```

If SPS is still too low, the next speed-focused variant to try is setting
`gnn_include_neighbors = false`, but that changes the graph observation more
than just making the encoder smaller.
