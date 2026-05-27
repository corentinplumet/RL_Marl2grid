# GINE Run Configs

These configs compare the current MLP recipes with thesis-style GNN encoders
using `gnn_type = "gine"`. Each family has deterministic and stochastic eval.

All runs use:

- `actor_encoder = "gnn"`
- `critic_encoder = "gnn"`
- `gnn_type = "gine"`
- bus14 topology with train/test split
- `test_chronics_pct = 0.2`
- `eval_train_chronics = true`
- `eval_all_split_chronics = false`

Launch all 8 runs:

```bash
sbatch job_jed.sh configs/gine/gine20_a1_decay_det.toml
sbatch job_jed.sh configs/gine/gine20_a1_decay_stoch.toml
sbatch job_jed.sh configs/gine/gine20_p999_decay_det.toml
sbatch job_jed.sh configs/gine/gine20_p999_decay_stoch.toml
sbatch job_jed.sh configs/gine/gine20_a3_fixed_det.toml
sbatch job_jed.sh configs/gine/gine20_a3_fixed_stoch.toml
sbatch job_jed.sh configs/gine/gine20_a3_logit_decay_det.toml
sbatch job_jed.sh configs/gine/gine20_a3_logit_decay_stoch.toml
```

| Config | Base idea | Eval | Batch | Entropy | p0 | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `gine20_a1_decay_det.toml` | A1 decay | det | 20x2000 | 0.01 -> 0.0 | 0.3 | Strong no-validation MLP deterministic family. |
| `gine20_a1_decay_stoch.toml` | A1 decay | stoch | 20x2000 | 0.01 -> 0.0 | 0.3 | Same training, sampled eval. |
| `gine20_p999_decay_det.toml` | high p0 decay | det | 20x2000 | 0.02 -> 0.001 | 0.999 | Tests strong idle prior with GINE. |
| `gine20_p999_decay_stoch.toml` | high p0 decay | stoch | 20x2000 | 0.02 -> 0.001 | 0.999 | Same training, sampled eval. |
| `gine20_a3_fixed_det.toml` | A3 fixed entropy | det | 40x1000 | 0.02 | 0.5 | Best old 40-env deterministic shape, now with split. |
| `gine20_a3_fixed_stoch.toml` | A3 fixed entropy | stoch | 40x1000 | 0.02 | 0.5 | Same training, sampled eval. |
| `gine20_a3_logit_decay_det.toml` | A3 logit decay | det | 40x1000 | 0.02 -> 0.001 | 0.5 | Adds temporary action-0 logit bonus. |
| `gine20_a3_logit_decay_stoch.toml` | A3 logit decay | stoch | 40x1000 | 0.02 -> 0.001 | 0.5 | Same training, sampled eval. |
