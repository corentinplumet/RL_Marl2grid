# No-Validation MLP Seed Sweep

These 12 configs repeat the no-validation MLP train/test setup for seeds 1 and 2.
The train/test chronic split is intentionally fixed with `chronic_split_seed = 0`,
so differences mostly reflect training seed sensitivity rather than a different
train/test partition.

All runs use:

- `actor_encoder = "mlp"`
- `critic_encoder = "mlp"`
- `split_chronics = true`
- `test_chronics_pct = 0.2`
- `eval_train_chronics = true`
- `eval_all_split_chronics = false`
- `eval_episodes = 10`

Launch all 12 runs:

```bash
sbatch job_jed.sh configs/noval_seed_sweep/noval20_mlp_a1_entropy_decay_s1_det.toml
sbatch job_jed.sh configs/noval_seed_sweep/noval20_mlp_a1_entropy_decay_s1_stoch.toml
sbatch job_jed.sh configs/noval_seed_sweep/noval20_mlp_a1_entropy_decay_s2_det.toml
sbatch job_jed.sh configs/noval_seed_sweep/noval20_mlp_a1_entropy_decay_s2_stoch.toml
sbatch job_jed.sh configs/noval_seed_sweep/noval20_mlp_a3_entropy_decay_s1_det.toml
sbatch job_jed.sh configs/noval_seed_sweep/noval20_mlp_a3_entropy_decay_s1_stoch.toml
sbatch job_jed.sh configs/noval_seed_sweep/noval20_mlp_a3_entropy_decay_s2_det.toml
sbatch job_jed.sh configs/noval_seed_sweep/noval20_mlp_a3_entropy_decay_s2_stoch.toml
sbatch job_jed.sh configs/noval_seed_sweep/noval20_mlp_a4_entropy_decay_s1_det.toml
sbatch job_jed.sh configs/noval_seed_sweep/noval20_mlp_a4_entropy_decay_s1_stoch.toml
sbatch job_jed.sh configs/noval_seed_sweep/noval20_mlp_a4_entropy_decay_s2_det.toml
sbatch job_jed.sh configs/noval_seed_sweep/noval20_mlp_a4_entropy_decay_s2_stoch.toml
```

| Recipe | Base idea | Batch | Entropy schedule | p0 | Seeds | Eval |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `a1_entropy_decay` | A1 entropy decay | 20x2000 | 0.01 -> 0.0 | 0.3 | seeds 1, 2 | det + stoch |
| `a3_entropy_decay` | A3 entropy decay | 20x2000 | 0.02 -> 0.001 | 0.5 | seeds 1, 2 | det + stoch |
| `a4_entropy_decay` | A4 entropy decay | 20x2000 | 0.02 -> 0.001 | 0.7 | seeds 1, 2 | det + stoch |

## Hyperparameters That Change

Everything not listed here is intentionally kept identical across the 12 runs.

| Axis | Config values | What changes |
| --- | --- | --- |
| Recipe | `a1_entropy_decay`, `a3_entropy_decay`, `a4_entropy_decay` | Selects the entropy schedule and initial action-0 prior. |
| Training seed | `seed = 1`, `seed = 2` | Changes model initialization, environment RNG, and sampling RNG. |
| Eval mode | `deterministic_eval = true`, `false` | Changes only evaluation action selection: greedy vs sampled. |
| Initial entropy | A1: `0.01`; A3/A4: `0.02` | Higher values keep the policy more exploratory early. |
| Final entropy | A1: `0.0`; A3/A4: `0.001` | Lower final values make the learned policy more confident late. |
| Initial action-0 prior | A1: `0.3`; A3: `0.5`; A4: `0.7` | Sets how strongly the initial policy favors do-nothing/action 0. |

| Recipe | `entropy_coef` | `entropy_coef_final` | `init_do_nothing_prob` | `seed` | `deterministic_eval` |
| --- | ---: | ---: | ---: | --- | --- |
| `a1_entropy_decay` | `0.01` | `0.0` | `0.3` | `1`, `2` | `true`, `false` |
| `a3_entropy_decay` | `0.02` | `0.001` | `0.5` | `1`, `2` | `true`, `false` |
| `a4_entropy_decay` | `0.02` | `0.001` | `0.7` | `1`, `2` | `true`, `false` |
