# No-Validation MLP Seed Sweep

These 24 configs repeat the no-validation MLP train/test setup for seeds 0, 1, and 2.
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

Launch the original 18 entropy-decay runs:

```bash
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a1_entropy_decay_s0_det.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a1_entropy_decay_s0_stoch.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a1_entropy_decay_s1_det.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a1_entropy_decay_s1_stoch.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a1_entropy_decay_s2_det.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a1_entropy_decay_s2_stoch.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a3_entropy_decay_s0_det.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a3_entropy_decay_s0_stoch.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a3_entropy_decay_s1_det.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a3_entropy_decay_s1_stoch.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a3_entropy_decay_s2_det.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a3_entropy_decay_s2_stoch.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a4_entropy_decay_s0_det.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a4_entropy_decay_s0_stoch.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a4_entropy_decay_s1_det.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a4_entropy_decay_s1_stoch.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a4_entropy_decay_s2_det.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a4_entropy_decay_s2_stoch.toml
```

Launch the added A3 logit-decay runs:

```bash
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a3_logit_decay_s0_det.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a3_logit_decay_s0_stoch.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a3_logit_decay_s1_det.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a3_logit_decay_s1_stoch.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a3_logit_decay_s2_det.toml
sbatch job_jed.sh configs/entropy_decay_s0_s1_s2/noval20_mlp_a3_logit_decay_s2_stoch.toml
```

| Recipe | Base idea | Batch | Entropy schedule | p0 | Seeds | Eval |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `a1_entropy_decay` | A1 entropy decay | 20x2000 | 0.01 -> 0.0 | 0.3 | seeds 0, 1, 2 | det + stoch |
| `a3_entropy_decay` | A3 entropy decay | 20x2000 | 0.02 -> 0.001 | 0.5 | seeds 0, 1, 2 | det + stoch |
| `a3_logit_decay` | A3 entropy decay + action-0 logit decay | 40x1000 | 0.02 -> 0.001 | 0.5 | seeds 0, 1, 2 | det + stoch |
| `a4_entropy_decay` | A4 entropy decay | 20x2000 | 0.02 -> 0.001 | 0.7 | seeds 0, 1, 2 | det + stoch |

## Hyperparameters That Change

Everything not listed here is intentionally kept identical within each recipe.

| Axis | Config values | What changes |
| --- | --- | --- |
| Recipe | `a1_entropy_decay`, `a3_entropy_decay`, `a3_logit_decay`, `a4_entropy_decay` | Selects the entropy schedule and initial action-0 prior. |
| Training seed | `seed = 0`, `seed = 1`, `seed = 2` | Changes model initialization, environment RNG, and sampling RNG. |
| Eval mode | `deterministic_eval = true`, `false` | Changes only evaluation action selection: greedy vs sampled. |
| Initial entropy | A1: `0.01`; A3/A4: `0.02` | Higher values keep the policy more exploratory early. |
| Final entropy | A1: `0.0`; A3/A4: `0.001` | Lower final values make the learned policy more confident late. |
| Initial action-0 prior | A1: `0.3`; A3: `0.5`; A4: `0.7` | Sets how strongly the initial policy favors do-nothing/action 0. |
| Action-0 logit bonus | A3 logit: `1.0 -> 0.0` over 40% training | Adds an explicit early do-nothing bias on top of the initial policy prior. |

| Recipe | `entropy_coef` | `entropy_coef_final` | `init_do_nothing_prob` | `seed` | `deterministic_eval` |
| --- | ---: | ---: | ---: | --- | --- |
| `a1_entropy_decay` | `0.01` | `0.0` | `0.3` | `0`, `1`, `2` | `true`, `false` |
| `a3_entropy_decay` | `0.02` | `0.001` | `0.5` | `0`, `1`, `2` | `true`, `false` |
| `a3_logit_decay` | `0.02` | `0.001` | `0.5` | `0`, `1`, `2` | `true`, `false` |
| `a4_entropy_decay` | `0.02` | `0.001` | `0.7` | `0`, `1`, `2` | `true`, `false` |
