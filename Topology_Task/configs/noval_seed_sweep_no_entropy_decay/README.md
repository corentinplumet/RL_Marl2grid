# No-Validation MLP Seed Sweep Without Entropy Decay

These 12 configs rerun the `noval_seed_sweep` recipes with entropy kept fixed.
They are designed as a direct A/B comparison against the entropy-decay runs.

The only intended change from `configs/noval_seed_sweep/` is:

```toml
entropy_coef_final = entropy_coef
```

So the entropy coefficient stays constant for the full 15M steps:

| Recipe | `entropy_coef` | `entropy_coef_final` | `init_do_nothing_prob` | Seeds | Eval |
| --- | ---: | ---: | ---: | --- | --- |
| `a1_no_entropy_decay` | `0.01` | `0.01` | `0.3` | `1`, `2` | det + stoch |
| `a3_no_entropy_decay` | `0.02` | `0.02` | `0.5` | `1`, `2` | det + stoch |
| `a4_no_entropy_decay` | `0.02` | `0.02` | `0.7` | `1`, `2` | det + stoch |

Everything else is kept identical, including `chronic_split_seed = 0`.

Launch all 12 runs:

```bash
sbatch job_jed.sh configs/noval_seed_sweep_no_entropy_decay/noval20_mlp_a1_no_entropy_decay_s1_det.toml
sbatch job_jed.sh configs/noval_seed_sweep_no_entropy_decay/noval20_mlp_a1_no_entropy_decay_s1_stoch.toml
sbatch job_jed.sh configs/noval_seed_sweep_no_entropy_decay/noval20_mlp_a1_no_entropy_decay_s2_det.toml
sbatch job_jed.sh configs/noval_seed_sweep_no_entropy_decay/noval20_mlp_a1_no_entropy_decay_s2_stoch.toml
sbatch job_jed.sh configs/noval_seed_sweep_no_entropy_decay/noval20_mlp_a3_no_entropy_decay_s1_det.toml
sbatch job_jed.sh configs/noval_seed_sweep_no_entropy_decay/noval20_mlp_a3_no_entropy_decay_s1_stoch.toml
sbatch job_jed.sh configs/noval_seed_sweep_no_entropy_decay/noval20_mlp_a3_no_entropy_decay_s2_det.toml
sbatch job_jed.sh configs/noval_seed_sweep_no_entropy_decay/noval20_mlp_a3_no_entropy_decay_s2_stoch.toml
sbatch job_jed.sh configs/noval_seed_sweep_no_entropy_decay/noval20_mlp_a4_no_entropy_decay_s1_det.toml
sbatch job_jed.sh configs/noval_seed_sweep_no_entropy_decay/noval20_mlp_a4_no_entropy_decay_s1_stoch.toml
sbatch job_jed.sh configs/noval_seed_sweep_no_entropy_decay/noval20_mlp_a4_no_entropy_decay_s2_det.toml
sbatch job_jed.sh configs/noval_seed_sweep_no_entropy_decay/noval20_mlp_a4_no_entropy_decay_s2_stoch.toml
```
