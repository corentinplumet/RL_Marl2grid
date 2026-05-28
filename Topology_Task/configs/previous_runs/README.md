# Previous JED Run Configs

These configs were generated from `wandb_export_2026-05-27T14_04_17.232+08_00.csv`. Each TOML corresponds to one W&B run row and can be submitted with:

```bash
sbatch job_jed.sh configs/previous_runs/<config>.toml
```

Validation split flags from older train/test/validation runs are intentionally omitted because the current code only supports train/test splitting.

| Config | W&B run | State | Step | Survival | Eval | Split | Batch |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `noval20_mlp_a3_logit_decay_stoch.toml` | `noval20_mlp_a3_logit_decay_stoch` | running | 8560000 | 29.6% | stoch | train/test | 40x1000 |
| `noval20_mlp_a3_logit_decay_det.toml` | `noval20_mlp_a3_logit_decay_det` | running | 5960000 | 19.1% | det | train/test | 40x1000 |
| `noval20_mlp_p999_decay_stoch.toml` | `noval20_mlp_p999_decay_stoch` | running | 7920000 | 19.7% | stoch | train/test | 20x2000 |
| `noval20_mlp_p999_decay_det.toml` | `noval20_mlp_p999_decay_det` | running | 6160000 | 77.2% | det | train/test | 20x2000 |
| `noval20_mlp_a1_decay_stoch.toml` | `noval20_mlp_a1_decay_stoch` | running | 8480000 | 45.1% | stoch | train/test | 20x2000 |
| `noval20_mlp_a1_decay_det.toml` | `noval20_mlp_a1_decay_det` | running | 5680000 | 100.0% | det | train/test | 20x2000 |
| `split_mlp_a3_logit_decay_stoch_s0.toml` | `split_mlp_a3_logit_decay_stoch_s0` | finished | 15000000 |  | stoch | train/test | 40x1000 |
| `split_mlp_a3_logit_decay_det_s0.toml` | `split_mlp_a3_logit_decay_det_s0` | killed | 13120000 |  | det | train/test | 40x1000 |
| `split_mlp_p999_decay_stoch_s0.toml` | `split_mlp_p999_decay_stoch_s0` | finished | 15000000 |  | stoch | train/test | 20x2000 |
| `split_mlp_p999_decay_det_s0.toml` | `split_mlp_p999_decay_det_s0` | killed | 13800000 |  | det | train/test | 20x2000 |
| `split_mlp_a1_decay_stoch_s0.toml` | `split_mlp_a1_decay_stoch_s0` | finished | 15000000 |  | stoch | train/test | 20x2000 |
| `split_mlp_a1_decay_det_s0.toml` | `split_mlp_a1_decay_det_s0` | finished | 14480000 |  | det | train/test | 20x2000 |
| `split_mlp_a3_stoch_s0.toml` | `split_mlp_a3_stoch_s0` | finished | 15000000 |  | stoch | train/test | 40x1000 |
| `split_mlp_a3_det_s0.toml` | `split_mlp_a3_det_s0` | finished | 14520000 |  | det | train/test | 40x1000 |
| `a0_stoch_20env_mb20_ep15_kl004.toml` | `a0_stoch_20env_mb20_ep15_kl004` | finished | 15000000 | 40.5% | stoch | none | 20x2000 |
| `a1_stoch_ent0015_mb20_ep15_kl004.toml` | `a1_stoch_ent0015_mb20_ep15_kl004` | finished | 15000000 | 23.2% | stoch | none | 40x1000 |
| `a1_stoch_mb20_ep15_kl004.toml` | `a1_stoch_mb20_ep15_kl004` | finished | 15000000 | 46.6% | stoch | none | 40x1000 |
| `a3_stoch_mb20_ep20_kl003.toml` | `a3_stoch_mb20_ep20_kl003` | finished | 15000000 | 9.1% | stoch | none | 40x1000 |
| `a3_stoch_ent003_mb20_ep15_kl004.toml` | `a3_stoch_ent003_mb20_ep15_kl004` | finished | 15000000 | 18.6% | stoch | none | 40x1000 |
| `a3_stoch_mb20_ep15_kl004.toml` | `a3_stoch_mb20_ep15_kl004` | finished | 15000000 | 28.9% | stoch | none | 40x1000 |
| `a1_40env_entropy_to0_stoch.toml` | `a1_40env_entropy_to0_stoch` | finished | 15000000 | 18.9% | stoch | none | 40x1000 |
| `a3_40env_entropy_decay_stoch.toml` | `a3_40env_entropy_decay_stoch` | finished | 15000000 | 43.0% | stoch | none | 40x1000 |
| `a3_40env_decay_margin005_stoch.toml` | `a3_40env_decay_margin005_stoch` | finished | 15000000 | 30.8% | stoch | none | 40x1000 |
| `p0999_entropy_decay_stoch.toml` | `p0999_entropy_decay_stoch` | finished | 15000000 | 59.3% | stoch | none | 20x2000 |
| `a3_20env_entropy_decay_stoch.toml` | `a3_20env_entropy_decay_stoch` | finished | 15000000 | 74.1% | stoch | none | 20x2000 |
| `a3_40env_decay_mb20_stoch.toml` | `a3_40env_decay_mb20_stoch` | finished | 15000000 | 51.5% | stoch | none | 40x1000 |
| `a1_20env_entropy_to0_stoch.toml` | `a1_20env_entropy_to0_stoch` | finished | 15000000 | 52.1% | stoch | none | 20x2000 |
| `a1_60env_660steps_det2.toml` | `a1_60env_660steps_det2` | finished | 14968800 | 100.0% | det | none | 60x660 |
| `a3_60env_660steps_det2.toml` | `a3_60env_660steps_det2` | finished | 14968800 | 27.0% | det | none | 60x660 |
| `a3_40env_1000steps_det2.toml` | `a3_40env_1000steps_det2` | finished | 15000000 | 100.0% | det | none | 40x1000 |
| `p0_0999_entropy001_20env_det2.toml` | `p0_0999_entropy001_20env_det2` | killed | 13120000 | 40.9% | det | none | 20x2000 |
| `a1_40env_1000steps_det2.toml` | `a1_40env_1000steps_det2` | killed | 9320000 | 80.3% | det | none | 40x1000 |
| `p0_0999_entropy002_20env_det2.toml` | `p0_0999_entropy002_20env_det2` | killed | 9840000 | 84.1% | det | none | 20x2000 |
| `b1_40env_1000steps_sto.toml` | `b1_40env_1000steps_sto` | finished | 15000000 | 24.5% | stoch | none | 40x1000 |
| `b0_short_rollout_sto.toml` | `b0_short_rollout_sto` | finished | 15000000 | 40.4% | stoch | none | 20x1000 |
| `a5_kl_004_sto.toml` | `a5_kl_004_sto` | finished | 15000000 | 44.1% | stoch | none | 20x2000 |
| `a4_entropy_002_p0_07_sto.toml` | `a4_entropy_002_p0_07_sto` | finished | 15000000 | 8.2% | stoch | none | 20x2000 |
| `a2_p0_07_sto.toml` | `a2_p0_07_sto` | finished | 15000000 | 46.7% | stoch | none | 20x2000 |
| `a3_entropy_002_sto.toml` | `a3_entropy_002_sto` | finished | 15000000 | 45.7% | stoch | none | 20x2000 |
| `a0_known_good_sto.toml` | `a0_known_good_sto` | finished | 15000000 | 55.4% | stoch | none | 20x2000 |
| `a1_p0_03_sto.toml` | `a1_p0_03_sto` | finished | 15000000 | 36.7% | stoch | none | 20x2000 |
| `a0_known_good_det.toml` | `a0_known_good_det` | finished | 15000000 | 20.0% | det | none | 20x2000 |
| `a4_entropy_002_p0_07_det.toml` | `a4_entropy_002_p0_07_det` | crashed | 9240000 | 76.5% | det | none | 20x2000 |
| `a1_p0_03_det.toml` | `a1_p0_03_det` | crashed | 9720000 | 92.7% | det | none | 20x2000 |
| `a3_entropy_002_det.toml` | `a3_entropy_002_det` | crashed | 9840000 | 59.9% | det | none | 20x2000 |
| `b0_short_rollout_det.toml` | `b0_short_rollout_det` | crashed | 9920000 | 0.0% | det | none | 20x1000 |
| `a2_p0_07_det.toml` | `a2_p0_07_det` | finished | 15000000 | 33.9% | det | none | 20x2000 |
| `a5_kl_004_det.toml` | `a5_kl_004_det` | crashed | 12240000 | 69.4% | det | none | 20x2000 |
| `mappo_baseline_nejjar.toml` | `MAPPO_baseline_nejjar` | finished | 25000000 | 58.3% | det | none | 20x2000 |
