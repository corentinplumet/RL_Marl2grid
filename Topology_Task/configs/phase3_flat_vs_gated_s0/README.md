# Phase 3 Flat vs Gated S0

Four self-contained configs generated from `best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update_s0.toml`.

The comparison isolates two factors:

- actor structure: original flat actor vs intervention-gated actor
- sparse objective: no action penalty vs threshold-free non-idle action penalty

No run uses the safe/rho-threshold penalty. All configs set:

```toml
safe_intervention_penalty = 0.0
safe_intervention_rho_threshold = 0.90
```

The threshold value is therefore inactive in this experiment.

| Config | Actor | `intervention_penalty` | `safe_intervention_penalty` |
| --- | --- | ---: | ---: |
| `phase3_00_flat_no_penalty_s0.toml` | flat | 0.0 | 0.0 |
| `phase3_01_flat_penalty001_s0.toml` | flat | 0.001 | 0.0 |
| `phase3_02_gated_no_penalty_s0.toml` | gated | 0.0 | 0.0 |
| `phase3_03_gated_penalty001_s0.toml` | gated | 0.001 | 0.0 |

Launch all from the repository root on JED:

```bash
sbatch job_jed.sh configs/phase3_flat_vs_gated_s0/phase3_00_flat_no_penalty_s0.toml
sbatch job_jed.sh configs/phase3_flat_vs_gated_s0/phase3_01_flat_penalty001_s0.toml
sbatch job_jed.sh configs/phase3_flat_vs_gated_s0/phase3_02_gated_no_penalty_s0.toml
sbatch job_jed.sh configs/phase3_flat_vs_gated_s0/phase3_03_gated_penalty001_s0.toml
```
