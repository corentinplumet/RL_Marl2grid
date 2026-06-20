# Phase 4 Sparse Control 24-Run Sweep

This sweep uses the new Phase 3 sparse-action objective and Phase 4 explainability logs.

Design:

- actor: original flat actor vs intervention-gated actor
- threshold-free action penalty: `0.0`, `0.001`, `0.003`, `0.01`
- seeds: `0`, `1`, `2`

No config in this folder uses the safe/rho-threshold penalty:

```toml
safe_intervention_penalty = 0.0
```

The useful comparison is whether the flat actor with an action penalty matches or beats the gated actor with the same penalty.

## Configs

| Config | Actor | Seed | intervention_penalty |
| --- | --- | ---: | ---: |
| `sparse16_flat_p000_s0.toml` | flat | 0 | 0.0 |
| `sparse16_flat_p001_s0.toml` | flat | 0 | 0.001 |
| `sparse16_flat_p003_s0.toml` | flat | 0 | 0.003 |
| `sparse16_flat_p010_s0.toml` | flat | 0 | 0.01 |
| `sparse16_gated_p000_s0.toml` | gated | 0 | 0.0 |
| `sparse16_gated_p001_s0.toml` | gated | 0 | 0.001 |
| `sparse16_gated_p003_s0.toml` | gated | 0 | 0.003 |
| `sparse16_gated_p010_s0.toml` | gated | 0 | 0.01 |
| `sparse16_flat_p000_s1.toml` | flat | 1 | 0.0 |
| `sparse16_flat_p001_s1.toml` | flat | 1 | 0.001 |
| `sparse16_flat_p003_s1.toml` | flat | 1 | 0.003 |
| `sparse16_flat_p010_s1.toml` | flat | 1 | 0.01 |
| `sparse16_gated_p000_s1.toml` | gated | 1 | 0.0 |
| `sparse16_gated_p001_s1.toml` | gated | 1 | 0.001 |
| `sparse16_gated_p003_s1.toml` | gated | 1 | 0.003 |
| `sparse16_gated_p010_s1.toml` | gated | 1 | 0.01 |
| `sparse16_flat_p000_s2.toml` | flat | 2 | 0.0 |
| `sparse16_flat_p001_s2.toml` | flat | 2 | 0.001 |
| `sparse16_flat_p003_s2.toml` | flat | 2 | 0.003 |
| `sparse16_flat_p010_s2.toml` | flat | 2 | 0.01 |
| `sparse16_gated_p000_s2.toml` | gated | 2 | 0.0 |
| `sparse16_gated_p001_s2.toml` | gated | 2 | 0.001 |
| `sparse16_gated_p003_s2.toml` | gated | 2 | 0.003 |
| `sparse16_gated_p010_s2.toml` | gated | 2 | 0.01 |

## Launch All

```bash
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_flat_p000_s0.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_flat_p001_s0.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_flat_p003_s0.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_flat_p010_s0.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_gated_p000_s0.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_gated_p001_s0.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_gated_p003_s0.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_gated_p010_s0.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_flat_p000_s1.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_flat_p001_s1.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_flat_p003_s1.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_flat_p010_s1.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_gated_p000_s1.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_gated_p001_s1.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_gated_p003_s1.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_gated_p010_s1.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_flat_p000_s2.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_flat_p001_s2.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_flat_p003_s2.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_flat_p010_s2.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_gated_p000_s2.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_gated_p001_s2.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_gated_p003_s2.toml
sbatch job_jed.sh configs/phase4_sparse_control_16/sparse16_gated_p010_s2.toml
```
