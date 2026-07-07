# Adaptive Intervention Budget: First 7 Runs

This folder contains a first small sweep for the adaptive intervention budget
implementation on top of `best_00`.

The goal is to test whether a learned Lagrangian penalty can increase action
`0` usage while preserving survival, without relying on an evaluation-only
rho threshold.

## Runs

| Config | Seed | Actor | Target | Purpose |
| --- | ---: | --- | ---: | --- |
| `aib_00_flat_local_t020_s0.toml` | 0 | flat/original | 0.20 | Main candidate |
| `aib_00_flat_local_t020_s1.toml` | 1 | flat/original | 0.20 | Main candidate repeat |
| `aib_00_flat_local_t020_s2.toml` | 2 | flat/original | 0.20 | Main candidate repeat |
| `aib_01_flat_local_t010_s0.toml` | 0 | flat/original | 0.10 | Stricter sparse-control stress test |
| `aib_02_flat_local_t035_s0.toml` | 0 | flat/original | 0.35 | Looser budget |
| `aib_02_flat_local_t035_s1.toml` | 1 | flat/original | 0.35 | Looser budget repeat |
| `aib_03_gate_hgreedy_sep_local_t020_s0.toml` | 0 | intervention gate | 0.20 | Diagnostic: does the gate help once sparsity is in the objective? |

## Additional 8-Job Batch

The first 5 configs complete the existing conditions up to the usual
`seed = 0,1,2` protocol. The main `t020` flat condition already had seeds
`0,1,2`, so it does not need extra seed runs.

The last 3 configs add one new 3-seed diagnostic setup: the flat actor with
`target = 0.20`, but `intervention_budget_cost_mode = "nonidle"`. This tests
whether a plain intervention budget is enough, or whether the local-safe
rho-weighted cost is important for preserving survival.

| Config | Seed | Actor | Target | Purpose |
| --- | ---: | --- | ---: | --- |
| `aib_01_flat_local_t010_s1.toml` | 1 | flat/original | 0.10 | Strict budget extra seed |
| `aib_01_flat_local_t010_s2.toml` | 2 | flat/original | 0.10 | Strict budget extra seed |
| `aib_02_flat_local_t035_s2.toml` | 2 | flat/original | 0.35 | Loose budget extra seed |
| `aib_03_gate_hgreedy_sep_local_t020_s1.toml` | 1 | intervention gate | 0.20 | Gated diagnostic extra seed |
| `aib_03_gate_hgreedy_sep_local_t020_s2.toml` | 2 | intervention gate | 0.20 | Gated diagnostic extra seed |
| `aib_04_flat_nonidle_t020_s0.toml` | 0 | flat/original | 0.20 | Nonidle-cost diagnostic |
| `aib_04_flat_nonidle_t020_s1.toml` | 1 | flat/original | 0.20 | Nonidle-cost diagnostic |
| `aib_04_flat_nonidle_t020_s2.toml` | 2 | flat/original | 0.20 | Nonidle-cost diagnostic |

All runs use:

```toml
adaptive_intervention_budget = true
intervention_budget_cost_mode = "local_safe"
intervention_budget_lr = 0.02
intervention_budget_rho_threshold = 0.90
intervention_budget_rho_sharpness = 25.0
```

`local_safe` is the decentralized cost: each agent pays for non-idle actions
mostly when its own local max rho is below the smooth safety threshold.

## Launch

From the repository root:

```bash
bash Topology_Task/configs/adaptive_intervention_budget_7/launch_all.sh
```

Or submit one run manually:

```bash
sbatch job_jed.sh configs/adaptive_intervention_budget_7/aib_00_flat_local_t020_s0.toml
```

Launch only the 8 additional jobs:

```bash
bash Topology_Task/configs/adaptive_intervention_budget_7/launch_next8.sh
```

## First Metrics To Check

- `charts/episodic_survival`, `test/episodic_survival`, `train_eval/episodic_survival`
- `test/explain/frac_action_0_agent_*`
- `train_eval/explain/frac_action_0_agent_*`
- `train/intervention_budget_lambda_agent_*`
- `train/intervention_budget_cost_agent_*`
- `train/intervention_budget_cost_violation_agent_*`
- `train/intervention_rate_when_budget_costly_agent_*`
- `train/intervention_rate_when_budget_free_agent_*`

The desired pattern is survival close to baseline, larger final executed
action-0 fractions, falling intervention rates in costly safe states, and
lambdas that stabilize instead of saturating immediately.
