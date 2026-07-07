# Adaptive Intervention Budget Mechanism Ablation

This folder contains 15 runs to separate the effects of:

1. adaptive Lagrangian intervention budgeting,
2. state-dependent safety weighting,
3. local vs global rho weighting,
4. the hard-coded local-safe rho threshold.

All setups use the flat/original actor, not the intervention gate. Seeds are
`0,1,2` for every setup.

## Existing Comparisons

These runs should be analyzed together with the existing
`adaptive_intervention_budget_7` runs:

| Existing condition | Meaning |
| --- | --- |
| `aib_00_flat_local_t020_s*` | Adaptive local-safe budget, threshold 0.90 |
| `aib_01_flat_local_t010_s*` | Adaptive local-safe budget, stricter target |
| `aib_04_flat_nonidle_t020_s*` | Adaptive non-idle budget without state-dependent safety weighting |

## New Runs

| Config prefix | Seeds | What it tests |
| --- | --- | --- |
| `aibm_00_fixed_nonidle_p006` | 0,1,2 | Fixed penalty only. No adaptive lambda, no state-dependent weighting. Coefficient is close to the learned AIB lambda scale. |
| `aibm_01_fixed_global_safe_p006` | 0,1,2 | Fixed state-dependent penalty. No adaptive lambda. Penalizes non-idle only when global max rho is below 0.90. |
| `aibm_02_adaptive_global_t020` | 0,1,2 | Adaptive budget with global-safe weighting. Tests whether local weighting matters. |
| `aibm_03_adaptive_local_t020_r085` | 0,1,2 | Adaptive local-safe budget with threshold 0.85. Tests sensitivity below 0.90. |
| `aibm_04_adaptive_local_t020_r095` | 0,1,2 | Adaptive local-safe budget with threshold 0.95. Tests sensitivity above 0.90. |

## Interpretation

Use these comparisons:

| Question | Compare |
| --- | --- |
| Does adaptivity matter beyond a fixed penalty? | `aibm_00_fixed_nonidle_p006` vs `aib_04_flat_nonidle_t020` |
| Does state-dependent safety weighting matter without adaptivity? | `aibm_00_fixed_nonidle_p006` vs `aibm_01_fixed_global_safe_p006` |
| Does local rho weighting matter? | `aibm_02_adaptive_global_t020` vs `aib_00_flat_local_t020` |
| Is threshold 0.90 special? | `aibm_03_adaptive_local_t020_r085`, `aib_00_flat_local_t020`, `aibm_04_adaptive_local_t020_r095` |

Expected useful metrics:

- `test/charts/episodic_survival`
- `test/explain/frac_action_0_agent_*`
- `train/intervention_penalty_mean`
- `train/intervention_budget_lambda_mean`
- `train/intervention_budget_cost_mean`
- `train/intervention_budget_safety_weight_mean`

## Launch

From the repository root:

```bash
bash Topology_Task/configs/adaptive_intervention_budget_mechanism_15/launch_all.sh
```

This uses 15 runs. Keep the 16th free slot for a failed Slurm job or for a
follow-up threshold/penalty after the first curves are visible.
