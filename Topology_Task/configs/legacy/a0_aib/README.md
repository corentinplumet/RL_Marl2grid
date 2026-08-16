# A0 AIB

Adaptive intervention-budget variants rebuilt on top of rerun_bias0_s* MLP base configs.

All configs in this folder use the matching seed from:

```text
Topology_Task/configs/rerun_bias0_obsstats_s0_s1_s2/rerun_bias0_s*.toml
```

The MLP rerun-bias base is preserved, including `actor_encoder = "mlp"`, `critic_encoder = "mlp"`, `init_do_nothing_prob = 0.0`, `total_timesteps = 20000000`, split chronics, and `eval_train_chronics = true`.

Only experiment-control keys are overlaid from the historical GINE sweep: `intervention_*`, `safe_intervention_*`, `adaptive_intervention_budget`, `eval_action_*`, and `trace_rollout_*`.

## Configs

| Config | Overlay keys |
| --- | --- |
| `a0_aib_00_flat_local_t020_s0.toml` | `adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_00_flat_local_t020_s1.toml` | `adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_00_flat_local_t020_s2.toml` | `adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_01_flat_local_t010_s0.toml` | `adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_01_flat_local_t010_s1.toml` | `adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_01_flat_local_t010_s2.toml` | `adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_02_flat_local_t035_s0.toml` | `adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_02_flat_local_t035_s1.toml` | `adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_02_flat_local_t035_s2.toml` | `adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_03_gate_hgreedy_sep_local_t020_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_gate_entropy_mode, intervention_gate_entropy_mult, intervention_nonidle_entropy_mult, adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_03_gate_hgreedy_sep_local_t020_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_gate_entropy_mode, intervention_gate_entropy_mult, intervention_nonidle_entropy_mult, adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_03_gate_hgreedy_sep_local_t020_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_gate_entropy_mode, intervention_gate_entropy_mult, intervention_nonidle_entropy_mult, adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_04_flat_nonidle_t020_s0.toml` | `adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_04_flat_nonidle_t020_s1.toml` | `adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |
| `a0_aib_04_flat_nonidle_t020_s2.toml` | `adaptive_intervention_budget, intervention_budget_cost_mode, intervention_budget_target, intervention_budget_lr, intervention_budget_rho_threshold, intervention_budget_rho_sharpness` |

## Launch

```bash
bash Topology_Task/configs/a0_aib/launch_all.sh
```
