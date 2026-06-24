# A0 HVG

Heuristic-vs-gate variants rebuilt on top of rerun_bias0_s* MLP base configs.

All configs in this folder use the matching seed from:

```text
Topology_Task/configs/rerun_bias0_obsstats_s0_s1_s2/rerun_bias0_s*.toml
```

The MLP rerun-bias base is preserved, including `actor_encoder = "mlp"`, `critic_encoder = "mlp"`, `init_do_nothing_prob = 0.0`, `total_timesteps = 20000000`, split chronics, and `eval_train_chronics = true`.

Only experiment-control keys are overlaid from the historical GINE sweep: `intervention_*`, `safe_intervention_*`, `adaptive_intervention_budget`, `eval_action_*`, and `trace_rollout_*`.

## Configs

| Config | Overlay keys |
| --- | --- |
| `a0_hvg_00_baseline_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_00_baseline_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_00_baseline_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_01_eval_rho090_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_01_eval_rho090_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_01_eval_rho090_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_02_gate_final_map_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_02_gate_final_map_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_02_gate_final_map_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_03_gate_hierarchical_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_03_gate_hierarchical_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_03_gate_hierarchical_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_04_eval_local_rho090_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_04_eval_local_rho090_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |
| `a0_hvg_04_eval_local_rho090_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, eval_action_heuristic, eval_action_rho_threshold` |

## Launch

```bash
bash Topology_Task/configs/a0_hvg/launch_all.sh
```
