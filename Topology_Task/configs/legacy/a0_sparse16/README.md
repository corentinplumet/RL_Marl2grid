# A0 Sparse16

Sparse-action-penalty variants rebuilt on top of rerun_bias0_s* MLP base configs.

All configs in this folder use the matching seed from:

```text
Topology_Task/configs/rerun_bias0_obsstats_s0_s1_s2/rerun_bias0_s*.toml
```

The MLP rerun-bias base is preserved, including `actor_encoder = "mlp"`, `critic_encoder = "mlp"`, `init_do_nothing_prob = 0.0`, `total_timesteps = 20000000`, split chronics, and `eval_train_chronics = true`.

Only experiment-control keys are overlaid from the historical GINE sweep: `intervention_*`, `safe_intervention_*`, `adaptive_intervention_budget`, `eval_action_*`, and `trace_rollout_*`.

## Configs

| Config | Overlay keys |
| --- | --- |
| `a0_sparse16_flat_p000_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_flat_p000_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_flat_p000_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_flat_p001_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_flat_p001_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_flat_p001_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_flat_p003_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_flat_p003_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_flat_p003_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_flat_p010_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_flat_p010_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_flat_p010_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_gated_p000_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_gated_p000_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_gated_p000_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_gated_p001_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_gated_p001_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_gated_p001_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_gated_p003_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_gated_p003_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_gated_p003_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_gated_p010_s0.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_gated_p010_s1.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |
| `a0_sparse16_gated_p010_s2.toml` | `intervention_gate, intervention_gate_eval_mode, intervention_penalty, safe_intervention_penalty, safe_intervention_rho_threshold, trace_rollout_actions, trace_rollout_env_idx, trace_rollout_max_steps, trace_rollout_every, trace_rollout_decode_actions` |

## Launch

```bash
bash Topology_Task/configs/a0_sparse16/launch_all.sh
```
