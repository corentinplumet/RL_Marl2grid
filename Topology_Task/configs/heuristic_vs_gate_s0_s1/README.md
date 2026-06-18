# Heuristic vs Gate, Seeds 0-1

This folder compares four variants using `best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update` as the base config.

Design:

- baseline flat actor
- baseline flat actor with an evaluation-only rho-threshold heuristic
- intervention gate with final-action MAP deterministic evaluation
- intervention gate with hierarchical-greedy deterministic evaluation
- seeds: `0`, `1`

The rho heuristic is evaluation-only: the trained policy proposes actions, then evaluation forces all agents to action `0` if the pre-action `max_rho < 0.90`.

Gate evaluation modes:

- `final_action_map`: chooses the most likely executed environment action under `P(action 0)=P(do nothing)` and `P(action a>0)=P(intervene) * P(non-idle action a)`.
- `hierarchical_greedy`: first compares `P(do nothing)` and `P(intervene)`; if do-nothing is higher it executes action `0`, otherwise it executes the highest-probability non-idle action.

| Config | Variant | Seed | intervention_gate | gate eval mode | eval_action_heuristic | rho threshold |
| --- | --- | ---: | --- | --- | --- | ---: |
| `hvg_00_baseline_s0.toml` | Baseline | 0 | false | `final_action_map` | `none` | 0.90 |
| `hvg_01_eval_rho090_s0.toml` | Baseline + eval rho heuristic | 0 | false | `final_action_map` | `rho_threshold` | 0.90 |
| `hvg_02_gate_final_map_s0.toml` | Intervention gate final-action MAP | 0 | true | `final_action_map` | `none` | 0.90 |
| `hvg_03_gate_hierarchical_s0.toml` | Intervention gate hierarchical greedy | 0 | true | `hierarchical_greedy` | `none` | 0.90 |
| `hvg_00_baseline_s1.toml` | Baseline | 1 | false | `final_action_map` | `none` | 0.90 |
| `hvg_01_eval_rho090_s1.toml` | Baseline + eval rho heuristic | 1 | false | `final_action_map` | `rho_threshold` | 0.90 |
| `hvg_02_gate_final_map_s1.toml` | Intervention gate final-action MAP | 1 | true | `final_action_map` | `none` | 0.90 |
| `hvg_03_gate_hierarchical_s1.toml` | Intervention gate hierarchical greedy | 1 | true | `hierarchical_greedy` | `none` | 0.90 |

## Launch

```bash
cd /home/plumet/RL_Marl2grid/Topology_Task
bash configs/heuristic_vs_gate_s0_s1/launch_all.sh
```
