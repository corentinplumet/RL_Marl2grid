# WCCI AIB 72x576

Adaptive intervention-budget variants for `bus36_wcci`, built on top of the
`wcci_reduced_mlp_baseline_72x576_s*` configs.

The base WCCI setup is preserved:

```text
env_id = "bus36_wcci"
n_envs = 72
n_steps = 576
eval_freq = 82944
actor_encoder = "mlp"
critic_encoder = "mlp"
reduced_action_space = outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk256.json
```

Only the AIB/intervention overlay is copied from `configs/a0_aib`.
Seeds are `0`, `1`, and `2`.

## Variants

| Prefix | Meaning |
| --- | --- |
| `wcci_aib_00_flat_local_t020_72x576_s*` | Flat policy, local-safe AIB target `0.20` |
| `wcci_aib_01_flat_local_t010_72x576_s*` | Flat policy, local-safe AIB target `0.10` |
| `wcci_aib_02_flat_local_t035_72x576_s*` | Flat policy, local-safe AIB target `0.35` |
| `wcci_aib_03_gate_hgreedy_sep_local_t020_72x576_s*` | Intervention gate, hierarchical greedy eval, separate entropy, local-safe AIB target `0.20` |
| `wcci_aib_04_flat_nonidle_t020_72x576_s*` | Flat policy, non-idle AIB cost target `0.20` |

## Launch

```bash
bash Topology_Task/configs/wcci_aib/launch_all.sh
```
