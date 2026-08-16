# WCCI AIB + Topology Distance

Topology-distance reward sweep for WCCI reduced-action MAPPO.

These configs are copied from the current `configs/wcci_aib` local-safe flat AIB
configs, then only `topology_reward_weight` and the run names are changed.
They preserve the WCCI setup used by the source configs, including:

```text
env_id = "bus36_wcci"
n_envs = 72
n_steps = 576
eval_freq = 82944
reduced_action_space = outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk256.json
```

## Variants

| Config prefix | AIB cost mode | AIB target | Topology reward weight |
| --- | --- | --- | --- |
| `wcci_aib_00_flat_local_t020_topo003_72x576_s*` | `local_safe` | `0.20` | `0.003` |
| `wcci_aib_00_flat_local_t020_topo010_72x576_s*` | `local_safe` | `0.20` | `0.01` |
| `wcci_aib_01_flat_local_t010_topo003_72x576_s*` | `local_safe` | `0.10` | `0.003` |
| `wcci_aib_01_flat_local_t010_topo010_72x576_s*` | `local_safe` | `0.10` | `0.01` |

Each variant has seeds `0`, `1`, and `2`.

## Launch

```bash
bash Topology_Task/configs/wcci_aib_topology/launch_all.sh
```

For diagnostic runs with chronic-subset printing, append `--eval-progress-print true`
to individual `sbatch` commands.
