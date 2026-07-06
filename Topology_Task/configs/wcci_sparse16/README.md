# WCCI Sparse16 Flat P0.003

WCCI reduced-action configs adapted from `a0_sparse16_flat_p003_s*`.

These configs preserve the WCCI reduced-action `72x576` setup:

```text
env_id = "bus36_wcci"
n_envs = 72
n_steps = 576
eval_freq = 82944
reduced_action_space = outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk256.json
```

The sparse-control overlay is:

```text
intervention_gate = false
intervention_penalty = 0.003
safe_intervention_penalty = 0.0
```

## Launch

```bash
bash Topology_Task/configs/wcci_sparse16/launch_all.sh
```
