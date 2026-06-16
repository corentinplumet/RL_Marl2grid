# Intervention Gate 15-Run Ablation

Fifteen configs generated from `configs/gine_s0_s1_s2/best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update_s*.toml`.

All runs use:

- `intervention_gate = true`
- `intervention_gate_eval_mode = "final_action_map"`
- `init_do_nothing_prob = 0.7` inherited from `best_00`
- `trace_rollout_actions = false` by default, to avoid heavy W&B tables in the 15-run sweep

To trace one run, append CLI overrides such as:

```bash
sbatch job_jed.sh configs/intervention_gate_15/ig_01_entropy_decay_s0.toml \
  --trace-rollout-actions true \
  --trace-rollout-max-steps 512 \
  --trace-rollout-every 5 \
  --trace-rollout-decode-actions true
```

## Conditions

| Config | Seed | Purpose | topology_reward_weight | entropy_coef_final |
| --- | ---: | --- | ---: | ---: |
| `ig_00_phase2_base_s0` | 0 | Intervention gate only; same entropy/topology reward as best_00. | 0.0 | 0.02 |
| `ig_01_entropy_decay_s0` | 0 | Intervention gate with entropy annealed to zero. | 0.0 | 0.0 |
| `ig_02_topo001_entropy_decay_s0` | 0 | Entropy decay plus weak topology-distance penalty. | 0.001 | 0.0 |
| `ig_03_topo005_entropy_decay_s0` | 0 | Entropy decay plus medium topology-distance penalty. | 0.005 | 0.0 |
| `ig_04_topo010_entropy_decay_s0` | 0 | Entropy decay plus stronger topology-distance penalty. | 0.01 | 0.0 |
| `ig_00_phase2_base_s1` | 1 | Intervention gate only; same entropy/topology reward as best_00. | 0.0 | 0.02 |
| `ig_01_entropy_decay_s1` | 1 | Intervention gate with entropy annealed to zero. | 0.0 | 0.0 |
| `ig_02_topo001_entropy_decay_s1` | 1 | Entropy decay plus weak topology-distance penalty. | 0.001 | 0.0 |
| `ig_03_topo005_entropy_decay_s1` | 1 | Entropy decay plus medium topology-distance penalty. | 0.005 | 0.0 |
| `ig_04_topo010_entropy_decay_s1` | 1 | Entropy decay plus stronger topology-distance penalty. | 0.01 | 0.0 |
| `ig_00_phase2_base_s2` | 2 | Intervention gate only; same entropy/topology reward as best_00. | 0.0 | 0.02 |
| `ig_01_entropy_decay_s2` | 2 | Intervention gate with entropy annealed to zero. | 0.0 | 0.0 |
| `ig_02_topo001_entropy_decay_s2` | 2 | Entropy decay plus weak topology-distance penalty. | 0.001 | 0.0 |
| `ig_03_topo005_entropy_decay_s2` | 2 | Entropy decay plus medium topology-distance penalty. | 0.005 | 0.0 |
| `ig_04_topo010_entropy_decay_s2` | 2 | Entropy decay plus stronger topology-distance penalty. | 0.01 | 0.0 |

## Launch All

```bash
sbatch job_jed.sh configs/intervention_gate_15/ig_00_phase2_base_s0.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_01_entropy_decay_s0.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_02_topo001_entropy_decay_s0.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_03_topo005_entropy_decay_s0.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_04_topo010_entropy_decay_s0.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_00_phase2_base_s1.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_01_entropy_decay_s1.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_02_topo001_entropy_decay_s1.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_03_topo005_entropy_decay_s1.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_04_topo010_entropy_decay_s1.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_00_phase2_base_s2.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_01_entropy_decay_s2.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_02_topo001_entropy_decay_s2.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_03_topo005_entropy_decay_s2.toml
sbatch job_jed.sh configs/intervention_gate_15/ig_04_topo010_entropy_decay_s2.toml
```
