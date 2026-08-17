# NLS WCCI mk64 seed-0 scratch versus conservative fine-tuning

This folder contains a matched 16-run WCCI campaign:

- eight shared candidate-scoring architectures;
- conservative fine-tuning (`ft64c`) versus random initialization (`sc64c`);
- seed 0 only;
- exactly 64 available actions per agent;
- 5M target-environment steps.

The two regimes use the same architecture, action space, WCCI chronic split,
optimizer settings, rollout schedule, critic initialization, and target budget.
The only intended difference is the actor initialization: `ft64c` transfers the
bus14 graph encoder and shared candidate scorer, while `sc64c` initializes them
randomly. Both actors remain fully trainable and both WCCI critics start fresh.

## Conservative settings

| Setting | Value |
|---|---:|
| Actor learning rate | `3e-5` |
| Critic learning rate | `1e-4` |
| Final LR fraction | `0.1` |
| PPO update epochs | 5 |
| Clip coefficient | `0.1` |
| Target KL | `0.01` |
| Entropy coefficient | `0.001 -> 0.0001` |
| Target steps | 5,000,000 |
| Local-rho during training | disabled |

The scratch learning rate is deliberately matched to fine-tuning so the
campaign isolates transferred initialization. It is not claimed to be the
optimal scratch learning rate; a later `1e-4` scratch control can test that
separately if needed.

## Launching on Izar

Run from the repository root. Preview all missing jobs:

```bash
DRY_RUN=true EXCLUDE_NODES=i39 \
Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_mk64_seed0/launch_izar.sh
```

Submit all missing jobs:

```bash
EXCLUDE_NODES=i39 \
Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_mk64_seed0/launch_izar.sh
```

The launcher skips exact job names already visible in `squeue` and runs with an
existing periodic, best-test, or final checkpoint. Thus, while
`ft64c_NLS_mean_f1_a0h0_s0` and `ft64c_NLS_tmean_f0_a0h0_s0` are active, the
unfiltered command plans the remaining 14 jobs.

Optional arguments are ANDed substring filters:

```bash
# All eight scratch controls
Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_mk64_seed0/launch_izar.sh sc64c

# The six fine-tuning jobs other than the two already active are skipped/planned
# automatically according to squeue and checkpoints.
Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_mk64_seed0/launch_izar.sh ft64c

# One matched architecture pair
Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_mk64_seed0/launch_izar.sh mean_f0_a0h0
```

Use `FORCE_LAUNCH=true` only when deliberately relaunching a checkpointed or
currently active experiment.

## Regeneration

The TOMLs are generated from the original matched WCCI transfer configs:

```bash
python Topology_Task/tools/generate_shared_wcci_mk64_seed0_configs.py
```

Use the completed `final_*.tar` checkpoints for the primary 50-chronic full-test
comparison. Apply local-rho only as a later evaluation-time ablation.
