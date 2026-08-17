# NLS WCCI mk64 seed-0 scratch versus conservative fine-tuning

This folder contains a matched 16-run WCCI campaign:

- eight shared candidate-scoring architectures;
- conservative fine-tuning (`ft64c`) versus random initialization (`sc64c`);
- seed 0 only;
- exactly 64 available actions per agent;
- a 5M-step conservative recipe for fine-tuning;
- the 15M-step bus14 recipe for training from scratch.

The two regimes use the same architecture, mk64 action space, seed, WCCI
chronic split, rollout schedule, and fresh WCCI critic initialization. They use
training recipes appropriate to their roles: `ft64c` transfers the bus14 graph
encoder and shared candidate scorer and adapts them conservatively, while
`sc64c` initializes the actor randomly and uses the original, stronger bus14
training recipe. Both actors remain fully trainable.

## Training settings

| Setting | `ft64c` fine-tuning | `sc64c` scratch |
|---|---:|---:|
| Actor learning rate | `3e-5` | `1e-4` |
| Critic learning rate | `1e-4` | `1e-4` |
| Final LR fraction | `0.1` | `0.0` |
| PPO update epochs | 5 | 10 |
| Clip coefficient | `0.1` | `0.2` |
| Target KL | `0.01` | `0.02` |
| Entropy coefficient | `0.001 -> 0.0001` | `0.01 -> 0.01` |
| Target steps | 5,000,000 | 15,000,000 |
| Local-rho during training | disabled | disabled |

This is a procedure-level comparison rather than a one-variable initialization
ablation: it compares conservative transfer against a credible from-scratch
recipe. A later conservative scratch control on only the leading architectures
can isolate initialization if needed.

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
