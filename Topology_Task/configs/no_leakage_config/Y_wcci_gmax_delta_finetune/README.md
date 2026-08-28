# Conservative WCCI Gmax-delta fine-tuning

This campaign fine-tunes the seven completed NLS Gmax-delta bus14 actors on
`bus36_wcci_nomaint`, using the `mk32` and `mk64` reduced action spaces. It
contains 14 seed-0 runs.

The missing eighth architecture, `tmean_f1_a0h1`, is not included because no
completed NLS source checkpoint exists. Do not silently replace it with random
weights or an NL checkpoint.

Each run transfers the graph encoder, action-delta token encoder, and shared
candidate scorer. They remain trainable. The WCCI critic, optimizer state,
normalization statistics, and environment state start from scratch.

The optimization protocol matches the conservative candidate-scorer
fine-tuning campaign: 5M target steps, actor learning rate `3e-5`, critic
learning rate `1e-4`, five PPO epochs, clip coefficient `0.1`, target KL
`0.01`, and entropy annealed from `0.001` to `0.0001`.

Preview or submit all runs from the repository root:

```bash
DRY_RUN=true bash Topology_Task/configs/no_leakage_config/Y_wcci_gmax_delta_finetune/launch.sh jed
bash Topology_Task/configs/no_leakage_config/Y_wcci_gmax_delta_finetune/launch.sh jed
```

Replace `jed` with `izar` to use Izar. Optional filename filters are ANDed, so
`launch.sh jed ftgd32 tmean` selects the three available typed-mean mk32 runs.

Regenerate the TOMLs with:

```bash
python Topology_Task/tools/generate_gmax_delta_wcci_finetune_configs.py
```
