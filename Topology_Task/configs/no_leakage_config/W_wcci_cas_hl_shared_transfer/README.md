# Shared-candidate WCCI scratch versus fine-tuning campaign

The later conservative mk64 protocol and the reasons for changing the original
fine-tuning settings are documented in
[`CONSERVATIVE_FINETUNING_MK64.md`](CONSERVATIVE_FINETUNING_MK64.md).

This folder contains the seed-0 target-training comparison for the shared
candidate actors previously evaluated zero-shot on WCCI. It contains 32 runs:

- 2 preprocessing families: `NL` (unscaled) and `NLS` (scaled);
- 8 candidate-scoring variants: `mean`/`typed_mean`, with or without candidate
  features, and with or without a separate do-nothing head;
- 2 target-training regimes: `scratch` and `finetune`.

## Controlled comparison

Each scratch/fine-tune pair uses the same WCCI no-maintenance environment,
reduced 256-action space, seed, architecture, 15M target-step budget, learning
rates, rollout schedule, and evaluation schedule.

- `scratch`: the actor encoder and shared candidate scorer start from random
  initialization.
- `finetune`: the bus14 encoder and shared candidate scorer are loaded, then
  both remain trainable. The WCCI critic and WCCI observation statistics start
  fresh.

This isolates the effect of transferred initialization. It does not resume the
bus14 optimizer, critic, training step, or observation normalization state.

The unscaled runs load from:

`checkpoint/no_leak/shared/NL_cas_hl_shared`

The scaled runs load from the Izar family used in the existing zero-shot test:

`checkpoint/no_leak/shared/NL_cas_hl_scaled_shared_izar`

The existing zero-shot full-test results are in:

- `outputs/full_test_eval/shared/NL_cas_hl_shared_wcci36`
- `outputs/full_test_eval/shared/NLS_cas_hl_izar_shared_wcci36`

The downloaded unscaled zero-shot results currently contain 7 of the 8 cells:
`mean_f0_a0h1` is missing. Run that evaluation as well if a complete three-way
zero-shot/scratch/fine-tune table is required.

## Launching on Izar

From the repository root, preview all 32 submissions:

```bash
DRY_RUN=true bash Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_transfer/launch_izar.sh
```

Submit all runs:

```bash
bash Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_transfer/launch_izar.sh
```

Optional filename filters are combined. Examples:

```bash
# The 16 fine-tuning runs only
bash Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_transfer/launch_izar.sh finetune

# The 8 scaled fine-tuning runs only
bash Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_transfer/launch_izar.sh NLS finetune

# One matched scratch/fine-tune pair
bash Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_transfer/launch_izar.sh NL mean_f0_a0h0
```

Before real submission, the launcher checks that the WCCI reduced-action-space
file exists. For every selected fine-tuning config it also checks the matching
source checkpoint. `DRY_RUN=true` skips those cluster-data checks and only
prints the commands.

## Regeneration

The TOMLs and launcher are generated from the corrected no-leakage source
configs:

```bash
python Topology_Task/tools/generate_shared_wcci_transfer_configs.py
```
