# Conservative WCCI fine-tuning with the mk64 action space

This note records the conservative fine-tuning protocol introduced after the
initial WCCI fine-tuning campaign. It is intentionally separate from the
original 15M-step mk256 configurations so the earlier experiment remains
reproducible.

## Motivation

The original target-training configurations used the same actor learning rate
for training from scratch and fine-tuning:

```text
actor learning rate = 1e-4
critic learning rate = 1e-4
PPO update epochs    = 10
clip coefficient     = 0.2
target KL            = 0.02
entropy coefficient  = 0.01
target budget        = 15M steps
action-space cap     = 256
```

Although the actor was initialized from a pretrained bus14 checkpoint, its
optimizer was new and its learning rate restarted at `1e-4`. The complete
actor was unfrozen. This protocol is therefore more accurately described as
aggressive warm-start retraining than as conservative fine-tuning, and it can
overwrite useful transferred features early in target training.

The revised protocol lowers and anneals the actor learning rate, makes each PPO
update more conservative, reduces exploration, uses a shorter target budget,
and trains on exactly 64 available actions per agent.

## Runs selected

The first conservative campaign contains two scaled (`NLS`) seed-0 runs:

| Target run | Source checkpoint role |
|---|---|
| `ft64c_NLS_mean_f1_a0h0_s0` | Best original mk64 difficult-chronic robustness |
| `ft64c_NLS_tmean_f0_a0h0_s0` | Strongest original mk64 result with local-rho evaluation gating |

Their source checkpoints are, respectively:

```text
checkpoint/no_leak/shared/NL_cas_hl_scaled_shared_izar/best_test_cas_hl_NLS_izar_shared_mean_f1_a0h0_s0.tar
checkpoint/no_leak/shared/NL_cas_hl_scaled_shared_izar/best_test_cas_hl_NLS_izar_shared_tmean_f0_a0h0_s0.tar
```

The `mean_f1_a0h0` run is the primary experiment. The `tmean_f0_a0h0` run is a
complementary architecture control rather than a separate hyperparameter
search.

## Transfer procedure

Each run starts by constructing a fresh WCCI `bus36_wcci_nomaint` actor with
the target mk64 candidate-action metadata. It then loads:

- the bus14 graph-encoder parameters;
- the learned candidate-action scorer parameters, using source `agent_0` as
  the representative shared scorer.

Grid-shaped topology buffers, target action IDs, and target action descriptors
are not copied from bus14. They remain the values constructed for WCCI.

The following state is deliberately initialized fresh on WCCI:

- the target MLP critic;
- actor and critic Adam optimizer states;
- observation-normalization statistics;
- reward-normalization statistics;
- environment trajectories and random-number state;
- training global step, which starts at zero.

This is transfer initialization, not checkpoint resume. Both the transferred
GNN encoder and candidate scorer remain trainable. The critic learns from
scratch at a higher learning rate.

## Conservative optimization changes

| Setting | Original fine-tuning | Conservative mk64 |
|---|---:|---:|
| Reduced action-space cap | 256 | 64 |
| Target steps | 15,000,000 | 5,000,000 |
| Actor learning rate | `1e-4` | `3e-5` |
| Final actor learning rate | 0 | `3e-6` |
| Critic learning rate | `1e-4` | `1e-4` |
| Final critic learning rate | 0 | `1e-5` |
| PPO update epochs | 10 | 5 |
| PPO clip coefficient | 0.2 | 0.1 |
| Target KL | 0.02 | 0.01 |
| Initial entropy coefficient | 0.01 | 0.001 |
| Final entropy coefficient | 0.01 | 0.0001 |
| Encoder frozen | No | No |
| Candidate scorer frozen | No | No |
| Intervention gate during training | No | No |
| Local-rho heuristic during training | No | No |

Learning rates are linearly annealed over the 5M-step target budget and retain
10% of their initial value at the end (`lr_final_frac=0.1`). The fresh critic
therefore learns more quickly than the transferred actor throughout training.

No local-rho gate, intervention penalty, safe-intervention penalty, or special
do-nothing prior is introduced during training. Local-rho remains a separate
evaluation-time mechanism so its contribution can be measured independently.

## Implementation

The protocol is implemented by:

```text
launch_nls_conservative_finetune_mk64_izar.sh
```

The launcher reuses the existing matching NLS fine-tuning TOMLs for the model
architecture and source checkpoint, then passes explicit command-line
overrides for mk64 and the conservative optimizer settings. It does not edit or
replace the original mk256/15M-step configurations.

Before submission, it verifies that the target reduced-action artifact contains
exactly 64 selected actions for every agent and that the required source
checkpoint exists. Existing run checkpoints are skipped unless
`FORCE_LAUNCH=true` is provided.

## Launch commands

Preview both jobs without submitting:

```bash
DRY_RUN=true EXCLUDE_NODES=i39 \
Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_transfer/launch_nls_conservative_finetune_mk64_izar.sh
```

Submit both jobs:

```bash
EXCLUDE_NODES=i39 \
Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_transfer/launch_nls_conservative_finetune_mk64_izar.sh
```

Submit only the primary `mean_f1_a0h0` run:

```bash
EXCLUDE_NODES=i39 \
Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_transfer/launch_nls_conservative_finetune_mk64_izar.sh \
  mean_f1_a0h0
```

The default settings can be overridden explicitly for a controlled ablation.
For example, an actor learning-rate control at `1e-5` can be submitted with:

```bash
ACTOR_LR=0.00001 RUN_SUFFIX=_lr10 EXCLUDE_NODES=i39 \
Topology_Task/configs/no_leakage_config/W_wcci_cas_hl_shared_transfer/launch_nls_conservative_finetune_mk64_izar.sh \
  mean_f1_a0h0
```

`RUN_SUFFIX` is appended to the experiment tag and checkpoint name, preventing
the ablation from overwriting or being confused with the default run.

## Checkpoints and evaluation rule

Expected completed checkpoints are:

```text
checkpoint/final_ft64c_NLS_mean_f1_a0h0_s0.tar
checkpoint/final_ft64c_NLS_tmean_f0_a0h0_s0.tar
```

The training loop also produces `best_test_...` checkpoints from periodic
10-episode train/test evaluation windows. Those windows rotate through the
chronic split, so their scores are not strictly comparable from one evaluation
time to another. The `best_test_...` checkpoint can be used for debugging, but
the primary conservative comparison should use the completed `final_...`
checkpoint selected without inspecting the final full-test result.

After training, evaluate each final checkpoint in two separate stages on the
same 50 held-out chronics:

1. deterministic mk64 evaluation with `eval_action_heuristic=none`;
2. deterministic mk64 local-rho evaluation, with thresholds treated as a
   separate deployment ablation.

Report at least overall mean survival, difficult-chronic mean and median,
rescues, non-rescue delta versus do-nothing, easy-chronic preservation, and
executed versus pre-gate policy intervention rates. Compare against the
matching original zero-shot mk64 checkpoint before comparing with any earlier
aggressive fine-tuning run.

## Current limitations

- Only seed 0 is included.
- Encoder and candidate scorer use one shared actor learning rate; differential
  encoder/head learning rates are not yet implemented.
- The source checkpoints themselves were saved using the historical
  `best_test_` convention.
- The periodic target checkpoint score observes the target test split, so it
  must not be used as the final unbiased performance estimate.
- The protocol tests gentle full-actor adaptation. A frozen-encoder/head-only
  run would be a separate ablation, not part of this initial campaign.
