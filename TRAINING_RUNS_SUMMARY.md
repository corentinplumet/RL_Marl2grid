# MAPPO Training Run Summary

Updated on 2026-05-25 from `wandb_export_2026-05-25T22_46_16.343+08_00.csv`.

The CSV export is the current source of truth for run results. The main metric is eval survival rate from W&B `charts/episodic_survival`, converted to percent. `SPS` is computed from exported global step divided by runtime. Runs marked `running`, `killed`, or `crashed` should be interpreted as partial or interrupted snapshots.

## Shared Setup

Most named runs share this base setup:

| Hyperparameter | Value |
| --- | --- |
| total timesteps | 15,000,000 |
| eval frequency | 40,000 steps, except 60-env runs used 42,000 |
| actor layers | 256 256 256 |
| critic layers | 256 256 256 |
| actor lr / critic lr | 0.0001 / 0.0001 |
| gamma | 0.9 |
| GAE lambda | 0.95 |
| PPO clip coef | 0.2 |
| value loss coef | 0.5 |
| max grad norm | 1.0 |
| reward normalization | True |
| seed | 0 |
| task | bus14 topology, time limit 1300 |

## Changed Hyperparameters

These are the hyperparameters varied across the experiments and what they change during training.

| Hyperparameter | What it changes | Higher value means | Lower value means |
| --- | --- | --- | --- |
| `--init-do-nothing-prob` / `p0` | Initial actor bias toward action 0, the "do nothing" action. | Agent starts safer and more idle, with fewer random risky actions. Too high can trap the policy in idle behavior. | Agent explores non-idle topology actions earlier, but can make more bad actions at the beginning. |
| `--entropy-coef` | Strength of the entropy bonus in the actor loss. | More exploration and more action diversity, but the final sampled policy can stay too uncertain. | More confident policy sooner, but higher risk of premature collapse to a narrow action pattern. |
| `--target-kl` | PPO's limit for how far the new policy can move from the old policy during an update. | More aggressive policy updates, potentially faster learning, but higher instability risk. | More conservative updates, usually stabler but slower. |
| `--n-envs` | Number of parallel environments collecting rollouts. | Faster wall-clock data collection and more parallel trajectories. | Slower data collection, less parallelism. |
| `--n-steps` | Rollout length per environment before each PPO update. | Larger batches and longer trajectories before updating; often stabler but updates less frequently. | More frequent updates, but noisier batches and less trajectory context. |
| `--n-minibatches` | Number of chunks each rollout batch is split into during PPO updates. | Smaller minibatches, more granular/noisier optimization steps. | Larger minibatches, smoother gradients but fewer separate update chunks. |
| `--update-epochs` | Number of times PPO reuses the same collected batch. | More learning from each sample and more probability-mass concentration, but more overfitting/instability risk. | Less reuse of each batch; relies more on fresh environment data. |
| `--deterministic-eval` | Eval-only action selection mode. This does not affect training. | Greedy eval: use the highest-probability action. Shows whether the policy has learned a good argmax action. | Stochastic eval: sample from the policy distribution. Shows whether the policy is confident enough to act well when sampling. |

The key tension in these runs is that the policy needs enough entropy and non-idle exploration to discover useful topology actions, but enough PPO pressure to concentrate probability mass on those actions so stochastic eval also improves.

## Why These Runs Were Chosen

| Family | Changed hyperparameters | Reason for the choice |
| --- | --- | --- |
| A0 known good | p0=0.5, entropy=0.01, 20 envs x 2000 steps, 16 minibatches, 10 epochs, KL=0.02 | Reproduce the best-looking baseline from the earlier logs before changing one variable at a time. |
| A1 / A2 do-nothing prior | p0=0.3 and p0=0.7 | Action 0 is a useful fallback, so this tests whether weaker or stronger initial bias improves early learning. |
| A3 / A4 exploration | entropy=0.02, with p0=0.5 or p0=0.7 | Test whether more exploration helps escape the "always do nothing" phase faster. |
| A5 larger KL | target_kl=0.04 instead of 0.02 | Let PPO move the policy further per update, hoping for faster improvement. |
| B0 shorter rollout | 20 envs x 1000 steps, 10 minibatches | More frequent updates with smaller rollout batches, testing sample efficiency. |
| B1 more envs | 40 envs x 1000 steps | Same batch size as A0, but more parallelism, testing wall-clock speed. |
| p0=0.999 | init do-nothing probability almost 1.0 | Test the hypothesis that since successful policies first move toward action 0, starting almost fully idle saves time. |
| 40-env A1/A3 | 40 envs x 1000 steps with A1/A3 hyperparameters | Keep the same total batch size as A0 while improving wall-clock speed. |
| 60-env A1/A3 | 60 envs x 660 steps, 18 minibatches | Push environment parallelism further while keeping batch size close to 40k transitions. |
| stochastic repeats | `--deterministic-eval False` | Check whether the learned action distribution is actually confident, not only whether the greedy action is good. |

## Current Run Results

This table comes from `wandb_export_2026-05-25T22_46_16.343+08_00.csv`. It is a W&B run-table snapshot, so survival here is the latest exported `charts/episodic_survival` value, not necessarily the best value reached during training.

| Run | State | Eval | Batch | Epochs | MB | Entropy | KL | p0 | Step | Current survival | SPS | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `a3_40env_1000steps_det2` | finished | det | 40x1000=40k | 10 | 16 | 0.02 | 0.02 | 0.5 | 15.00M | 100.0% | 201 | Best deterministic export result. |
| `a1_p0_03_det` | crashed | det/pre-flag | 20x2000=40k | 10 | 16 | 0.01 | 0.02 | 0.3 | 9.72M | 92.7% | 201 | Strong, but interrupted. |
| `p0_0999_entropy002_20env_det2` | killed | det | 20x2000=40k | 10 | 16 | 0.02 | 0.02 | 0.999 | 9.84M | 84.1% | 234 | High p0 works only with higher entropy, still interrupted. |
| `a1_60env_660steps_det2` | running | det | 60x660=39.6k | 10 | 18 | 0.01 | 0.02 | 0.3 | 11.05M | 82.2% | 228 | Partial 60-env run, promising but not final. |
| `a1_40env_1000steps_det2` | killed | det | 40x1000=40k | 10 | 16 | 0.01 | 0.02 | 0.3 | 9.32M | 80.3% | 222 | Similar to earlier A1 40-env behavior. |
| `a4_entropy_002_p0_07_det` | crashed | det/pre-flag | 20x2000=40k | 10 | 16 | 0.02 | 0.02 | 0.7 | 9.24M | 76.5% | 192 | Greedy can be good, but this family is unstable. |
| `a5_kl_004_det` | crashed | det/pre-flag | 20x2000=40k | 10 | 16 | 0.01 | 0.04 | 0.5 | 12.24M | 69.4% | 254 | Higher KL alone still not best. |
| `p0999_entropy_decay_stoch` | running | stoch | 20x2000=40k | 10 | 16 | 0.02 -> 0.001 | 0.02 | 0.999 | 14.24M | 62.7% | 320 | Best current stochastic export value, but still running. |
| `a1_20env_entropy_to0_stoch` | running | stoch | 20x2000=40k | 10 | 16 | 0.01 -> 0 | 0.02 | 0.3 | 13.24M | 61.9% | 299 | Strongest low-p0 entropy-schedule stochastic run so far. |
| `a3_entropy_002_det` | crashed | det/pre-flag | 20x2000=40k | 10 | 16 | 0.02 | 0.02 | 0.5 | 9.84M | 59.9% | 203 | Interrupted deterministic repeat. |
| `MAPPO_baseline_nejjar` | finished | unknown | 20x2000=40k | 10 | 8 | 0.01 | 0.02 | 0.5 | 25.00M | 58.3% | 347 | Older long baseline. |
| `a0_known_good_sto` | finished | stoch | 20x2000=40k | 10 | 16 | 0.01 | 0.02 | 0.5 | 15.00M | 55.4% | 299 | Finished stochastic baseline latest value. |
| `a3_20env_entropy_decay_stoch` | running | stoch | 20x2000=40k | 10 | 16 | 0.02 -> 0.001 | 0.02 | 0.5 | 12.72M | 51.6% | 286 | Entropy decay helps but not enough yet. |
| `a3_40env_decay_mb20_stoch` | finished | stoch | 40x1000=40k | 10 | 20 | 0.02 -> 0.001 | 0.02 | 0.5 | 15.00M | 51.5% | 344 | More minibatches with decay did not break through. |
| `a2_p0_07_sto` | finished | stoch | 20x2000=40k | 10 | 16 | 0.01 | 0.02 | 0.7 | 15.00M | 46.7% | 290 | Better than A1/A3 fixed stochastic latest values. |
| `a3_entropy_002_sto` | finished | stoch | 20x2000=40k | 10 | 16 | 0.02 | 0.02 | 0.5 | 15.00M | 45.7% | 316 | Fixed entropy 0.02 not enough for stochastic confidence. |
| `a5_kl_004_sto` | finished | stoch | 20x2000=40k | 10 | 16 | 0.01 | 0.04 | 0.5 | 15.00M | 44.1% | 307 | Higher KL alone underperforms. |
| `a3_40env_entropy_decay_stoch` | finished | stoch | 40x1000=40k | 10 | 16 | 0.02 -> 0.001 | 0.02 | 0.5 | 15.00M | 43.0% | 360 | Good SPS, mediocre stochastic survival. |
| `p0_0999_entropy001_20env_det2` | killed | det | 20x2000=40k | 10 | 16 | 0.01 | 0.02 | 0.999 | 13.12M | 40.9% | 313 | Confirms p0=0.999 with low entropy is weak. |
| `b0_short_rollout_sto` | finished | stoch | 20x1000=20k | 10 | 10 | 0.01 | 0.02 | 0.5 | 15.00M | 40.4% | 297 | Frequent updates do not solve stochastic confidence. |
| `a1_p0_03_sto` | finished | stoch | 20x2000=40k | 10 | 16 | 0.01 | 0.02 | 0.3 | 15.00M | 36.7% | 340 | Stable deterministic family, poor stochastic latest value. |
| `a2_p0_07_det` | finished | det/pre-flag | 20x2000=40k | 10 | 16 | 0.01 | 0.02 | 0.7 | 15.00M | 33.9% | 338 | Poor deterministic final value. |
| `a3_40env_decay_margin005_stoch` | finished | stoch | 40x1000=40k | 10 | 16 | 0.02 -> 0.001 | 0.02 | 0.5 | 15.00M | 30.8% | 357 | Line-margin reward 0.05 hurt here. |
| `a3_60env_660steps_det2` | finished | det | 60x660=39.6k | 10 | 18 | 0.02 | 0.02 | 0.5 | 14.97M | 27.0% | 315 | 60-env A3 finished poorly. |
| `b1_40env_1000steps_sto` | finished | stoch | 40x1000=40k | 10 | 16 | 0.01 | 0.02 | 0.5 | 15.00M | 24.5% | 375 | Fast wall-clock, poor stochastic performance. |
| `a0_stoch_20env_mb20_ep15_kl004` | running | stoch | 20x2000=40k | 15 | 20 | 0.01 | 0.04 | 0.5 | 12.44M | 22.3% | 332 | Stronger PPO pressure not helping so far. |
| `a0_known_good_det` | finished | det/pre-flag | 20x2000=40k | 10 | 16 | 0.01 | 0.02 | 0.5 | 15.00M | 20.0% | 315 | Collapse after earlier greedy peak. |
| `a1_40env_entropy_to0_stoch` | finished | stoch | 40x1000=40k | 10 | 16 | 0.01 -> 0 | 0.02 | 0.3 | 15.00M | 18.9% | 484 | Fastest SPS, but bad survival. |
| `a1_stoch_ent0015_mb20_ep15_kl004` | running | stoch | 40x1000=40k | 15 | 20 | 0.015 | 0.04 | 0.3 | 14.08M | 18.6% | 376 | Fixed entropy + stronger PPO poor so far. |
| `a1_stoch_mb20_ep15_kl004` | running | stoch | 40x1000=40k | 15 | 20 | 0.01 | 0.04 | 0.3 | 13.64M | 16.9% | 364 | Fixed entropy + stronger PPO poor so far. |
| `a3_stoch_mb20_ep15_kl004` | running | stoch | 40x1000=40k | 15 | 20 | 0.02 | 0.04 | 0.5 | 14.00M | 16.0% | 372 | Fixed entropy + stronger PPO poor so far. |
| `a4_entropy_002_p0_07_sto` | finished | stoch | 20x2000=40k | 10 | 16 | 0.02 | 0.02 | 0.7 | 15.00M | 8.2% | 400 | Bad stochastic policy. |
| `a3_stoch_ent003_mb20_ep15_kl004` | running | stoch | 40x1000=40k | 15 | 20 | 0.03 | 0.04 | 0.5 | 14.36M | 7.5% | 380 | Too much fixed entropy appears harmful. |
| `a3_stoch_mb20_ep20_kl003` | running | stoch | 40x1000=40k | 20 | 20 | 0.02 | 0.03 | 0.5 | 12.48M | 5.6% | 333 | More epochs did not help so far. |
| `b0_short_rollout_det` | crashed | det/pre-flag | 20x1000=20k | 10 | 10 | 0.01 | 0.02 | 0.5 | 9.92M | 0.0% | 205 | Crashed/collapsed run. |

## Analysis

The strongest pattern in the current CSV is still the gap between deterministic and stochastic eval. The best deterministic finished run is `a3_40env_1000steps_det2`, which reaches 100.0% current survival at 15M steps. The best stochastic rows are much lower: `p0999_entropy_decay_stoch` is at 62.7% and `a1_20env_entropy_to0_stoch` is at 61.9%, and both are still running.

The export suggests that the policy can learn a good greedy action, but stochastic sampling still suffers because too much probability mass remains on bad actions. In practical terms, deterministic eval answers "does the argmax action work?", while stochastic eval answers "is the policy distribution confident enough to sample good actions reliably?"

The `p0` results are mixed. Very high idle bias is not enough by itself: `p0_0999_entropy001_20env_det2` only has 40.9% current survival, while `p0_0999_entropy002_20env_det2` is much better at 84.1% but was killed. The best current stochastic row, `p0999_entropy_decay_stoch`, combines very high `p0` with entropy decay, suggesting that high idle bias may need strong exploration early and lower entropy later.

Entropy scheduling now looks more promising than fixed entropy. The strongest stochastic rows in the export use entropy decay: `p0999_entropy_decay_stoch`, `a1_20env_entropy_to0_stoch`, `a3_20env_entropy_decay_stoch`, and `a3_40env_decay_mb20_stoch`. They still do not reach the 90%-100% target, but they are clearly ahead of the fixed-entropy stronger-PPO group.

Fixed stronger PPO pressure is currently weak. The `mb20_ep15_kl004` and `ep20_kl003` stochastic runs are at only 5.6%-22.3% current survival around 12.4M-14.4M steps. Increasing minibatches, epochs, and KL without scheduling did not solve stochastic confidence.

Throughput improvements are not the same as learning improvements. `b1_40env_1000steps_sto` has good SPS but only 24.5% current survival. `a1_40env_entropy_to0_stoch` is very fast at 484 SPS but only reaches 18.9%. The 60-env A3 deterministic run also finished poorly at 27.0%, so pushing parallelism too far is not automatically useful.

The next direction should focus on stochastic confidence with scheduling or curriculum-like shaping. Fixed entropy with stronger PPO updates looks like a bad direction in this export. The most useful next comparisons are around entropy decay, `p0` choice, and possibly mechanisms that reduce bad-action probability late in training without blocking early exploration.

## Current Best Bets

| Goal | Best current run | Reason |
| --- | --- | --- |
| Best finished deterministic run | `a3_40env_1000steps_det2` | Finished at 15M with 100.0% current survival. |
| Best interrupted deterministic run | `a1_p0_03_det` | Crashed at 9.72M but already had 92.7% current survival. |
| Best current stochastic runs | `p0999_entropy_decay_stoch`, `a1_20env_entropy_to0_stoch` | Both are still running and are the only stochastic rows above 60% current survival. |
| Best finished stochastic run | `a0_known_good_sto` | Finished at 15M with 55.4% current survival. |
| Avoid for now | fixed stronger-PPO stochastic runs, `a3_60env_660steps_det2`, line-margin reward run | These underperform clearly in the current export. |

## Fixed-Entropy Test Status

These were proposed to target stochastic eval without an entropy scheduler. The W&B export suggests this direction is currently weak.

| Tag | Key change | Export status | Current survival | Interpretation |
| --- | --- | --- | ---: | --- |
| `a3_stoch_mb20_ep15_kl004` | A3 with 40 envs, 20 minibatches, 15 epochs, KL=0.04 | running at 14.00M | 16.0% | Stronger PPO pressure did not help. |
| `a3_stoch_ent003_mb20_ep15_kl004` | Same as above, entropy=0.03 | running at 14.36M | 7.5% | Too much fixed entropy appears harmful. |
| `a3_stoch_mb20_ep20_kl003` | A3 with 20 epochs, KL=0.03 | running at 12.48M | 5.6% | More epochs did not improve stochastic survival. |
| `a1_stoch_mb20_ep15_kl004` | A1 with stronger PPO updates | running at 13.64M | 16.9% | Not better than older A1 stochastic. |
| `a1_stoch_ent0015_mb20_ep15_kl004` | A1 with entropy=0.015 | running at 14.08M | 18.6% | Middle entropy point is still weak. |
| `a0_stoch_20env_mb20_ep15_kl004` | A0 baseline with stronger PPO updates | running at 12.44M | 22.3% | Best of this fixed-entropy group, but still poor. |
