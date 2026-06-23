# Rerun Bias 0 With Obs Stats

These configs rerun the `rerun_bias_0` setup on seeds 0, 1, and 2. They keep
the same training/evaluation hyperparameters as:

```text
configs/deterministic_evaluation/rerun_bias_0.toml
```

The folder also contains `rerun_a0_known_good.toml`, which keeps the original
`a0_known_good_det` initialization bias (`init_do_nothing_prob = 0.5`) but uses
the same split/20M/eval protocol. This isolates the protocol change from the
zero-bias change.

The only intended changes are:

- `run.name`
- `exp_tag`
- `seed` for the seed 1 and seed 2 copies

The current code now saves `training_state["obs_stats"]` in checkpoints, so
these reruns can later be evaluated faithfully with `full_test_eval` and
`--obs-normalization require`.

## Launch

```bash
sbatch job_jed.sh configs/rerun_bias0_obsstats_s0_s1_s2/rerun_bias0_obsstats_s0.toml
sbatch job_jed.sh configs/rerun_bias0_obsstats_s0_s1_s2/rerun_bias0_obsstats_s1.toml
sbatch job_jed.sh configs/rerun_bias0_obsstats_s0_s1_s2/rerun_bias0_obsstats_s2.toml
sbatch job_jed.sh configs/rerun_bias0_obsstats_s0_s1_s2/rerun_a0_known_good.toml
```

## What Changed Versus The Earlier `a0_known_good_det` Config

`rerun_bias_0` is not a GNN/gate/heuristic model. It is a flat MLP MAPPO model:

- `actor_encoder = "mlp"`
- `critic_encoder = "mlp"`
- `eval_action_heuristic` is not set, so no rho-threshold override is used
- `intervention_gate = false`

Compared with `configs/deterministic_evaluation/a0_known_good_det.toml`, the
important differences are:

- `split_chronics = true` instead of `false`
- `test_chronics_pct = 0.2` and `chronic_split_seed = 0`
- `total_timesteps = 20000000` instead of `15000000`
- `eval_freq = 80000` instead of `40000`
- `init_do_nothing_prob = 0.0` instead of `0.5`
- `eval_train_chronics = true`

The most important causal suspect is `init_do_nothing_prob = 0.0`.
`rerun_bias_05.toml` and `rerun_bias_07.toml` keep the same split/longer-run
setup as `rerun_bias_0`, but change only the initial do-nothing prior to 0.5
and 0.7. If those runs are much worse than `rerun_bias_0`, then the likely
story is that forcing an early do-nothing prior hurt exploration or trapped the
policy in overly conservative behavior.

Be careful comparing `rerun_bias_0` directly to old non-split runs:
`split_chronics = true` changes the evaluation protocol. The clean comparison is
therefore mostly:

```text
rerun_bias_0  vs  rerun_bias_05  vs  rerun_bias_07
```

not only `rerun_bias_0` versus `a0_known_good_det`.

`rerun_a0_known_good` is the explicit same-folder control for:

```text
a0_known_good_det + split_chronics + 20M timesteps + eval_freq 80k + train-split eval
```

while preserving:

```text
init_do_nothing_prob = 0.5
```
