# HVG 04 Local-Rho Reruns With Obs Stats

These configs rerun the historical `hvg_04_eval_local_rho090` setup for seeds
0, 1, and 2, but with new run names / `exp_tag`s so the new checkpoints can be
distinguished from the old checkpoints that did not contain observation
normalization statistics.

The hyperparameters are copied from:

```text
configs/heuristic_vs_gate_s0_s1_s2/hvg_04_eval_local_rho090_s*.toml
```

Launch:

```bash
sbatch job_jed.sh configs/hvg_04_obsstats_s0_s1_s2/hvg_04_eval_local_rho090_obsstats_s0.toml
sbatch job_jed.sh configs/hvg_04_obsstats_s0_s1_s2/hvg_04_eval_local_rho090_obsstats_s1.toml
sbatch job_jed.sh configs/hvg_04_obsstats_s0_s1_s2/hvg_04_eval_local_rho090_obsstats_s2.toml
```

After the runs save checkpoints, use `full_test_eval/evaluate_checkpoint.py`
with `--obs-normalization require` to confirm that the checkpoint contains the
normalization stats needed for faithful standalone evaluation.
