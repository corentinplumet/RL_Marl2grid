# JED Run Configs

These TOML files define one training run each. The Slurm job activates the
environment, reads the TOML, and forwards `[args]` to `Topology_Task/main.py`.
Paths passed to `job_jed.sh` are resolved relative to `Topology_Task`.

Submit the default old MLP training:

```bash
sbatch job_jed.sh
```

Submit a specific config:

```bash
sbatch job_jed.sh configs/old_training.toml
```

Submit one of the reproduced historical W&B runs:

```bash
sbatch job_jed.sh configs/previous_runs/a3_40env_1000steps_det2.toml
```

The `previous_runs/` directory contains one TOML per row from the latest W&B
CSV export, plus a README table with the W&B state, exported step, survival,
eval mode, split mode, and rollout batch.

Submit one of the new GINE runs:

```bash
sbatch job_jed.sh configs/gine/gine20_a1_decay_det.toml
```

The `gine/` directory contains 8 GINE configs: four training recipes, each with
deterministic and stochastic eval.

Append one-off overrides after the config path:

```bash
sbatch job_jed.sh configs/old_training.toml --deterministic-eval false
```

For Slurm arrays, set `seed_from_slurm_array = true` in `[run]` and submit:

```bash
sbatch --array=0-7 job_jed.sh configs/old_training.toml
```

Environment variables can override values from `[args]` by using the uppercase
argument name, for example `N_ENVS`, `TOTAL_TIMESTEPS`, `SEED`, or
`TORCH_THREADS` for `n_threads`.
