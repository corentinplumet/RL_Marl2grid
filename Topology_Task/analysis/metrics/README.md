# Metrics Analysis

This folder contains notebooks and helper modules for plotting and inspecting
training, evaluation, action-distribution, and teacher-student metrics.

It used to live under `Topology_Task/configs/zz_print_metrics`, but it is not a
configuration folder. Keeping it under `analysis/metrics` makes the repo layout
clearer:

- `helpers/`: reusable Python modules used by the notebooks.
- `notebooks/wandb/`: W&B history plots and run comparisons.
- `notebooks/action_distribution/`: action-0, non-idle, override, and action
  distribution analysis.
- `notebooks/teacher_student/`: teacher-student dataset, BC checkpoint, and
  full-test comparison analysis.
- `notebooks/custom/`: configurable dashboard where you can type config names,
  run names, or teacher-student checkpoint names and compare the matching logs.

The notebooks locate `helpers/` automatically whether they are run from this
folder, the repository root, or inside `Topology_Task`.

## Run Data Source

The plotting helpers first look for permanent W&B downloads under:

```text
Topology_Task/outputs/run_data/
```

This is the preferred source. If no permanent run data is found, the helpers
fall back to the older cache under:

```text
Topology_Task/outputs/wandb_cache/
```
