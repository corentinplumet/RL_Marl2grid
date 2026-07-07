# Permanent W&B Run Downloads

This folder is for permanent local W&B data downloads. It is separate from the
older notebook cache under:

```text
Topology_Task/outputs/wandb_cache/
```

The goal is simple: choose a W&B group or regex once, download all matching run
histories into a stable `run_data/` folder, and reuse those files from notebooks
without depending on cache state.

## Quick Start

Edit:

```text
Topology_Task/analysis/download/download_config.env
```

Set the W&B group name:

```bash
GROUP_NAME=my_wandb_group
```

Then run from the repository root:

```bash
python Topology_Task/analysis/download/download_wandb_group.py
```

The data is written to:

```text
Topology_Task/outputs/run_data/<group_name>/
```

`outputs/run_data/` is meant to stay local because it can become very large.

## Selecting Runs

The preferred selector is the exact W&B group:

```bash
python Topology_Task/analysis/download/download_wandb_group.py \
  --group wcci_reduced_mlp
```

You can also use run-name regex:

```bash
python Topology_Task/analysis/download/download_wandb_group.py \
  --run-name-regex '^wcci_reduced_mlp_baseline_72x576_s[0-2]$' \
  --download-name wcci_reduced_mlp_baseline_72x576
```

Or `exp_tag` regex:

```bash
python Topology_Task/analysis/download/download_wandb_group.py \
  --exp-tag-regex '^wcci_aib_00_flat_local_t020_72x576_s[0-2]$' \
  --download-name wcci_aib_00_flat_local_t020
```

You can combine selectors. For example, use a group and a run-name regex to
download only a subset of that group.

## Output Layout

For a download called `my_group`, the layout is:

```text
Topology_Task/outputs/run_data/my_group/
  download_request.json
  manifest.csv
  manifest.json
  skipped.csv
  runs/
    <run_name>__<run_id>/
      metadata.json
      config.json
      summary.json
      history.parquet
      history.csv.gz
      files_manifest.csv
```

Per-run files:

- `history.parquet`: full scalar history with all W&B history keys.
- `history.csv.gz`: compressed CSV copy, useful if parquet support is missing.
- `config.json`: W&B config.
- `summary.json`: W&B summary.
- `metadata.json`: download metadata, paths, row count, column count, source.
- `files_manifest.csv`: list of W&B run files.

The downloader first tries the W&B full-history artifact:

```text
run-<run_id>-history:latest
```

If the artifact is unavailable, it falls back to:

```text
run.scan_history()
```

## Reusing Existing Downloads

By default, existing run folders are reused. This is intentional: `run_data/`
is permanent, not a temporary cache.

To refresh everything:

```bash
FORCE=true python Topology_Task/analysis/download/download_wandb_group.py
```

Or:

```bash
python Topology_Task/analysis/download/download_wandb_group.py \
  --group my_wandb_group \
  --force
```

## Optional: Download W&B Run Files

Scalar history, config, summary, and file manifests are always saved. Actual
W&B run files are not downloaded by default because they can be large.

To download them too:

```bash
DOWNLOAD_FILES=true python Topology_Task/analysis/download/download_wandb_group.py
```

They will be saved under:

```text
runs/<run_name>__<run_id>/files/
```

This is useful if you need W&B tables or media files such as rollout action
trace tables.

## Loading In A Notebook

Example:

```python
from pathlib import Path
import pandas as pd

root = Path("Topology_Task/outputs/run_data/my_group")
manifest = pd.read_csv(root / "manifest.csv")

histories = []
for path in manifest["history_parquet"].dropna():
    histories.append(pd.read_parquet(path))

history = pd.concat(histories, ignore_index=True, sort=False)
history["step_millions"] = pd.to_numeric(history["_step"], errors="coerce") / 1_000_000
history.head()
```

If parquet loading is unavailable:

```python
histories = [
    pd.read_csv(path)
    for path in manifest["history_csv"].dropna()
]
history = pd.concat(histories, ignore_index=True, sort=False)
```

## Useful Commands

Smoke test with only two runs:

```bash
python Topology_Task/analysis/download/download_wandb_group.py \
  --group my_wandb_group \
  --max-runs 2 \
  --download-name smoke_my_group
```

Download all states:

```bash
python Topology_Task/analysis/download/download_wandb_group.py \
  --group my_wandb_group \
  --states all
```

Download only finished runs:

```bash
python Topology_Task/analysis/download/download_wandb_group.py \
  --group my_wandb_group \
  --states finished
```

Force the slower API fallback:

```bash
python Topology_Task/analysis/download/download_wandb_group.py \
  --group my_wandb_group \
  --prefer-scan-history
```
