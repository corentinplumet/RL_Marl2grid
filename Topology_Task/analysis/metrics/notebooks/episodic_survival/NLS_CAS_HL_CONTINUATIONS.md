# NLS candidate-action continuation workflow

This workflow keeps each resumed segment as a separate W&B run, while creating
one canonical local history per experiment for analysis.

## 1. Publish continuation archives on JED

From `/home/plumet/RL_Marl2grid/Topology_Task`, first inspect the plan:

```bash
python scripts/publish_continuation_runs.py \
  --archive-root /home/plumet/RL_Marl2grid/outputs \
  --group NLS_cas_hl
```

The script only selects core `cas_hl_NLS` archives whose first history step is
above one million, so ordinary runs starting at step zero are not republished.
It also prints every source archive, step range, and new W&B ID. If the list is
correct, execute it:

```bash
python scripts/publish_continuation_runs.py \
  --archive-root /home/plumet/RL_Marl2grid/outputs \
  --group NLS_cas_hl \
  --execute
```

For extra protection, restrict the execution to the job IDs shown by the dry
run by repeating `--job-id`, for example:

```bash
python scripts/publish_continuation_runs.py \
  --archive-root /home/plumet/RL_Marl2grid/outputs \
  --group NLS_cas_hl \
  --job-id 65995442 \
  --job-id 65995443 \
  --execute
```

Each new run receives:

- a unique ID ending in `-cont-<job-id>`;
- a display name ending in `[continuation <job-id>]`;
- group `NLS_cas_hl`;
- `continuation_of_run_id` and `continuation_of_run_name` config fields;
- its actual first and last logged environment steps.

The eight original runs are also assigned to `NLS_cas_hl`. A successfully
published continuation is marked `continuation_sync_completed`, making normal
reruns of the command idempotent. Use `--force` only to intentionally resend an
archive.

## 2. Download the complete W&B group locally

From the local repository root:

```bash
python Topology_Task/analysis/download/download_wandb_group.py \
  --group NLS_cas_hl \
  --download-name NLS_cas_hl \
  --states all \
  --prefer-scan-history \
  --force
```

`--prefer-scan-history` is intentional immediately after an offline sync: it
reads the current server history instead of relying on a possibly stale history
artifact.

## 3. Reconstruct canonical histories

```bash
python Topology_Task/scripts/merge_downloaded_continuations.py \
  --group-dir Topology_Task/outputs/run_data/NLS_cas_hl
```

The output is written to:

```text
Topology_Task/outputs/run_data/NLS_cas_hl/merged_runs/
```

The merge uses `_step`, which is the explicitly logged environment step. At the
first step of each continuation, the old segment is cut and the continuation
wins any overlap. This is safe even if a continuation was previously appended
to its original W&B run. Previous canonical output is moved to a timestamped
backup before being regenerated.

## 4. Run the analysis notebook

Open:

```text
Topology_Task/analysis/metrics/notebooks/episodic_survival/nls_cas_hl_complete_summary.ipynb
```

The notebook plots all eight reconstructed periodic test curves, marks the
continuation boundaries, and reports full-test results as they become available.
