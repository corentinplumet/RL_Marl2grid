# Full Test Evaluation

This folder contains a standalone checkpoint evaluator for running a trained
MAPPO policy on the full test chronic split from a Slurm job.

The evaluator rebuilds the actors from the checkpoint's saved training
arguments, loads the per-agent actor weights, evaluates deterministically by
default, and prints the mean survival rate over the whole test split.

## Main Commands

Use an exact checkpoint path or checkpoint stem when you know it:

```bash
sbatch Topology_Task/full_test_eval/job_full_test_eval.sh \
  --checkpoint checkpoint/best_test_hvg_04_eval_local_rho090_s0.tar
```

On Izar, use the Izar wrapper instead:

```bash
sbatch Topology_Task/full_test_eval/job_full_test_eval_izar.sh \
  --checkpoint checkpoint/best_test_hvg_04_eval_local_rho090_s0.tar
```

Or select by model name and checkpoint global step:

```bash
sbatch Topology_Task/full_test_eval/job_full_test_eval.sh \
  --model hvg_04_eval_local_rho090_s0 \
  --step 15000000
```

List the available checkpoints before choosing one:

```bash
python Topology_Task/full_test_eval/evaluate_checkpoint.py --list-checkpoints
```

Filter the list by model/run-name substring:

```bash
python Topology_Task/full_test_eval/evaluate_checkpoint.py \
  --list-checkpoints \
  --model hvg_04_eval_local_rho090_s0
```

If `--model` and `--step` are used, the script scans `--checkpoint-dir`
(`checkpoint` by default), first looking for files whose checkpoint
`global_step` equals `--step`, then falling back to filenames containing the
step value.

If the requested step does not exist exactly, choose a policy:

```bash
sbatch Topology_Task/full_test_eval/job_full_test_eval.sh \
  --model hvg_04_eval_local_rho090_s0 \
  --step 10160000 \
  --step-policy nearest
```

`before` uses the latest checkpoint at or before the requested step, `after`
uses the earliest checkpoint at or after it, and `nearest` uses the closest
available checkpoint.

## Useful Options

- `--checkpoint PATH_OR_STEM`: exact checkpoint file, or a stem inside
  `Topology_Task/checkpoint`.
- `--list-checkpoints`: print checkpoint files and saved metadata, then exit.
- `--model NAME`: substring used to find a checkpoint by filename,
  `exp_tag`, or stored W&B run path/name.
- `--step N`: requested training step. If omitted, the newest matching model
  checkpoint is used.
- `--step-policy exact|before|after|nearest`: behavior when `--step` does not
  exist exactly. Defaults to `exact`.
- `--checkpoint-dir DIR`: directory containing `.tar` checkpoints.
- `--split test`: chronic split to evaluate. Defaults to `test`.
- `--eval-all-split-chronics true`: evaluate every chronic in the selected
  split. This is enabled by default.
- `--eval-episodes N`: override the number of evaluated episodes instead of
  using the whole split.
- `--eval-action-heuristic checkpoint|none|rho_threshold|local_rho_threshold`:
  keep the checkpoint setting by default, or override it for evaluation.
- `--eval-action-rho-threshold 0.90`: threshold for the evaluation heuristic.
- `--intervention-gate-eval-mode checkpoint|final_action_map|hierarchical_greedy`:
  keep or override the deterministic rule for gated actors.
- `--deterministic-eval true`: greedy evaluation by default.
- `--device auto|cpu|cuda|mps`: inference device.
- `--progress true`: print one line per completed chronic. Enabled by default.
- `--obs-normalization auto|disable|require`: handling for checkpoints trained
  with normalized observations. `auto` uses saved normalization stats when
  present; for older checkpoints that do not contain stats, it disables
  normalization with a warning instead of crashing. `require` aborts if stats
  are missing.
- `--output-json PATH`: optional explicit JSON summary path.

The default JSON summary is written under:

```text
Topology_Task/outputs/full_test_eval/
```

## Shared-candidate zero-shot WCCI evaluation with 64 actions

The NL/NLS shared candidate checkpoints can be evaluated with the WCCI
`mk64` action space using the dedicated Izar launcher. It preserves the
existing `mk256` protocol (50 deterministic test episodes, frozen transferred
encoder and scorer, disabled cross-grid observation normalization) and changes
only the target reduced action space.

First generate the `mk64` artifact if it is not already present on Izar:

```bash
TOP_KS=64 Topology_Task/teacher_student/reduce_wcci_action_space_sizes.sh
```

Preview all 16 NL/NLS evaluations:

```bash
DRY_RUN=true Topology_Task/full_test_eval/launch_shared_zero_shot_wcci_mk64_izar.sh
```

Submit all 16:

```bash
Topology_Task/full_test_eval/launch_shared_zero_shot_wcci_mk64_izar.sh
```

Optional arguments filter run names. For example, `NL_` selects the eight
unscaled checkpoints and `NLS_` selects the eight scaled checkpoints:

```bash
Topology_Task/full_test_eval/launch_shared_zero_shot_wcci_mk64_izar.sh NLS_
Topology_Task/full_test_eval/launch_shared_zero_shot_wcci_mk64_izar.sh NL_ _mean_f0_a0h0
```

Results are kept separate from `mk256` under directories ending in `_mk64`.
Existing results are skipped unless `FORCE_RESULTS=true` is set. Exact
per-step action traces are disabled by default, as in the original evaluation;
set `SAVE_ACTION_TRACE=true` if they are needed.

For the corresponding 128-actions-per-agent evaluation, use the dedicated
entry point (assuming the `mk128` JSON already exists):

```bash
DRY_RUN=true Topology_Task/full_test_eval/launch_shared_zero_shot_wcci_mk128_izar.sh
Topology_Task/full_test_eval/launch_shared_zero_shot_wcci_mk128_izar.sh
```

It uses the same evaluation protocol and writes to separate `_mk128`
directories. Internally, both launchers use the same validated implementation;
`ACTION_SIZE=N` can also select another available `mkN` artifact.

For the `mk512` artifact, preview and submit with:

```bash
DRY_RUN=true Topology_Task/full_test_eval/launch_shared_zero_shot_wcci_mk512_izar.sh
Topology_Task/full_test_eval/launch_shared_zero_shot_wcci_mk512_izar.sh
```

The results and action summaries are written to separate `_mk512`
directories. To avoid a problematic node, set `EXCLUDE_NODES`; the launcher
passes it as an explicit `sbatch --exclude` option:

```bash
EXCLUDE_NODES=i39 Topology_Task/full_test_eval/launch_shared_zero_shot_wcci_mk512_izar.sh
```

For the `mk1024` artifact, use the corresponding entry point:

```bash
EXCLUDE_NODES=i39 DRY_RUN=true Topology_Task/full_test_eval/launch_shared_zero_shot_wcci_mk1024_izar.sh
EXCLUDE_NODES=i39 Topology_Task/full_test_eval/launch_shared_zero_shot_wcci_mk1024_izar.sh
```

Its summaries and action artifacts are isolated in `_mk1024` directories.

Here `mkN` means a top-k cap, not necessarily exactly `N` retained actions for
every agent. If fewer candidates satisfy the reduction filters, an agent can
have a smaller set. A per-agent cap can also be below `N` when the agent's
original action space is smaller. The launcher accepts non-empty sets up to the
global and per-agent caps and prints the actual sizes.

## Output

The important terminal line looks like:

```text
test at step 15000000, survival=97.321%, return=[...]
```

With progress enabled, the sbatch output also contains lines like:

```text
test chronic 3/20: 003 survival=100.000% steps=8064/8064 running_mean=96.481%
```

The JSON summary also records the checkpoint path, checkpoint step, split,
number of evaluated episodes, heuristic settings, deterministic mode, and
survival as both a fraction and a percentage.

## Observation Normalization

Older checkpoints did not store the training observation normalization stats.
If such a checkpoint has `norm_obs = true`, standalone eval cannot reconstruct
the exact normalized inputs used during training. By default the script prints a
warning and evaluates on raw observations so the job still runs:

```bash
--obs-normalization auto
```

For strict reproducibility checks, use:

```bash
--obs-normalization require
```

Future checkpoints saved by this code include `training_state["obs_stats"]`, so
standalone full-test eval can use the same normalization stats as training-time
eval.
