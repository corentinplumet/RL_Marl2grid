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
- `--output-json PATH`: optional explicit JSON summary path.

The default JSON summary is written under:

```text
Topology_Task/outputs/full_test_eval/
```

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
