# No-Leak Candidate-Action Scoring with Corrected Encoder Inputs

This folder pairs [`no_leak`](../no_leak) one-for-one. Same 33 configurations,
same leakage-free information boundary, with the two input corrections of
[`transfer_scaled`](../../transfer_scaled) applied on top. The `cas_hl_NLS`
prefix keeps W&B metadata, checkpoints, and output directories separate from
the `cas_hl_NL` control arm.

## What changes

Relative to its `no_leak` counterpart, each config changes only:

```toml
name    = "cas_hl_NLS_<variant>"
exp_tag = "cas_hl_NLS_<variant>"
gnn_angle_representation = "edge_diff"   # angle drop per line, not absolute per-asset angle
gnn_physical_scaling     = true          # power divided by the grid's own gen_pmax
```

plus the Izar wall clock (below). Optimizer, GINE encoder, critic, reward,
rollout, evaluation, and the 15M-step budget are untouched, so `NLS` against
`NL` is a clean two-factor comparison on otherwise identical settings.

## Why this needed a code change

`gnn_angle_representation = "edge_diff"` used to raise on
`heterogeneous_line`. In the other two schemas the angle drop moves onto the
physical line edge, but here every line is already a node and the attachment
edges carry no physical channel at all, so there was no edge to move it to.

The drop now replaces the absolute angle **on the line node**, in place: same
column, same node width, renamed to `theta_diff` so the normalization layer
scales it as an angle. Measured on `bus14` under this exact configuration:

| agent | angle-carrying rows, before | after | of which line nodes |
|-------|------------------------------|-------|---------------------|
| 0     | 5 / 22 (23%)                 | 8 / 22 (36%) | 8 |
| 1     | 3 / 20 (15%)                 | 8 / 20 (40%) | 8 |
| 2     | 8 / 29 (28%)                 | 9 / 29 (31%) | 9 |

Before, angles reached the encoder only through the agent's own generators and
loads, and agent 1 saw three values in total. After, every line the agent
touches carries its own drop, and the channel is gauge-invariant: shifting
every angle in the grid by a constant leaves it unchanged.

Because these runs set `gnn_running_norm = false`, physical scaling is the only
input conditioning they get; without it the encoder reads raw megawatts and raw
degrees.

## Wall clock

`job_izar.sh` requests `3-00:12:00` (4332 minutes), so these configs set
`time_limit = 4260` and leave roughly an hour to stop and write a final
checkpoint rather than being killed mid-write. The `no_leak` control uses
`10080` because `launch_all.sh` asks JED for seven days.

A 15M-step run will not finish inside one Izar allocation. Resume it with
`scripts/prepare_resume_runs.py`, or relaunch from the last checkpoint.

`n_envs` stays at 72 to match the control, while `job_izar.sh` allocates 40
CPUs. The runs oversubscribe and step more slowly; nothing about the
optimization changes.

## Validate before submitting

From the repository root:

```bash
DRY_RUN=true bash Topology_Task/configs/gnn_action_scoring/no_leak_scaled/launch_izar.sh
```

The dry run must report 33 submissions.

## Launch

```bash
bash Topology_Task/configs/gnn_action_scoring/no_leak_scaled/launch_izar.sh
```

The launcher takes substring filters, and every filter must match. Seed 0
alone is 11 jobs, and the three adaptive typed-attention runs at seed 0 are
three:

```bash
bash Topology_Task/configs/gnn_action_scoring/no_leak_scaled/launch_izar.sh _s0
```

```bash
bash Topology_Task/configs/gnn_action_scoring/no_leak_scaled/launch_izar.sh tattn _s0
```

Worth doing first. 33 runs is a large commitment before seeing a curve.

Override the wall clock only deliberately:

```bash
IZAR_WALLTIME=1-00:00:00 \
  bash Topology_Task/configs/gnn_action_scoring/no_leak_scaled/launch_izar.sh _s0
```

## Requirements

Run from the `pooling` branch, or any descendant containing the line-node
`edge_diff` implementation. On an older checkout every one of these configs
fails at startup with

```
angle_representation='edge_diff' is not implemented for the heterogeneous line-node graph.
```

which is the intended behaviour there: the flag used to be refused rather than
silently ignored.

## Note on checkpoints

Node width is unchanged, so a `cas_hl_NL` checkpoint still *loads* under these
configs — but the angle column means something different than it did during
training. These are from-scratch runs, so it does not arise; it matters only if
an `NL` run is resumed with an `NLS` config.
