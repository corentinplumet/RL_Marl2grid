# Graph-Design Screens Under the Corrected Construction

Seed-0 rerun of Screens A, B, C and D on the corrected local-graph
construction, with Screen B extended to include the controlled readouts.
**44 runs.** The `nl_` prefix keeps W&B metadata, checkpoints and
output directories separate from the original `gs_` families.

| folder | screen | source family | runs |
|---|---|---|---|
| `screen_a_depth_width` | A — encoder depth and width | `stage3_depth_width` | 12 |
| `screen_b_encoder_pooling` | B — operator and readout | `stage3_encoder_pooling` | 15 |
| `screen_c_structure` | C — structural augmentations | `stage2_structure` | 8 |
| `screen_d_message_direction` | D — asset message direction | `heterogenous_message_direction` | 9 |

Screen B is the one screen whose design changes. It becomes a full
full $2\times4$ factorial over the message-passing operator and the readout:

|            | `mean` | `max` | `controlled_mean` | `controlled_max` |
|------------|--------|-------|-------------------|------------------|
| **GINE**   | reused from Screen A | new | new | new |
| **GAT**    | new    | new   | new               | new              |

Seven configs plus one reused cell. The GINE + `mean` cell is Screen A's
`mp2_h128` config, exactly as in the original design, so compare against
`nl_s3dw_bus_n0_none_e0n0v0_mp2_h128_s0` rather than running a duplicate.

The readout dimension is screened rather than assumed because every config here sets
`gnn_include_neighbors = true`, which makes an ordinary mean pool a set whose
size follows the topology: on agent 0 of `bus14` it holds 10 nodes nominally,
11 once a contextual substation splits, 9 after a boundary line trips. A
`controlled_*` readout pools the action domain instead, whose size is fixed for
the episode, while contextual nodes continue to reach the readout through
message passing, and an energized readout drops nodes that carry no current.
The three restrictions are plausible for different reasons and there is no
measurement that separates them, so all of them are run rather than argued
about. The energized levels are the ones most likely to lose: on agent 0 of
`bus14` four of eight busbars are de-energized under the nominal topology, and
those are precisely the empty busbars a splitting action would move an element
onto. That is a prediction the screen can refute.

It belongs in Screen B and not elsewhere: Screen C's `v = 1` cells replace
terminal pooling with a virtual node, so a `controlled_*` readout has no
meaning for half of that screen.

## What differs from the original screens

Only the name, the output directory, the wall clock, and two flags stated
explicitly:

```toml
gnn_context_requires_connection      = true
gnn_structural_relations_controlled_only = true
```

Every screened factor, the optimizer, the reward, the rollout, the evaluation
protocol, the chronic split and the 15M-step budget are byte-identical to the
originals. Both flags default to `true` in the current code; they are written
out so the configs record the construction rather than depending on a default.

## What the corrected construction changes

Three defects that all affected these screens, none of which was a screened
factor:

1. **Contextual visibility.** A contextual node is now valid only while an
   active, non-structural relation connects it to the controlled region.
   Previously every busbar of a neighboring substation was permanently valid,
   so unreachable rows entered the graph readout. On agent 0 of `bus14` the
   pooled set drops from 12 to 10 (`bus`) and from 22 to 20 (`heterogeneous`).
2. **Structural relations are controlled-only.** Same-substation edges and
   substation summary nodes are built only inside the agent's own substations.
   A same-substation relation asserts that an element can be moved between two
   busbars, which outside the controlled domain is another agent's action.
   **This hits Screen C hardest:** under the old construction its `e=1` cells
   joined a neighbor's two busbars, which upgraded the visibility defect from a
   readout perturbation into a message-passing channel.
3. **Summary nodes are initialised from structure, not identity.** The learned
   per-substation matrix was shaped `[n_sub, d_h]` and made any `n=1` checkpoint
   impossible to transfer — the loader raises on the shape mismatch rather than
   rebuilding it. Summary nodes now start from a projection of the substation's
   busbar, line, generator and load counts, whose learned tensors are the same
   shape on any grid.

Because of (1) and (2), results here are **not** comparable cell-by-cell with
the original screens; they are a new baseline.

## Wall clock

`job_izar.sh` requests `3-00:12:00` (4332 minutes), so every config sets
`time_limit = 4260`, leaving about an hour to stop and write a final
checkpoint. The originals used 5760 (`bus`) and 2880 (`heterogeneous`); 5760
overruns both JED and Izar.

A 15M-step run does not finish in one Izar allocation. Resume with
`scripts/prepare_resume_runs.py`, or relaunch from the last checkpoint.

The JED launchers use the fuller allocation by default, setting the internal
guard to 4980 minutes. Override `TIME_LIMIT` before the launcher if needed.
For a single manual submission, use:

```bash
TIME_LIMIT=4980 sbatch --time=3-12:00:00 job_jed.sh <config>
```

## Validate before submitting

From the repository root:

```bash
DRY_RUN=true bash Topology_Task/configs/no_leakage_config/launch_all_izar.sh
DRY_RUN=true bash Topology_Task/configs/no_leakage_config/launch_all_jed.sh
```

Must report 12, 15, 8 and 9.

## Launch

All four screens, 44 jobs:

```bash
bash Topology_Task/configs/no_leakage_config/launch_all_izar.sh
bash Topology_Task/configs/no_leakage_config/launch_all_jed.sh
```

One screen:

```bash
bash Topology_Task/configs/no_leakage_config/screen_c_structure/launch_izar.sh
bash Topology_Task/configs/no_leakage_config/screen_c_structure/launch_jed.sh
```

The launchers take substring filters, and every filter must match:

```bash
bash Topology_Task/configs/no_leakage_config/screen_a_depth_width/launch_izar.sh mp2
bash Topology_Task/configs/no_leakage_config/screen_a_depth_width/launch_jed.sh mp2
```

## Requirements

Run from a checkout containing the corrected construction: the visibility rule,
`--gnn-structural-relations-controlled-only`, and the structural summary-node
descriptor. On an older checkout the two explicit flags are unknown arguments
and the runs fail at startup, which is the intended behaviour.

## Deliberately unchanged

Two options discussed alongside these fixes are **not** applied here, so that
each screen isolates the construction change:

- `gnn_readout_aggr` keeps its original value in Screens A, C and D, so those
  three isolate the construction change exactly. Only Screen B varies it, where
  it is the screened factor.
- `gnn_angle_representation` stays at the absolute-angle default. The angle
  drop is an input change, not a construction fix.
