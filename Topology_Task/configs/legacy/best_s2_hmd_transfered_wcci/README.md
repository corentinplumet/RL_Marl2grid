# bus14 → WCCI (no maintenance) encoder transfer

Does a GNN actor encoder trained on `bus14` still carry useful structure on a
36-substation grid? Each architecture is run twice on `bus36_wcci_nomaint`:

| arm | encoder | heads |
| --- | --- | --- |
| `frozen` | loaded from the paired bus14 run, `requires_grad=False` | trained |
| `scratch` | random init | trained |

Four architectures × two arms × three seeds = **24 runs**. The two arms of a
pair differ only in `transfer_encoder_checkpoint` and
`transfer_freeze_encoder`; every other setting is copied verbatim from the
bus14 screening config, which is what makes the encoder weights loadable at
all.

| variant | bus14 source config |
| --- | --- |
| `hmd_gb2a_la2b` | `gnn_graph_screening/heterogenous_message_direction/gs_hmd_hetero_n0_none_gb2a_la2b_s*` |
| `hmd_gb2a_lb2a` | `gnn_graph_screening/heterogenous_message_direction/gs_hmd_hetero_n0_none_gb2a_lb2a_s*` |
| `s2_e0n0v0` | `gnn_graph_screening/stage2_structure/gs_s2_bus_n0_none_e0n0v0_s*` |
| `s2_e1n0v0` | `gnn_graph_screening/stage2_structure/gs_s2_bus_n0_none_e1n0v0_s*` |

Relative to the bus14 bases, only `env_id`, `name`/`exp_tag` and the wall-clock
budget change (`time_limit = 5040`, `MAX_TIME_LIMIT_MINUTES = 10080`, matching
the other WCCI configs). `total_timesteps` stays at 15M.

## Prerequisite

Each `frozen` config points at `checkpoint/best_test_<bus14 exp_tag>.tar`. Those
files must be present under `Topology_Task/checkpoint/` before launching, or the
run fails immediately with a `FileNotFoundError` naming the missing path. Check
all twelve first:

```bash
cd Topology_Task && grep -h transfer_encoder_checkpoint configs/transfer_bus14_to_wcci_nomaint/*_frozen_*.toml | sed 's/.*= "//;s/"//' | xargs ls -l
```

## Pre-flight

The encoder only loads if the source and target graph specs agree on
`node_dim` and `edge_dim`. Compare them once per graph type before spending
cluster time — a mismatch raises `ValueError: Encoder architecture mismatch`:

```bash
cd Topology_Task && python tools/inspect_graph_specs.py --env-id bus14 --actor-encoder gnn --gnn-graph-type bus --gnn-include-neighbors true
```

```bash
cd Topology_Task && python tools/inspect_graph_specs.py --env-id bus36_wcci_nomaint --actor-encoder gnn --gnn-graph-type bus --gnn-include-neighbors true
```

Repeat with `--gnn-graph-type heterogeneous` for the `hmd_*` variants.

## Launch

From the repository root:

```bash
Topology_Task/configs/transfer_bus14_to_wcci_nomaint/launch_izar.sh _s0
```

The optional filter is a substring match on the config name, so `_s0` launches
the eight seed-0 runs, `frozen` launches all twelve frozen runs, and no
argument launches all 24.

## What actually transfers

Only `encoder.graph_encoder.*` is reused. The action heads cannot transfer —
bus14 splits into 3 agents and WCCI into 4, with different action-space sizes —
and the centralized critic is an MLP over a grid-sized flat observation, so both
always start fresh. The encoder's `edge_index` / `node_ids` buffers are skipped
during the load: they describe the source grid, and `GraphAndFlatEncoder`
overrides them per agent from the target env's own spec at call time.

## Action space

Unlike the bus14 bases, these configs set
`reduced_action_space = ".../reduced_action_space_wcci_full2048a_90_v3_mk256.json"`,
matching the other WCCI configs. The full WCCI action space is ~67k actions
(agent_1 alone is ~65.6k, see `teacher_student/ACTION_SPACE_REDUCTION_README.md`),
which is not trainable within this budget. This does not affect the transfer:
the action head never crosses grids, and with `actor_action_head = "mlp"` the
encoder does not depend on `n_actions`.

Override per run without editing the configs — a different reduction:

```bash
REDUCED_ACTION_SPACE=outputs/.../reduced_action_space_wcci_full2048a_90_v3_mk64.json sbatch job_izar.sh <config>
```

or the empty string for the unreduced space, since `run_from_config.py` drops
empty values and `main.py` then falls back to its default.

## Caveat: input scale

These configs inherit `gnn_physical_scaling = false` and
`gnn_running_norm = false` from the bus14 screening protocol, so the encoder
sees raw physical features. bus14 and WCCI do not share a power or thermal-limit
scale, which means the frozen arm is fed a shifted input distribution and may
lose for reasons that have nothing to do with representation quality. Turning
scaling on only for the target would break weight compatibility with the source
checkpoints; testing that properly needs bus14 source runs with
`gnn_physical_scaling = true`.
