# Stage 1e: Heterogeneous Structure and Explicit-Line Scout

This folder contains eight jobs:

- seven non-baseline combinations of same-substation edges (`e`), substation
  summary nodes (`n`), and virtual-node readout (`v`) for the direct
  heterogeneous representation;
- one unaugmented `heterogeneous_line` run at the full 8M-step budget.

The Stage 1b direct heterogeneous `e0n0v0` run supplies the eighth member of its
structural factorial.  The explicit-line scout replaces no result: it gives the
slow representation a fair full-budget run after its original screen stopped
near 3.3M steps.

Every job uses no physical or running input normalization, GINE, bidirectional
generator/load/line relations, seed 0, the centralized MLP critic, 72
environments, 576 rollout steps, evaluation every 82,944 steps, and a
2,880-minute training limit.

Regenerate and validate from `Topology_Task`:

```bash
python tools/gnn_graph_screening.py heterogeneous-structure --force
python tools/gnn_graph_screening.py validate \
  configs/gnn_graph_screening/stage1e_heterogeneous_structure
```

Launch all eight jobs from the repository root:

```bash
bash Topology_Task/configs/gnn_graph_screening/stage1e_heterogeneous_structure/launch_all.sh
```

Interpret the structural results together with the Stage 1b heterogeneous
baseline.  Promote the explicit-line graph only if its completed trajectory is
competitive enough to justify slower training and another multi-seed test.
