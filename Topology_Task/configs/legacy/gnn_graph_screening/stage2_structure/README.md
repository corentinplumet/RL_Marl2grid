# Stage 2: Bus Structural Factorial

This folder contains a complete `2 x 2 x 2 x 3` experiment on the bus graph:

- `e`: same-substation busbar edges off/on;
- `n`: substation summary nodes off/on;
- `v`: mean readout/virtual-node readout;
- seed: 0, 1, or 2.

All 24 configurations are new runs, including the three unaugmented `e0n0v0`
baselines. They keep the bus representation, no physical scaling, no running
normalization, GINE actor, and MLP critic. Same-substation `e` edges and
busbar/substation `n` edges are bidirectional. Virtual-node `v` edges are
one-way toward the virtual node: substation-to-virtual when `n=1`, otherwise
busbar-to-virtual. Thus the virtual node aggregates graph context without
sending messages back into the physical or substation nodes.

Each run has 15,000,000 training steps and a 5,760-minute (four-day) limit in
both `[args].time_limit` and
`[environment].MAX_TIME_LIMIT_MINUTES`.

The runs are assigned by a reproducible constrained randomization
(`split_seed=20260726`) to JED and Izar. Each cluster receives:

- 12 runs;
- four runs for each training seed;
- six enabled and six disabled cases for each of `e`, `n`, and `v`;
- one or two seeds from every structural configuration.

The `cluster` column in `manifest.csv` records the assignment.

Regenerate and validate from `Topology_Task`:

```bash
python tools/gnn_graph_screening.py bus-structure --force
python tools/gnn_graph_screening.py validate \
  configs/gnn_graph_screening/stage2_structure
```

Launch each half from the repository root on its assigned cluster:

```bash
# On JED
bash Topology_Task/configs/gnn_graph_screening/stage2_structure/launch_jed.sh

# On Izar
bash Topology_Task/configs/gnn_graph_screening/stage2_structure/launch_izar.sh
```
