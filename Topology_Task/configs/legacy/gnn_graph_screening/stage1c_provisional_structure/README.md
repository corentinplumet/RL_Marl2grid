# Stage 1c: Provisional Structural Screen

This folder uses seven spare cluster slots to test the non-baseline structural
combinations for the provisional `bus + no input preprocessing + GINE` winner.
The matching Stage 1b `e0n0v0` run supplies the eighth, unaugmented baseline.

The filename encodes three binary factors:

- `e`: direct edges between busbars belonging to the same substation;
- `n`: learned substation summary nodes connected to their busbars;
- `v`: virtual-node graph readout instead of mean pooling.

All seven runs keep seed 0, the MLP critic, bidirectional relations, 8M steps,
72 environments, 576 rollout steps, evaluation every 82,944 steps, and a
2,880-minute training limit.  GINE remains fixed so the experiment changes the
graph structure and pooling without introducing an encoder confound.

Regenerate and validate the configs from `Topology_Task`:

```bash
python tools/gnn_graph_screening.py provisional-structure --force
python tools/gnn_graph_screening.py validate \
  configs/gnn_graph_screening/stage1c_provisional_structure
```

Launch all seven jobs from the repository root:

```bash
bash Topology_Task/configs/gnn_graph_screening/stage1c_provisional_structure/launch_all.sh
```

Treat the results as provisional until Stage 1b identifies the winning base
representation and normalization.  A different Stage 1b winner requires a
focused structural confirmation under that winning input configuration.
