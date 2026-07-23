# Stage 1d: Heterogeneous Asset-Edge Directions

This folder contains the eight non-baseline combinations of generator and load
attachment directions in the direct heterogeneous graph.  Each relation uses
one of:

- `bidirectional`: asset and busbar exchange messages;
- `asset_to_busbar`: the busbar aggregates asset state without sending context
  back to the asset;
- `busbar_to_asset`: the asset receives busbar context without contributing its
  state to the busbar.

The matching Stage 1b heterogeneous `bidirectional/bidirectional` run supplies
the ninth factorial combination.  All Stage 1d runs use no input preprocessing,
GINE, mean pooling, no substation edges or summary nodes, seed 0, 8M steps, and
a 2,880-minute training limit.  The transmission-line and hierarchy directions
remain bidirectional and are outside this screen.

Regenerate and validate from `Topology_Task`:

```bash
python tools/gnn_graph_screening.py heterogeneous-directions --force
python tools/gnn_graph_screening.py validate \
  configs/gnn_graph_screening/stage1d_heterogeneous_directions
```

Launch all eight jobs from the repository root:

```bash
bash Topology_Task/configs/gnn_graph_screening/stage1d_heterogeneous_directions/launch_all.sh
```

Compare all nine direction pairs at the common 8M-step endpoint.  The primary
hypothesis is that `asset_to_busbar/asset_to_busbar` is sufficient because the
busbar should aggregate generator and load state.  The complete factorial also
tests whether generators and loads benefit from different directions.
