# Stage 3 Encoder And Pooling Screen

This folder contains nine runs derived from the Stage 2 unaugmented bus
baseline `e0n0v0`:

- GINE with max pooling, seeds 0, 1, and 2;
- GAT with max pooling, seeds 0, 1, and 2;
- GAT with mean pooling, seeds 0, 1, and 2.

Every other setting is inherited from the matching Stage 2 baseline config:

```text
configs/gnn_graph_screening/stage2_structure/
  gs_s2_bus_n0_none_e0n0v0_s{seed}.toml
```

In particular, all runs use two message-passing layers, hidden and output
dimensions of 128, four GAT heads, 15 million timesteps, and the same chronic
split. GINE with mean pooling is not repeated because it is the existing Stage
2 baseline.

Validate from `Topology_Task`:

```bash
python tools/gnn_graph_screening.py validate \
  configs/gnn_graph_screening/stage3_encoder_pooling
```

Launch all nine runs on JED from `Topology_Task`:

```bash
bash configs/gnn_graph_screening/stage3_encoder_pooling/launch_all.sh
```
