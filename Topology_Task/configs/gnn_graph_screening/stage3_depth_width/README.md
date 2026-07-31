# Stage 3 Depth/Width Screen

This folder contains the depth/width grid derived from the Stage 2 unaugmented bus baseline `e0n0v0`.

Only the actor GNN message-passing depth and embedding width are screened:

- `gnn_layers`: 1, 2, or 3;
- `gnn_hidden_dim`: 16, 32, 64, or 128;
- `gnn_out_dim`: matched to `gnn_hidden_dim` for a clean width comparison;
- seeds: 0, 1, and 2.

Everything else is inherited from the corresponding Stage 2 baseline config:
`configs/gnn_graph_screening/stage2_structure/gs_s2_bus_n0_none_e0n0v0_s{seed}.toml`.

This produces 12 architecture settings and 36 runnable configs. The `mp2_h128`
configs reproduce the original Stage 2 baseline width/depth under the new naming.

Validate from `Topology_Task`:

```bash
python tools/gnn_graph_screening.py validate configs/gnn_graph_screening/stage3_depth_width
```

Launch from the repository root on JED:

```bash
bash Topology_Task/configs/gnn_graph_screening/stage3_depth_width/launch_all.sh
```
