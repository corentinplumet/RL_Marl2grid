# `nl_s2em`: corrected Screen C with the selected readout

This folder reproduces the eight seed-0 cells of Screen C on the corrected
information boundary. The fixed encoder is GINE with two message-passing
layers and hidden/output width 128. The baseline readout is `energized_max`.

The factorial remains `(e, n, v) in {0, 1}^3`:

- `e`: same-substation edges;
- `n`: substation-summary nodes;
- `v`: virtual-node readout.

For `v=0`, `gnn_readout_aggr = "energized_max"`. For `v=1`, the readout is
intentionally `virtual_node`, because changing the terminal readout is the
screened `v` factor. Contextual visibility still requires an active connection,
and structural relations are restricted to controlled substations in every
cell.

Validate or submit the eight jobs from the repository root:

```bash
DRY_RUN=true bash Topology_Task/configs/nl_s2em_structure/launch_izar.sh
bash Topology_Task/configs/nl_s2em_structure/launch_izar.sh
```

Use `launch_jed.sh` for JED. Both launchers accept substring filters.
