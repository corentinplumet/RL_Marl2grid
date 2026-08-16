# `nl_hmdem`: corrected Screen D with the selected readout

This folder reproduces the nine seed-0 cells of Screen D on the corrected
information boundary. Every cell uses GINE with two message-passing layers,
hidden/output width 128, and `energized_max` readout.

The full `3 x 3` factorial crosses generator and load relation directions:
`bidirectional`, `asset_to_busbar`, and `busbar_to_asset`. All other graph,
optimization, evaluation, and chronic-split settings are held fixed. Contextual
visibility requires an active connection and structural relations are limited
to controlled substations.

Validate or submit the nine jobs from the repository root:

```bash
DRY_RUN=true bash Topology_Task/configs/nl_hmdem_message_direction/launch_izar.sh
bash Topology_Task/configs/nl_hmdem_message_direction/launch_izar.sh
```

Use `launch_jed.sh` for JED. Both launchers accept substring filters.
