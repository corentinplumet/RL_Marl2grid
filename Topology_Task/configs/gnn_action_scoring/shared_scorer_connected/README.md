# Shared candidate-scorer bus14 sweep

This sweep contains the eight combinations of:

- `mean` or `typed_mean` candidate pooling;
- action features disabled or enabled;
- the separate do-nothing head disabled or enabled.

Each combination is run once with the original no-leak inputs (`NL`) and once
with physical scaling and edge-difference angles (`NLS`). Both groups retain
the explicit `connected` node feature, giving the explicit-line graph 13 node
features, and set `share_candidate_scorer = true`.

From `Topology_Task`, validate the 16 submissions with:

```bash
DRY_RUN=true bash configs/gnn_action_scoring/shared_scorer_connected/launch_izar.sh
```

Remove `DRY_RUN=true` to submit them. Optional filename filters can be passed
as arguments, for example `NL_shared` or `tmean`.
