# Shared candidate-scorer bus14 sweep

This sweep contains the eight combinations of:

- `mean` or `typed_mean` candidate pooling;
- action features disabled or enabled;
- the separate do-nothing head disabled or enabled.

Each combination is run once with the original no-leak inputs (`NL`) and once
with physical scaling and edge-difference angles (`NLS`). Both groups retain
the explicit `connected` node feature, giving the explicit-line graph 13 node
features, and set `share_candidate_scorer = true`.

The folder also contains eight identical NLS configurations whose filenames,
run names, and experiment tags include `NLS_izar`. These provide a separately
named Izar batch without changing any model or training parameter.

From `Topology_Task`, validate only the eight separately named Izar NLS runs
with:

```bash
DRY_RUN=true bash configs/gnn_action_scoring/shared_scorer_connected/launch_izar.sh NLS_izar
```

Remove `DRY_RUN=true` to submit them. The launcher accepts other filename
filters such as `NL_shared` or `tmean`; without a filter, it submits every TOML
configuration in this folder.
