# Max-readout candidate-architecture ablation

This screen carries the supervised capacity result back to bus14 MAPPO. It has
two paired seed-0 folders:

- `NL`: original no-leak input preprocessing;
- `NLS`: edge-difference angles and physical power scaling.

Every configuration keeps the transferable shared candidate architecture,
action features enabled, the separate action-0 head disabled, two GINE
message-passing layers, and max aggregation at both levels:

```toml
candidate_action_pool = "max"          # nodes touched by each action
gnn_readout_aggr = "energized_max"     # energized nodes in the local graph
```

Only these remaining architectural factors vary:

| Variant | Action-delta encoder | Residual GNN | Jumping Knowledge |
|---|---:|---:|---|
| `base` | no | no | none |
| `delta` | yes | no | none |
| `resjk` | no | yes | concat |
| `delta_resjk` | yes | yes | concat |

The two families otherwise retain their source 15M-step MAPPO recipe from
`Z_nl_cas_hl_shared`. This gives eight runs in total and keeps NL versus NLS a
paired preprocessing comparison.

Validate without submitting from the repository root:

```bash
DRY_RUN=true bash Topology_Task/configs/no_leakage_config/F_nl_cas_hl_max_arch/NL/launch_izar.sh
DRY_RUN=true bash Topology_Task/configs/no_leakage_config/F_nl_cas_hl_max_arch/NLS/launch_izar.sh
```

Remove `DRY_RUN=true` to submit. Each launcher accepts filename substring
filters such as `delta` or `resjk`.

Regenerate and validate all TOMLs with:

```bash
python Topology_Task/tools/generate_max_action_delta_arch_configs.py
```
