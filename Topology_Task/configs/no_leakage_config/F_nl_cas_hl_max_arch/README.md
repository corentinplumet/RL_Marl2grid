# Global-max/action-delta shared-candidate screen

This folder contains two paired eight-run seed-0 screens:

- `NL`: original no-leak input preprocessing;
- `NLS`: edge-difference angles and physical power scaling.

All 16 configurations use the strongest architecture from the supervised
capacity study as their fixed backbone:

```toml
gnn_layers = 2
gnn_readout_aggr = "energized_max"       # global local-graph readout
candidate_action_delta_encoder = true
candidate_action_delta_dim = 0
gnn_residual = false
gnn_jumping_knowledge = "none"
share_actor_gnn = true
share_candidate_scorer = true
```

Here, `max` applies to the global graph readout. The candidate-action pooling
remains an experimental factor. Each NL/NLS folder contains the complete
2 x 2 x 2 shared-scorer factorial inherited from `Z_nl_cas_hl_shared`:

| Factor | Values | Filename token |
|---|---|---|
| Candidate-action pool | `mean`, `typed_mean` | `mean`, `tmean` |
| Static action features | disabled, enabled | `f0`, `f1` |
| Separate do-nothing head | disabled, enabled | `a0h0`, `a0h1` |

The two families otherwise retain the same 15M-step bus14 MAPPO recipe as the
source screen, making NL versus NLS a paired preprocessing comparison.

Validate without submitting from the repository root:

```bash
DRY_RUN=true bash Topology_Task/configs/no_leakage_config/F_nl_cas_hl_max_arch/NL/launch_izar.sh
DRY_RUN=true bash Topology_Task/configs/no_leakage_config/F_nl_cas_hl_max_arch/NLS/launch_izar.sh

DRY_RUN=true bash Topology_Task/configs/no_leakage_config/F_nl_cas_hl_max_arch/NL/launch_jed.sh
DRY_RUN=true bash Topology_Task/configs/no_leakage_config/F_nl_cas_hl_max_arch/NLS/launch_jed.sh
```

Remove `DRY_RUN=true` to submit all eight runs in a family. Each launcher also
accepts filename substring filters such as `tmean`, `f1`, or `a0h0`.

Regenerate and validate all 16 TOMLs with:

```bash
python Topology_Task/tools/generate_max_action_delta_arch_configs.py
```
