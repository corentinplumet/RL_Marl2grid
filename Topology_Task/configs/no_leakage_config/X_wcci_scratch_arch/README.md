# WCCI from-scratch architecture screen

Five bus14-screened encoders trained from scratch on `bus36_wcci_nomaint`, at
two reduced action spaces. 10 runs, seed 0.

The premise: an architecture that learns bus14 well is a better bet for learning
WCCI than one that does not. This is a *retraining* question, not a transfer
question, so the r = -0.01 correlation between bus14 survival and WCCI zero-shot
transfer does not apply here.

## Cells

| config | schema | bus14 full-test | source |
|---|---|---|---|
| `wsc_bus_e0n0v0_*` | busbar, no augmentation | 95.45 % | Screen C baseline |
| `wsc_bus_e1n0v0_*` | busbar + same-substation edges | 97.19 % | Screen C best cell |
| `wsc_bus_e0n1v0_*` | busbar + substation summary nodes | 96.09 % | Screen C |
| `wsc_het_ga2b_lb2a_*` | disaggregated, gen `a2b` / load `b2a` | 99.12 % | Screen D best cell |
| `wsc_het_gbi_lbi_*` | disaggregated, both bidirectional | not downloaded | Screen D baseline |

Each config's encoder and training block are **byte-identical** to the
corresponding bus14 screen config. Only three things change: `env_id`, the
`reduced_action_space` line, and the run name. All cells keep
`gnn_readout_aggr = "energized_max"` from Screen B, as the corrected C and D
reruns did.

## Action head

These encoders use the `bus` and `heterogeneous` schemas, and
`common/action_metadata.py` requires `gnn_graph_type = "heterogeneous_line"` for
candidate-action pooling. So every cell uses `actor_action_head = "mlp"` over the
reduced space, exactly as Screens A-D did on bus14.

**Consequence:** these runs are *not* comparable to the `cas_hl` / `sc64c`
candidate-pool line, and the resulting weights are not transfer-compatible. This
screen answers "which encoder learns WCCI best", not "which encoder transfers".

## What mk64 and mk256 actually mean

`mk` is a per-agent cap, not a count. Agents whose full action set is smaller
than the cap expose fewer actions:

| agent | substations | full | mk64 | mk256 |
|---|---|---|---|---|
| agent_0 | 5 | 77 | 64 | 77 |
| agent_1 | 5 | 65642 | 64 | 256 |
| agent_2 | 9 | 127 | 64 | 127 |
| agent_3 | 17 | 1119 | 64 | 256 |
| **total** | | | **256** | **716** |

So mk256 is not "4x mk64": two agents barely move (77 and 127), and the increase
is concentrated in agents 1 and 3. Note also that agent_3 covers 17 substations
against bus14's largest agent at 6, so the shared encoder must serve subgraphs
from 5 to 17 substations.

## Launch

```bash
DRY_RUN=true bash configs/no_leakage_config/X_wcci_scratch_arch/launch_izar.sh
bash configs/no_leakage_config/X_wcci_scratch_arch/launch_izar.sh
```

Filters match on the config name, e.g. `... launch_izar.sh mk64` submits the five
mk64 cells only. The launcher refuses to submit if any referenced action-space
artifact is missing locally.

## Before trusting the results

The existing WCCI-from-scratch run (`sc64c_NLS_tmean_f0_a0h0_s0`) selected its
best-test checkpoint at **165,888 of 15,000,000 steps** and never improved. If
WCCI-from-scratch training genuinely peaks that early, this screen will produce
ten early-peaking runs and separate nothing. Check that curve first.

Everything here is seed 0, matching the screens it derives from. Add seeds for
whichever cells look promising rather than tripling the grid up front.
