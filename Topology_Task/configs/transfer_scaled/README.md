# bus14 → WCCI transfer with corrected encoder inputs

Same study as [`transfer_bus14_to_wcci_nomaint`](../transfer_bus14_to_wcci_nomaint),
run on inputs that are comparable across the two grids. Two settings change
relative to that study, in **both** the source and the target:

```toml
gnn_angle_representation = "edge_diff"   # angle drop per line, not per-asset absolute angle
gnn_physical_scaling = true              # power divided by each grid's own gen_pmax
```

Measured effect on the distribution mismatch (6000 do-nothing steps per grid):

| channel | ratio before | KS before | ratio after | KS after |
|---|---|---|---|---|
| `load_theta` | 0.85 | 0.331 | *replaced* | |
| `gen_theta` | 0.92 | 0.115 | *replaced* | |
| `theta_diff` | — | — | **1.15** | **0.231** |
| `gen_p` | 2.44 | 0.089 | **0.85** | 0.115 |
| `load_p` | 2.36 | 0.040 | **0.83** | 0.244 |
| `rho` | 1.05 | 0.141 | 1.05 | 0.142 |

## Two stages, in order

A frozen encoder's weights are bound to the input scale they were trained on, so
the bus14 sources must be retrained before the frozen arm means anything. The
existing `best_test_gs_*` checkpoints were trained on raw features and cannot be
reused here.

### Stage 1 — 12 bus14 sources

`sources/trsrc_<variant>_s<seed>.toml`, 4 variants × 3 seeds. Each is its
screening config from `gnn_graph_screening` with only the two input settings and
the run name changed, so it stays comparable to the original screening result.

```bash
Topology_Task/configs/transfer_scaled/sources/launch_izar.sh
```

### Stage 2 — 24 WCCI arms

`wcci/trs_<variant>_<arm>_s<seed>.toml`, 4 variants × {frozen, scratch} × 3
seeds. Each frozen arm loads `checkpoint/best_test_trsrc_<variant>_s<seed>.tar`,
produced by its stage 1 counterpart. Verify they exist first:

```bash
Topology_Task/configs/transfer_scaled/check_sources.sh
```

```bash
Topology_Task/configs/transfer_scaled/wcci/launch_izar.sh
```

Both launchers take a substring filter, so `launch_izar.sh _s0` runs seed 0 only
— 4 jobs in stage 1, 8 in stage 2. Worth doing first: 36 runs is a lot of
cluster time to commit before seeing a curve.

## What the pairing buys

Three comparisons come out of running this alongside the raw study:

1. **frozen vs scratch, within this study** — does a pretrained encoder help,
   once its inputs are calibrated? This is the actual research question.
2. **scratch here vs scratch in the raw study** — does the input correction help
   or hurt on WCCI by itself? Identical env, action space, architecture and
   budget; the features are the only difference, and no transfer is involved.
3. **frozen here vs frozen in the raw study** — how much of any transfer deficit
   was distribution shift rather than the representation failing to generalize.

Keep the raw study's runs for 2 and 3.

## Reference points

On the 50-chronic held-out sample: do-nothing survives 0.560 of the average
episode and completes 52% of chronics; simulator-greedy over the same `mk256`
action space reaches 0.976 and 92%. Final numbers should be taken on the full
576-chronic test split, since at n=50 the 95% interval on mean survival is
±0.13.
