# Staged GNN Graph Screening

This folder implements the graph screening protocol described in the Methods
chapter. The screen starts on `bus14` with one paired seed, a fixed GINE actor
and MLP critic, and 8M training steps (approximately half of the 15M-step bus14
reference budget).

Each screening run uses 72 parallel environments, 576 rollout steps, and an
evaluation frequency of 82,944 environment steps. This yields 41,472 samples
per policy update, close to the previous 20-by-2,000 batch size while exposing
more environment work in parallel.

The settings used to isolate graph generalization are:

- one actor GNN shared by all agents;
- no learned global node-ID embeddings;
- no flattened observation concatenated to the actor;
- two message-passing layers, which allow information to cross an explicit
  transmission-line node;
- LayerNorm enabled in every screening condition;
- all equipment, line-node, and hierarchy relations kept bidirectional for the
  backwards-compatible screening baseline;
- the same chronic split, action space, critic, and PPO settings.

The heterogeneous builders also support relation-specific one-way message
passing. The four controls are:

- `gnn_generator_edge_direction`: `bidirectional`, `asset_to_busbar`, or
  `busbar_to_asset`;
- `gnn_load_edge_direction`: the same three choices;
- `gnn_line_node_edge_direction`: `bidirectional`, `line_to_busbar`, or
  `busbar_to_line` for the explicit-line representation;
- `gnn_summary_edge_direction`: `bidirectional` or `toward_summary` for
  busbar/substation links;
- `gnn_virtual_edge_direction`: `inherit`, `bidirectional`, or
  `toward_virtual` for the graph-level virtual-node links.

Direct physical transmission relations and same-substation busbar relations
remain bidirectional. A useful follow-up to the main screen is to take its
winning heterogeneous configuration and compare the bidirectional baseline
with `asset_to_busbar`, `line_to_busbar`, `toward_summary`, and
`toward_virtual`. These controls allow the busbar/hierarchy backbone to
aggregate equipment state without sending higher-level context back to the
lower-level nodes.

The MLP critic is deliberately kept fixed during screening. The critic is used
only for centralized training, while the graph representation being screened
is the decentralized actor input. Changing both at once would confound actor
representation quality with critic optimization and add substantial compute.
A GNN critic should be tested afterward as a separate ablation using the winning
actor configuration. It becomes especially relevant if the complete
actor--critic checkpoint, rather than only the actor encoder, must transfer
between grid sizes.

## Stage 1: representation x normalization

The committed folder `stage1_representation_normalization` contains the 12
runnable configurations formed by:

- graph type: `bus`, `heterogeneous`, or `heterogeneous_line`;
- graph input preprocessing: none, physical scaling, running normalization, or
  physical scaling followed by running normalization.

Launch all stage-1 runs from `Topology_Task`:

```bash
bash configs/gnn_graph_screening/stage1_representation_normalization/launch_all.sh
```

The files can be recreated and checked with:

```bash
python tools/gnn_graph_screening.py init --force
python tools/gnn_graph_screening.py validate \
  configs/gnn_graph_screening/stage1_representation_normalization
```

Select the stage-1 winner using held-out deterministic evaluation, with
survival/completion as the primary metric and reward, overloads, topology-action
count, training stability, runtime, and memory as secondary diagnostics.

## Stage 1b: focused normalization confirmation

The first screen did not reach the configured 8M steps before its 12-hour
training limit.  The focused confirmation therefore retains only the three
credible finalists:

- bus graph without input preprocessing;
- bus graph with deterministic physical scaling;
- direct heterogeneous graph without input preprocessing.

Each finalist is trained from scratch with seeds 0, 1, and 2 for 8M steps.  The
training limit is 2,880 minutes (48 hours) so the comparison is governed by environment
steps rather than the previous 720-minute cutoff.  Create, validate, and launch
the nine runs with:

```bash
python tools/gnn_graph_screening.py confirm-normalization
python tools/gnn_graph_screening.py validate \
  configs/gnn_graph_screening/stage1b_normalization_confirmation
bash configs/gnn_graph_screening/stage1b_normalization_confirmation/launch_all.sh
```

Rank configurations at the common 8M-step endpoint using the mean and standard
deviation across seeds.  The periodic evaluation remains at ten episodes to
control training cost; evaluate the selected checkpoints afterward on at least
30 episodes or the complete held-out chronic split. Keep this as the
representation/normalization comparison; Stage 2 separately tests structural
augmentation on fresh bus-graph runs.

## Stage 1c: provisional structural screen

Seven spare cluster slots can be used while Stage 1b is running to screen the
three binary structural factors on the provisional `bus + n0_none + GINE`
winner.  The Stage 1b `e0n0v0` run is the baseline, so Stage 1c contains exactly
the other seven combinations of same-substation edges (`e`), substation summary
nodes (`n`), and virtual-node readout (`v`).  All runs use seed 0, 8M steps, and
the 2,880-minute training limit.

Create, validate, and launch the seven runs with:

```bash
python tools/gnn_graph_screening.py provisional-structure
python tools/gnn_graph_screening.py validate \
  configs/gnn_graph_screening/stage1c_provisional_structure
bash configs/gnn_graph_screening/stage1c_provisional_structure/launch_all.sh
```

This is an early single-seed pilot, not the final structural comparison.
Stage 2 repeats all eight combinations—including a new unaugmented
baseline—with three seeds, 15M steps, and a longer training limit before any
structural conclusion is drawn.

## Stage 1d: heterogeneous asset-edge directions

The direct heterogeneous graph has two independently configurable asset
relations: generator attachments and load attachments.  Each relation can be
bidirectional, asset-to-busbar, or busbar-to-asset.  Their full factorial has
nine combinations; the Stage 1b heterogeneous `bidirectional/bidirectional`
run supplies the baseline, leaving eight Stage 1d jobs.

All eight runs keep the direct heterogeneous representation, no input
preprocessing, GINE, mean pooling, no hierarchy augmentation, seed 0, 8M steps,
and the 2,880-minute limit.  Create, validate, and launch them with:

```bash
python tools/gnn_graph_screening.py heterogeneous-directions
python tools/gnn_graph_screening.py validate \
  configs/gnn_graph_screening/stage1d_heterogeneous_directions
bash configs/gnn_graph_screening/stage1d_heterogeneous_directions/launch_all.sh
```

This screen tests whether equipment should only send state toward its busbar,
only receive busbar context, or exchange messages in both directions.  It keeps
the encoder fixed so a direction effect cannot be mistaken for a GAT-versus-GINE
effect.  As with Stage 1c, treat the result as provisional until Stage 1b
confirms the base representation and normalization.

## Stage 1e: heterogeneous structure and explicit-line scout

Stage 1e applies the same seven non-baseline `e/n/v` structural combinations
from Stage 1c to the direct heterogeneous graph.  Its Stage 1b
`e0n0v0` run supplies the baseline.  An eighth job trains the unaugmented
explicit-line representation for the complete 8M-step budget because its first
screen stopped at approximately 3.3M steps.

All eight jobs use no input preprocessing, GINE, bidirectional relations, seed
0, and the 2,880-minute limit.  Create, validate, and launch them with:

```bash
python tools/gnn_graph_screening.py heterogeneous-structure
python tools/gnn_graph_screening.py validate \
  configs/gnn_graph_screening/stage1e_heterogeneous_structure
bash configs/gnn_graph_screening/stage1e_heterogeneous_structure/launch_all.sh
```

Together, Stages 1c and 1e provide provisional structural screens for the bus
and direct heterogeneous finalists.  The explicit-line scout determines
whether that slower representation warrants additional seeds.  Stop adding
new factors after this stage and use the observed results to choose which
configurations deserve encoder and multi-seed confirmation.

## Stage 2: bus structural factors

Stage 2 first tests graph augmentation on the bus representation, without
assuming that an earlier unaugmented run can serve as its baseline. It creates
all eight binary combinations of same-substation edges (`e`), substation
summary nodes (`n`), and virtual-node readout (`v`), each as fresh runs with
seeds 0, 1, and 2. All 24 runs use no input preprocessing, 15M steps, and a
5,760-minute (four-day) limit. The `e` and `n` links are bidirectional, while
the `v` links carry messages only toward the virtual node:

```bash
python tools/gnn_graph_screening.py bus-structure
bash configs/gnn_graph_screening/stage2_structure/launch_all.sh
```

The generated filename contains `e0/1`, `n0/1`, `v0/1`, and `s0/1/2`.
Substation summary nodes currently use learned substation-indexed initial
embeddings, so report their in-grid performance separately from the
transfer/generalization claim.

## Stage 3: encoder comparison

Promote the actual stage-2 winner. This creates GCN, GAT, GINE, GraphSAGE, and
sparse graph transformer variants with the selected graph, normalization, and
structural settings held fixed:

```bash
python tools/gnn_graph_screening.py promote-encoder \
  configs/gnn_graph_screening/stage2_structure/WINNER.toml
bash configs/gnn_graph_screening/stage3_encoder/launch_all.sh
```

Only mean and virtual-node readout are used because they are implemented by all
five encoders. Attention and controlled-attention readouts remain
sparse-transformer-only experiments.

## Full-budget confirmation

After screening, promote the selected configurations to fresh 15M-step bus14
runs with seeds 0, 1, and 2. Up to four winner paths may be supplied in one
command:

```bash
python tools/gnn_graph_screening.py confirm \
  configs/gnn_graph_screening/stage3_encoder/SELECTED_1.toml \
  configs/gnn_graph_screening/stage3_encoder/SELECTED_2.toml
bash configs/gnn_graph_screening/stage4_confirmation/launch_all.sh
```

Each generated stage folder contains a `manifest.csv` recording the varied
factors. The promotion commands refuse to overwrite changed files unless
`--force` is passed.

After bus14 confirmation, the winning actor design can be retrained on WCCI as
a separate transfer/scale experiment. Agent-specific action heads must be
rebuilt for the WCCI action spaces even when the shared graph encoder is reused.
