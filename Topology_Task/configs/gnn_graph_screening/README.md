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
- the same chronic split, action space, critic, and PPO settings.

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

## Stage 2: structural factors

Promote the actual stage-1 winner. This creates all eight binary combinations
of same-substation edges (`e`), substation summary nodes (`n`), and virtual-node
readout (`v`) while preserving the winning representation and normalization:

```bash
python tools/gnn_graph_screening.py promote-structure \
  configs/gnn_graph_screening/stage1_representation_normalization/WINNER.toml
bash configs/gnn_graph_screening/stage2_structure/launch_all.sh
```

The generated filename contains `e0/1`, `n0/1`, and `v0/1`. Substation summary
nodes currently use learned substation-indexed initial embeddings, so report
their in-grid performance separately from the transfer/generalization claim.

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
