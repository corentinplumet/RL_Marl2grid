# Stage 1b: Focused Normalization Confirmation

This folder contains a paired 3-by-3 confirmation screen on `bus14`:

- finalists: bus/no preprocessing, bus/physical scaling, and heterogeneous/no
  preprocessing;
- seeds: 0, 1, and 2;
- budget: 8,000,000 environment steps per run;
- training limit: 2,880 minutes (48 hours);
- actor: shared two-layer GINE encoder with agent-specific MLP action heads;
- critic: the unchanged centralized MLP.

Running normalization and the explicit transmission-line-node representation
are not repeated because they were not competitive in the first screen.  All
other PPO, graph, chronic-split, and evaluation settings remain paired with
Stage 1.

Regenerate and validate the configs from `Topology_Task`:

```bash
python tools/gnn_graph_screening.py confirm-normalization --force
python tools/gnn_graph_screening.py validate \
  configs/gnn_graph_screening/stage1b_normalization_confirmation
```

Launch all nine cluster jobs from the repository root:

```bash
bash Topology_Task/configs/gnn_graph_screening/stage1b_normalization_confirmation/launch_all.sh
```

Compare the three-seed mean and standard deviation at the common 8M-step
endpoint.  After selecting the winner, run a larger deterministic evaluation
on at least 30 episodes or the complete held-out chronic split before promoting
it to Stage 2.
