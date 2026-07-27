# Heterogeneous Message-Direction Evaluation

This folder contains the 27 training configurations from the complete
generator-direction by load-direction factorial, with three seeds per
direction pair.

The evaluation launchers preserve the original cluster assignment:

- `evaluate_best_cpu_jed.sh`: 15 checkpoints trained on JED;
- `evaluate_best_gpu_izar.sh`: 12 checkpoints trained on Izar.

If all checkpoint files have been copied or synchronized to JED,
`evaluate_best_all_jed.sh` submits all 27 evaluations there instead.

Each launcher first checks that every expected
`checkpoint/best_test_<run>.tar` file exists. If any checkpoint is missing, it
exits before submitting jobs. Every submitted job uses:

- the complete held-out `test` chronic split;
- deterministic policy actions;
- no evaluation-time heuristic;
- the observation-normalization statistics stored in the checkpoint;
- one JSON summary under `Topology_Task/outputs/full_test_eval/`.

Run from the repository root on JED:

```bash
bash Topology_Task/configs/gnn_graph_screening/heterogenous_message_direction/evaluate_best_cpu_jed.sh
```

Alternatively, evaluate all 27 checkpoints on JED:

```bash
bash Topology_Task/configs/gnn_graph_screening/heterogenous_message_direction/evaluate_best_all_jed.sh
```

Run from the repository root on Izar:

```bash
bash Topology_Task/configs/gnn_graph_screening/heterogenous_message_direction/evaluate_best_gpu_izar.sh
```

Monitor the submitted jobs with:

```bash
squeue -u "$USER"
```

The `best_test_` checkpoints were selected during training using periodic
evaluation performance. Therefore, evaluating them again on the same test
split is useful for a standardized full-split comparison, but it is not an
unbiased final test estimate. A strictly held-out estimate requires checkpoint
selection on a separate validation split.
