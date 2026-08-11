# Candidate-actor preprocessing transfer study

This directory contains the seed-0 WCCI no-maintenance comparison requested for
the three candidate configurations that performed well with both preprocessing
pipelines:

- `mean_f1_a0h0`
- `mean_f0_a0h1`
- `mean_f0_a0h0`

For each candidate, the study crosses `NL` and `NLS` preprocessing with three
actor regimes:

| Regime | Initialization | WCCI adaptation |
|---|---|---|
| `frozen` | bus14 GNN and candidate scorer | Actor learning rate is zero; run only to the first evaluation/checkpoint |
| `scratch` | Random GNN and candidate scorer | Train the complete actor for 15M steps |
| `finetune` | bus14 GNN and candidate scorer | Fine-tune the complete actor for 15M steps |

The critic is always WCCI-specific and starts from scratch. `NL` source actors
receive the matching unprocessed WCCI graph inputs; `NLS` source actors receive
edge-difference angles and physical scaling based on the target grid's largest
installed generator.

The source checkpoints predate removal of the redundant `connected` node
column. All target arms, including scratch, therefore retain that column so the
actor architecture remains identical within every comparison.

Candidate scorers were independently trained per bus14 agent, while WCCI has a
different number of agents. The transfer rule is explicit and fixed across all
conditions: learned scorer parameters from bus14 `agent_0` are broadcast to all
WCCI actors, while every target actor retains its own WCCI action metadata.

Launch all conditions from `Topology_Task` with:

```bash
bash configs/transfer_candidate_preprocessing/launch_izar.sh
```

Regenerate the files after changing the matrix with:

```bash
python tools/generate_candidate_transfer_configs.py
```
