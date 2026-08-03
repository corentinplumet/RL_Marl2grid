# Candidate-Action Attention Stages

This folder contains the learned adaptive-pooling ladder for the heterogeneous-line candidate-action scorer.

The configs are copied from the strongest a-priori hard-pooling control:

```text
configs/gnn_action_scoring/candidate_variants/cas_hl_mean_f1_a0h0_s*.toml
```

Only the pooling mechanism is changed from uniform `mean` to learned `typed_attention`. The encoder, critic, PPO settings, reward, chronic split, and training budget remain paired with `candidate_variants`.

## Stages

| Stage | Scope | Meaning |
|---|---|---|
| `affected` | `candidate_action_attention_scope = "affected"` | Learn weights only inside the hardcoded affected-node mapping. |
| `softprior` | `candidate_action_attention_scope = "soft_prior"` | Attend to all nodes, but add a score bonus to hardcoded affected nodes. |
| `all` | `candidate_action_attention_scope = "all"` | Attend to all graph nodes with no physical prior. |

Every run uses:

```toml
actor_action_head = "candidate_pool"
candidate_action_pool = "typed_attention"
candidate_action_use_features = true
candidate_action_do_nothing_head = false
candidate_action_attention_heads = 1
candidate_action_attention_dim = 0
candidate_action_attention_temperature = 1.0
candidate_action_attention_query = "global_action_features"
candidate_action_attention_normalizer = "softmax"
candidate_action_attention_prior_bias = 2.0
candidate_action_attention_chunk_size = 64
```

## Recommended Launch Order

Start with seed 0 for all three stages, since you have cluster room and this gives a fast read on stability and memory:

```bash
for cfg in Topology_Task/configs/gnn_action_scoring/attention_stages/*_s0.toml; do
  sbatch --time=7-00:00:00 job_jed.sh "${cfg#Topology_Task/}"
done
```

If you want the strictest incremental test first, launch only `affected` seed 0:

```bash
sbatch --time=7-00:00:00 job_jed.sh configs/gnn_action_scoring/attention_stages/cas_hl_tattn_affected_f1_a0h0_s0.toml
```

Once a stage looks stable, promote the same stage to seeds 1 and 2:

```bash
for cfg in Topology_Task/configs/gnn_action_scoring/attention_stages/cas_hl_tattn_affected_f1_a0h0_s{1,2}.toml; do
  sbatch --time=7-00:00:00 job_jed.sh "${cfg#Topology_Task/}"
done
```

The complete factor table is in `manifest.csv`.
