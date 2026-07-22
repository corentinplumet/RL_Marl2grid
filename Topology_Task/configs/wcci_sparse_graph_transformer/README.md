# WCCI Sparse Graph Transformer Configs

These configs test the substation-aware sparse graph transformer on `bus36_wcci`
with the learned reduced action space.

Variants:

- `wcci_sparse_gt_mean_72x576_s*`: sparse transformer with physical-line,
  self, and same-substation busbar edges, using mean graph pooling.
- `wcci_sparse_gt_controlled_attn_72x576_s*`: same encoder, but graph readout
  attends only over controlled busbar nodes when neighbor context is included.

Launch examples:

```bash
sbatch job_jed.sh configs/wcci_sparse_graph_transformer/wcci_sparse_gt_mean_72x576_s0.toml
sbatch job_jed.sh configs/wcci_sparse_graph_transformer/wcci_sparse_gt_controlled_attn_72x576_s0.toml
```

