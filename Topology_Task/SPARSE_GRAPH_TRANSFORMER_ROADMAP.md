# Substation-Aware Sparse Graph Transformer Roadmap

This roadmap describes a new architecture direction for decentralized MAPPO on Grid2Op topology control. It is written so another LLM or engineer can implement the idea from the existing codebase without needing the original discussion.

The goal is not only to tune hyperparameters. The goal is to add an inductive bias that matches the physical decision structure of topology control:

1. overloaded or risky lines are local physical objects;
2. topology actions modify busbar assignments inside substations;
3. useful information mostly travels through physical line connections and same-substation busbar alternatives;
4. the policy should remain decentralized, without inter-agent communication.

The proposed architecture is a **Substation-Aware Sparse Graph Transformer**. It replaces or complements the current GINE/GNN encoder with a sparse attention encoder whose attention graph is constrained to physically meaningful neighbors.

---

## 0. Implementation Status

This document started as a roadmap. It now also records the current implementation state on branch:

```text
codex/sparse-graph-transformer-roadmap
```

### 0.1 Implemented

The core busbar sparse graph-transformer path has been implemented.

Implemented files:

- `Topology_Task/common/graph.py`
  - Added edge type constants:
    - `self`
    - `physical_line`
    - `same_substation_busbar`
  - Added optional explicit self edges.
  - Added optional same-substation busbar edges.
  - Added `edge_type` to graph specs and graph observations.
  - Added `controlled_node_mask` to graph specs and graph observations.
  - Kept old physical-line graph behavior unchanged unless the sparse graph-transformer path requests the new edges.

- `Topology_Task/env/utils.py`
  - Automatically enables self edges and same-substation busbar edges when:

    ```toml
    gnn_type = "sparse_transformer"
    ```

  - Existing GINE/GAT/GCN/GraphSAGE configs keep the previous graph structure by default.

- `Topology_Task/common/gnn.py`
  - Added `SparseGraphTransformerLayer`.
  - Added `SparseGraphTransformerEncoder`.
  - Integrated it through:

    ```toml
    actor_encoder = "gnn"
    gnn_type = "sparse_transformer"
    ```

  - Implemented relation-aware sparse attention with:
    - node queries, keys, and values;
    - edge-attribute key/value projections;
    - edge-type key/value embeddings;
    - relation-specific attention bias;
    - residual connections;
    - layer normalization;
    - feed-forward transformer block.

  - Implemented graph readout options:
    - `mean`
    - `sum`
    - `max`
    - `attention`
    - `controlled_mean`
    - `controlled_attention`

- `Topology_Task/alg/mappo/config.py`
  - Added `sparse_transformer` to `--gnn-type`.
  - Added sparse graph-transformer flags:

    ```text
    --sparse-gt-pooling
    --sparse-gt-add-self-edges
    --sparse-gt-add-substation-edges
    --sparse-gt-use-edge-attr
    --sparse-gt-use-edge-type-embeddings
    --sparse-gt-relation-bias
    --sparse-gt-dropout
    --sparse-gt-attention-dropout
    --sparse-gt-ffn-multiplier
    ```

- `Topology_Task/tools/inspect_graph_specs.py`
  - Added a graph inspection utility to print graph size and edge-type counts per agent.

- `Topology_Task/configs/wcci_sparse_graph_transformer/`
  - Added WCCI configs for three seeds with mean pooling:

    ```text
    wcci_sparse_gt_mean_72x576_s0.toml
    wcci_sparse_gt_mean_72x576_s1.toml
    wcci_sparse_gt_mean_72x576_s2.toml
    ```

  - Added WCCI configs for three seeds with controlled-attention pooling:

    ```text
    wcci_sparse_gt_controlled_attn_72x576_s0.toml
    wcci_sparse_gt_controlled_attn_72x576_s1.toml
    wcci_sparse_gt_controlled_attn_72x576_s2.toml
    ```

### 0.2 Validated Locally

The following checks were run locally:

- Python compile checks for the edited files.
- Direct tensor forward pass through `SparseGraphTransformerEncoder`.
- Integration tensor forward pass through `GraphAndFlatEncoder`.
- TOML dry-run command expansion for the WCCI sparse-transformer configs.

Full Grid2Op training smoke was **not** run locally because the local environment used during implementation did not have the full Grid2Op runtime installed. The first full environment test should be run on the cluster.

### 0.3 Not Implemented Yet

The following roadmap items are intentionally not implemented yet:

- Heterogeneous line-node graph variant.
  - Reason: it changes the graph schema more deeply by introducing explicit line nodes and heterogeneous relation types. It should be implemented only after the busbar sparse transformer has been tested.

- Action-conditioned decoder.
  - Reason: it needs a clean action-metadata layer that describes each action physically. Implementing it without this layer would make the actor fragile and hard to verify.

- Sparse-transformer W&B attention diagnostics.
  - Reason: correct logging requires retaining and aggregating attention statistics by relation type during actor forwards. This should be added carefully to avoid bloating rollout memory or slowing training.

- Formal unit tests.
  - Reason: smoke and tensor tests were run manually, but no permanent pytest-style tests were added yet.

### 0.4 Phase-By-Phase Status

| Phase | Status | What exists now | What remains |
|---|---:|---|---|
| Phase 0: diagnostics | Partial | `tools/inspect_graph_specs.py` prints node counts, edge counts, edge-type counts, dimensions, and controlled-node counts. | Add richer boundary-line diagnostics if needed. |
| Phase 1: graph builder | Implemented | `GridGraphBuilder` supports typed self, physical-line, and same-substation busbar edges. | None for the busbar-only sparse transformer. |
| Phase 2: sparse transformer encoder | Implemented | `SparseGraphTransformerEncoder` is available through `gnn_type = "sparse_transformer"`. | Cluster smoke training still required. |
| Phase 3: pooling variants | Implemented | `mean`, `sum`, `max`, `attention`, `controlled_mean`, and `controlled_attention` are supported. | Compare variants empirically. |
| Phase 4: heterogeneous line-node graph | Not implemented | Only specified in this roadmap. | Needs graph schema extension with explicit line nodes. |
| Phase 5: action-conditioned decoder | Not implemented | Only specified in this roadmap. | Needs action metadata extraction and decoder integration. |
| Phase 6: experiment matrix | Partial | Two WCCI sparse-transformer variants are configured for seeds 0, 1, 2. | Add stronger GINE and ablation configs if needed. |
| Phase 7: W&B explainability logs | Not implemented | No sparse-transformer attention logs yet. | Add relation-level attention mass and max-rho attention diagnostics. |
| Phase 8: testing | Partial | Compile and tensor-forward smoke checks were run manually. | Add permanent tests and cluster smoke. |

### 0.5 Immediate Next Step

Before launching long WCCI jobs, run graph inspection and one short smoke training on the cluster:

```bash
python Topology_Task/tools/inspect_graph_specs.py \
  --env-id bus36_wcci \
  --actor-encoder gnn \
  --critic-encoder mlp \
  --gnn-type sparse_transformer \
  --gnn-include-neighbors true \
  --sparse-gt-add-self-edges true \
  --sparse-gt-add-substation-edges true
```

Then run a short training override:

```bash
sbatch job_jed.sh \
  configs/wcci_sparse_graph_transformer/wcci_sparse_gt_mean_72x576_s0.toml \
  --total-timesteps 200000 \
  --eval-freq 82944 \
  --track false
```

If this smoke run works, launch the full seed set.

---

## 1. Current Code Context

Important files in the current repository:

- `Topology_Task/common/graph.py`
  - Contains `GridGraphBuilder`.
  - Builds busbar-level graph observations.
  - Current nodes are busbars.
  - Current graph edges mostly correspond to physical power lines.
  - Local graph construction is controlled by `gnn_include_neighbors`.

- `Topology_Task/common/gnn.py`
  - Contains `GraphEncoder` and `GraphAndFlatEncoder`.
  - Supports `gat`, `gine`, `gcn`, `graphsage`.
  - Current implementation is message passing, not sparse transformer attention.

- `Topology_Task/common/token_transformer.py`
  - Contains an existing token transformer.
  - This transformer uses token attention, but it is not the physical sparse attention proposed here.

- `Topology_Task/common/tokenizer.py`
  - Contains tokenization utilities for line/load/generator/substation-like features.
  - Useful reference if implementing a future heterogeneous token/line-node variant.

- `Topology_Task/alg/mappo/agent.py`
  - Contains `Actor` and `Critic`.
  - Actor currently encodes observation and outputs one categorical logit per action.
  - Existing encoder choices are `mlp`, `gnn`, and `transformer`.

- `Topology_Task/env/scenario.json`
  - Defines environments and multi-agent zones.
  - `bus36_wcci` has four agents and a much larger action space.

The first implementation should keep all existing configs working. New behavior must only activate when the new encoder/config flags are used.

---

## 2. Core Idea

Current GNNs pass messages along physical lines. This is useful, but topology control has another important relation: two busbars inside the same substation are not connected by a physical line, yet choosing between them is exactly what topology actions manipulate.

The new encoder should therefore build a sparse attention graph with three relation types:

1. `self`
   - A busbar attends to itself.

2. `physical_line`
   - A busbar attends to busbars connected by active or candidate power lines.

3. `same_substation_busbar`
   - A busbar attends to the other busbar(s) belonging to the same substation.

For each busbar node \(i\), the attention neighborhood is:

```latex
\[
\mathcal{A}(i)
=
\{i\}
\cup
\mathcal{N}_{\mathrm{line}}(i)
\cup
\mathcal{N}_{\mathrm{sub}}(i).
\]
```

Where:

- \(\mathcal{N}_{\mathrm{line}}(i)\) are busbars connected to \(i\) by physical lines;
- \(\mathcal{N}_{\mathrm{sub}}(i)\) are busbars in the same substation as \(i\);
- the graph remains sparse and physically meaningful.

This is closer to the proposed architecture than a full Graph Transformer. Full attention over all busbars would be expensive and physically weak because it allows every busbar to attend to every other busbar without respecting grid locality.

---

## 3. Mathematical Definition

Let:

- \(h_i^\ell\) be the hidden state of node \(i\) at transformer layer \(\ell\);
- \(e_{ij}\) be the edge feature for edge \(i \to j\);
- \(r_{ij}\) be the relation type of the edge;
- \(h\) index attention heads.

The relation-aware sparse attention can be implemented as:

```latex
\[
q_i^h = W_Q^h h_i^\ell,
\]

\[
k_{ij}^h = W_K^h h_j^\ell + W_{K,e}^{h,r_{ij}} e_{ij},
\]

\[
v_{ij}^h = W_V^h h_j^\ell + W_{V,e}^{h,r_{ij}} e_{ij}.
\]
```

The attention score is:

```latex
\[
s_{ij}^h
=
\frac{(q_i^h)^\top k_{ij}^h}{\sqrt{d_h}}
+ b_{r_{ij}}^h.
\]
```

The normalized attention weights are computed only over the sparse neighborhood:

```latex
\[
\alpha_{ij}^h
=
\frac{
\exp(s_{ij}^h)
}{
\sum_{m \in \mathcal{A}(i)}
\exp(s_{im}^h)
}.
\]
```

The node update is:

```latex
\[
\tilde{h}_i^{\ell+1}
=
\bigg\Vert_h
\sum_{j \in \mathcal{A}(i)}
\alpha_{ij}^h v_{ij}^h.
\]
```

Then use the standard transformer block:

```latex
\[
h_i^{\ell+1}
=
\mathrm{LayerNorm}
\left(
h_i^\ell
+
\mathrm{Dropout}
\left(
W_O \tilde{h}_i^{\ell+1}
\right)
\right),
\]

\[
h_i^{\ell+1}
=
\mathrm{LayerNorm}
\left(
h_i^{\ell+1}
+
\mathrm{FFN}(h_i^{\ell+1})
\right).
\]
```

This gives the model transformer-style dynamic attention while preserving physical locality.

---

## 4. Why This Is Different From Current GINE

Current GINE is a message passing GNN:

```latex
\[
h_i^{\ell+1}
=
f_\theta
\left(
h_i^\ell,
\sum_{j \in \mathcal{N}(i)}
g_\theta(h_j^\ell, e_{ij})
\right).
\]
```

The sparse graph transformer instead computes an adaptive weighted mixture:

```latex
\[
h_i^{\ell+1}
\approx
\sum_{j \in \mathcal{A}(i)}
\alpha_{ij}(x,e,r) v_{ij}.
\]
```

The important difference is that \(\alpha_{ij}\) is state-dependent. The model can learn that when one line is overloaded, some neighboring busbar or same-substation alternative is more relevant than others.

The same-substation edge is also new. It explicitly represents the topology-control alternative:

```latex
\[
\text{current busbar}
\leftrightarrow
\text{alternative busbar in same substation}.
\]
```

This is exactly the relation manipulated by busbar switching actions.

---

## 5. Decentralization Constraint

The architecture must remain decentralized:

- each agent receives only its own observation graph;
- no hidden state is exchanged between agents;
- no communication layer is added between agents;
- each actor outputs its own action independently.

It is acceptable for an agent to include one-hop physical neighbors in its observation if this is already allowed by the environment wrapper and graph builder. This is not communication between agents; it is local grid observation.

Recommended default for WCCI/bus36:

```toml
gnn_include_neighbors = true
```

Reason: boundary lines between zones are important. If a line connects two regions, both endpoint regions should be able to see enough context to reason about the overloaded asset.

---

## 6. Phase 0: Diagnostics Before Implementation

Before implementing the new encoder, add or run diagnostics to understand the existing graph shapes.

### 6.1 Report Graph Sizes

For each environment and agent, print:

- number of nodes;
- number of physical line edges;
- number of candidate same-substation edges;
- number of local lines;
- number of boundary lines;
- number of actions in the agent action space.

Target environments:

- `bus14`;
- `bus36`;
- `bus36_wcci`.

### 6.2 Validate Boundary Behavior

For `gnn_include_neighbors = false`:

- local graph should include only lines whose two endpoints are inside the agent domain.

For `gnn_include_neighbors = true`:

- local graph should include lines touching the agent domain;
- endpoint substations outside the domain may be included as neighbor context;
- this should be logged clearly.

### 6.3 Expected Output

Create a simple command or script that can print:

```text
env=bus36_wcci agent=agent_1
nodes=...
physical_edges=...
same_substation_edges=...
edge_types={self: ..., physical_line: ..., same_substation_busbar: ...}
actions=...
```

This is useful for debugging and for thesis reporting.

---

## 7. Phase 1: Extend Graph Builder

Modify `Topology_Task/common/graph.py`.

### 7.1 Add Edge Type IDs

Add edge type IDs to graph observations:

```python
EDGE_TYPE_SELF = 0
EDGE_TYPE_PHYSICAL_LINE = 1
EDGE_TYPE_SAME_SUBSTATION = 2
```

The graph observation should contain:

- `edge_index`;
- `edge_attr`;
- `edge_type`;
- existing masks and node features.

Keep backward compatibility:

- existing GNNs can ignore `edge_type`;
- if an old model/config does not use the sparse transformer, behavior should stay unchanged.

### 7.2 Add Same-Substation Busbar Edges

For each substation, add directed edges between busbar nodes belonging to the same substation.

If each substation has two busbars:

```text
(sub_id, busbar_0) -> (sub_id, busbar_1)
(sub_id, busbar_1) -> (sub_id, busbar_0)
```

If the internal representation has more busbars, connect all distinct pairs inside the same substation:

```latex
\[
(s,b_a) \to (s,b_b)
\quad
\forall b_a \ne b_b.
\]
```

### 7.3 Edge Features For Same-Substation Edges

Same-substation edges do not have line-specific features like `rho`.

Options:

1. Use zero edge attributes and rely on `edge_type`.
2. Add a binary feature `is_same_substation`.
3. Add substation-level features if available.

Recommended first implementation:

- keep the same edge feature dimension as physical edges;
- fill unavailable line features with zero;
- set `edge_type = EDGE_TYPE_SAME_SUBSTATION`;
- add a relation embedding in the encoder.

### 7.4 Add Self Edges

Self edges can be added either:

- explicitly in `edge_index`; or
- implicitly inside the encoder.

Recommended:

- explicitly add self edges in the graph builder for the sparse transformer only;
- set `edge_type = EDGE_TYPE_SELF`;
- use zero edge attributes.

This makes the attention implementation simpler because all attention happens over a single edge list.

### 7.5 Proposed New Config Flags

Add config flags:

```toml
sparse_gt_add_self_edges = true
sparse_gt_add_substation_edges = true
sparse_gt_use_edge_type_embeddings = true
sparse_gt_edge_dropout = 0.0
```

If possible, reuse existing `gnn_include_neighbors`.

---

## 8. Phase 2: Implement Sparse Graph Transformer Encoder

Create a new file:

```text
Topology_Task/common/sparse_graph_transformer.py
```

or add the class to:

```text
Topology_Task/common/gnn.py
```

Recommended class name:

```python
class SparseGraphTransformerEncoder(nn.Module):
    ...
```

### 8.1 Integration Choice

Prefer integrating it through the existing GNN path:

```toml
actor_encoder = "gnn"
critic_encoder = "mlp"
gnn_type = "sparse_transformer"
```

This minimizes changes in `Actor` and `Critic`.

Alternative:

```toml
actor_encoder = "sparse_graph_transformer"
```

This is cleaner conceptually but requires more agent/config changes.

Recommended first implementation: use `gnn_type = "sparse_transformer"`.

### 8.2 Required Inputs

The encoder must support graph batches produced by the current wrapper.

Required fields:

- node features;
- edge index;
- edge attributes;
- edge type;
- node mask if padded;
- agent-local domain mask if available.

The existing `GraphEncoder._to_pyg_batch` can be used as a reference for converting padded graph tensors into a PyTorch Geometric batch.

### 8.3 Attention Implementation

Use PyTorch Geometric utilities if available:

- `torch_geometric.utils.softmax` for edge-wise softmax grouped by destination node;
- `torch_scatter` or PyG scatter utilities for aggregation.

The attention should be grouped by target node \(i\). If `edge_index` is represented as `[src, dst]`, then normalize over incoming edges per `dst`.

Pseudo-code:

```python
src, dst = edge_index
q = q_proj(h)[dst]
k = k_proj(h)[src] + edge_k_proj(edge_attr, edge_type)
v = v_proj(h)[src] + edge_v_proj(edge_attr, edge_type)

score = (q * k).sum(-1) / sqrt(head_dim)
score = score + relation_bias[edge_type]
alpha = pyg_softmax(score, dst)
message = alpha.unsqueeze(-1) * v
out = scatter_sum(message, dst, dim=0, dim_size=num_nodes)
```

### 8.4 Encoder Hyperparameters

Add config fields:

```toml
gnn_type = "sparse_transformer"
gnn_hidden_dim = 128
gnn_out_dim = 128
gnn_layers = 2
gnn_heads = 4
sparse_gt_dropout = 0.0
sparse_gt_attention_dropout = 0.0
sparse_gt_ffn_multiplier = 4
sparse_gt_use_edge_attr = true
sparse_gt_use_edge_type_embeddings = true
sparse_gt_relation_bias = true
```

### 8.5 First Safe Defaults

For WCCI/bus36:

```toml
actor_encoder = "gnn"
critic_encoder = "mlp"
gnn_type = "sparse_transformer"
gnn_hidden_dim = 128
gnn_out_dim = 128
gnn_layers = 2
gnn_heads = 4
gnn_concat_flat = true
gnn_include_neighbors = true
sparse_gt_add_substation_edges = true
sparse_gt_use_edge_type_embeddings = true
```

Rationale:

- hidden dimension 64 and one layer were probably too small for WCCI;
- flat observation concatenation is useful because not everything is naturally represented in the graph;
- neighbor context is important for boundary lines;
- same-substation edges encode the topology-control action structure.

---

## 9. Phase 3: Pooling Variants

The encoder produces node embeddings. The actor needs one fixed-size embedding per agent.

Implement these pooling options:

### 9.1 Mean Pooling

```latex
\[
z = \frac{1}{|V|}
\sum_{i \in V}
h_i.
\]
```

This is simple and should reproduce current GNN behavior.

### 9.2 Attention Pooling

Learn a pooling score:

```latex
\[
\beta_i
=
\operatorname{softmax}_i(w^\top \tanh(W h_i)),
\]

\[
z
=
\sum_i
\beta_i h_i.
\]
```

This lets the actor focus on a few important busbars.

### 9.3 Controlled-Node Pooling

If a mask of nodes controlled by the agent is available, pool only controlled nodes:

```latex
\[
z_{\mathrm{local}}
=
\frac{1}{|V_i^{\mathrm{ctrl}}|}
\sum_{v \in V_i^{\mathrm{ctrl}}}
h_v.
\]
```

This is important when `gnn_include_neighbors = true`, because neighbor nodes are context, not directly controlled assets.

### 9.4 Combined Pooling

Recommended final pooling:

```latex
\[
z
=
\left[
z_{\mathrm{controlled}},
z_{\mathrm{all}},
z_{\mathrm{max}}
\right].
\]
```

Then project with an MLP before the actor head.

This can be added as:

```toml
gnn_readout_aggr = "controlled_attention"
```

or:

```toml
sparse_gt_pooling = "controlled_attention"
```

---

## 10. Phase 4: Heterogeneous Line-Node Variant

The busbar graph represents lines as edges. This is natural, but overloaded lines are central objects in topology control. A stronger variant is to add explicit line nodes.

### 10.1 Graph Entities

Node types:

1. `busbar`
2. `line`

Edges:

1. `busbar_to_line_or`
2. `line_to_busbar_or`
3. `busbar_to_line_ex`
4. `line_to_busbar_ex`
5. `same_substation_busbar`
6. `self`

### 10.2 Line Node Features

Line nodes should include:

- `rho`;
- `line_status`;
- `timestep_overflow`;
- `time_before_cooldown_line`;
- maintenance indicators if available;
- origin substation ID embedding;
- extremity substation ID embedding;
- whether line touches the agent domain.

### 10.3 Why This Helps

The actor can attend directly to overloaded lines:

```latex
\[
\text{busbar} \leftrightarrow \text{line} \leftrightarrow \text{busbar}.
\]
```

This makes explainability easier:

- attention mass on the max-rho line;
- selected action touches a substation adjacent to max-rho line;
- selected action changes topology near a line-node that received high attention.

### 10.4 Implementation Recommendation

Do not implement this before the busbar-only sparse transformer works.

The line-node variant changes the graph format more deeply and should be Phase 4, not Phase 1.

---

## 11. Phase 5: Action-Conditioned Decoder

The current actor maps the final embedding \(z\) to logits over action IDs:

```latex
\[
\ell = f_\theta(z),
\quad
\ell \in \mathbb{R}^{|\mathcal{A}|}.
\]
```

This treats action IDs as arbitrary classes. For a huge action space, especially WCCI, this is weak because action IDs do not encode what the action physically does.

The action-conditioned decoder scores each action using action features:

```latex
\[
\ell_a
=
g_\theta(z, \psi(a)),
\]
```

where \(\psi(a)\) is an embedding or feature vector describing action \(a\).

### 11.1 Action Features

For each action \(a\), precompute:

- original action ID;
- reduced action ID if using reduced action space;
- affected substation(s);
- number of topology changes;
- whether it changes busbar of a line endpoint;
- whether it changes load/generator attachment;
- local agent ID;
- one-hot or embedding of affected substation;
- binary vector of controlled substations touched by the action.

### 11.2 Decoder Options

MLP decoder:

```latex
\[
\ell_a
=
\mathrm{MLP}
\left(
\left[
z,
\psi(a),
z \odot W\psi(a)
\right]
\right).
\]
```

Bilinear decoder:

```latex
\[
\ell_a
=
z^\top W \psi(a)
+
u^\top [z, \psi(a)].
\]
```

Recommended first version:

- MLP decoder, easier to implement and debug.

### 11.3 Why This Matters

For WCCI, some agents have very large raw action spaces. A categorical head must learn each action logit independently. An action-conditioned decoder can generalize across structurally similar actions.

This is especially relevant if we compare:

- full action space;
- reduced action space;
- 64/128/256/512 per-agent reduced spaces.

---

## 12. Phase 6: Experiment Matrix

Use WCCI/bus36 as the main target, because this is where the current performance is low.

### 12.1 Minimal Architecture Ablations

Run each on seeds `0, 1, 2`.

#### A. Current MLP Baseline

Purpose: reference.

```toml
actor_encoder = "mlp"
critic_encoder = "mlp"
```

#### B. Current GINE Light

Purpose: current graph baseline.

```toml
actor_encoder = "gnn"
gnn_type = "gine"
gnn_hidden_dim = 64
gnn_out_dim = 64
gnn_layers = 1
gnn_concat_flat = false
gnn_include_neighbors = false
```

#### C. Stronger GINE

Purpose: distinguish architecture idea from underpowered GINE.

```toml
actor_encoder = "gnn"
gnn_type = "gine"
gnn_hidden_dim = 128
gnn_out_dim = 128
gnn_layers = 2
gnn_concat_flat = true
gnn_include_neighbors = true
```

#### D. Sparse Graph Transformer Without Same-Substation Edges

Purpose: test whether attention alone helps.

```toml
actor_encoder = "gnn"
gnn_type = "sparse_transformer"
gnn_hidden_dim = 128
gnn_out_dim = 128
gnn_layers = 2
gnn_heads = 4
gnn_concat_flat = true
gnn_include_neighbors = true
sparse_gt_add_substation_edges = false
```

#### E. Sparse Graph Transformer With Same-Substation Edges

Purpose: main proposed architecture.

```toml
actor_encoder = "gnn"
gnn_type = "sparse_transformer"
gnn_hidden_dim = 128
gnn_out_dim = 128
gnn_layers = 2
gnn_heads = 4
gnn_concat_flat = true
gnn_include_neighbors = true
sparse_gt_add_substation_edges = true
```

#### F. Sparse Graph Transformer With Controlled Attention Pooling

Purpose: handle neighbor context better.

```toml
actor_encoder = "gnn"
gnn_type = "sparse_transformer"
sparse_gt_pooling = "controlled_attention"
```

#### G. Sparse Graph Transformer With Action-Conditioned Decoder

Purpose: address huge action space.

```toml
actor_encoder = "gnn"
gnn_type = "sparse_transformer"
actor_action_decoder = "action_conditioned"
```

### 12.2 Optional Reward/Constraint Overlay

After the architecture is validated, combine it with existing sparse-action methods:

- fixed intervention penalty;
- adaptive intervention budget;
- topology distance penalty;
- reduced action space.

Do not start with all overlays at once. First establish whether the encoder improves baseline learning.

---

## 13. Phase 7: W&B Logs And Explainability

The new architecture should add logs that directly explain whether the inductive bias is being used.

### 13.1 Attention Relation Mass

For each agent:

```text
train/sparse_gt_attention_self_agent_0
train/sparse_gt_attention_physical_line_agent_0
train/sparse_gt_attention_same_substation_agent_0
```

Interpretation:

- high `physical_line` attention means the model is looking through grid connectivity;
- high `same_substation` attention means the model is using busbar alternatives;
- high `self` attention may mean it is behaving more like local MLP/message passing.

### 13.2 Attention To Max-Rho Context

At evaluation time, log:

```text
test/sparse_gt_attention_to_max_rho_line_agent_0
test/sparse_gt_selected_action_touches_max_rho_zone_agent_0
```

If using line-node variant:

```text
test/sparse_gt_attention_to_max_rho_line_node_agent_0
```

### 13.3 Action Decoder Logs

If using action-conditioned decoder:

```text
train/action_decoder_selected_action_score_agent_0
train/action_decoder_selected_action_touched_substation_agent_0
train/action_decoder_selected_action_num_topo_changes_agent_0
```

### 13.4 Existing Logs To Compare Against

Always compare against existing logs:

- episodic survival;
- action-0 fraction per agent;
- non-idle count per step;
- illegal action rate;
- line margin reward;
- topology reward;
- intervention penalty or adaptive budget terms if enabled.

---

## 14. Phase 8: Testing Plan

### 14.1 Unit Tests

Add tests or smoke scripts for:

1. graph builder creates same-substation edges;
2. edge types have correct values;
3. edge attributes have consistent dimensions;
4. sparse transformer forward pass returns correct shape;
5. batched graph forward pass works;
6. old GINE configs still run.

### 14.2 Smoke Training

Run a tiny training:

```bash
python Topology_Task/main.py \
  --env-id bus14 \
  --actor-encoder gnn \
  --gnn-type sparse_transformer \
  --total-timesteps 10000 \
  --n-envs 2 \
  --n-steps 64 \
  --track false
```

Then WCCI smoke:

```bash
python Topology_Task/main.py \
  --env-id bus36_wcci \
  --actor-encoder gnn \
  --gnn-type sparse_transformer \
  --total-timesteps 10000 \
  --n-envs 2 \
  --n-steps 64 \
  --track false
```

### 14.3 Safety Checks

Verify:

- no NaNs in logits;
- no NaNs in value loss;
- action distribution is not collapsed at initialization;
- old configs produce the same command-line arguments and do not require new fields.

---

## 15. Recommended Implementation Order

Implement in this exact order:

1. Add graph diagnostics.
2. Add edge types to graph observations.
3. Add same-substation busbar edges.
4. Implement sparse graph transformer layer.
5. Integrate as `gnn_type = "sparse_transformer"`.
6. Add basic mean pooling.
7. Run bus14 smoke test.
8. Run bus36_wcci smoke test.
9. Add controlled-node pooling.
10. Add attention/explainability logs.
11. Run WCCI seed `0` short training.
12. Run full WCCI seeds `0, 1, 2`.
13. Implement heterogeneous line-node variant only if busbar sparse transformer is stable.
14. Implement action-conditioned decoder only after sparse transformer is stable.

This order avoids mixing too many moving pieces at once.

---

## 16. Minimal Config Template

Example WCCI sparse graph transformer config:

```toml
exp_tag = "wcci_sparse_gt_busbar_s0"
seed = 0

env_id = "bus36_wcci"
split_chronics = true
test_chronics_pct = 0.2
chronic_split_seed = 0

actor_encoder = "gnn"
critic_encoder = "mlp"

gnn_type = "sparse_transformer"
gnn_hidden_dim = 128
gnn_out_dim = 128
gnn_layers = 2
gnn_heads = 4
gnn_concat_flat = true
gnn_include_neighbors = true
gnn_graph_type = "bus"

sparse_gt_add_self_edges = true
sparse_gt_add_substation_edges = true
sparse_gt_use_edge_attr = true
sparse_gt_use_edge_type_embeddings = true
sparse_gt_relation_bias = true
sparse_gt_attention_dropout = 0.0
sparse_gt_dropout = 0.0
sparse_gt_ffn_multiplier = 4
sparse_gt_pooling = "mean"

actor_layers = [128, 128]
critic_layers = [256, 256, 256]

total_timesteps = 20000000
n_envs = 72
n_steps = 576
eval_freq = 80000

gamma = 0.9
gae_lambda = 0.95
actor_lr = 0.0001
critic_lr = 0.0001
entropy_coef = 0.01
vf_coef = 0.5
target_kl = 0.02
```

If reduced action space is used:

```toml
reduced_action_space = "outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk128.json"
```

---

## 17. Variants To Implement Later

### 17.1 Sparse GT Plus Adaptive Intervention Budget

Purpose:

- combine stronger physical encoder with sparse decision behavior.

Config additions:

```toml
intervention_penalty = 0.0
adaptive_intervention_budget = true
intervention_budget_target = 0.20
intervention_budget_scope = "local"
```

### 17.2 Sparse GT Plus Topology Distance Cost

Purpose:

- encourage staying close to the reference topology.

Config additions:

```toml
topology_reward_weight = 0.001
```

or use any existing topology-distance penalty fields in the codebase.

### 17.3 Sparse GT Plus Reduced Action Space

Purpose:

- reduce WCCI action space from tens of thousands to a manageable set.

Config additions:

```toml
reduced_action_space = "path/to/reduced_action_space.json"
```

### 17.4 Sparse GT Plus Action-Conditioned Decoder

Purpose:

- make logits depend on action semantics.

Config additions:

```toml
actor_action_decoder = "action_conditioned"
action_embedding_dim = 64
action_decoder_hidden_dim = 128
```

---

## 18. Success Criteria

The architecture should be considered useful if it improves at least one of these without unacceptable regression in the others:

1. higher full-test episodic survival on WCCI;
2. better stability across seeds;
3. better action-0 usage without destroying survival;
4. lower illegal action rate;
5. improved survival with smaller reduced action spaces;
6. interpretable attention patterns around overloaded lines and affected substations.

Minimum useful comparison:

```text
MLP baseline
vs
current GINE
vs
sparse graph transformer without same-substation edges
vs
sparse graph transformer with same-substation edges
```

If the version with same-substation edges improves over the version without them, this supports the thesis claim that topology-control-specific graph structure matters.

---

## 19. Thesis Explanation

A concise way to describe this in the thesis:

> I introduce a substation-aware sparse graph transformer for decentralized topology control. Unlike a standard GNN that only propagates messages over physical transmission lines, the proposed encoder also connects alternative busbars inside the same substation, which are the objects directly manipulated by topology actions. Attention is restricted to physically meaningful neighborhoods: self, line-connected busbars, and same-substation busbars. This preserves decentralization while giving each regional policy a stronger inductive bias for reasoning about overloaded lines and local switching alternatives.

The key distinction is:

```latex
\[
\text{standard GNN locality}
=
\text{physical line neighbors},
\]

\[
\text{proposed topology-control locality}
=
\text{physical line neighbors}
\cup
\text{same-substation busbar alternatives}.
\]
```

This makes the model more aligned with how topology control actually works.
