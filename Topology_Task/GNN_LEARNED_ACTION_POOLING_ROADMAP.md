# Learned Candidate-Action Pooling Roadmap

## Status

Implementation progress:

```text
[done] candidate_action_pool = "mean"
[done] candidate_action_pool = "typed_mean"
[done] typed_attention + affected scope
[done] typed_attention + soft_prior scope
[next] typed_attention + all scope
```

The later sparse-normalization and candidate-action-node experiments remain
proposals rather than part of the current implementation.

The goal is to preserve the candidate-action classifier introduced in
`GNN_ACTION_SCORING_ROADMAP.md`, while replacing uniform pooling with learned,
state-dependent pooling that can be visualized as an action-specific heatmap.

## Main Idea

The current typed pooling assigns the same weight to every affected node of a
given type:

```text
line_context(action) = mean(affected line-node embeddings)
```

The learned version assigns a different weight to each node:

```text
line_context(action, state)
  = sum_i attention_weight(action, state, line_i) * line_embedding_i
```

For each candidate action, an action token acts like a CLS query:

```text
action metadata + global graph context
              |
              v
       action token / query
              |
              v
     cross-attention over nodes
              |
              v
 learned action-specific graph context
              |
              v
       shared action scorer
              |
              v
          action logit
```

A single global CLS token is not sufficient because it would produce the same
pooled graph representation for every candidate. The implementation needs one
query per action, either generated from physical action features or represented
by a learned action embedding.

## Recommended Progression

| Stage | Pooling | Attention scope | Purpose |
|---|---|---|---|
| 0 | `typed_mean` | Hardcoded affected nodes | Current baseline |
| 1 | `typed_attention` | Hardcoded affected nodes | Learn relative importance inside the physical mapping |
| 2 | `typed_attention` | All nodes with a hardcoded-node score bonus | Let the model extend the physical mapping |
| 3 | `typed_attention` | All nodes without a prior | Test whether the model rediscovers the mapping |
| 4 | Sparse attention or action nodes | Learned sparse neighborhood | Optional stronger graph classifier |

The stages should be implemented and validated in this order. Stage 1 has the
lowest risk and is the cleanest comparison against `typed_mean`. Stage 2 is the
most useful architecture for comparing a learned heatmap with the hardcoded
mapping without forcing them to be identical.

## Mathematical Formulation

For batch item `b`, action `a`, and graph node `i`:

```text
g_b       global graph embedding
h_bi      node embedding after GNN message passing
f_a       static physical action features
q_ba      action query generated from [g_b, f_a]
k_bi      node key generated from h_bi
v_bi      node value generated from h_bi
```

Scaled dot-product attention is:

```text
score_bai = dot(q_ba, k_bi) / sqrt(attention_dim)
alpha_bai = normalize_i(score_bai)
context_ba = sum_i alpha_bai * v_bi
```

The action logit remains:

```text
logit_ba = shared_scorer([g_b, context_ba, f_a])
```

PPO remains unchanged. The policy still receives:

```python
Categorical(logits=logits)
```

## Stage 0: Freeze The Current Baseline

Before changing the scorer, preserve the current reference:

```toml
actor_action_head = "candidate_pool"
candidate_action_pool = "typed_mean"
candidate_action_use_features = true
candidate_action_do_nothing_head = true
```

Record:

- parameter count;
- initial action probabilities;
- training throughput;
- peak memory;
- survival and return curves;
- action-0 frequency;
- complete full-test results.

This run is the control for every learned-pooling comparison.

## Stage 1: Masked Typed Attention

### Behavior

Attention is restricted to the same hardcoded nodes used by `typed_mean`.

For each action, compute one attention distribution per physical type:

```text
busbar attention over affected busbars only
line attention over affected lines only
load attention over affected loads only
generator attention over affected generators only
```

The resulting context keeps the current typed structure:

```text
local_context = concat(
    attended_busbar_context,
    attended_line_context,
    attended_load_context,
    attended_generator_context,
)
```

Nodes outside the action metadata receive a score of negative infinity before
softmax and therefore exactly zero weight.

### Meaning Of The Heatmap

This mode answers:

```text
Among the objects that the physical action mapping says are relevant,
which ones does the policy consider most important in this state?
```

It cannot discover additional nodes. Its overlap with the hardcoded set is
always 100 percent, so the useful analysis is the relative weight assigned to
each affected node and node type.

### Efficient Tensor Path

The existing metadata already contains padded indices and masks:

```text
line_indices: [n_actions, max_lines_per_action]
line_mask:    [n_actions, max_lines_per_action]
```

For one node type:

```text
node embeddings: [batch, n_nodes, node_dim]
gathered nodes:  [batch, n_actions, max_items, node_dim]
action queries:  [batch, n_actions, attention_dim]
scores:          [batch, n_actions, max_items]
weights:         [batch, n_actions, max_items]
context:         [batch, n_actions, node_dim]
```

This avoids materializing attention over every graph node and should be the
first implementation used on bus14 and WCCI reduced action spaces.

### Empty Node Types

Some actions do not touch a load, generator, or line. For an empty type:

```text
context = zero vector
attention weights = zero
```

The masked normalization must never receive a row containing only negative
infinity. Detect empty rows before softmax and explicitly return zeros.

## Stage 2: Soft-Prior Attention

### Behavior

Every action can attend to every node in the local graph, but nodes from the
hardcoded physical mapping receive an additive score bonus:

```text
score_bai = learned_score_bai + beta * affected_mask_ai
```

where:

```text
affected_mask_ai = 1 if node i is hardcoded as affected by action a
affected_mask_ai = 0 otherwise
```

Interpretation of `beta`:

```text
large positive beta  strongly trusts the hardcoded mapping
small positive beta  treats the mapping as a weak prior
beta = 0             becomes free all-node attention
```

Start with a fixed `beta`. A learnable value can be tested later, preferably
one value per node type rather than one unconstrained value per action.

### Meaning Of The Heatmap

This mode answers:

```text
Does the model mainly use the objects directly changed by the action,
or does it rely on neighboring lines, substations, loads, or generators?
```

The main metric is:

```text
hardcoded attention mass
  = sum of learned attention weights on hardcoded affected nodes
```

A value near 1 means the learned pooling agrees with the hardcoded mapping. A
smaller value identifies additional physical context used by the policy.

### Dense Attention Mask

Build a static boolean tensor once per agent:

```text
affected_node_mask: [n_actions, n_nodes]
```

It can be derived from the existing padded busbar, line, load, and generator
indices. Store it as a non-trainable module buffer so it follows the actor to
CPU or GPU and is checked during checkpoint loading.

### Memory Control

All-node attention creates scores with shape:

```text
[batch, n_actions, n_nodes]
```

For WCCI, score actions in chunks:

```text
actions 0:chunk_size
actions chunk_size:2*chunk_size
...
```

The chunk size should be configurable and benchmarked. Start with `32` or `64`
candidate actions per chunk.

## Stage 3: Free All-Node Attention

### Behavior

Set the hardcoded prior bonus to zero:

```text
score_bai = learned_score_bai
```

The action query must contain action-specific information. Otherwise every
action would use the same attention query and differ only in the final scorer.

Recommended query:

```text
q_ba = query_mlp([global_graph_embedding_b, action_features_a])
```

Avoid using a learned action-ID embedding in the first experiment. It can
memorize action identities and weakens the physical transfer argument. Add it
only as a separate ablation.

### Scientific Question

This stage tests whether the learned classifier independently rediscovers the
physical action mapping. It is the most direct heatmap comparison, but also the
least constrained and most likely to overfit.

## Stage 4A: Sparse Learned Attention

Softmax assigns nonzero weight to every eligible node. More readable heatmaps
can be obtained with:

- `sparsemax`;
- `entmax15`;
- differentiable top-k attention.

This should be implemented only after ordinary softmax attention works. Entmax
is a reasonable first sparse normalizer because it remains differentiable and
often produces exact zero weights.

## Stage 4B: Action Nodes Inside The GNN

A stronger alternative is to add explicit candidate-action nodes:

```text
busbar nodes
line nodes
load nodes
generator nodes
candidate-action nodes
```

Each candidate-action node can be connected to:

- only its hardcoded affected nodes;
- its affected nodes plus neighboring nodes;
- every local graph node through attention edges.

After message passing:

```text
final candidate-action-node embedding -> shared scalar head -> action logit
```

This gives the GNN direct classification responsibility, but increases the
graph by `n_actions` nodes and can be expensive for WCCI. Treat it as a later
architecture, not as part of the initial learned-pooling implementation.

## Alternative: Learned Scalar Gating

Cross-attention is not the only learned pooling mechanism. A simpler scorer is:

```text
score_bai = gate_mlp([node_embedding_bi, action_query_ba])
alpha_bai = softmax_i(score_bai)
```

This produces the same kind of heatmap but avoids explicit multi-head query,
key, and value projections. It is a useful fallback if transformer-style
attention is too expensive or unstable.

## Proposed Configuration Flags

### Initial Stage-1 Flags

```text
--candidate-action-pool
  choices: mean, typed_mean, typed_attention
  default: typed_mean

--candidate-action-attention-scope
  choices: affected
  default: affected

--candidate-action-attention-heads
  default: 1

--candidate-action-attention-dim
  default: 0
  meaning: use the GNN node dimension

--candidate-action-attention-temperature
  default: 1.0
```

### Stage-2 And Stage-3 Flags

```text
--candidate-action-attention-scope
  choices: affected, soft_prior, all

--candidate-action-attention-prior-bias
  default: 2.0

--candidate-action-attention-query
  choices: global_action_features, learned_action, global_only
  default: global_action_features

--candidate-action-attention-chunk-size
  default: 64

--candidate-action-attention-normalizer
  choices: softmax, entmax15
  default: softmax
```

Do not overload `candidate_action_use_features`. That existing flag controls
whether static action features are concatenated into the final shared scorer.
The attention query should have its own explicit configuration because an
all-node query needs action-specific information even when the final scorer
does not receive the raw action features.

## Proposed Module Design

Add a dedicated module, for example:

```text
alg/mappo/action_pooling.py
```

Conceptual API:

```python
@dataclass
class CandidateAttentionOutput:
    context: torch.Tensor
    weights: Optional[Dict[str, torch.Tensor]]


class CandidateActionAttentionPool(nn.Module):
    def __init__(
        self,
        node_dim,
        graph_dim,
        action_feature_dim,
        metadata,
        node_types,
        scope="affected",
        heads=1,
        attention_dim=None,
        prior_bias=2.0,
        temperature=1.0,
        normalizer="softmax",
        action_chunk_size=64,
    ):
        ...

    def forward(
        self,
        graph_embedding,
        node_embeddings,
        return_weights=False,
    ):
        ...
```

`CandidateActionScorer` should own the pooling module:

```text
CandidateActionScorer
  -> CandidateActionAttentionPool
  -> shared scoring MLP
  -> dedicated action-0 head
```

The training path should request only the context. Evaluation and analysis
tools can request attention weights explicitly. This avoids retaining large
attention tensors during normal PPO training.

## Query Design

### Recommended Query

Use a state-dependent physical query:

```text
query_input_ba = concat(global_graph_embedding_b, action_features_a)
query_ba = query_mlp(query_input_ba)
```

This lets the same action attend to different nodes in different grid states.

### Learned Action-ID Query

An optional ablation is:

```text
query_ba = learned_action_embedding[a] + state_projection(g_b)
```

Advantages:

- expressive;
- easy to implement.

Disadvantages:

- tied to the exact action ordering;
- can memorize IDs;
- less transferable to another reduced action space;
- weaker physical interpretation.

Do not make this the default.

### Global-Only Query

For masked attention, the hardcoded candidate set already distinguishes
actions. A global-only query is therefore a valid ablation:

```text
query_ba = query_mlp(g_b)
```

It asks every action to apply the same state-dependent importance rule to its
own physical candidate set. It is not suitable for free all-node attention,
where every action would otherwise see the same node set and query.

## Attention Scoring Choices

### Scaled Dot Product

```text
score = dot(Wq(query), Wk(node)) / sqrt(d)
```

This matches transformer cross-attention and supports multiple heads.

### Additive Attention

```text
score = w^T tanh(Wq(query) + Wk(node))
```

This is slightly more expensive but can be easier to initialize close to
uniform pooling. Initializing the final vector `w` close to zero starts the
model near `typed_mean` while still allowing gradients to learn nonuniform
weights.

For the first implementation, single-head additive attention is the simplest
debuggable option. Multi-head scaled dot-product attention can follow.

## Preserving A Fair Baseline

The learned pool introduces parameters that `typed_mean` does not have. Report:

- total actor parameter count;
- GNN parameter count;
- pooling parameter count;
- shared scorer parameter count;
- training throughput and peak memory.

Where practical, keep the output of each typed pool equal to `node_dim`, so the
input dimension of the shared scoring MLP remains unchanged. This isolates the
pooling mechanism from scorer width.

## Do-Nothing Action

Action 0 has no physical node set. Keep the existing behavior:

```text
global graph embedding -> dedicated action-0 head -> action-0 logit
```

Do not invent a node heatmap for action 0. Its decision is global by design.

If `candidate_action_do_nothing_head = false`, action 0 can retain the existing
empty local context and learned scalar bias, but its attention weights should
be reported as empty.

## Heatmap And Diagnostic API

Add an explicit evaluation method rather than changing PPO return values:

```python
Actor.get_candidate_attention(observation, action_ids=None)
```

Suggested output:

```text
logits:                    [batch, n_actions]
attention weights by type:[batch, n_actions, max_items]  # affected scope
or
attention weights:        [batch, n_actions, n_nodes]    # all-node scope
hardcoded affected mask:  [n_actions, n_nodes]
```

Allow `action_ids` to restrict diagnostics to:

- the selected action;
- the top-k policy actions;
- a user-specified action;
- the best greedy action from a comparison policy.

Never log the full `[episodes, steps, actions, nodes]` tensor during training.
It would be too large. Compute detailed heatmaps only for selected evaluation
states and aggregate lightweight metrics during ordinary evaluation.

## Mapping Heatmaps Back To The Grid

Use the existing graph spec metadata:

```text
node_type
node_type_names
node_substation_ids
line_id_to_node_row
gen_id_to_node_row
load_id_to_node_row
substation_busbar_node_rows
```

For each node row, recover a display label such as:

```text
busbar: substation 6, busbar 1
line: line 17
generator: generator 4
load: load 9
```

A graph heatmap can encode:

- node color: attention weight;
- node size: attention weight or rho for line nodes;
- node border: hardcoded affected versus learned context;
- line style: graph relation or electrical connection;
- annotation: selected action and action logit.

For line nodes, show both attention weight and current `rho`. This makes it
possible to determine whether the policy attends to overloaded lines or merely
to nodes identified by the action mapping.

## Quantitative Comparison With The Hardcoded Mapping

### Hardcoded Attention Mass

```text
sum of attention weights assigned to hardcoded affected nodes
```

This is the primary agreement metric for soft-prior and free attention.

### Top-K Precision And Recall

Treat the highest-weight nodes as the learned set:

```text
precision@k = fraction of top-k nodes in the hardcoded set
recall@k    = fraction of hardcoded nodes recovered in the top-k set
```

Choose `k` equal to the number of hardcoded affected nodes for a direct set
comparison.

### Distribution Divergence

Convert the hardcoded mask into a uniform distribution over affected nodes and
compare it with learned attention using:

- Jensen-Shannon divergence;
- cross-entropy;
- total variation distance.

Do not use raw KL divergence without epsilon smoothing because the hardcoded
distribution contains exact zeros.

### Attention Entropy

Measure whether attention is concentrated or diffuse:

```text
entropy = -sum_i alpha_i * log(alpha_i)
```

Report entropy by node type and action category.

### Causal Deletion Test

Attention weights are not automatically causal explanations. Test them by
masking node embeddings at evaluation time:

1. remove the highest-attention nodes;
2. remove the same number of lowest-attention nodes;
3. recompute the action logits;
4. compare the change in the selected-action logit and action ranking.

If removing high-attention nodes has a substantially larger effect, the
heatmap is more likely to reflect actual decision dependence.

## Implementation Roadmap

### Phase 1: Refactor Pooling Behind One Interface

Goal:

```text
mean, typed_mean, and future attention modes share one pooling API
```

Tasks:

1. Move pooling logic out of `CandidateActionScorer` into
   `alg/mappo/action_pooling.py`.
2. Implement `CandidateActionMeanPool` for the current modes.
3. Keep state-dict names compatible where practical.
4. Verify old candidate configs produce numerically identical logits after the
   refactor.

Acceptance criteria:

- all existing 48 tests still pass;
- existing `mean` and `typed_mean` checkpoints load;
- no parameter changes for the old modes;
- no change to PPO code.

### Phase 2: Implement Masked Typed Attention

Tasks:

1. Add `CandidateActionAttentionPool`.
2. Generate action queries from `[global_context, action_features]`.
3. Reuse existing padded per-type indices and masks.
4. Implement safe masked normalization for empty node types.
5. Return typed contexts with the same dimensions as `typed_mean`.
6. Add `typed_attention` to the config parser.
7. Keep action 0 on the existing dedicated head.

Acceptance criteria:

- logits have shape `[batch, n_actions]`;
- weights sum to one for nonempty typed candidate sets;
- weights are zero outside each hard mask;
- empty sets return finite zero contexts;
- gradients reach query, key, value, scorer, and GNN parameters;
- checkpoint save/load is exact.

### Phase 3: Add Attention Diagnostics

Tasks:

1. Add `return_weights=False` to the pooling module.
2. Add `Actor.get_candidate_attention()`.
3. Build node-row-to-physical-label helpers.
4. Add aggregate agreement and entropy metrics.
5. Add a small analysis script or notebook for graph heatmaps.

Acceptance criteria:

- normal training does not retain attention maps;
- selected-action heatmaps can be generated from a checkpoint;
- hardcoded nodes are visually distinguishable;
- metrics work for batched and unbatched evaluation.

### Phase 4: Implement Soft-Prior Attention

Tasks:

1. Build `affected_node_mask [n_actions, n_nodes]` from action metadata.
2. Add all-node typed key/value banks.
3. Add the configurable prior bias.
4. Implement action chunking.
5. Measure peak memory on bus14 and WCCI reduced action spaces.

Acceptance criteria:

- `prior_bias = 0` matches free all-node attention;
- increasing the prior bias increases hardcoded attention mass in a controlled
  synthetic test;
- action chunking does not change logits;
- no out-of-memory failure for the target reduced WCCI action space.

### Phase 5: Implement Free Attention

Tasks:

1. Expose `scope = "all"` as an explicit configuration.
2. Require an action-specific query mode.
3. Add tests preventing `global_only` queries with all-node scope unless the
   user explicitly overrides the warning.
4. Compare attention overlap with the hardcoded mapping without adding an
   alignment loss.

Do not add an attention-alignment loss in the first free-attention experiment.
The purpose is to observe whether the mapping is rediscovered naturally.

### Phase 6: Optional Alignment Regularization

Only if free attention is unstable or ignores obvious physical structure, add:

```text
attention_alignment_coef
```

Possible loss:

```text
alignment_loss = cross_entropy(
    learned_attention,
    uniform_distribution_over_hardcoded_nodes,
)
```

Integrating this loss changes the PPO actor objective, so it must be logged
separately and tested carefully. Keep the default coefficient at zero.

### Phase 7: Sparse Normalization Or Action Nodes

After Stages 1-3 are evaluated:

1. add `entmax15` as a sparse attention normalizer; or
2. prototype candidate-action nodes on bus14 only.

Do not develop both simultaneously. Entmax is the smaller extension; action
nodes are a separate architecture study.

## Files Expected To Change

### `alg/mappo/agent.py`

- route candidate scoring through the new pooling module;
- expose attention diagnostics;
- preserve action-0 bonus and categorical policy behavior.

### `alg/mappo/action_pooling.py`

- new pooling classes;
- safe masked softmax;
- typed attention;
- all-node attention and chunking;
- optional attention-weight return object.

### `common/action_metadata.py`

- optionally expose a dense affected-node mask;
- validate that masks agree with padded typed indices;
- preserve reduced-action original IDs.

### `alg/mappo/config.py`

- add attention pooling and scope flags;
- keep all existing defaults unchanged.

### `tests/test_candidate_action_scoring.py`

- mean-pooling compatibility;
- masked attention normalization;
- soft-prior behavior;
- chunking equivalence;
- gradients and checkpoint round trips.

### Analysis tooling

Add either:

```text
analysis/metrics/notebooks/action_attention/
```

or a reusable helper plus notebook under the existing action-distribution
analysis area.

## Test Plan

### Unit Tests

1. Uniform synthetic scores reproduce a mean pool.
2. Hard-masked weights are exactly zero outside affected nodes.
3. Nonempty attention rows sum to one.
4. Empty typed sets produce zero contexts and weights.
5. Action 0 produces no local attention map.
6. Soft-prior bias increases attention on the hardcoded set.
7. Action chunking matches unchunked logits and gradients.
8. Node-order permutations preserve pooled results when metadata is permuted
   consistently.
9. Multi-head outputs have stable dimensions.
10. State-dict round trips preserve logits and attention maps.

### Actor Integration Tests

1. Sample and score actions with PPO's existing API.
2. Evaluate deterministically and stochastically.
3. Apply `init_do_nothing_prob`.
4. Apply the runtime action-0 logit bonus.
5. Test shared and non-shared actor GNNs.
6. Test GINE and sparse-transformer node embeddings.

### Environment Tests

1. Build metadata for all bus14 actions.
2. Build metadata for a reduced WCCI action space.
3. Verify every hardcoded index is inside the local graph.
4. Verify reduced exposed IDs map to the correct original action IDs.

### Training Smokes

Run seed 0 with:

```text
typed_mean
masked typed_attention
soft-prior typed_attention
free typed_attention
```

Start with `100k-200k` bus14 timesteps and verify:

- finite PPO losses;
- finite attention weights;
- nonzero gradients;
- action diversity;
- checkpoint save/load;
- acceptable throughput and memory.

## Experimental Progression

### Experiment A: Does Learned Weighting Help?

```text
typed_mean
masked typed_attention
```

This changes only uniform versus learned weights inside the same hardcoded
candidate sets.

### Experiment B: Is The Hardcoded Mapping Complete?

```text
masked typed_attention
soft-prior typed_attention
```

Compare survival and hardcoded attention mass.

### Experiment C: Does The Model Rediscover Physics?

```text
soft-prior typed_attention with beta > 0
free typed_attention with beta = 0
```

Compare overlap, entropy, and causal deletion metrics.

### Experiment D: Is Sparsity Useful?

```text
softmax attention
entmax attention
```

Compare performance and heatmap stability, not only visual sharpness.

### Seed Strategy

For every stage:

1. run seed 0 as a smoke;
2. promote viable variants to seeds 0, 1, and 2;
3. keep `chronic_split_seed = 0` fixed for paired comparisons;
4. evaluate all complete test chronics deterministically.

## Recommended First Deliverable

Implement only:

```text
candidate_action_pool = "typed_attention"
candidate_action_attention_scope = "affected"
candidate_action_attention_heads = 1
candidate_action_attention_query = "global_action_features"
candidate_action_attention_normalizer = "softmax"
```

Keep:

```text
candidate_action_do_nothing_head = true
intervention_gate = false
```

This first deliverable directly tests the main hypothesis:

```text
Can an action-conditioned query improve classification by learning which
physically affected line, busbar, load, or generator matters most in the
current grid state?
```

After that result is stable, implement the soft-prior scope. That is the stage
that enables the most informative comparison between learned heatmaps and the
current hardcoded action-node mapping.
