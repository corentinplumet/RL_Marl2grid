# Heterogeneous Line Graphs and Candidate-Action Scoring

This note summarizes how the `heterogeneous_line` representation can make the
actor GNN act more like an action classifier, instead of only acting as a graph
encoder followed by an arbitrary action-ID MLP.

The high-level goal is:

```text
current:
  graph -> GNN -> one global embedding -> MLP -> logits over action IDs

target:
  graph -> GNN -> node/line embeddings
        -> gather objects touched by each action
        -> score each action from its physical graph context
```

The PPO objective can stay unchanged. The actor still outputs categorical
logits. Only the way those logits are parameterized changes.

## Implementation Status

The minimal implementation described in this roadmap is available on branch
`codex/gnn-candidate-action-scoring`.

Implemented:

- stable physical-ID-to-node-row maps in `heterogeneous_line` graph specs;
- one-time decoding of full or reduced per-agent discrete action spaces;
- strict action-to-busbar/line/load/generator metadata validation;
- `forward_with_nodes()` for GINE/GCN/GAT/GraphSAGE and sparse-transformer
  graph encoders;
- simple mean and typed mean candidate pools;
- one shared candidate scorer for all non-idle actions;
- a dedicated global-context head for action 0;
- compatibility with `init_do_nothing_prob` and the runtime action-0 bonus;
- focused metadata, gradient, encoder, and Actor integration tests.

Intentionally deferred:

- attention candidate pooling;
- auxiliary line-risk or intervention losses;
- combining candidate pooling with the hierarchical intervention gate.

The current actor remains the default. Enable the new path with:

```text
actor_encoder = "gnn"
gnn_graph_type = "heterogeneous_line"
gnn_include_neighbors = true
actor_action_head = "candidate_pool"
candidate_action_pool = "typed_mean"
candidate_action_use_features = true
candidate_action_do_nothing_head = true
intervention_gate = false
```

For a paired command-line run from an existing heterogeneous-line config:

```bash
python Topology_Task/run_from_config.py \
  Topology_Task/configs/gnn_graph_screening/stage1_representation_normalization/gs_s1_hetero_line_n0_none_s0.toml \
  --actor-action-head candidate_pool \
  --candidate-action-pool typed_mean \
  --candidate-action-use-features true \
  --candidate-action-do-nothing-head true
```

Keep the same config without these overrides as the global-MLP control. The
strict metadata builder requires neighboring lines to be present in each local
agent graph, hence `gnn_include_neighbors = true`.

## Current Actor

The current actor architecture in `alg/mappo/agent.py` is:

```text
grid observation
  -> graph builder
  -> GraphAndFlatEncoder
  -> GraphEncoder or SparseGraphTransformerEncoder
  -> graph-level readout
  -> actor MLP
  -> one logit per discrete action
```

In code, the important path is:

```text
Actor._encode(x)
  -> self.encoder(x, graph_key="graph")
  -> graph_embedding

self.actor(graph_embedding)
  -> logits over action IDs
```

This means the GNN produces a single graph embedding. The final MLP then learns
to map that embedding to action IDs. This works, but it treats actions mostly
as arbitrary classes. The model must learn that action `173` is useful from
data, even though the action ID itself does not expose which substation, line
endpoint, load, or generator it affects.

## Existing Heterogeneous + Line Representation

The `heterogeneous_line` representation already exists in `common/graph.py`.
It creates explicit typed graph nodes:

```text
busbar nodes
generator nodes
load nodes
transmission-line nodes
optional hierarchy nodes added later by the encoder
```

The static graph spec contains useful metadata:

```text
node_ids
entity_ids
node_type
node_type_names
node_substation_ids
line_ids
gen_ids
load_ids
controlled_nodes
controlled_node_mask
edge_type
edge_type_names
edge_line_ids
edge_line_endpoint
edge_asset_kind
edge_asset_ids
```

The runtime graph observation contains the dynamic tensors:

```text
node_features
edge_features
node_mask
edge_mask
node_type
edge_type
controlled_node_mask
```

The important difference from the busbar graph is the explicit line node.
Instead of representing a powerline only as an edge between two busbars, each
transmission line has its own node embedding. That node can carry features such
as:

- line flow;
- rho or thermal loading;
- line status;
- time before cooldown;
- time next maintenance;
- maintenance duration;
- endpoint context through message passing.

After message passing, a line embedding can represent:

```text
this line is stressed, connected to these busbars, near these assets,
and affected by this local grid context
```

This is useful because many topology actions are line-driven. If a line is
overloaded or close to maintenance or attack risk, actions around neighboring
substations should become easier to score.

## Why The Current Readout Loses Information

The current GNN readout compresses all node embeddings into one graph vector:

```text
node embeddings -> mean/sum/max/virtual-node pooling -> graph embedding
```

Then the actor MLP produces all logits:

```text
logits = MLP(graph_embedding)
```

This has two weaknesses:

1. The actor head does not know which physical objects each action touches.
2. The final layer is tied to action IDs, so it can memorize ID-specific
   behavior instead of learning a reusable physical scoring rule.

For WCCI or reduced action spaces, this is especially limiting because the
action space is large and the action IDs are not physically meaningful.

## Desired Actor

The proposed actor scores each candidate action from the graph objects it
touches:

```text
grid observation
  -> heterogeneous line graph
  -> GNN message passing
  -> node and line embeddings
  -> for each candidate action: gather affected embeddings
  -> candidate-action pooling
  -> shared scoring network
  -> one logit per candidate action
```

The action logit becomes:

```text
logit(action a) = score(global_context, local_context(a), action_features(a))
```

where:

```text
global_context:
  graph-level embedding, as today

local_context(a):
  pooled embeddings of the nodes and lines associated with action a

action_features(a):
  static features describing what action a physically does
```

This makes the classification decision depend on the physical neighborhood of
the action.

## Candidate-Action Metadata

The central new object is a static metadata table for each agent action space.
For every discrete action ID, precompute what the action physically touches.

One row per action:

```text
agent_id
action_id
is_do_nothing
affected_substations
affected_busbar_node_ids
affected_line_node_ids
affected_line_endpoint_ids
affected_generator_node_ids
affected_load_node_ids
n_topology_changes
n_line_endpoint_changes
n_generator_changes
n_load_changes
local_agent_or_zone
```

For a first implementation, the most important fields are:

```text
action_id
is_do_nothing
affected_substations
affected_line_ids
affected_gen_ids
affected_load_ids
affected_busbar_ids
action_feature_vector
```

The mapping should be stable and deterministic. It should be created once when
the environment/action space is built, not recomputed during each rollout step.

## Mapping Actions To Graph Nodes

The `heterogeneous_line` graph spec already gives stable IDs for graph rows:

```text
line_ids -> rows of transmission-line nodes
gen_ids  -> rows of generator nodes
load_ids -> rows of load nodes
node_substation_ids -> substation attached to each node row
node_type -> busbar / generator / load / line
```

The action metadata should use the same indexing convention. For each action:

1. Decode the Grid2Op topology action.
2. Identify which topology-vector positions change.
3. Map those positions to physical objects:
   - line origin endpoint;
   - line extremity endpoint;
   - generator;
   - load.
4. Map those physical objects to graph-node rows in the agent graph spec.
5. Store padded index arrays and masks.

Example conceptual metadata for action `a`:

```text
action a:
  line_ids = [4, 17]
  gen_ids = []
  load_ids = [2]
  substations = [6]
  busbar_node_rows = [12, 13]
  line_node_rows = [22, 35]
  load_node_rows = [41]
```

The candidate scorer then gathers:

```text
node_embeddings[line_node_rows]
node_embeddings[load_node_rows]
node_embeddings[busbar_node_rows]
```

and pools them into one action context.

## Candidate-Action Pooling Options

### Version 1: Simple Mean Pool

The simplest version pools all affected nodes together:

```text
local_context(a) = mean(embeddings of nodes touched by action a)
```

Advantages:

- easiest to implement;
- smallest scorer input;
- good first smoke test.

Disadvantages:

- mixes line, load, generator, and busbar roles;
- less interpretable.

### Version 2: Typed Mean Pool

A stronger first serious version keeps node types separated:

```text
local_context(a) = concat(
    mean(affected_busbar_embeddings),
    mean(affected_line_embeddings),
    mean(affected_load_embeddings),
    mean(affected_generator_embeddings),
    mean(affected_substation_or_summary_embeddings)
)
```

Advantages:

- still simple;
- interpretable;
- lets the scorer distinguish "this is a stressed line" from "this is a load";
- better aligned with the heterogeneous graph.

This should probably be the first implementation used for real experiments.

### Version 3: Attention Pool

Attention pooling lets the action decide which touched objects matter most:

```text
query(a) = action_embedding(a) or MLP(action_features(a))

local_context(a) = attention_pool(
    affected_node_embeddings,
    query = query(a)
)
```

Advantages:

- can focus on the most relevant touched line or endpoint;
- potentially more expressive.

Disadvantages:

- more moving parts;
- harder to debug;
- should come after typed mean pooling works.

## Shared Action Scorer

Once the local action context is built, use the same scorer for every action:

```text
logit_a = shared_mlp(
    concat(global_graph_embedding, local_context(a), action_features(a))
)
```

This changes the learning problem. The policy no longer has to memorize only:

```text
action ID 173 was good in this kind of state
```

It can learn a reusable rule:

```text
actions touching substations near stressed lines are better,
especially when they move the relevant asset or line endpoint
```

That is what makes the GNN more classifier-like. The classification decision is
conditioned on the physical graph neighborhood of each action.

## Do-Nothing Action

Action `0` does not touch a physical object. Treat it separately:

```text
logit(action 0) = score_do_nothing(global_graph_embedding, action0_embedding)
```

or include it in the same scorer with an empty local context:

```text
local_context(action 0) = learned_empty_action_context
```

The first option is cleaner and easier to interpret. It makes action `0`
classify whether the grid is safe enough to avoid intervention.

## Proposed Code Structure

### New Module: `common/action_metadata.py`

Responsible for building static action-to-object mappings.

Suggested objects:

```python
@dataclass
class ActionGraphMetadata:
    action_features: torch.Tensor
    busbar_indices: torch.Tensor
    busbar_mask: torch.Tensor
    line_indices: torch.Tensor
    line_mask: torch.Tensor
    load_indices: torch.Tensor
    load_mask: torch.Tensor
    generator_indices: torch.Tensor
    generator_mask: torch.Tensor
    substation_indices: torch.Tensor
    substation_mask: torch.Tensor
    is_do_nothing: torch.Tensor
```

The tensors should be per agent:

```text
action_features: [n_actions, action_feature_dim]
line_indices:    [n_actions, max_lines_per_action]
line_mask:       [n_actions, max_lines_per_action]
```

Use `-1` for padded indices and `0` in the corresponding mask.

### Extend Graph Specs

The action scorer needs stable node-row mappings. The current
`heterogeneous_line` spec already contains most of this:

```text
line_ids
gen_ids
load_ids
node_type
node_substation_ids
```

If needed, add explicit row maps to the graph spec:

```text
line_id_to_node_row
gen_id_to_node_row
load_id_to_node_row
substation_id_to_node_rows
busbar_id_to_node_row
```

This avoids recomputing searches in the action-metadata builder.

### Extend GNN Encoder Output

Today `GraphEncoder.forward()` returns only:

```text
graph_embedding: [batch, gnn_out_dim]
```

Candidate-action scoring also needs node embeddings:

```text
node_embeddings: [batch, n_nodes, hidden_dim]
graph_embedding: [batch, gnn_out_dim]
```

Add a flag or a new method rather than changing all existing behavior:

```python
graph_embedding = encoder(graph_obs)

graph_embedding, node_embeddings = encoder.forward_with_nodes(graph_obs)
```

This keeps current configs backwards-compatible.

### New Actor Head: `CandidateActionScorer`

Add a small module, probably in `alg/mappo/agent.py` or a new file such as
`alg/mappo/action_scorer.py`.

Conceptual API:

```python
class CandidateActionScorer(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        node_dim: int,
        action_feature_dim: int,
        metadata: ActionGraphMetadata,
        pool_mode: str = "typed_mean",
    ):
        ...

    def forward(self, graph_embedding, node_embeddings):
        # returns logits [batch, n_actions]
        ...
```

The actor then becomes:

```text
if actor_action_head == "mlp":
    current behavior

if actor_action_head == "candidate_pool":
    graph_embedding, node_embeddings = encoder.forward_with_nodes(...)
    logits = candidate_action_scorer(graph_embedding, node_embeddings)
```

### Config Flags

Add flags in `alg/mappo/config.py`:

```text
--actor-action-head
  choices: mlp, candidate_pool
  default: mlp

--candidate-action-pool
  implemented choices: mean, typed_mean
  default: typed_mean

--candidate-action-use-features
  default: true

--candidate-action-do-nothing-head
  default: true
```

`attention` remains the planned third pooling mode after the typed-mean
baseline has been trained and checked.

Keep defaults equal to the current behavior.

## Implementation Roadmap

### Phase 0: Decide Scope

Start with the smallest useful target:

```text
environment: bus14 first
graph type: heterogeneous_line
actor encoder: gnn
critic encoder: unchanged
action head: candidate_pool
pooling: typed_mean
PPO objective: unchanged
intervention gate: disabled at first
```

Do not add auxiliary losses or attention pooling yet. They can come later.

### Phase 1: Action Metadata Prototype

Goal:

```text
for each agent and each discrete action ID, know which graph nodes it touches
```

Tasks:

1. Add `common/action_metadata.py`.
2. Create a builder that receives:
   - Grid2Op environment;
   - agent action spaces;
   - graph specs;
   - observation/action domain metadata already available in the wrapper.
3. Decode every discrete topology action.
4. Extract touched physical objects.
5. Convert touched physical objects to graph-node rows.
6. Return padded tensors and masks.

Validation:

- action `0` has empty touched-object masks;
- every non-empty node index is inside `[0, n_nodes)`;
- every line/load/generator touched by an action exists in the graph spec;
- run this for `bus14` and `bus36_wcci`.

Deliverable:

```text
ActionGraphMetadata for each agent
```

### Phase 2: Encoder Node Embedding API

Goal:

```text
allow the actor to access both global graph embedding and per-node embeddings
```

Tasks:

1. Add `forward_with_nodes()` to `GraphEncoder`.
2. Add the same method to `SparseGraphTransformerEncoder` only if needed later.
3. Add a matching method to `GraphAndFlatEncoder`.
4. Preserve current `forward()` exactly for existing configs.

Important detail:

The node embeddings used for action scoring should be the post-message-passing
embeddings before graph readout:

```text
node_features
  -> node pre-encoder
  -> GNN message passing
  -> node_embeddings
  -> graph pooling
  -> graph_embedding
```

Validation:

- `forward()` output remains bitwise or numerically equivalent for old configs;
- `forward_with_nodes()` returns shapes:

```text
graph_embedding: [batch, gnn_out_dim]
node_embeddings: [batch, n_nodes, hidden_dim]
```

### Phase 3: Candidate Pooling Module

Goal:

```text
pool the node embeddings associated with each action
```

Tasks:

1. Implement masked gather:

```text
node_embeddings: [batch, n_nodes, hidden_dim]
indices:         [n_actions, max_items]
mask:            [n_actions, max_items]
output:          [batch, n_actions, hidden_dim]
```

2. Implement typed pools:

```text
busbar_pool
line_pool
load_pool
generator_pool
substation_pool
```

3. Concatenate typed pools:

```text
local_context: [batch, n_actions, n_types * hidden_dim]
```

4. Concatenate global and action features:

```text
global_context expanded: [batch, n_actions, gnn_out_dim]
action_features expanded: [batch, n_actions, action_feature_dim]
```

5. Score each action:

```text
logits = shared_mlp(concat(...)).squeeze(-1)
```

Validation:

- no NaNs when an action touches no nodes;
- action `0` logit is finite;
- logits shape is `[batch, n_actions]`;
- gradients flow into the GNN node embeddings.

### Phase 4: Integrate Into `Actor`

Goal:

```text
swap only the actor head, not the training loop
```

Tasks:

1. Add config flag `actor_action_head`.
2. In `Actor.__init__`, if:

```text
actor_encoder == "gnn"
actor_action_head == "candidate_pool"
```

then build:

```text
self.encoder = GraphAndFlatEncoder(...)
self.actor = CandidateActionScorer(...)
```

3. In `get_discrete_action()`, route to:

```text
logits = self._candidate_action_logits(x)
```

4. In `get_eval_discrete_action()`, use the same logits.
5. Keep `action0_logit_bonus` compatible.
6. Initially reject `intervention_gate=True` with candidate pooling, or support
   it later as a separate phase.

Validation:

- old configs still instantiate the old MLP head;
- candidate-pool configs instantiate only for GNN observations;
- PPO still receives `Categorical(logits=logits)`;
- checkpoint save/load works.

### Phase 5: Tests

Add focused tests before launching training.

Suggested tests:

1. Metadata construction:
   - action `0` has empty masks;
   - nonzero actions have at least one touched object when expected;
   - all indices are in range.

2. Encoder compatibility:
   - old `GraphEncoder.forward()` shape unchanged;
   - `forward_with_nodes()` returns expected shapes.

3. Candidate scorer tensor test:
   - fake `node_embeddings`;
   - fake metadata;
   - output logits shape `[batch, n_actions]`;
   - backprop works.

4. Actor smoke test:
   - build a `bus14` env with `heterogeneous_line`;
   - instantiate `Actor`;
   - call `get_discrete_action()`;
   - call `get_eval_discrete_action()`;
   - verify no NaNs.

5. One short training smoke:

```bash
python Topology_Task/main.py \
  --actor-encoder gnn \
  --gnn-graph-type heterogeneous_line \
  --actor-action-head candidate_pool \
  --candidate-action-pool typed_mean \
  --total-timesteps 200000 \
  --track false
```

### Phase 6: First Experiments

Run in increasing order of risk:

1. `bus14`, seed 0, short smoke.
2. `bus14`, seed 0, full budget.
3. `bus14`, seeds 0/1/2.
4. WCCI reduced action space, seed 0.
5. WCCI reduced action space, seeds 0/1/2.
6. WCCI full action space only if memory and runtime are acceptable.

Primary comparison:

```text
heterogeneous_line + current global MLP head
vs
heterogeneous_line + candidate_pool typed_mean head
```

Keep the critic, PPO hyperparameters, reward, normalization, chronic split, and
evaluation protocol unchanged.

## Optional Phase: Auxiliary Heads

Auxiliary heads can make the GNN embeddings more decision-relevant, but they
should come after the candidate scorer is working.

Possible auxiliary tasks:

- line-risk head: predict which lines will overload soon;
- substation-risk head: predict which substations are likely to need action;
- intervention head: predict whether any non-zero action is needed;
- candidate-improvement head: predict whether a candidate action improves local
  rho or survival compared with do-nothing.

These can use data already available from:

```text
full-test traces
greedy reduced-action evaluation
teacher-student action-outcome datasets
```

The safest first auxiliary head is probably line-risk prediction because line
nodes already carry rho/status/cooldown/maintenance features.

## Risks And Mitigations

### Risk: Action Metadata Is Wrong

If action-to-node mapping is wrong, the model will learn misleading local
contexts.

Mitigation:

- create debug tables for randomly sampled actions;
- print decoded Grid2Op action strings next to touched graph nodes;
- add tests against known small action examples.

### Risk: Too Much Memory On WCCI

Candidate contexts have shape:

```text
[batch, n_actions, context_dim]
```

For large WCCI action spaces this can be heavy.

Mitigation:

- start on reduced action spaces;
- compute logits in chunks over actions;
- keep typed mean pooling simple before attention pooling;
- optionally cache static action features on CPU and move once at actor init.

### Risk: Do-Nothing Becomes Poorly Calibrated

If action `0` uses an empty context while nonzero actions use rich local
contexts, the do-nothing logit might be miscalibrated.

Mitigation:

- use a dedicated do-nothing scorer;
- keep `action0_logit_bonus` support;
- compare action-0 frequency and survival against the current baseline.

### Risk: Policy Becomes Less Comparable

Changing the actor head changes capacity, not only inductive bias.

Mitigation:

- compare parameter counts;
- keep GNN hidden dimension and PPO settings fixed;
- compare against a width-matched MLP head if needed.

## Minimal First Implementation

The smallest useful implementation is:

```text
graph type:
  heterogeneous_line

encoder:
  existing GINE GraphEncoder

new encoder API:
  forward_with_nodes()

action metadata:
  line/load/generator/busbar node-row masks

pooling:
  typed mean

scorer:
  shared MLP over [global graph embedding, typed local pools, action features]

do-nothing:
  dedicated scalar head from global graph embedding

training:
  unchanged PPO
```

This is the cleanest version to test the hypothesis:

```text
Can the GNN classify actions better when each action logit is conditioned on
the graph nodes and line nodes that the action physically affects?
```

## Main Benefit

The current GNN actor is mostly:

```text
graph encoder + action-ID classifier
```

The proposed actor is:

```text
graph encoder + physical candidate-action classifier
```

This should improve interpretability and transfer, especially for larger
environments such as WCCI, because the model scores actions from the physical
objects they affect rather than from action IDs alone.
