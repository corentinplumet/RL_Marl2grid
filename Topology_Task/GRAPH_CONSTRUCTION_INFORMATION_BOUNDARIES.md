# Local Graph Construction and Information Boundaries

## Status and purpose

This document is the authoritative definition of which physical objects may
appear in each agent's local graph. These rules are enforced during graph
construction, before the actor encoder and before graph-level pooling.

The primary requirement is to prevent state leakage between agents. Pooling is
an architectural choice and cannot be used as the security boundary: a node
excluded from a readout may still transmit information through message passing.
For that reason, private neighboring assets are either not constructed at all,
or their inactive fixed-shape candidate rows are stripped of state and have no
active edge into the agent graph.

The global `state` graph is intentionally unchanged. It represents the whole
grid for centralized training components such as the critic. The restrictions
below apply to each decentralized actor's local graph.

## Terminology

- A **controlled substation** belongs to the agent's observation/action domain.
- A **boundary** or **shared line** has one endpoint in the controlled domain
  and one endpoint in another agent's domain.
- A **neighboring busbar** is the far-end busbar to which a live boundary line
  is currently attached.
- A **neighboring asset** is a generator or load at a substation outside the
  agent's controlled domain.

## Fixed shapes, masks, and what "zeroed" means

Local graph tensors must keep a constant shape throughout an episode. The graph
spec therefore enumerates candidate nodes and candidate edges for topology
assignments that might become active later. Candidate membership in the tensor
does not by itself make a node observable.

At every environment step, construction proceeds in this order:

1. Build the current node and edge feature matrices.
2. Compute a preliminary `edge_mask` from line status and the current
   `topo_vect` bus assignments.
3. Compute `node_mask`. Every controlled node has `node_mask = 1`. A contextual
   busbar has `node_mask = 1` only if an active physical relation connects it to
   the controlled domain. Self and same-substation relations cannot establish
   visibility.
4. For every row with `node_mask = 0`, assign
   `node_features[row, :] = 0`.
5. Deactivate every edge whose source or destination has `node_mask = 0`, then
   assign `edge_features[edge, :] = 0` for every inactive edge.

"The node feature vector is zeroed" therefore means that every dynamic and
role feature supplied to the encoder for that candidate row is numerically
zero. For the busbar schema this includes `gen_p`, `gen_theta`, `load_p`,
`load_theta`, substation cooldown, and `domain_mask`. For the heterogeneous
schemas it includes the type indicator columns, `p`, `theta`, cooldown,
`connected`, and `domain_mask`.

The static graph spec is not erased: `node_ids`, `node_type`, `edge_index`,
`edge_type`, `node_mask`, and `controlled_node_mask` still describe the fixed
candidate layout. This can reveal static grid structure, but not the hidden
candidate's dynamic electrical state. Since all incident edges are inactive,
the placeholder also cannot affect visible nodes through node-pre-encoder
biases, node-ID embeddings, type embeddings, self edges, or same-substation
edges. The readout receives `node_mask` as an additional exclusion mechanism.

The explicit compatibility setting `gnn_context_requires_connection = false`
intentionally restores the legacy behavior: contextual rows remain visible and
are not zeroed. It exists only for historical control runs and does not provide
the leakage guarantees described here.

### Central invariant: an unconnected contextual node is unused

For a contextual neighboring node, "unconnected" means that no active physical
relation connects it to the agent's controlled domain. Such a node receives
`node_mask = 0` and is completely excluded from both computations:

- **No message passing:** every edge incident to the node receives
  `edge_mask = 0`, so the encoder removes those edges before applying any graph
  convolution or attention layer. The node cannot send a message, receive a
  message, or influence another node through an optional self or
  same-substation relation.
- **No pooling:** every supported readout applies `node_mask`. For a mean
  readout, the node contributes neither an embedding to the numerator nor one
  unit to the denominator. It is therefore equivalent to computing the mean
  over visible nodes only, not adding a zero vector to the mean. Max and
  attention-based readouts likewise exclude the masked row.
- **No dynamic state:** its complete `node_features` row is zero, as described
  above.

For example, if two visible nodes have embeddings `x1` and `x2` and one fixed
candidate row is unconnected, the readout is:

```text
(x1 + x2) / 2
```

and not:

```text
(x1 + x2 + 0) / 3
```

This invariant concerns **contextual neighboring nodes**, not every object that
is electrically disconnected:

- A controlled busbar, generator, or load remains visible because it belongs
  to the agent's own domain. Its unavailable physical attachments are masked,
  but the controlled node itself is retained.
- A disconnected shared-line node in `heterogeneous_line` also remains visible
  because every line touching the controlled domain belongs to that agent's
  observation/action graph. It reports `line_status = 0`, while all of its
  busbar attachment edges are inactive.
- The legacy control setting `gnn_context_requires_connection = false`
  intentionally disables this invariant for contextual rows so old behavior
  can be reproduced.

## Exact edge-activation rules

### Physical-line edges in `bus` and `heterogeneous`

For every included line, the fixed spec enumerates every candidate pair
`(origin busbar, extremity busbar)` and both directed edges for that pair. A
candidate physical edge is preliminarily active exactly when:

```text
line_status[line] > 0
and current_origin_bus[line] == candidate_origin_bus
and current_extremity_bus[line] == candidate_extremity_bus
```

It remains active in the final graph only if both endpoint nodes have
`node_mask = 1`. Thus, with bidirectional physical relations, a live line has
exactly two active directed edges: origin-to-extremity and
extremity-to-origin. A disconnected line has none. The line feature vector is
copied onto the active pair; all inactive candidate edge vectors are zero.

### Generator/load attachment edges in heterogeneous schemas

For every locally constructed generator or load, candidate attachment edges
are enumerated for every busbar at its substation. Depending on the configured
direction, the spec contains asset-to-busbar, busbar-to-asset, or both. A
candidate attachment is active exactly when:

```text
current_asset_bus[asset] == candidate_bus
```

Only controlled generators and loads are constructed in local heterogeneous
graphs, so no such edge can originate from or reach a neighboring private
asset. A disconnected controlled asset has no active attachment edge, while
its controlled node remains in the agent graph with `connected = 0` and zero
`p`/`theta`.

### Line-node attachment edges in `heterogeneous_line`

Every line touching the controlled domain is a line node. Candidate attachment
edges are created only for endpoints whose substation is controlled by the
agent. For each such endpoint, candidates are enumerated for all of its
busbars. A candidate is active exactly when:

```text
line_status[line] > 0
and current_endpoint_bus[line] == candidate_bus
```

With bidirectional line-node relations:

- a live boundary line has two active directed edges: controlled busbar to line
  node and line node to controlled busbar;
- a live internal line has four active directed edges, two at each controlled
  endpoint;
- a disconnected line has no active attachment edge.

With one-way relations those counts are respectively one, two, and zero. The
line node itself remains visible and controlled even when disconnected; only
its attachment edges are removed. No candidate edge to the non-controlled
endpoint is constructed.

### Optional structural edges

Self and same-substation busbar edges are structural rather than electrical.
They cannot make a contextual node visible. They are active only when both
endpoint nodes are visible, so an inactive contextual placeholder has no active
structural or physical edge.

## Required construction by representation

### 1. Busbar representation (`bus`)

The local graph contains:

- every busbar in the controlled substations;
- every line touching a controlled substation;
- for each live boundary line, the single far-end busbar to which that line is
  currently attached.

The neighboring busbar is needed because the physical line is an edge in this
representation. Message passing needs the edge's two endpoint nodes in order to
transfer the shared-line features into the controlled busbar.

Busbar node features aggregate the generators and loads currently attached to
that busbar. Consequently, the live neighboring busbar exposes aggregated
neighbor power information (`gen_p`, `load_p`, and the associated angle
aggregates) in addition to the shared-line edge features. This is intentional
for the busbar representation and makes it the representation with the most
neighbor information.

The graph spec contains candidate rows for both busbars of a neighboring
substation so tensor shapes remain fixed across topology changes. Only the row
attached by a live boundary line is active. Every inactive contextual row and
all of its incident edges are zeroed and masked as defined above. A line outage
therefore removes all neighboring busbar state from the local observation while
preserving the controlled busbars.

### 2. Heterogeneous representation (`heterogeneous`)

The local graph contains:

- every busbar, generator, and load in the controlled substations;
- every line touching a controlled substation;
- for each live boundary line, the single far-end neighboring busbar.

It never contains a generator or load from a neighboring substation, even when
that asset is connected to the visible neighboring busbar. Neighbor generators
and loads are excluded when the local graph spec is created, so they cannot
participate in message passing or pooling.

The neighboring busbar still provides limited busbar-level context: its type,
identity where enabled, current boundary attachment, and busbar/substation
features. The power and angle of neighboring generators and loads are not
available because those quantities live on the excluded asset nodes.

As in the busbar representation, inactive contextual busbar candidates have
zero feature vectors and no active physical edge to the controlled domain.

### 3. Heterogeneous representation with line nodes (`heterogeneous_line`)

The local graph contains:

- every busbar, generator, and load in the controlled substations;
- every line touching a controlled substation as an explicit transmission-line
  node.

It contains no neighboring busbar, generator, or load. A boundary line node is
part of the agent's own observation/action graph because the line touches a
controlled substation. The line node carries the shared measurements, including
line status, `rho`, overflow duration, cooldown, and maintenance features when
configured. It attaches only to its endpoint busbar inside the controlled
domain. No edge to the far-end busbar is created in the local graph.

This rule is independent of `gnn_include_neighbors`: explicit shared-line nodes
are always included because they belong to the agent's boundary action space,
while neighboring equipment is never added. If a boundary line is disconnected,
its node remains present with `line_status = 0`; its busbar attachment edge is
inactive.

## The representations are not information-equivalent

The three representations intentionally expose different amounts of
neighboring information:

| Representation | Neighbor information available to the local actor |
|---|---|
| `bus` | Most: shared-line features plus the neighboring busbar's aggregated generator/load power and angle features |
| `heterogeneous` | Intermediate: shared-line features plus the neighboring busbar, but no neighboring generator/load nodes or their power/angle |
| `heterogeneous_line` | Least: shared-line node features only; no neighboring busbar or neighboring asset |

Therefore, an empirical comparison between these representations is not a pure
comparison of neural architecture. It also compares different information
sets. Results must be described with this qualification, and checkpoints trained
under the older construction are not strictly comparable to checkpoints trained
under these boundaries.

## Pooling is separate from construction

`mean` and other ordinary readouts may aggregate every visible node, whereas a
`controlled_*` readout aggregates only nodes belonging to the agent's domain.
That choice changes the graph summary but does not define what information the
agent can access during message passing.

The construction rules above hold for every pooling mode:

- neighboring private assets are absent from the graph;
- inactive contextual candidates carry no state;
- inactive candidate edges cannot transmit messages;
- the explicit-line representation has no far-end neighboring equipment.

Pooling experiments can consequently change aggregation without reopening the
cross-agent leakage paths closed by graph construction.

## Regression coverage

The information boundaries are covered by:

- `tests/test_context_visibility.py`, which checks dynamic busbar visibility,
  zeroed inactive candidate rows, exclusion of heterogeneous neighbor assets,
  and absence of all neighbor equipment from explicit-line local graphs;
- `tests/test_heterogeneous_graph.py`, which checks the exact local node-type
  membership, row maps, boundary-line attachments, and active-edge counts;
- `tests/test_candidate_action_scoring.py`, which checks that a boundary line
  remains available to action metadata without adding its far-end equipment.
