# Contextual Node Visibility in Local Actor Graphs

Branch: `pooling`. Commit: *Gate contextual graph nodes on a live electrical
connection*.

This note explains a state-leakage defect in how local actor graphs were built,
the rule that replaces it, and what changed in the code.

## The defect

An agent's local graph is built in two steps. First a **region** is fixed: with
one-hop context enabled, every line with at least one endpoint in the agent's
controlled substations, plus the substations at both ends of those lines.
Second, that region is filled with nodes.

The second step expanded every substation of the region into *all* of its
busbars:

```python
# common/graph.py
def _bus_node_ids(self, sub_ids):
    return np.asarray(
        [self._bus_node_id(sub_id, bus)
         for sub_id in sub_ids for bus in range(self.n_busbar)], ...
    )
```

and every builder then emitted a constant valid-node mask:

```python
"node_mask": np.ones((len(spec["node_ids"]),), dtype=np.float32),
```

Because `GraphEncoder._pool_nodes` averages over `node_mask` and never receives
the controlled mask, **every node of the local graph entered the readout**. That
includes the busbars of a neighbouring substation that no live line attaches to
the agent's own region.

Concretely, for a neighbouring substation with `B = 2`:

- the line arrives on one of its two busbars;
- the *other* busbar is electrically separate from the agent's region, and may
  carry its own load and generation;
- both were pooled into the agent's embedding, at all times.

The same held after a line outage: a neighbouring substation stayed in the
average even with no live path to it. This is state the agent has no way of
observing, so it is leakage rather than context.

Two related facts make the leak purely a pooling problem, not a message-passing
one. Edges are only created for lines that touch the controlled domain, so a
contextual node never has an edge to another contextual node; and an inactive
candidate edge is masked before message passing. A contextual busbar with no
live attachment was therefore an isolated node that received no messages, yet
still contributed its raw features to the mean.

## The rule now implemented

> A controlled node is always valid. A contextual node is valid only when
> active, non-structural relations connect it to the controlled region.

Details that matter:

- **Structural relations do not carry visibility.** `self` and
  `same_substation_busbar` are computational relations, not electrical ones. Two
  busbars of one substation are not joined, so a same-substation edge must not
  make the unattached busbar visible.
- **Visibility propagates to a fixed point**, not one hop. The explicit-line
  schema reaches a neighbouring busbar *through* the line node, which is two
  hops; contextual generators and loads become visible through the busbar they
  attach to, and only while that busbar is itself connected.
- **It is dynamic.** The node set stays fixed for the episode, as the candidate
  graph design requires; only the mask changes per step. Switching a line onto
  the other busbar moves visibility with it, and a line outage removes the
  neighbouring substation from the average.

## What changed in the code

All in `common/graph.py`.

1. `GridGraphBuilder.__init__` takes `context_requires_connection: bool = True`,
   forwarded by `HeterogeneousGridGraphBuilder` and
   `HeterogeneousLineGraphBuilder`. `make_grid_graph_builder` passes `**kwargs`
   through, so no factory change was needed.
2. New `GridGraphBuilder._visible_node_mask(spec, edge_mask)` implements the
   rule. It reads relation ids from `spec["edge_type_names"]` rather than
   hard-coding them, because the three builders number their edge types
   differently.
3. The three `_build_*_for_spec` methods now call it instead of emitting ones.

Setting `context_requires_connection=False` restores the previous behaviour,
which is what every run before this branch was trained with.

### One implementation trap

The first version propagated visibility with

```python
grown[dst] |= visible[src]      # WRONG
```

NumPy buffers fancy-index assignment, so when a node appears several times in
`dst` only the last write survives and nodes with several active edges silently
lose visibility. The `bus` schema happened to pass and `heterogeneous` failed,
which is what exposed it. The unbuffered form is required:

```python
np.logical_or.at(grown, dst, visible[src])
```

## Verification

`tests/test_context_visibility.py`, 8 tests on a three-substation chain where
the agent controls the middle substation:

- only the attached busbar of a neighbour is visible;
- visibility follows a busbar switch;
- a disconnected line hides its whole substation;
- controlled nodes are never hidden, even with every line out;
- same-substation edges do not make a neighbour visible;
- disabling the flag restores the old behaviour;
- contextual generators follow their busbar;
- the explicit-line schema reaches neighbours through the line node.

The eight existing graph test modules still pass unchanged: 87 tests across
`test_heterogeneous_graph`, `test_candidate_action_scoring`,
`test_angle_representation`, `test_graph_normalization`,
`test_runtime_graph_config`, `test_encoder_transfer`, `test_feature_stats` and
`test_gnn_graph_screening`.

Measured on the real `bus14` partition under the nominal topology:

| agent | controlled subs | `bus` before | `bus` after | `heterogeneous` before | after |
|-------|-----------------|--------------|-------------|------------------------|-------|
| 0     | 0, 1, 2, 4      | 12           | 10          | 22                     | 20    |
| 1     | 3, 6, 7, 8      | 18           | 13          | 28                     | 23    |
| 2     | 5, 9–13         | 16           | 14          | 26                     | 24    |

Agent 1 is affected most: five of its eighteen busbar rows were leakage.

## Consequences for existing results

Every run reported so far was trained and evaluated with the leak, so the fix
changes the observation and invalidates strict comparability with them. Two
things follow:

- the graph-design screens should be re-run, or at least spot-checked, before
  their conclusions are carried forward under the corrected observation;
- `context_requires_connection=False` makes an explicit A/B possible on
  otherwise identical settings, which is the cleanest way to report the change.

The effect is not obviously in one direction. Removing unobservable rows should
help, but it also shrinks the pooled set and makes its size vary with topology,
which changes the scale of a mean readout from step to step.

## Not addressed

Pooling only nodes that are currently energised is a separate question. It would
also change what an agent sees of *its own* controlled region, not just of its
context, so it is a modelling choice rather than a leak fix and needs its own
decision and its own experiment.
