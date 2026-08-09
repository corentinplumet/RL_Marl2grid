# Code Changes Behind the Corrected Runs

What changed in the graph construction and the encoder, why, and which flag
controls it. These changes govern every run launched from this folder and from
[`gnn_action_scoring/no_leak_scaled`](../gnn_action_scoring/no_leak_scaled).

Every number below was measured, not estimated. Guarded by 141 tests, of which
29 are in `tests/test_context_visibility.py`.

## Flags at a glance

| flag | default | effect |
|---|---|---|
| `gnn_context_requires_connection` | `true` | a contextual node is visible only while a live line ties it to the agent |
| `gnn_structural_relations_controlled_only` | `true` | same-substation edges and summary nodes exist only inside controlled substations |
| `gnn_angle_representation` | `"node"` | `"edge_diff"` carries the angle as a per-line drop; now works on every schema |
| `gnn_readout_aggr` | `"mean"` | 13 names, composed as `[controlled_][energized_]{mean,sum,max}` plus `virtual_node` |

Four further changes have no flag, because the previous behaviour was not worth
preserving: the `connected` column is gone (6), neutral relations carry zero
loading (7), node-identifier embeddings are grid-independent (8), and the
obsolete `relation_self` column and its schema-version option are removed.

The first two default to the corrected behaviour, so a run gets it without
asking. Setting either to `false` reproduces the previous behaviour for an A/B.

---

## 1. Contextual nodes must be electrically connected

**The defect.** The region was expanded into *all* busbars of every substation
it touched, and every one was marked permanently valid. An agent therefore
pooled busbars that no live line attached to its own region — including, after a
neighbour split, a busbar carrying load on a bus the agent had no path to.

**Why that was wrong** is subtler than "it saw too much". The old gate was a
*labelling* rule: you may see anything sharing a substation label with something
you touch. But two busbars of one substation are separate electrical nodes, so
the rule had no physical content. Worse, such a node has no active edge, so it
never entered message passing — it contributed its raw feature row straight to
the mean readout, bypassing the graph entirely.

**The rule now.** A controlled node is always valid. A contextual node is valid
only when active, *non-structural* relations connect it to the controlled
region. `self` and `same_substation_busbar` do not qualify: they are
computational relations, not conductors. Visibility propagates to a fixed point,
because the explicit-line schema reaches a neighbouring busbar through a line
node, which is two hops.

**Measured**, `bus14`, nominal topology, pooled nodes per agent:

| agent | `bus` before → after | `heterogeneous` before → after |
|---|---|---|
| 0 | 12 → 10 | 22 → 20 |
| 1 | 18 → 13 | 28 → 23 |
| 2 | 16 → 14 | 26 → 24 |

Five of agent 1's eighteen busbar rows were leakage.

**Where:** `GridGraphBuilder._visible_node_mask` in `common/graph.py`.

---

## 2. Structural relations stay inside the controlled domain

**The reasoning.** A same-substation relation asserts that an element can be
moved between two busbars. That is a statement about an *action space* — and for
a contextual substation, about *another agent's* action space, which this agent
can neither exercise nor anticipate. Between two contextual busbars it is
neither electrical nor reachable, and it fabricates a message path that crosses
the boundary twice: information can leave the agent on one line and return on
another through a coupling that does not exist.

**Both routes had to close together.** Restricting only the direct edges leaves
the exchange open through the summary node, since `A → u_s → B` relates the same
two busbars.

**Measured**, agent 0 of `bus14` with contextual substation 3 split so both its
busbars are attached:

| setting | active same-substation edges | between two contextual busbars |
|---|---|---|
| `false` (previous) | 10 | 2 |
| `true` (now) | 8 | 0 |

**This matters most for Screen C.** Under the old construction its `e = 1` cells
joined a neighbour's two busbars, which promoted the visibility defect from a
readout perturbation into a real message-passing channel. Those are the cells
whose old results are least safe to carry forward.

**Where:** the three `_make_*_spec` builders in `common/graph.py`, and
`_append_hierarchy_nodes` in `common/gnn.py`.

---

## 3. Summary nodes describe structure, not identity

**The defect.** A summary node was initialised from row *s* of a trainable
matrix `[n_sub, d_h]` — 14×128 on `bus14`, 36×128 on WCCI. That matrix is not a
graph-describing buffer, so the transfer loader treats the shape mismatch as an
architecture difference and **raises**. Any `n = 1` checkpoint was therefore
impossible to transfer, which would have broken the thesis's own method:
screen on `bus14`, carry the winner to WCCI.

**Now.** The node starts from a projection of what the substation is made of:

```
u_s(0) = W_sub . phi_s + b_sub        W_sub in R^{d_h x 4},  b_sub in R^{d_h}
```

with `phi_s` holding the busbar, incident-line, generator and load counts, each
divided by a constant identical on every grid. Counts are structural rather than
physical, so the same descriptor means the same thing on any network.

This *generalises* the obvious alternative of one shared vector: `b_sub` is that
vector, and `W_sub` adds the modulation that separates a busy junction from a
radial end point.

```
3 subs: substation_node_encoder.weight=(6, 4)  bias=(6,)
6 subs: substation_node_encoder.weight=(6, 4)  bias=(6,)
state-dict entries differing in shape across grids:   (none)
```

**Still open:** `gnn_node_id_embeddings` builds `sub_id_embedding` as
`[n_sub, emb_dim]` and has the identical defect. It is `false` in every config
here, but 225 configs elsewhere set it `true`.

**Where:** `_init_hierarchy_nodes` in `common/gnn.py`,
`_substation_structural_features` in `common/graph.py`.

---

## 4. The angle drop works on the explicit-line schema

Not used by these screens — they keep the absolute angle — but active in
`no_leak_scaled` and recorded here because it is the same code.

`gnn_angle_representation = "edge_diff"` used to raise on `heterogeneous_line`.
In the other schemas a line is an edge, so the drop moves onto it; here a line
is a *node* and the attachment edges carry no physical channel at all, so there
was no edge to move it to. Removing the guard without more would have deleted
the angle silently: the correction strips the node angles and adds an edge
column that this schema's empty edge block then discards.

The drop is now written on the line node, in the column the absolute angle
occupied, renamed `theta_diff` so the scaling layer treats it as an angle. Node
width is unchanged. It is masked by line status, because the simulator zeroes
`rho` on a dead line but leaves its endpoint angles at their last solved values
— an unmasked drop reported `6.56` degrees on a line whose loading read exactly
zero.

Angle-carrying rows on `bus14`:

| agent | absolute | drop | of which line nodes |
|---|---|---|---|
| 0 | 5 / 22 | 8 / 22 | 8 |
| 1 | 3 / 20 | 8 / 20 | 8 |
| 2 | 8 / 29 | 9 / 29 | 9 |

---

## 5. One list of readout names

`[controlled_][energized_]{mean,sum,max}` is parsed compositionally by
`_pool_nodes`, so every combination already worked. Four separate hand-written
copies of the accepted names disagreed about which existed, and
`energized_max`, `controlled_energized_max` and the `sum` variants were missing
from all of them. One of the copies lived in a launcher guard, so a config could
pass the parser and the encoder and still be rejected after the job started.

The names are now generated once in `common/readouts.py`, a module with no
imports so a launcher can validate a config without loading torch. Both
encoders, the argument parser and the launcher guard read that set;
`test_every_gate_accepts_every_declared_readout` keeps them in step.

The observation also carries `energized_node_mask`: a node is energized when
incident to at least one active *electrical* relation, structural relations
excluded.

---

(The "Audited and deliberately left alone" section below records three feature
oddities that were examined and not changed.)

## 6. The `connected` column is gone

It was a constant on busbar rows and a copy of `line_status` on line rows, so it
carried information on asset rows alone — and there it never varied. A *change*
action toggles an element between busbars and cannot detach it, and a split that
islands an asset ends the episode. Verified on **both** grids:

| grid | random-action steps | episode ends | assets seen detached | lines seen disconnected |
|---|---|---|---|---|
| `bus14` | 600 | 180 | **0** | 150 |
| WCCI | 400 | 8 | **0** | 493 |

The column was removed. A detached asset would still be identifiable — its
measured channels zero while its type indicator stays set — so what is lost is
only the distinction from an attached asset reading zero, which would matter
under an action space exposing `set_bus`.

Node widths: `heterogeneous` 8 → 7, `heterogeneous_line` 13 → 12.

---

## 7. Neutral relations carry zero loading

The neutral physical block set `rho = 1` on structural and attachment
relations. That is a multiplicative identity for the weighted-GCN edge weight,
not a measurement, and it made a relation carrying no current read as a line at
its thermal limit for every encoder that consumes the edge vector as an
attribute — which is all of them in this work.

The attribute is now `0`, and the GCN weight substitution moved to the encoder:
`_gcn_edge_weight` returns the loading on physical lines and unit weight
elsewhere, deriving "physical" from the relation one-hot, or from `line_status`
in the busbar schema. The attribute stays an honest measurement and the
weighted-GCN variant still works. The same applied to the summary and virtual
edges appended by the encoder, which set the column to `1.0` as well; they now
carry a zero physical block too.

---

## 8. Node-identifier embeddings transfer

`sub_id_embedding` was `nn.Embedding(n_sub, emb_dim)` — the identical defect
fixed for summary nodes in change 3, and `true` in 225 configurations
elsewhere. It now projects the same structural descriptor:

```
3 subs: sub_id_embedding.weight=(4, 4)
6 subs: sub_id_embedding.weight=(4, 4)
learned tensors differing in shape across grids: none
```

`bus_id_embedding` is indexed by busbar position rather than substation, so it
was already grid-independent and is unchanged.

---

## What this means for the results

Changes 1, 2, 6 and 7 all alter the observation, so these runs are **not**
comparable cell-by-cell with the original screens, and no single one of them
explains a difference. They are a new baseline.

Changes 3 and 8 alter parameter shapes rather than inputs: they make an `n = 1`
or node-id checkpoint transferable, which it previously was not. Change 5 is
what makes the extended Screen B runnable at all.

## Reproducing the previous behaviour

```toml
gnn_context_requires_connection          = false   # contextual nodes always valid
gnn_structural_relations_controlled_only = false   # structural relations across the region
```

Changes 3, 6, 7 and 8 have no flag. Each removed something that was either
untransferable or uninformative, so keeping the previous behaviour reachable
would preserve a defect rather than an option.

---

## Background on the removals

All three were fixed, as changes 6 and 7 and the removal below. The reasoning
is kept here because it is what justifies the removals. They are recorded here so a reader can tell "looked at and
judged harmless" apart from "overlooked".

### Self relations

**Removed.** The always-zero `relation_self` input column, the
`include_legacy_self_relation_feature` builder option and the
`--gnn-edge-feature-schema-version` flag are all gone: no experiment uses self
edges, so the column existed only to rebuild checkpoints predating the schema
change. Loading such a checkpoint still works — the transfer loader's
`_migrate_edge_input_weight` splices the obsolete column out.

The misleading `sparse_gt_add_self_edges = true` line was also dropped from all
44 configs, where `sparse_transformer_self_edges_enabled` silently ignored it
for GINE and GAT.

### `connected`: the reasoning, kept for the record

The column separates "this asset is not on the grid" from "this asset is on the
grid and currently at zero" — a real distinction that never arises in these
experiments. Actions are built from `change_bus` and `change_line_status`, and a
*change* action toggles an element between busbars without ever detaching it;
separately, a split that islanded a load or generator ends the episode. Measured
over 600 random topology actions on `bus14` spanning 180 episode ends: **zero**
generator or load rows observed detached, against 150 line disconnections.

Where the column does vary it duplicates something already present — on busbar
rows it is a constant `1.0`, on transmission-line rows it equals `line_status`,
and on a contextual row that loses visibility it falls to zero with the rest of
that row, which the visibility mask already reports. Being constant, it is
absorbed by the first linear layer at a cost of one weight per output unit, so
it is retained rather than removed, which also keeps the input width stable.

Note it is **not** the same thing as energization: `connected` reads the
element's entry in the topology vector, while `energized_node_mask` reads
incidence to an active electrical relation. On agent 0 of `bus14` all eight
busbar rows read `connected = 1` while only four are energized.

### `rho = 1`: why it existed

The neutral physical block sets `rho = 1` rather than `0`. It is a
multiplicative identity, not a loading measurement: the weighted-GCN variant
reads that column as a scalar edge weight, and zero is the value that *deletes*
a message, so a structural or attachment relation weighted zero would vanish
from message passing.

**No reported run uses that path.** All of these use GINE or GAT, which consume
the edge vector as an attribute through an MLP and never touch `edge_weight`.
For them the convention buys nothing, and `0` — the value every other column of
the neutral block already takes — would have been the consistent choice.

It destroys no information. An active physical line always has
`line_status = 1` while the neutral block sets it to `0`; a disconnected line's
edge is masked, zeroed and filtered out before any convolution, so an *active*
row with `line_status = 0` can only be structural; and the disaggregated schemas
state it explicitly in the relation one-hot. The cost is that the
disambiguation is learned rather than given. Over 6,000 live-line samples on
`bus14`, `rho` has mean 0.415 and never reaches 1, so the neutral value at least
sits outside the range real loadings occupy.

Exposure is small, and change 2 reduced it further — active edges carrying the
neutral value, agent 0 of `bus14`, nominal topology:

| schema | old construction | new construction |
|---|---|---|
| busbar, no augmentation | 0 of 16 | 0 of 16 |
| busbar + same-substation edges | 12 of 28 | **8 of 24** |
| disaggregated | 12 of 28 | 12 of 28 |
| disaggregated + line nodes | no rho column on edges | no rho column on edges |

So the plain busbar screens never see it, the candidate-action runs never see
it, and restricting structural relations to the controlled domain removed a
third of the occurrences in the one cell family that did.
