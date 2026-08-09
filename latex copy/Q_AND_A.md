# Thesis Q&A

Detailed answers to questions raised while reviewing the LaTeX, kept here in
full. Each answer also exists in condensed form in the report itself, written in
teal (`\color{clarif}`), so a reader does not need this file.

Every claim below was checked against the code rather than recalled. File and
line references point at `Topology_Task/`.

---

## Q1. Why keep the graph tensors at a fixed shape instead of rebuilding the graph after each topology action?

**Where:** Section 4.2.1, *Candidate graph and dynamic mask*.

Rebuilding the graph after every action is the more obvious implementation. It
is rejected for four reasons. The first three are engineering; the fourth is the
one to lead with, because it is about what the model can represent.

### 1. The interface makes it mandatory, not merely preferable

Each agent's observation is declared once as a fixed-size `Box`
(`env/utils.py:1110`). More decisively, the rollout buffer is allocated **once**
from the shape of the first observation and then written into at every step:

```python
# alg/mappo/core.py:662
observations[id] = zeros_like_with_leading(
    strip_state_graph(next_obs[id]), (args.n_steps,), device=device
)
```

`zeros_like_with_leading` builds `tuple(leading_shape) + tuple(obj.shape)`
(`common/utils.py:93`). With 72 environments stepping in parallel, an
observation whose node or edge count changed mid-episode could neither be
stacked across environments nor stored in that buffer. It would raise, not
degrade gracefully.

### 2. Batching and indexing stay trivial

The encoder registers the message-passing index once:

```python
# common/gnn.py:375
self.register_buffer("edge_index", th.tensor(graph_spec["edge_index"], dtype=th.long))
```

A rebuilt graph would mean constructing a variable-size graph and re-deriving
its index for every environment, agent and step, then dynamically re-batching —
the standard ragged-graph cost, paid on every forward pass.

### 3. Row identity is load-bearing

Row *i* always denotes the same physical busbar, line or asset. The
candidate-action head depends on this directly: `ActionGraphMetadata` is
documented as a *"Static mapping from discrete action IDs to physical graph
rows"* (`common/action_metadata.py:26`), built once before training. If the node
set were rebuilt per step, that mapping would have to be recomputed at every
step and would refer to different objects at different times.

### 4. The substantive reason: a rebuilt graph hides what actions can do

A rebuilt graph contains only the connections that *currently* exist. The empty
busbar an element would be switched onto would simply not be in the graph, so no
encoder reading it could evaluate that action at all.

Keeping the candidate set fixed means the reachable topologies are present as
masked structure, and the mask makes the current topology an **explicit input**
rather than something inferable only from which messages happen to exist. This
is exactly the *busbar information asymmetry* identified by de Jong, Viebahn and
Shapovalova (arXiv:2501.07186), already cited in the introduction — which lets
the design choice rest on a named problem in the literature rather than on
implementation convenience.

### The cost, stated honestly

With `B = 2` a line reserves `2B² = 8` directed candidates, of which at most two
are ever active. The edge tensor is therefore roughly four times larger than the
live topology requires. On the grids used here that is affordable; it is also
why the scheme does not scale indefinitely in the number of busbars per
substation.

---

## Q2. Why do the disaggregated representations carry a `connected` attribute when the busbar representation does not?

**Where:** Section 4.2.3, *Disaggregated Representation*.

### The reason it is needed

Both disaggregated schemas give every generator and load **its own node**, and
that node stays in the fixed candidate graph for the whole episode whether or
not the asset is attached. When the asset detaches, its attachment candidates
are masked and its measured channels are zeroed.

Verified on a three-substation fixture, varying only the generator's bus
assignment:

| state | `connected` | `p` |
|---|---|---|
| attached | 1.0 | 5.0 |
| detached (`topo_vect = -1`) | 0.0 | 0.0 |

A generator that is *attached but injecting nothing* also reads `p = 0`. So
`connected` is the only channel separating **"this asset is not on the grid"**
from **"this asset is on the grid and currently at zero"**. Without it the two
states are the same input vector.

### Why the busbar graph does not need it

There is no asset row to disambiguate. A detached asset simply stops
contributing to the sum aggregated onto its busbar
(`_put_bus_asset_feature` in `common/graph.py`), and the busbar node itself is
switchgear — it is never "connected" or "disconnected".

There is also a representational reason. The busbar graph has a **single node
type**, so a column that took the same value on every node would carry no
information and would be pure width. The disaggregated schemas share **one
feature vector across all node types**, with the type given by the `is_*`
one-hots, so a column only has to be meaningful for *some* of them.

### Three caveats worth volunteering before they are asked

1. **On busbar rows the flag is uninformative.** It is written as a constant
   `1.0` (`features[bus_rows, col["connected"]] = 1.0`), confirmed across every
   busbar row of the real `bus14` graph.
2. **On transmission-line rows it duplicates `line_status`.** Both are set from
   the same `line_status > 0` test; I verified the two columns are elementwise
   equal on the built graph.
3. **The busbar graph keeps an ambiguity the disaggregated ones resolve.** A
   busbar reading zero generation may have no generator attached, or an attached
   generator producing zero, and nothing in that representation distinguishes
   them. This is a genuine information difference between schemas, not just a
   difference of architecture — the same point Section 4.2.5 makes about the
   three representations not being information-equivalent.

---

## Q3. What do "attached" and "detached" actually refer to? Is `connected` about a busbar being energized?

**Where:** Section 4.2.3, following Q2.

No — and the name is misleading, which is why this needs saying in the report.

### What `connected` reads

It is a property of an **element**, taken straight from Grid2Op's topology
vector. For a generator, load or line endpoint, `topo_vect` holds `-1` when the
element is disconnected and `1..B` for the busbar it sits on. The builder maps
that to a busbar index and tests it:

```python
# common/graph.py, _put_entity_node_features
connected = (asset_bus[asset_ids] >= 0) & (asset_bus[asset_ids] < self.n_busbar)
```

So `connected` answers **"is this element currently switched onto one of its
substation's busbars?"** It says nothing about whether that busbar carries
current.

### The notion you were thinking of is a different one

Three quantities are easy to conflate, and only the last is electrical:

| quantity | kind | read from | question |
|---|---|---|---|
| `connected` | node **feature** | the element's `topo_vect` entry | is this element switched onto a busbar? |
| visibility (`node_mask`) | per-node **mask** | active non-structural relations reaching the controlled region | may this agent observe this node? |
| `energized_node_mask` | per-node **mask** | incidence to ≥1 active electrical relation | is this node electrically live? |

`energized_node_mask` is the one that means "attached to the grid". Its rule
excludes self and same-substation relations, because those are computational
aids rather than electrical connections.

### Where the two come apart

For a **generator or load** they coincide, verified on the chain fixture:

| generator state | `connected` | energized |
|---|---|---|
| attached | 1.0 | True |
| detached (`topo_vect = -1`) | 0.0 | False |

For a **busbar** they diverge completely. Measured on agent 0 of `bus14` at the
nominal topology, `heterogeneous` schema:

| row | type | `connected` | energized |
|---|---|---|---|
| 0 | busbar | 1 | 1 |
| 1 | busbar | 1 | **0** |
| 2 | busbar | 1 | 1 |
| 3 | busbar | 1 | **0** |

All **8 busbar rows read `connected = 1`; only 4 are energized.** The dead ones
are the second busbar of each substation — empty because everything sits on
busbar 1, and precisely the busbar a splitting action would move an element
onto.

So a busbar is never "detached" in the sense `connected` measures; it is
switchgear that is always present. Whether it is *live* is what the energized
mask reports, and that is a separate array added alongside the node features,
not a column inside them.

---

## Q4. Is `connected` actually useful? A disconnected generator or load should end the chronic.

**Where:** Section 4.2.3, following Q3.

Correct — and the consequence is that the flag is **inert in these experiments**.
The distinction it encodes is real, but the state it distinguishes never occurs.

### Why an asset is never observed detached

Two independent reasons:

1. **The action space cannot detach one.** Actions are built from
   `act_attr_to_keep = ["change_bus", "change_line_status"]`
   (`env/utils.py:1127`). A *change* action toggles an element between busbar 1
   and busbar 2; it has no way to write `-1`. Only `set_bus`, which is not
   exposed, can disconnect.
2. **Islanding ends the episode.** A substation split that isolated a load or
   generator terminates the chronic, so the agent never conditions on such an
   observation.

### The measurement

600 random topology actions on `bus14` with that exact action space:

```
600 random-action steps, 180 episode ends, action space 387
  gen/load rows seen with topo_vect < 1 : 0
  line rows seen disconnected           : 150
```

Zero asset detachments across 180 episodes, against 150 line disconnections.

### Where the column does vary, it duplicates something else

- **Busbar rows:** written as a constant `1.0`, always.
- **Transmission-line rows** (explicit-line schema only): equals `line_status`
  elementwise — verified on the built graph.
- **Contextual rows that lose visibility:** the whole row is zeroed, so
  `connected` drops to 0 with it. Verified on the chain fixture, where the
  `connected` column reproduces `node_mask` exactly:

  ```
  both lines live    node_mask=[1, 0, 1, 1, 1, 0]
                     connected=[1, 0, 1, 1, 1, 0]
  line 1 out         node_mask=[1, 0, 1, 1, 0, 0]
                     connected=[1, 0, 1, 1, 0, 0]
  ```

  That information is already carried by the visibility mask.

### So is it useless?

In the reported experiments, yes — it is a channel the construction supports
rather than one the experiments exercise. It would carry information under an
action space including `set_bus`, or in an environment where assets themselves
trip.

It is not harmful. A constant input column is absorbed by the first linear
layer's weights and costs one parameter per output unit. It is retained rather
than removed, which also keeps the input layout comparable with runs already
completed.

**Scope of the measurement:** taken on `bus14`. The `bus36` environment adds an
opponent and maintenance, but both act on *lines*, not on assets, so the
conclusion is expected to hold there; it has not been measured directly.

---

## Q5. Why does the neutral physical block set `rho = 1`? The relation one-hot already makes the vector non-zero, and `rho = 1` reads as an overloaded line.

**Where:** Section 4.2.3, Equation for `p_neutral`.

Short answer: it is a **weight**, not a measurement — and it is an artifact of an
encoder that no reported result uses.

### What it is for

Only the weighted GCN variant reads `rho` as a scalar message weight
(`edge_weight=edge_attr[:, idx]`, `common/gnn.py:529`). For that consumer, zero
is the value that *deletes* a message:

```python
# common/graph.py, _heterogeneous_edge_features
# If rho is selected as a GCN message weight, structural relations need
# unit weight so generator/load and optional relation messages survive.
edge_features[~physical, rho_idx] = 1.0
```

A structural or attachment relation weighted zero would silently vanish from
message passing. One is the multiplicative identity.

### Does it make sense given only GINE was used?

**No.** GINE consumes the edge vector as an *attribute* through an MLP and never
touches the weight path, so the justification does not apply to a single
reported result. Zero — the value already taken by every other column of the
neutral block (`line_status`, `timestep_overflow`, `cooldown`) — would have been
the consistent choice.

One factual correction worth having straight: the repository does contain three
GCN runs, `best_08_shared_actor_gnn_weighted_gcn_..._s0/s1/s2` in the
`gine_s0_s1_s2` group from the early architecture search. **No GCN result is
reported in the thesis**, so "GINE and GAT only" is accurate for everything
presented, but the runs exist.

### It does not destroy information

Three independent identifiers separate a neutral row from a loaded line:

1. An active physical line always has `line_status = 1`; the neutral block sets
   it to `0`.
2. A disconnected line's edge is masked, its row zeroed
   (`edge_features[~active] = 0.0`) and filtered out before any convolution — so
   an *active* row with `line_status = 0` can only be structural.
3. In the disaggregated schemas the relation one-hot states it explicitly.

Measured over 6,000 live-line samples on `bus14`: `mean rho = 0.415`,
`median = 0.396`, `p95 = 0.759`, `fraction >= 1: 0.0000`. So the neutral value
also sits outside the range real loadings occupy.

The cost is that the disambiguation is **learned rather than given**.

### How much of the work is exposed

Active edges carrying the neutral value, agent 0 of `bus14`, nominal topology:

| schema | affected |
|---|---|
| busbar, no structural augmentation | 0 of 16 |
| busbar + same-substation edges | 8 of 24 |
| disaggregated | 12 of 28 |
| disaggregated with line nodes | none — edges hold no physical block |

So the candidate-action experiments (`heterogeneous_line`) are **completely
untouched**, and so are the busbar screens that leave the structural
augmentations off. It bites only in Screen C's structural-edge arm and in the
disaggregated runs.

### Position to take

The neutral value is correct for the consumer it was written for, unnecessary
for the encoder actually used, and its cost is a learnable disambiguation rather
than lost information. Zero would have been cleaner. It is retained rather than
corrected because changing it would alter the observation and break
comparability with every completed run.

---

## Q6. Is it meaningful for the disaggregated schema not to give the neighbouring node its aggregated power? Does it make sense?

**Where:** Section 4.2.3, *Local information boundary*.

### What actually happens

Built on agent 0 of `bus14`, nominal topology, one-hop context enabled — the
visible contextual rows are:

```
bus            row  6: [ 0.0,  0.0, 44.3, -5.51, 0.0, 0.0 ]
               row 10: [ 5.3, -9.67, 11.9, -9.67, 0.0, 0.0 ]
               max |measured channel| over contextual rows: 44.300

heterogeneous  row  6: [1, 0, 0, 0, 0, 0, 1, 0]
               row 10: [1, 0, 0, 0, 0, 0, 1, 0]
               max |measured channel| over contextual rows: 0.000
```

In `bus` the far-end busbar carries 44.3 MW of neighbouring load and its angle.
In `heterogeneous` every measured channel is exactly zero — the row reduces to
`is_busbar = 1`, `connected = 1`, `domain_mask = 0`. The two contextual rows are
also **identical to each other**, so they are not even distinguishable as
different substations.

This is not a corner case: 287 configurations in the repository set
`gnn_include_neighbors = true`, against 257 with it false.

### Why it happens

Power and angle live on *asset* nodes in the disaggregated schema, and the
boundary rule never constructs a neighbouring generator or load. So nothing is
left to write onto the far-end busbar. In `bus`, by contrast, asset power is
aggregated *onto the busbar* by construction, so including the neighbouring
busbar automatically includes the neighbour's dispatch.

Same boundary rule, two very different information sets.

### Does it make sense? Two answers, pointing opposite ways

**As an information boundary — yes, it is the defensible one.** A neighbouring
substation's generation and load are another region's private state. A regional
operator sees the flow on the tie line, not the neighbour's dispatch. On that
reading `bus` is the leaky schema and `heterogeneous` is closer to the
decentralized setting the thesis claims to model.

**As a graph design — it is incoherent.** Having decided to withhold the
neighbour's power, keeping its busbar as a node buys very little:

- the row is a constant, so it carries no state;
- it is anonymous — all contextual busbars share the same vector;
- it is a dead end, because edges are only created for lines touching the
  controlled domain, so a contextual node never connects to another contextual
  node.

What the agent learns about the neighbour arrives through the **edge**
attributes of the tie line, chiefly its loading. The node contributes a
nonlinear round trip, not new information.

### The consistent alternative already exists

`heterogeneous_line` does not construct the far-end busbar at all, and puts the
shared-line state on the line node. So the three schemas are **not** a clean
ladder of decreasing context — the disaggregated one occupies an awkward middle
position, keeping the node while dropping its content.

### Consequence for the comparisons

A `bus` versus `heterogeneous` result is not a pure architecture comparison: one
agent sees 44.3 MW of neighbouring load and the other sees zero. This is the
concrete version of the non-equivalence argued in Section 4.2.5, and it should
be stated whenever those two schemas are compared.

Note also that before the visibility fix these constant contextual rows entered
the mean readout, diluting it with a fixed vector whose weight varied with how
many neighbours happened to be visible.

---

## Q7. In a real technical scenario, is it meaningful *not* to give the agent the whole neighbouring substation? The agent cannot control it anyway.

**Where:** Section 4.2.5, *Local actor information boundary*.

### The premise needs splitting

"Cannot control it, so need not observe it" conflates two independent things.
Real regional control rooms observe far beyond what they operate: state
estimation runs on an extended model, tie-line telemetry is continuous, and
neighbouring TSOs exchange boundary data. On realism grounds the local graphs
used here are **more** restrictive than a real control room, not less.

### The power-systems answer is not "one hop"

The established treatment of "how much exterior do I need" is an **external
network equivalent** (Ward, REI and relatives): reduce everything outside the
area to a small model that preserves *boundary* behaviour. Note what such an
equivalent deliberately discards — the neighbour's internal dispatch — and what
it keeps: the boundary injections and the stiffness seen from the boundary.

Applied here:

- The neighbour's `gen_p` / `load_p` is a single bus of an unbounded remainder.
  It is the part real equivalents throw away, and on its own it says little
  about how the exterior responds to a switch.
- The **tie-line loading and the angle across it** are the boundary condition.
  That is the load-bearing quantity, and it is carried on the line itself
  (Section 4.1.4).

Whether a switching action is safe inside the controlled region depends on the
post-action flows there. Those depend on the region's own topology and
injections — fully observed — plus the exterior, which enters only through the
boundary. Power flow has no natural one-hop boundary, so one hop is an arbitrary
truncation.

### The argument that *does* support a local actor

It is methodological, not physical. An actor that observes everything is a
centralized controller with extra steps, and the question the thesis asks is how
far a regionally-conditioned policy can go.

The centralized critic already reads the global state, so the value estimate is
not blind — only the policy is. That compensates in part. It does not compensate
fully: where the optimal action depends on exterior state the actor cannot see,
the policy can only learn an average over it. That is the standard decentralized
partially-observed limitation, and cascading overload is exactly the regime where
a remote contingency decides whether a local action was correct.

### Existing evidence, and why it is weak

An A/B already exists in `configs/a0_gine_light_nonshared/`:

| arm | seeds | final-window test survival |
|---|---|---|
| one-hop context | 3 | 0.884, 1.000, 1.000 |
| no context | 3 | 1.000, ~0.48, ~0.94 |

Suggestive but not usable as evidence, for four reasons:

1. it is the **final-window** statistic, not the best-checkpoint full-test
   statistic used elsewhere in the thesis;
2. n = 3, with one no-context seed collapsing to 0.48 — within seed noise;
3. the context arm predates the visibility fix, so its extra context included
   rows the agent should not have seen;
4. it is the early `a0_gine_light_nonshared` family, not the current reference
   configuration.

### Position to take

Do not defend one-hop context on realism grounds; it is neither more nor less
realistic than the alternatives, it is an arbitrary cut. Frame it as the
ablation it is — *how much exterior does a local policy need?* — and observe
that the boundary telemetry, rather than the neighbour's dispatch, is the part
that carries information. Re-running the A/B leakage-free, on the current
reference configuration and scored by best-checkpoint full-test evaluation,
would convert an unjustified constant into a measured result.

---

## Q8. Was the old "leaking" construction actually wrong? Observing a neighbour you cannot control is not obviously cheating.

**Where:** Section 4.2.5, *Local actor information boundary*.

Yes it was wrong — but not because it revealed more. The premise is right:
observation and control are separate, and a real operator does observe
neighbours. The defect lies elsewhere.

### The old gate was a naming convention, not an electrical one

The earlier rule expanded every substation of the region into **all** of its
busbars and marked them permanently valid. In effect: *you may see anything that
shares a substation label with something you touch.* But two busbars of one
substation are separate electrical nodes whenever that substation is split.
"Same substation" is an artifact of how the grid is catalogued.

### The demonstration

Agent 0 of `bus14`. Contextual substation 3 is split so its 44.3 MW load sits on
busbar 2, while the agent's lines attach to busbar 1:

```
OLD   row  6 (busbar 1)  deg=6   gen_p=0.0  load_p= 0.0
      row  7 (busbar 2)  deg=0   gen_p=0.0  load_p=44.3   <-- visible, no path
      row 10             deg=2   gen_p=5.3  load_p=11.9
      row 11             deg=0   gen_p=0.0  load_p= 0.0

NEW   row  7 absent from the visible set
```

Note `deg=0`: **zero active edges**. Electrically that busbar is as remote from
the agent as any other bus in the grid.

### Two consequences, the second worse than the first

1. **The channel usually could not process it.** With same-substation edges
   *off*, an isolated node receives and sends no messages, so the 44.3 MW never
   entered message passing — it contributed its raw feature row directly to the
   **mean readout**, perturbing the pooled embedding with unstructured numbers
   that bypassed the graph structure entirely.

   **With same-substation edges on this is no longer true, and the leak is
   worse** — see Q9. The neighbour's two busbars are joined, the unreachable
   busbar gains degree 2, and its load propagates into the agent's own region
   through message passing.
2. **It injected observation-level non-stationarity.** The split is an action of
   the *neighbouring* agent. Under the old rule, agent 1 reconfiguring its own
   substation moved 44.3 MW between two rows of *agent 0's* observation with no
   change whatsoever in agent 0's electrical situation. That is a
   non-stationarity added on top of the one decentralized training already has
   from co-adapting policies.

   Under the new rule agent 0's observation still changes — row 6 falls to zero —
   but that change is real: the load genuinely left the busbar its tie line
   attaches to.

### The correct framing

The problem was never "too much information". Had the intent been to give the
agent the neighbour's full state, the right implementation would be proper edges
or an explicit exterior summary — not isolated nodes floating in the pooling.
The old scheme was wrong on **the criterion** (a label instead of a physical
connection) and wrong on **the delivery path** (pooling rather than message
passing), independently of volume.

It does not follow that the new boundary is the *correct* amount of exterior. As
argued in Q7 it is conservative — more restrictive than a real control room —
and an explicit external network equivalent would be the principled alternative.
The defensible claim is "an arbitrary and unusable channel was removed", not
"the right amount of context was found".

---

## Q9. For the same-substation edge augmentation, what happens at a neighbouring busbar? Is an edge created between the two busbars of a neighbouring substation?

**Where:** Section 4.3.1, *Direct same-substation edges*.

**Yes.** The augmentation loops over every substation of the region —
`for sub_id in sub_ids` in `common/graph.py` — and `sub_ids` includes contextual
substations. A neighbouring substation's busbars are joined to each other
exactly as controlled ones are.

### Why that matters, and a correction to Q8

Measured on agent 0 of `bus14`, contextual substation 3 split so its 44.3 MW
load sits on the busbar the agent's lines do **not** attach to:

| construction | same-substation edges | visible | active degree | load seen |
|---|---|---|---|---|
| old | off | yes | 0 | 44.3 |
| old | **on** | yes | **2** | **44.3** |
| new | off | no | 0 | 0 |
| new | on | no | 0 | 0 |

Q8 stated that the leaked value never entered message passing. **That is true
only with this augmentation off.** With it on, the neighbour's two busbars are
joined, the unreachable busbar gains degree 2, and its load propagates through
the structural edge into the attached busbar and from there across the tie line
into the agent's own region.

So under the old construction this augmentation *upgraded* the leak from a
readout perturbation to a genuine information channel. **The `e = 1` cells of
Screen C are the most affected results in the screening chapter.**

### Why the current rule excludes it

Grid2Op has no bus coupler. Busbar assignment *is* the topology, and two busbars
of one substation are **separate electrical nodes**. A same-substation relation
is a computational device encoding that an element can be moved between them —
a statement about an *action space*, not about conductivity. For a contextual
substation it is a statement about *another agent's* action space.

The visibility rule therefore excludes it, along with self relations:

```python
STRUCTURAL_EDGE_TYPE_NAMES = ("self", "same_substation_busbar")
```

Covered by `tests/test_context_visibility.py::test_same_substation_edges_do_not_make_a_neighbour_visible`.

Under the current construction the node is invisible and its structural edges
are inactive in both settings, so the augmentation changes message passing only
among nodes the agent is entitled to see.

---

## Q10. Is the new construction also safe for substation summary nodes? And does a neighbouring substation really get its own summary node connected to a single busbar?

**Where:** Section 4.3.2, *Substation summary nodes*.

Two separate answers: **safe, but largely pointless for contextual substations.**

### Safe — but by a different mechanism

Summary nodes are appended by the **encoder** (`_append_hierarchy_nodes` in
`common/gnn.py`), after the builder has decided visibility, so the visibility
rule cannot gate them. The protection is in the wiring instead:

```python
active_busbar_ids = busbar_ids[node_mask[busbar_ids] > 0]
```

Only visible busbars are connected to their summary node. Verified on the chain
fixture, where rows 1 and 5 are the unattached busbars of the two contextual
substations:

```
node_mask (physical rows) : [1, 0, 1, 1, 1, 0]
invisible rows            : [1, 5]
degree AFTER substation nodes appended : {1: 0, 5: 0}
```

Still isolated. An unreachable busbar's state cannot reach the summary node, so
it cannot propagate onward. The controlled mask is also derived correctly —
`[0, 1, 0]` for the three appended nodes — so a contextual summary node is
itself contextual and a `controlled_*` readout excludes it.

Locked in by
`tests/test_context_visibility.py::test_substation_nodes_do_not_wire_an_invisible_busbar`,
added because the hierarchy code lives in a different module from the visibility
rule and nothing previously connected the two.

### But the contextual summary node does almost nothing

Agent 0 of `bus14`, nominal topology:

| substation | kind | busbars aggregated |
|---|---|---|
| 0, 1, 2, 4 | controlled | 2 of 2 |
| 3 | contextual | **1 of 2** |
| 5 | contextual | **1 of 2** |

Summarising one node is an identity up to the learned transform. The round trip
`busbar -> summary -> busbar` returns to its origin while consuming two
message-passing steps — the entire depth at the two-layer setting used
throughout — and adds a learned constant row to an ordinary mean readout.

It is not degenerate in every topology. Splitting contextual substation 3 so its
lines land on both busbars makes both visible, and its summary node then
aggregates 2 of 2. But that is the exception.

### The fixed-shape argument does not justify it

Summary nodes cannot be created conditionally on visibility — the candidate
graph must keep a constant shape. But **controlled substations are a static set
too**. Restricting the augmentation to them would:

- preserve the fixed shape;
- keep every summary that aggregates more than one busbar;
- remove exactly the degenerate ones.

**This is now implemented** — see Q11. Summary nodes are built only for
controlled substations, so the degenerate contextual ones no longer exist.

**Coupling to record for any later change:** under the virtual-node readout the
graph summary attaches to *substation* nodes when they exist, so removing
contextual summary nodes would also disconnect contextual busbars from that
readout. That may be desirable under a controlled-only philosophy, but it is a
second effect rather than a free simplification.

---

## Q11. Should two busbars of a *neighbouring* substation exchange information at all, even when both are energized? The agent cannot act on them.

**Where:** Section 4.3.1, following Q9.

**The intuition is right**, and there is a stronger argument for it than the one
it starts from.

### The case is real

With contextual substation 3 split so both its busbars are attached to agent 0
by live lines, both are visible and the relation between them is active:

```
row 6 (sub 3) -> row 7 (sub 3)   both visible=True
row 7 (sub 3) -> row 6 (sub 3)   both visible=True
2 such active edges  <- they exchange messages
```

### Argument 1 — the stated one, and it holds

The relation encodes "an element can be moved between these two busbars". That
is a statement about an **action space**. For a contextual substation it is a
statement about *another agent's* action space, which this agent can neither
exercise nor anticipate.

It is also **inconsistent with a decision already taken**: structural relations
are excluded from carrying *visibility* on exactly this ground
(`STRUCTURAL_EDGE_TYPE_NAMES`). Allowing them to carry *messages* between two
contextual nodes contradicts that.

### Argument 2 — physical, and stronger

Two busbars of one substation are **separate electrical nodes**. Inside the
controlled domain the relation is justified because the agent can join them by
acting. Between two contextual busbars it is neither electrical nor reachable by
any action available to this agent.

Worse, it **fabricates a shortcut across the boundary**. Information leaving the
agent at busbar X on one line can return to its busbar Y on another via
`X -> A -> B -> Y`, where A and B are the neighbour's two separate buses. That
path asserts an electrical coupling that does not exist; the true path runs
through the agent's own internal topology.

This reframes the objection from "uninformative" to "incorrect", which is a much
better position to argue from.

### The distinction to keep straight

**This is not a leak.** Both endpoints are legitimately visible — each is
attached to the agent's region by a live line — so no unobservable state enters.
The claim is that the edge asserts a relation which does not hold: a modelling
error, not an information-boundary violation. Conflating the two would weaken
both claims.

### A counter-argument that does not survive

One might say the neighbour *will* reconfigure, so modelling the possibility
aids anticipation. It does not: a static structural edge encodes "these nodes
are related", not "the neighbour might act", and the agent can neither observe
nor influence that intent.

### What it would take, and what it would buy

Restrict both structural augmentations to controlled substations:

- `add_substation_edges`: iterate controlled substations instead of the whole
  region;
- `add_substation_nodes`: restrict the included substation set likewise — which
  also removes the degenerate contextual summary nodes of Q10.

The fixed candidate shape does not prevent this, since controlled substations
are a static set.

It would also **decouple a factor Screen C currently confounds**: enabling `e`
today changes relations inside the controlled domain *and* between contextual
nodes simultaneously, so a positive or negative effect cannot be attributed. The
same holds for `n`.

**Implemented**, since all screens are being rerun. New flag
`--gnn-structural-relations-controlled-only`, default `True`; set it `False` to
reproduce the previous behaviour for an A/B.

Verified on agent 0 of `bus14` with contextual substation 3 split:

```
controlled_only=False : 10 active same-substation edges, 2 between two contextual busbars
controlled_only=True  :  8 active same-substation edges, 0 between two contextual busbars
```

Covered by three tests in `tests/test_context_visibility.py`:
`test_same_substation_edges_stay_inside_the_controlled_domain`,
`test_summary_nodes_gather_only_controlled_busbars`, and
`test_legacy_flag_restores_contextual_same_substation_edges`.

**Coupling accepted:** the virtual-node readout attaches to summary nodes where
they exist, so contextual busbars no longer reach the graph summary directly.
They influence it through the controlled busbars they connect to, which is
consistent with reading the graph embedding as a description of the region the
agent controls.

---

## Q12. The trainable matrix Q is indexed by substation, so it cannot transfer. Would one learned vector shared by every substation be better?

**Where:** Section 4.3.2, *Substation summary nodes*.

The diagnosis is right, and the consequence is harder than "does not transfer
well": **transfer fails outright.**

### The measurement

```
small (3 subs)   substation_node_embedding.weight=(3, 6)   virtual_node_embedding=(1, 6)
large (6 subs)   substation_node_embedding.weight=(6, 6)   virtual_node_embedding=(1, 6)

shape differs between grids:
   substation_node_embedding.weight: (3, 6) vs (6, 6)   -> TRANSFER RAISES
```

The matrix was `[n_sub, d_h]` — 14×128 on `bus14`, 36×128 on WCCI. It is **not**
in `TOPOLOGY_BUFFER_NAMES` (`edge_index`, `node_ids`, `edge_type`,
`controlled_node_mask`), so `common/transfer.py` treats the mismatch as an
architecture difference and raises rather than rebuilding it.

Note the contrast in the same file: `virtual_node_embedding` is
`nn.Parameter(th.empty(1, d_h))` — one shared vector, identical on any grid. The
virtual node already did the right thing.

**It was latent but it blocked the pipeline.** All 36 `transfer_scaled` configs
set `gnn_add_substation_nodes = false`, so nothing failed in practice — but
Screen C's `n = 1` arm could never have been carried to WCCI. If `n = 1` had won
the screen, the thesis's own method (screen on `bus14`, transfer the winner)
would have broken for that winner.

### What was implemented

The shared-vector proposal works, but it throws away something real: a
per-substation vector can encode "this one is a five-line junction", and that is
*not* recoverable from busbar features, which carry no structural degree.

So the summary node is now initialised from a **structural descriptor** rather
than an identity:

```
u_s(0) = W_sub . phi_s + b_sub,     W_sub in R^{d_h x 4},  b_sub in R^{d_h}
```

where `phi_s` collects the substation's busbar count, incident line count,
attached generator count and attached load count, each divided by a constant
identical on every grid.

This **strictly generalises** the shared-vector proposal: `b_sub` *is* the single
vector every substation shares, and `W_sub` adds the structural modulation on
top. Because the entries are counts rather than physical magnitudes, the same
descriptor means the same thing on any network.

Sanity check on the three-substation chain (gen at sub 0, load at sub 2):

```
feature names: ['busbar_count', 'line_count', 'generator_count', 'load_count']
 [[1.   0.25 0.25 0.  ]     <- sub 0: one line, one generator
  [1.   0.5  0.   0.  ]     <- sub 1: two lines, nothing attached
  [1.   0.25 0.   0.25]]    <- sub 2: one line, one load
```

And after the change, across a 3-substation and a 6-substation grid:

```
3 subs: substation_node_encoder.weight=(6, 4)  bias=(6,)
6 subs: substation_node_encoder.weight=(6, 4)  bias=(6,)

state-dict entries differing in shape across grids:   (none)
```

### Tests

- `test_summary_node_parameters_are_grid_independent` — every learned tensor
  keeps its shape on a larger grid; only graph-describing buffers may differ.
  This is the regression that would have caught the original design.
- `test_summary_node_reads_its_substation_structure` — the descriptor actually
  separates substations, so the projection does not degenerate to one shared
  vector.

The descriptor buffer is registered with `persistent=False`, so it never enters
the state dict and transfer never sees it.

### Still open

`gnn_node_id_embeddings` builds `sub_id_embedding` as `[n_sub, emb_dim]` and has
exactly the same defect. **225 configurations set it `true`.** Any checkpoint
trained with it is equally untransferable, and it has not been changed here.

---

## Q13. In equation (4.16), why are the optional parameters in the middle of the vector rather than at the end?

**Where:** Section 4.2.4, explicit-line node vector.

The optional block is the two maintenance countdowns, and the placement is
deliberate — the builder inserts them at a named anchor rather than appending:

```python
insert_at = self.node_features.index("time_before_cooldown_sub")
self.node_features[insert_at:insert_at] = self.MAINTENANCE_EDGE_FEATURES
```

### Why there: the vector is grouped by owner

| block | columns | which rows carry it |
|---|---|---|
| type indicators | `is_busbar … is_transmission_line` | all |
| asset measurements | `p`, `theta` | generator / load |
| **line measurements** | `line_status`, `rho`, `timestep_overflow`, `time_before_cooldown_line` | **transmission line** |
| substation measurement | `time_before_cooldown_sub` | busbar |
| role flags | `connected`, `domain_mask` | all |

Maintenance is a per-**line** quantity. Appending it would put a line
measurement after the substation measurement and the role flags, splitting the
line block in two. Inserting keeps every line-owned column contiguous.

### It does not matter to the model

The first linear layer sees the whole vector. Permuting input columns permutes
the weight columns; initialisation is random. Column order is a readability
choice, not a modelling one.

### Where it does matter

**Checkpoint compatibility across the flag.** If the optional block were
appended, the 13-column layout would be a strict *prefix* of the 15-column one,
so a source checkpoint's first-layer weights would align with the target's first
13 columns and could be loaded with a zero-pad. With mid-insertion, three columns
change meaning and no correspondence exists.

Not hypothetical in this codebase: `common/transfer.py` already contains
`_migrate_edge_input_weight`, which splices the obsolete `relation_self` column
out of an old checkpoint's weight matrix so it fits the current edge vector. The
"a feature moved, realign the matrix" problem has been paid for once already.

### Whether it bites here

```
bus14: maintenance=False        bus36: maintenance=True
bus36_wcci: maintenance=True    bus36_wcci_nomaint: maintenance=False
```

A `bus14` → WCCI transfer would cross this boundary, but Section 7.1.1 removes
maintenance from WCCI — for a better reason than layout: a policy trained
without it would face an unseen hazard, and do-nothing survival is 21.7 % with
against 56.0 % without. The transfer runs use `bus36_wcci_nomaint`, so both
sides have 13 columns and the ordering is never exercised.

**Verdict:** leave it. The grouping rationale is sound, the cost is confined to a
checkpoint-reuse case the design already avoids, and changing it would alter the
observation for no result-level benefit.

---

## Q14. For mean/max pooling, what should be included? Visible nodes, controlled only, energized only, or controlled and energized?

**Where:** Section 4.4.1, *Mean, maximum, and sum pooling*.

**`controlled_mean`.** It is both the most sensible and the easiest to defend.

### The principle

The readout produces the vector the action head reads, and in the
candidate-scoring head it is the global context concatenated to every action. It
should answer one question: *what is the state of the region this agent is
deciding about?* That region is the action domain.

Excluding context from the average does not discard it — it arrives through
message passing, which is what the graph is for. Verified: on the chain fixture
the `controlled_mean` embedding differs across topologies that differ *only* in
context (0.6925 / 0.7002 / 0.7475). The question is not whether context informs
the summary, but whether it does so **as structure or as an addend**. Structure
uses the model instead of bypassing it.

### The mechanical argument, which is stronger

A **fixed** pooled set makes any systematic offset — including the zeros from
empty controlled busbars — a constant that the readout projection absorbs during
training. A **varying** pooled set turns that offset into a topology-dependent
quantity the network must separate from real signal.

Measured on agent 0 of `bus14`: the visible set is **10** nodes nominally,
**11** once a contextual substation splits, **9** after a boundary line trips.
An ordinary mean therefore rescales every node it keeps when a *neighbouring*
agent acts.

### Why the energized variants are the weakest

They remove exactly what actions target. **Four of agent 0's eight busbars are
de-energized at nominal topology** — the empty second busbar of each substation,
which is the busbar a splitting action moves an element onto. A readout that
drops them cannot represent "a free busbar is available", the most directly
actionable fact in the observation. And the denominator moves with every switch,
making it the least stable of the four.

`controlled_energized_mean` inherits that objection *and* reintroduces a varying
denominator inside a set that would otherwise be fixed.

### Two qualifications

- **Vacuous without context.** With `gnn_include_neighbors = false`, controlled
  and visible coincide (measured 22 = 22 = 22 on the `cas_hl` graphs), so
  `mean` already *is* `controlled_mean`. The choice bites only in the 287
  configurations that enable context.
- **Max is less exposed.** It does not depend on pool size, and a de-energized
  row of zeros seldom wins a maximum, so the energized variants matter far less
  there — though not never, since `theta_diff` can be negative. Its own risk is
  domination: with context included, a neighbour under greater stress owns the
  channel maxima, so the summary reports the neighbour's worst channel rather
  than the agent's. That argues for `controlled_max` on the same grounds.

### A bonus: it sharpens the argument against sum

Section 4.4.1 rules out sum because its magnitude follows the node count. Under a
controlled readout that count is fixed for the episode, so that objection
dissolves. What remains is that controlled sets differ **between agents** —
8, 8 and 12 busbars on `bus14` — while the encoder is shared, so a sum would
hand different agents systematically different magnitudes. Same conclusion,
better reason.

### Recommendation

Make `controlled_mean` the default, and since all screens are being rerun, add
the controlled dimension to Screen B rather than assuming it. That converts the
most defensible choice into a measured one.
