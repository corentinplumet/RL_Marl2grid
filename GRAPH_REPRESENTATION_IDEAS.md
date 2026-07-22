# Alternative Power-Grid Graph Representations

The current implementation already supports several useful representations:

- a busbar-only graph with transmission lines represented as edges;
- a heterogeneous graph containing busbars, generators, and loads;
- a heterogeneous graph with explicit transmission-line nodes;
- virtual graph nodes for learned global readout;
- learned substation summary nodes; and
- direct edges between busbars belonging to the same substation.

The main remaining directions are therefore not simply additional equipment
types. They concern electrical physics, temporal information, operational
constraints, and the way possible actions affect the grid.

## Candidate Representations

| Representation | Main idea | Particularly useful for |
|---|---|---|
| **Physics-weighted graph** | Weight physical edges with resistance, reactance, admittance, transformer ratio, phase shift, and thermal limit. Complex quantities can be represented using separate real and imaginary channels. | Power flow, overload prediction, and topology control |
| **Spatio-temporal graph** | Process the last \(K\) grid states, connecting each entity to itself over time or combining a spatial GNN with a GRU or Transformer. | Anticipating overloads, cooldowns, and cascading effects |
| **Factor or constraint graph** | Create variable nodes for voltages, injections, and flows, together with factor nodes for power balance, branch equations, and operational limits. | Physics-informed learning and state estimation |
| **Electrical-sensitivity graph** | Add nonphysical relations between electrically influential buses using PTDF, LODF, effective resistance, or another electrical-distance metric. | Learning the distant consequences of switching actions |
| **Line graph** | Make every transmission line a node and directly connect lines that share a busbar. | Congestion, line outages, and cascade prediction |
| **Hypergraph** | Represent a substation as a hyperedge joining all its busbars, line terminals, generators, and loads. | Multi-element topology actions |
| **Action-conditioned graph** | Add candidate-action nodes connected to every component affected by the corresponding action. | Comparing possible interventions inside the policy network |
| **Multi-scale graph** | Organize the grid into equipment, busbar, substation, area, and global levels. | Large grids and multi-agent control |
| **Contingency graph set** | Generate one graph for the current state and additional graphs for candidate \(N-1\) outages. | Security assessment and robust policies |

## Recommended Priorities

### 1. Physics-Weighted Edges

This is probably the easiest high-value extension. Pure adjacency treats a
weak electrical connection and a strong one almost identically, although their
influence on the grid can be very different. A physical edge could contain
features such as

\[
(r, x, g, b, \tau, \phi, S^{\max}, p_{or}, p_{ex}, q_{or}, q_{ex}),
\]

where \(r\) and \(x\) are resistance and reactance, \(g+jb\) is the complex
admittance, \(\tau\) and \(\phi\) describe a transformer, and \(S^{\max}\) is
the thermal limit. Endpoint flow values preserve the direction of the current
operating state.

Complex-valued and physics-informed graph models have been proposed
specifically for power-system applications:

- [Complex-Value Spatio-temporal Graph Convolutional Neural Networks and its Applications to Electric Power Systems AI](https://arxiv.org/abs/2208.08485)
- [Physics-informed Graphical Neural Network for Power System State Estimation](https://arxiv.org/abs/2312.17738)

### 2. Spatio-Temporal Representation

This may be the largest missing source of information for reinforcement
learning. Overflow duration, cooldowns, previous actions, and changing demand
cannot always be inferred from a single grid snapshot.

Two possible implementations are:

1. encode each of the last \(K\) graph observations with a shared GNN and pass
   the resulting embeddings through a GRU or Transformer; or
2. construct a space-time graph containing temporal edges between consecutive
   observations of the same entity.

Relevant work includes [PowerGNN: A Topology-Aware Graph Neural Network for
Electricity Grids](https://arxiv.org/abs/2503.22721).

### 3. Electrical-Sensitivity Relations

A switching action can influence a distant line even when the affected
components are separated by many graph hops. Additional PTDF-, LODF-, or
electrical-distance-based relations could expose these dependencies directly.

These relations should remain distinguishable from physical transmission
lines. For example, a multi-relational GNN could process `physical_line` and
`electrical_influence` edges with different message functions. Each bus could
be connected only to its top-\(k\) most influential buses to prevent the graph
from becoming dense.

### 4. Factor or Constraint Graph

This is a more principled physics representation, but it requires a larger
architectural change. Message passing can take place between electrical
variables and constraints such as power balance, branch equations, and thermal
limits. Constraint-node embeddings can also explicitly indicate which physical
conditions are close to being violated.

Power-system factor graphs have already been used to combine bus and branch
measurements in graph-based state estimation:

- [Robust and Fast Data-Driven Power System State Estimator Using Graph Neural Networks](https://arxiv.org/abs/2206.02731)

### 5. True Line Graph

The existing explicit-transmission-line-node representation is an incidence or
bipartite representation: busbar nodes exchange messages with line nodes. A
true line graph is different because two transmission-line nodes are connected
directly whenever the original lines meet at a busbar.

This representation makes quantities such as loading ratio, overflow duration,
disconnection state, and thermal limit natural node features. It is therefore
well suited to congestion and cascading-outage objectives.

### 6. Hypergraph and Multi-Scale Variants

The learned substation-summary-node representation already approximates a
hypergraph through a bipartite expansion: the substation node connects all
busbars that belong to the same substation. Consequently, implementing a
dedicated hypergraph may provide less improvement than physics-weighted,
temporal, or sensitivity-based representations.

For larger grids, however, the hierarchy could be extended as follows:

\[
\text{equipment} \longleftrightarrow
\text{busbar} \longleftrightarrow
\text{substation} \longleftrightarrow
\text{area} \longleftrightarrow
\text{global node}.
\]

## Suggested First Experiment

The first experiment should combine

\[
\text{busbar/equipment graph}
+ \text{electrical edge attributes}
+ \text{PTDF relations}
+ \text{short temporal history}.
\]

A fair ablation would compare:

1. the current graph baseline;
2. the physics-weighted graph;
3. the temporal graph; and
4. the combined physics-weighted and temporal graph.

The model size, training budget, random seeds, and action space should remain
approximately constant so that differences can be attributed to the graph
representation rather than additional model capacity.

