# Decentralized Sparse Intervention Control

This document merges the original intervention-gate roadmap with the adaptive
intervention-budget notes. It describes the sparse-control experiments that were
implemented around the `best_00` MAPPO/GNN setup:

- evaluation-time rho heuristic overrides,
- intervention-gated actors,
- fixed sparse-action penalties (`sparse16`),
- adaptive Lagrangian intervention budgets (`aib_*` and `aibm_*`).

The common goal is to make decentralized topology-control agents more
operator-like:

```text
act rarely,
act locally,
act when the local grid state makes action useful,
and keep the final policy decentralized.
```

## Scientific Motivation

The gate direction was inspired by hierarchical topology-control work, in
particular [Manczak et al., 2023][manczak-hrl]. The relevant idea is that
topology control can be decomposed into:

```text
do nothing vs propose topology change
choose where to act
choose local topology configuration
```

In this repository, only the first level was adapted to decentralized MAPPO:

```text
each regional agent decides locally whether to do nothing or intervene.
```

This is also related to hierarchical MARL formulations for topology
optimization, such as [van der Sar et al., 2023][vandersar-marl], but the
implementation here deliberately avoids a central coordinator or communication
between agents at action time.

The later sparse-control direction is closer to constrained and budgeted RL:

- constrained MDPs / Constrained Policy Optimization: maximize reward subject
  to a cost constraint, following the general formulation in
  [Achiam et al., 2017][achiam-cpo],
- Lagrangian constrained RL: learn a multiplier for the constraint instead of
  choosing one fixed penalty by hand; the multiplier update used here is the
  simple integral-style Lagrangian update discussed as the baseline form in
  [Stooke et al., 2020][stooke-pid],
- budgeted RL: treat non-idle interventions as a resource that should remain
  under a budget, in the spirit of [Carrara et al., 2019][carrara-budgeted],
- power-grid multi-objective topology control: topology switching frequency and
  topology deviation are operational objectives, not just cosmetic metrics, as
  emphasized by [Lautenbacher et al., 2025][lautenbacher-morl].

## Baseline Action Selection

For the original flat actor, each agent has one categorical policy over its full
local action space:

$$
\pi_i(a_i \mid o_i),
\qquad
a_i \in \{0,1,\dots,|\mathcal A_i|-1\}.
$$

Action `0` is the no-op action.

During training, actions are sampled:

$$
a_{i,t} \sim \pi_i(\cdot \mid o_{i,t}).
$$

During deterministic evaluation, the default behavior is greedy:

$$
a_{i,t}^{eval}
= \arg\max_a \pi_i(a \mid o_{i,t}).
$$

This distinction matters. Training remains stochastic, while evaluation usually
uses the most likely action.

## Phase 0: Baseline Diagnostics

Before modifying the objective or the actor, the first step was to measure how
often agents actually take non-idle actions.

Implemented logs:

```text
train/non_idle_agents_count_0_frac
train/non_idle_agents_count_1_frac
train/non_idle_agents_count_2_frac
train/non_idle_agents_count_3_frac
train/frac_any_non_idle
train/frac_multi_agent_non_idle
train/frac_action_0_agent_*
train/illegal_action_rate_agent_*
```

These logs answer:

```text
How often does no agent act?
How often does exactly one agent act?
How often do multiple agents act simultaneously?
Does high survival come from sparse action or frequent topology churn?
```

## Evaluation-Time Heuristic Override

The heuristic override is evaluation-only. It does not modify the training
objective and it does not teach the policy to act sparsely. The policy proposes
actions, then the evaluator may replace some of them by action `0`. It should
be interpreted as a rule-based diagnostic in the style of operational
topology-control analyses, not as the final learned method. This is closest in
spirit to using explicit operational rules to filter or revert topology
actions, as discussed for power-grid agents by [Lehna et al.,
2023][lehna-comparative].

Let:

$$
\bar a_{i,t}
= \arg\max_a \pi_i(a \mid o_{i,t})
$$

be the policy action proposed by agent $i$ at evaluation time.

### Global Rho Heuristic

The global heuristic uses the pre-action global maximum line loading:

$$
\rho_t^{global}
= \max_{\ell \in \mathcal L} \rho_{\ell,t}.
$$

The executed action is:

$$
a_{i,t}^{exec}
=
\begin{cases}
0,
& \text{if } \rho_t^{global} < \rho_{safe},\\
\bar a_{i,t},
& \text{otherwise.}
\end{cases}
$$

with:

$$
\rho_{safe}=0.90.
$$

This forces all agents to no-op when the whole grid is considered safe.

### Local Rho Heuristic

The local heuristic is decentralized. Each agent computes a local maximum rho
from the lines touching its own regional substations:

$$
\rho_{i,t}^{local}
= \max_{\ell \in \mathcal L_i} \rho_{\ell,t}.
$$

The executed action is:

$$
a_{i,t}^{exec}
=
\begin{cases}
0,
& \text{if } \rho_{i,t}^{local} < \rho_{safe},\\
\bar a_{i,t},
& \text{otherwise.}
\end{cases}
$$

If a line lies between two regions, it belongs to the local neighborhood of both
adjacent agents because it touches substations from both zones. This does not
require communication: both agents can independently observe that boundary line
through their local physical neighborhood.

### Interpretation

The heuristic is useful as a diagnostic because it shows what happens if actions
are blocked in safe states. However, it is not a learned solution:

```text
training: unchanged
evaluation: hard action override
```

Therefore, heuristic action-0 rates must be interpreted as final executed
evaluation actions, not as proof that the policy learned to prefer action `0`.

## Intervention-Gated Actor

The gated actor is the local decentralized adaptation of the top level of the
hierarchy in [Manczak et al., 2023][manczak-hrl]: before choosing a topology
change, the agent first decides whether an intervention is needed at all. The
gated actor changes the actor architecture. Each agent has two heads after the
encoder:

$$
\pi_i^{gate}(g_i \mid o_i),
\qquad
g_i \in \{0,1\}
$$

where:

```text
0 = do nothing
1 = intervene
```

and:

$$
\pi_i^{nonidle}(b_i \mid o_i),
\qquad
b_i \in \{1,\dots,|\mathcal A_i|-1\}.
$$

During training:

$$
g_{i,t} \sim \pi_i^{gate}(\cdot \mid o_{i,t}).
$$

If:

$$
g_{i,t}=0,
$$

then:

$$
a_{i,t}=0.
$$

If:

$$
g_{i,t}=1,
$$

then:

$$
a_{i,t} \sim \pi_i^{nonidle}(\cdot \mid o_{i,t}).
$$

The PPO log-probability of the final executed action is:

$$
\log P(a_i=0)
=
\log P(g_i=0),
$$

and for a non-idle action $a_i>0$:

$$
\log P(a_i)
=
\log P(g_i=1)
+
\log P(b_i=a_i \mid g_i=1).
$$

### Gated Evaluation Modes

Two deterministic decoders were implemented.

`final_action_map` computes final executed-action probabilities:

$$
P(a_i=0)=P(g_i=0),
$$

$$
P(a_i=a>0)
=
P(g_i=1)P(b_i=a \mid g_i=1),
$$

then returns:

$$
a_i^{eval}
=
\arg\max_a P(a_i=a).
$$

`hierarchical_greedy` follows the hierarchy strictly:

$$
\text{if } P(g_i=0) > P(g_i=1),
\quad
a_i^{eval}=0,
$$

otherwise:

$$
a_i^{eval}
=
\arg\max_{a>0} P(b_i=a \mid g_i=1).
$$

The `sparse16_gated_*` runs use a hierarchical actor, but their evaluation mode
is `final_action_map`, not `hierarchical_greedy`.

### Entropy Issue

The original coupled entropy was:

$$
H_i
=
H(g_i)
+
P(g_i=1)H(b_i).
$$

This can accidentally reward intervention because increasing $P(g_i=1)$
unlocks more non-idle entropy.

The corrected optional entropy is:

$$
H_i
=
\alpha_{gate}H(g_i)
+
\alpha_{nonidle}H(b_i).
$$

This is enabled with:

```bash
--intervention-gate-entropy-mode separate
--intervention-gate-entropy-mult 0.2
--intervention-nonidle-entropy-mult 1.0
```

### Gate Interpretation

The gate can restrict training exploration more strongly than a reward penalty.
If the gate collapses early to no-op, then the non-idle action head is rarely
executed and receives little useful learning signal.

Empirically, this made the gate less convincing than the flat actor with a
sparse objective.

## Sparse16: Fixed Sparse-Action Penalty

The `phase4_sparse_control_16` sweep tests a fixed penalty for non-idle actions.
This is the simplest way to encode that switching frequency is an operational
objective. It is less principled than a constrained formulation, but it is a
strong baseline because power-grid topology-control papers explicitly treat
switching frequency and topology deviation as objectives that trade off against
congestion relief [Lautenbacher et al., 2025][lautenbacher-morl].

The reward used for training is:

$$
r'_{i,t}
=
r_{i,t}
-
\lambda_{fixed}
\mathbf{1}[a_{i,t}\neq 0].
$$

where:

$$
\lambda_{fixed}
\in
\{0.0,0.001,0.003,0.01\}.
$$

This penalty is applied to the final executed action of each agent, so it works
for both:

```text
flat actors
gated actors
```

All Sparse16 configs set:

```toml
safe_intervention_penalty = 0.0
```

so there is no rho threshold and no state-dependent weighting.

### Sparse16 Experiment Grid

The sweep contains:

```text
actor in {flat, gated}
lambda_fixed in {0.0, 0.001, 0.003, 0.01}
seed in {0,1,2}
```

This gives:

$$
2 \times 4 \times 3 = 24
$$

runs.

The key comparison is:

```text
sparse16_flat_p010
vs
sparse16_gated_p010
```

because it asks whether the gate adds value once a simple sparse-action
objective already exists.

### Training-Time And Evaluation-Time Effect

Sparse16 affects training rewards only:

```text
training: reward is changed
evaluation: no action override
```

Actions are still sampled during training. Therefore Sparse16 does not hard-mask
exploration. It merely says:

```text
a non-idle action must be useful enough to pay its fixed cost.
```

## Adaptive Intervention Budget

The adaptive intervention budget turns sparse topology control into a local
constraint. This is the part most directly connected to constrained RL
[Achiam et al., 2017][achiam-cpo], adaptive Lagrangian methods
[Stooke et al., 2020][stooke-pid], and budgeted RL
[Carrara et al., 2019][carrara-budgeted].

For each agent $i$, define an intervention cost:

$$
c_i(s_t,a_{i,t})
=
\mathbf{1}[a_{i,t}\neq 0]w_i(s_t).
$$

The desired constraint is:

$$
\mathbb E_{\pi_i}
\left[
c_i(s_t,a_{i,t})
\right]
\leq d.
$$

The policy should maximize reward while respecting this budget:

$$
\max_{\pi_i} J_i(\pi_i)
$$

subject to:

$$
\bar C_i(\pi_i) \leq d.
$$

The Lagrangian is:

$$
\mathcal L_i(\pi_i,\lambda_i)
=
J_i(\pi_i)
-
\lambda_i
\left(
\bar C_i(\pi_i)-d
\right),
\qquad
\lambda_i \geq 0.
$$

The constant term $\lambda_i d$ does not affect the PPO action-gradient
inside a rollout, so the implemented reward is:

$$
r'_{i,t}
=
r_{i,t}
-
\lambda_i c_i(s_t,a_{i,t}).
$$

After each rollout, the multiplier is updated as:

$$
\lambda_i
\leftarrow
\mathrm{clip}
\left(
\lambda_i
+
\eta
\left(
\hat C_i-d
\right),
0,
\lambda_{\max}
\right),
$$

with:

$$
\hat C_i
=
\frac{1}{T}
\sum_{t=1}^{T}
c_i(s_t,a_{i,t}).
$$

If:

$$
\hat C_i > d,
$$

then:

$$
\lambda_i \uparrow,
$$

and future interventions become more expensive.

If:

$$
\hat C_i < d,
$$

then:

$$
\lambda_i \downarrow,
$$

and the intervention penalty relaxes.

All AIB runs start from:

$$
\lambda_i(0)=0.
$$

The usual settings are:

```toml
intervention_budget_lr = 0.02
intervention_budget_init_lambda = 0.0
intervention_budget_max_lambda = 10.0
```

## AIB Cost Modes

### Nonidle Cost

The simplest cost mode is:

$$
w_i(s_t)=1.
$$

Therefore:

$$
c_i(s_t,a_{i,t})
=
\mathbf{1}[a_{i,t}\neq 0].
$$

This is used by:

```text
aib_04_flat_nonidle_t020
```

It has the same cost definition as Sparse16, but an adaptive coefficient:

$$
\text{Sparse16: } \lambda_i = \lambda_{fixed}
$$

$$
\text{AIB nonidle: } \lambda_i \text{ is updated during training.}
$$

For the cached `aib_04` runs, the final learned mean multipliers were roughly:

```text
s0: 0.029
s1: 0.036
s2: 0.063
```

with mean values over training around:

```text
0.066, 0.117, 0.076
```

So `aib_04` is effectively an automatically tuned sparse penalty.

### Global-Safe Cost

The global-safe cost uses global pre-action max rho:

$$
\rho_t^{global}
=
\max_{\ell \in \mathcal L}
\rho_{\ell,t}.
$$

The safety weight is:

$$
w^{global}(s_t)
=
\sigma
\left(
k(\rho_{safe}-\rho_t^{global})
\right).
$$

The cost is:

$$
c_i(s_t,a_{i,t})
=
\mathbf{1}[a_{i,t}\neq 0]
w^{global}(s_t).
$$

This penalizes non-idle actions mostly when the whole grid is safe.

### Local-Safe Cost

The local-safe cost is the decentralized version. Each agent computes:

$$
\rho_{i,t}^{local}
=
\max_{\ell \in \mathcal L_i}
\rho_{\ell,t}.
$$

Then:

$$
w_i^{local}(s_t)
=
\sigma
\left(
k(\rho_{safe}-\rho_{i,t}^{local})
\right).
$$

The cost is:

$$
c_i(s_t,a_{i,t})
=
\mathbf{1}[a_{i,t}\neq 0]
w_i^{local}(s_t).
$$

When the local grid is safe:

$$
\rho_{i,t}^{local} < \rho_{safe}
\quad \Rightarrow \quad
w_i^{local}(s_t) \approx 1,
$$

so actions are expensive.

When the local grid is stressed:

$$
\rho_{i,t}^{local} \approx 1
\quad \Rightarrow \quad
w_i^{local}(s_t) \approx 0,
$$

so actions are cheap.

The default settings are:

```toml
intervention_budget_cost_mode = "local_safe"
intervention_budget_target = 0.20
intervention_budget_rho_threshold = 0.90
intervention_budget_rho_sharpness = 25.0
```

Important: $\rho_{safe}=0.90$ is not a hard action override in AIB. It is the
center of a smooth sigmoid.

## AIB First Runs

The first AIB folder is:

```text
configs/adaptive_intervention_budget_7
```

It should be read as a first ablation around the adaptive budget.

### `aib_00_flat_local_t020`

Flat actor, local-safe cost:

$$
d=0.20.
$$

This is the main candidate:

```text
Can the original flat actor learn sparse behavior from an adaptive local budget?
```

### `aib_01_flat_local_t010`

Flat actor, local-safe cost:

$$
d=0.10.
$$

This is stricter:

$$
0.10 < 0.20.
$$

It should increase action `0`, but may reduce survival if the budget is too
restrictive.

### `aib_02_flat_local_t035`

Flat actor, local-safe cost:

$$
d=0.35.
$$

This is looser:

$$
0.35 > 0.20.
$$

It should preserve survival more easily, but may allow more non-idle actions.

### `aib_03_gate_hgreedy_sep_local_t020`

Gated actor, local-safe cost:

$$
d=0.20.
$$

Evaluation uses:

```text
hierarchical_greedy
```

Entropy uses the separated gate/non-idle entropy mode. This tests:

```text
Does the gate help once sparsity is already in the objective?
```

Empirically, this was mostly a negative diagnostic because the gate could
collapse toward no-op.

### `aib_04_flat_nonidle_t020`

Flat actor, nonidle cost:

$$
d=0.20,
\qquad
c_i(s_t,a_{i,t})
=
\mathbf{1}[a_{i,t}\neq 0].
$$

This tests:

```text
Is an adaptive plain non-idle budget enough, or do we need local rho weighting?
```

The comparison:

```text
aib_00_flat_local_t020
vs
aib_04_flat_nonidle_t020
```

isolates the value of the local-safe state-dependent cost.

## Mechanism Ablation: 15 Runs

The folder:

```text
configs/adaptive_intervention_budget_mechanism_15
```

contains 15 additional runs, all using the flat actor and seeds `0,1,2`.

The purpose is to isolate which part of AIB matters:

```text
fixed vs adaptive coefficient
state-dependent vs plain non-idle cost
global vs local rho weighting
rho threshold sensitivity
```

### `aibm_00_fixed_nonidle_p006`

Fixed plain non-idle penalty:

$$
r'_{i,t}
=
r_{i,t}
-
0.06
\mathbf{1}[a_{i,t}\neq 0].
$$

No adaptive lambda and no rho weighting.

This tests whether a fixed coefficient near the learned AIB lambda scale is
enough.

### `aibm_01_fixed_global_safe_p006`

Fixed global-safe penalty:

$$
r'_{i,t}
=
r_{i,t}
-
0.06
\mathbf{1}[a_{i,t}\neq 0]
\mathbf{1}[\rho_t^{global}<0.90].
$$

This tests whether state-dependent safety weighting helps without adaptive
lambda.

### `aibm_02_adaptive_global_t020`

Adaptive global-safe budget:

$$
c_i(s_t,a_{i,t})
=
\mathbf{1}[a_{i,t}\neq 0]
\sigma(k(0.90-\rho_t^{global})).
$$

with:

$$
d=0.20.
$$

This tests whether global-safe adaptive weighting works, and should be compared
against local-safe AIB.

### `aibm_03_adaptive_local_t020_r085`

Adaptive local-safe budget with:

$$
\rho_{safe}=0.85,
\qquad
d=0.20.
$$

This tests a lower safety threshold. Fewer states are considered safe, so fewer
actions are strongly penalized.

### `aibm_04_adaptive_local_t020_r095`

Adaptive local-safe budget with:

$$
\rho_{safe}=0.95,
\qquad
d=0.20.
$$

This tests a higher safety threshold. More states are considered safe, so more
actions are strongly penalized.

## Main Comparisons

| Question | Compare |
| --- | --- |
| Does a fixed sparse penalty already work? | baseline vs `sparse16_flat_p010` |
| Does the gate help beyond a sparse objective? | `sparse16_flat_p010` vs `sparse16_gated_p010` |
| Fixed vs adaptive coefficient? | `sparse16_flat_p010` or `aibm_00_fixed_nonidle_p006` vs `aib_04_flat_nonidle_t020` |
| Does local-safe weighting matter? | `aib_00_flat_local_t020` vs `aib_04_flat_nonidle_t020` |
| Does global-safe weighting matter? | `aibm_02_adaptive_global_t020` vs `aib_00_flat_local_t020` |
| Is threshold `0.90` special? | `aibm_03` $0.85$, `aib_00` $0.90$, `aibm_04` $0.95$ |
| Is the heuristic merely an evaluation trick? | heuristic runs vs learned sparse/AIB runs |

## Training-Time vs Evaluation-Time Effects

| Method | Training effect | Evaluation effect | Exploration risk |
| --- | --- | --- | --- |
| Baseline | Sample from flat policy | Greedy argmax by default | Standard entropy-controlled exploration |
| Heuristic override | None | Hard no-op override in safe states | No training restriction, but evaluation is manually changed |
| Sparse16 | Fixed reward penalty | No action override | Softly discourages non-idle actions |
| AIB nonidle | Adaptive reward penalty | No action override | Softly discourages non-idle actions, coefficient can grow |
| AIB local-safe | Adaptive state-dependent reward penalty | No action override | Penalizes safe-state interventions more than stressed-state interventions |
| Gate | Changes action factorization and sampling | Final-action MAP or hierarchical greedy | Can starve non-idle exploration if the gate collapses |

The safest mechanisms, from an exploration perspective, are the reward-shaping
ones:

```text
Sparse16
AIB nonidle
AIB local-safe
```

because they do not hard-mask actions. They still allow the policy to sample
non-idle actions during training; those actions are simply penalized if they are
not useful enough.

The heuristic is useful as a diagnostic, but it is not learned. The gate is
learned, but it can restrict exploration more aggressively than a reward
penalty.

## Logs To Use

Action sparsity:

```text
train/frac_action_0_agent_i
train/explain/action_nonidle_agent_i
test/explain/frac_action_0_agent_i
test/explain/action_nonidle_agent_i
train_eval/explain/frac_action_0_agent_i
train_eval/explain/action_nonidle_agent_i
```

Joint action sparsity:

```text
train/non_idle_agents_count_0_frac
train/non_idle_agents_count_1_frac
train/non_idle_agents_count_2_frac
train/non_idle_agents_count_3_frac
train/frac_any_non_idle
train/frac_multi_agent_non_idle
```

Sparse fixed penalty:

```text
train/intervention_penalty_coef
train/intervention_penalty_mean
train/intervention_penalty_total
train/intervention_penalty_mean_agent_i
```

Adaptive budget:

```text
train/intervention_budget_lambda_agent_i
train/intervention_budget_lambda_before_agent_i
train/intervention_budget_lambda_updated_agent_i
train/intervention_budget_cost_agent_i
train/intervention_budget_cost_violation_agent_i
train/intervention_budget_penalty_mean_agent_i
train/intervention_budget_safety_weight_agent_i
train/intervention_rate_when_budget_costly_agent_i
train/intervention_rate_when_budget_free_agent_i
train/intervention_budget_lambda_mean
train/intervention_budget_cost_mean
train/intervention_budget_cost_violation_mean
train/intervention_budget_penalty_mean
train/intervention_budget_safety_weight_mean
```

Heuristic override:

```text
test/heuristic/rho_threshold
test/heuristic/is_local
test/heuristic/force_noop_frac
test/heuristic/force_noop_any_agent_frac
```

Gate:

```text
train/intervention_gate_do_nothing_frac_agent_i
train/intervention_gate_intervene_frac_agent_i
train/intervention_gate_prob_do_nothing_agent_i
train/intervention_gate_prob_intervene_agent_i
train/intervention_gate_entropy_agent_i
train/nonidle_action_entropy_agent_i
```

Physical explainability:

```text
train/explain/pre_max_rho
train/explain/post_max_rho
train/explain/delta_max_rho
train/explain/topology_distance_before
train/explain/topology_distance_after
train/explain/topology_distance_delta
test/explain/*
train_eval/explain/*
```

## Current Interpretation

The empirical story so far is:

```text
The gate is not the strongest contribution.
Sparse reward shaping already improves action-0 usage.
The adaptive budget makes the sparse penalty self-tuning.
The local-safe cost makes the penalty more physically meaningful.
```

In particular:

```text
sparse16_flat_p010
```

is a strong baseline because it shows that a small fixed non-idle cost can
produce high action-0 usage while preserving survival.

The most principled learned sparse-control method is:

```text
aib_00_flat_local_t020
```

because it keeps the original decentralized actor, avoids evaluation-time
overrides, learns the intervention penalty adaptively, and uses a local
state-dependent safety weight.

## References

- [Manczak et al., 2023][manczak-hrl]: hierarchical power-network topology
  control. Used here to motivate the local `do nothing / intervene` decision.
- [van der Sar et al., 2023][vandersar-marl]: hierarchical MARL for power-grid
  topology optimization. Related motivation for decomposing large topology
  action spaces, though this implementation remains decentralized.
- [Lehna et al., 2023][lehna-comparative]: rule-based and RL topology-control
  comparison with operational diagnostics. Used here to contextualize the
  evaluation-time rho heuristic as a diagnostic rule, not as learned control.
- [Achiam et al., 2017][achiam-cpo]: constrained policy optimization. Used here
  for the constrained objective
  $\mathbb E[c_i(s_t,a_{i,t})]\leq d$.
- [Stooke et al., 2020][stooke-pid]: PID Lagrangian methods. Used here to
  justify adapting the multiplier $\lambda_i$ from observed constraint
  violation; this code implements the simpler integral-style Lagrangian update.
- [Carrara et al., 2019][carrara-budgeted]: budgeted RL. Used here to frame
  non-idle interventions as a budgeted resource.
- [Lautenbacher et al., 2025][lautenbacher-morl]: multi-objective RL for power
  grid topology control. Used here to justify switching frequency and topology
  deviation as operational objectives.

## Non-Goals

- No central coordinator.
- No inter-agent communication.
- No Gibbs risk-prior surrogate.
- No action mask during training.
- No hard TopK action restriction.
- No evaluation-only heuristic as the final learned method.

[manczak-hrl]: https://arxiv.org/abs/2311.02129
[vandersar-marl]: https://arxiv.org/abs/2310.02605
[lehna-comparative]: https://arxiv.org/abs/2304.00765
[achiam-cpo]: https://arxiv.org/abs/1705.10528
[stooke-pid]: https://arxiv.org/abs/2007.03964
[carrara-budgeted]: https://arxiv.org/abs/1903.01004
[lautenbacher-morl]: https://arxiv.org/abs/2502.00040
