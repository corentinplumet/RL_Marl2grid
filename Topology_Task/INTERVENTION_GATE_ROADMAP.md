# Decentralized Intervention Gate Roadmap

## Motivation

The current independent MAPPO agents can all choose non-idle topology actions at
the same environment step. This is powerful on small grids, but it is not very
operator-like: real grid operation is sparse, and most timesteps should result
in no topology change. When an intervention is needed, the intervention should
be local, auditable, and physically meaningful.

The proposed direction is:

```text
We adapt hierarchical topology-control ideas to decentralized MAPPO by giving
each regional agent an explicit learned intervention gate. Each agent first
learns whether intervention is necessary in its own region, and only then
chooses a topology action. This encourages sparse, operator-like control without
introducing inter-agent communication.

The agents do not just learn high reward.
They learn when not to act, when to act, and we can explain the physical effect
of each intervention.
```

This is inspired mainly by hierarchical topology-control work that decomposes
grid operation into:

```text
do nothing vs propose topology change
choose where to act
choose local topology configuration
```

Our adaptation keeps the first hierarchy local to each regional agent. There is
no central coordinator and no message passing between agents at action time.

Relevant papers:

- "Hierarchical Reinforcement Learning for Power Network Topology Control"
  (Manczak, Viebahn, van Hoof, 2023): source of the hierarchical
  "do nothing vs topology change" framing.
- "Multi-Agent Reinforcement Learning for Power Grid Topology Optimization"
  (van der Sar, Zocca, Bhulai, 2023): related motivation for hierarchical MARL
  in topology control.
- "Managing power grids through topology actions: A comparative study between
  advanced rule-based and reinforcement learning agents" (Lehna et al., 2023):
  motivation for operational analysis, N-1 reasoning, and topology reversion.

## Design Principle

Each agent owns its decision. At time `t`, agent `i` receives its usual local
observation and produces two decisions:

```text
gate_i:
  0 = do nothing
  1 = intervene

local_action_i:
  one non-idle topology action from the agent's local action space
```

Execution rule:

```text
if gate_i == do nothing:
    executed_action_i = 0
else:
    executed_action_i = sampled non-idle local action
```

This avoids duplicate no-op paths. Action `0` belongs to the gate, while the
local action head is responsible only for actions `1..n_actions_i-1`.

## Phase 0: Baseline Diagnostics

Goal: establish what the current independent MAPPO policy is doing before we
change the architecture.

Status: implemented in the clean intervention branch through W&B training logs.

Add or reuse logs:

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

Expected use:

- Quantify how often each agent acts.
- Quantify how often multiple agents act simultaneously.
- Identify whether high reward comes from sparse operator-like actions or from
  frequent topology churn.

Success criterion:

```text
We can show the current intervention pattern before introducing the gate.
```

## Phase 1: Add The Intervention-Gated Actor

Goal: replace the single flat categorical action head with a hierarchical local
policy.

Current actor:

```text
obs_i -> encoder_i -> logits over [0, 1, ..., n_actions_i-1]
```

New actor:

```text
obs_i -> encoder_i -> gate_logits over [do_nothing, intervene]
                 \-> nonidle_action_logits over [1, ..., n_actions_i-1]
```

Sampling:

```text
gate_i ~ Categorical(gate_logits)

if gate_i == do_nothing:
    action_i = 0
else:
    action_i = 1 + Categorical(nonidle_action_logits).sample()
```

For PPO, the final action log-probability is:

```text
if action_i == 0:
    logprob_i = log P(gate_i = do_nothing)
else:
    logprob_i = log P(gate_i = intervene)
              + log P(nonidle_action_i = action_i - 1)
```

The exact entropy of this mixture is:

```text
entropy_i = H(gate_i) + P(gate_i = intervene) * H(nonidle_action_i)
```

Implementation notes:

- Keep the old actor as the default.
- Add a flag such as `--intervention-gate true`.
- Do not change the environment action space.
- Do not add communication between agents.
- Store only the final executed discrete action in the rollout buffer; the gate
  decision can be inferred from `action == 0`.

Success criterion:

```text
The gated actor trains with PPO and can reproduce the old behavior when the
gate strongly prefers intervention.
```

## Phase 2: PPO Integration And Logging

Goal: make the gated action distribution fully compatible with rollout sampling,
PPO log-prob recomputation, entropy regularization, checkpointing, and
deterministic evaluation.

Training path:

```text
actor.get_action(obs_i)
  -> action_i, logprob_i, entropy_i
```

PPO update path:

```text
actor.get_action(obs_i, stored_action_i)
  -> recomputed_logprob_i, entropy_i
```

Deterministic evaluation:

```text
if gate_argmax == do_nothing:
    action_i = 0
else:
    action_i = 1 + argmax(nonidle_action_logits)
```

New logs:

```text
train/intervention_gate_do_nothing_frac_agent_*
train/intervention_gate_intervene_frac_agent_*
train/intervention_gate_prob_intervene_agent_*
train/intervention_gate_entropy_agent_*
train/nonidle_action_entropy_agent_*
```

Success criterion:

```text
We can distinguish "agent did nothing because the gate chose no-op" from
"agent happened to assign high probability to action 0 inside a flat action
space."
```

## Phase 3: Sparse-Intervention Objective

Goal: make the learned behavior more operator-like without forcing a hard rule.

Start with logging only. Then test one optional reward regularizer:

```text
r_t' = r_t - lambda_intervention * I[action_i != 0]
```

A more targeted version penalizes unnecessary action in safe states:

```text
r_t' = r_t
     - lambda_safe_intervention
       * I[max_rho_t < rho_safe_threshold]
       * I[action_i != 0]
```

Recommended flags:

```text
--intervention-penalty 0.0
--safe-intervention-penalty 0.0
--safe-intervention-rho-threshold 0.90
```

Ablation plan:

```text
baseline MAPPO
gated MAPPO, no intervention penalty
gated MAPPO, small safe-intervention penalty
gated MAPPO, larger safe-intervention penalty
```

Success criterion:

```text
The policy keeps or improves survival while reducing unnecessary non-idle
actions, especially in safe states.
```

## Phase 4: Explainability Logs Without Extra Simulation

Goal: explain the behavior using quantities already available from real
transitions.

Per rollout or evaluation episode, log:

```text
explain/pre_max_rho
explain/post_max_rho
explain/delta_max_rho
explain/worst_line_before
explain/worst_line_after
explain/topology_distance_before
explain/topology_distance_after
explain/topology_distance_delta
explain/action_nonidle_agent_*
explain/gate_intervened_agent_*
```

Definitions:

```text
pre_max_rho  = max rho before executing the joint action
post_max_rho = max rho after executing the joint action
delta_max_rho = post_max_rho - pre_max_rho

topology_distance = distance from reference topology
topology_distance_delta = distance_after - distance_before
```

Interpretation:

- Negative `delta_max_rho`: the step reduced the worst overload.
- Positive `topology_distance_delta`: the step moved away from reference
  topology.
- Negative `topology_distance_delta`: the step restored the grid toward the
  reference topology.

Success criterion:

```text
We can describe each agent's learned strategy in physical terms, not only by
reward.
```

## Phase 5: Counterfactual No-Op Explainability

Goal: measure whether an intervention helped compared with doing nothing.

For selected evaluation steps, simulate the no-op joint action from the same
pre-action observation:

```text
actual_post_risk = max rho after executed action
noop_post_risk   = max rho after no-op

improvement_vs_noop = noop_post_risk - actual_post_risk
```

Interpretation:

```text
improvement_vs_noop > 0:
    the intervention reduced immediate risk compared with no-op

improvement_vs_noop < 0:
    the intervention made immediate risk worse than no-op
```

This should be evaluation-only or subsampled during training to avoid slowing
down PPO.

Recommended flags:

```text
--explain-counterfactual-noop true
--explain-counterfactual-every 20
```

Logs:

```text
explain/noop_post_max_rho
explain/improvement_vs_noop
explain/helpful_intervention_frac
explain/harmful_intervention_frac
explain/noop_equivalent_intervention_frac
```

Success criterion:

```text
We can say whether an action was operationally useful relative to doing nothing.
```

## Phase 6: Strategy Comparison Across Agents

Goal: explain why different agents learn different strategies.

Aggregate per-agent metrics:

```text
agent_i/intervention_rate
agent_i/intervention_rate_when_safe
agent_i/intervention_rate_when_hazard
agent_i/mean_improvement_vs_noop
agent_i/helpful_intervention_frac
agent_i/topology_restoration_frac
agent_i/most_common_actions
agent_i/most_common_changed_substations
```

Useful plots:

- Intervention rate by agent.
- Gate intervention probability vs `max_rho`.
- Improvement vs no-op by agent.
- Topology distance before/after interventions.
- Most frequently selected local actions.

Success criterion:

```text
We can explain not only that agents differ, but how they differ physically:
one restores topology, one handles overloads, one rarely acts, etc.
```

## Phase 7: Optional Local Action Factorization

Goal: improve scalability further if the local action spaces grow.

Instead of:

```text
gate_i -> local action id
```

factor into:

```text
gate_i -> local substation -> local topology configuration
```

This is closer to the full hierarchy in the HRL paper, but still decentralized.

Do not implement this until the simpler gate is evaluated.

Success criterion:

```text
The local action space scales with substations/configurations instead of one
flat categorical over all local topology actions.
```

## Recommended First Implementation

Implement only:

```text
Phase 1: intervention-gated actor
Phase 2: PPO compatibility and gate logs
Phase 4: lightweight explainability logs
```

Then run:

```text
best_00 baseline, seeds 0/1/2
best_00 gated actor, no penalty, seeds 0/1/2
best_00 gated actor, small safe-intervention penalty, seeds 0/1/2
```

Primary evaluation questions:

```text
1. Does the gate preserve survival/reward?
2. Does it reduce unnecessary non-idle actions?
3. Does it reduce simultaneous multi-agent interventions?
4. Are actions more explainable in terms of max rho and topology distance?
```

## Non-Goals For The First Version

- No central coordinator.
- No communication between agents.
- No risk-prior surrogate.
- No hard rule that forbids action in safe states.
- No TopK action restriction.
- No exhaustive counterfactual simulation during training.
