# Adaptive Intervention Budget

This note documents the sparse-control replacement for the first intervention
gate design.

## Motivation

The first gate implementation changed the actor architecture but not the
objective. Each agent had a learned `do nothing / intervene` gate, but PPO still
optimized only the usual task reward. If frequent topology actions helped
survival, the gate had no reason to become sparse. The result was expected:
the gate often learned to intervene and did not reliably increase action `0`.

The global/local rho evaluation heuristic showed a useful diagnostic result:
forcing action `0` in safe states can preserve performance better than the
learned gate. But the heuristic is evaluation-only and hand-coded. The new
implementation turns the idea into a training objective:

> preserve survival while respecting an intervention budget.

This is closer to a constrained MDP formulation than to a fixed reward hack.

## Implemented Components

### 1. Corrected Gated Entropy

The old gated actor entropy was:

```text
H(gate) + P(intervene) H(nonidle action)
```

This couples exploration with intervention. Increasing `P(intervene)` unlocks
more non-idle action entropy, so entropy regularization can accidentally reward
intervening.

The new optional mode is:

```text
gate_mult H(gate) + nonidle_mult H(nonidle action)
```

It is enabled with:

```bash
--intervention-gate-entropy-mode separate
--intervention-gate-entropy-mult 0.2
--intervention-nonidle-entropy-mult 1.0
```

The old behavior remains the default:

```bash
--intervention-gate-entropy-mode coupled
```

### 2. Adaptive Lagrangian Intervention Budget

For each agent `i`, define a cost:

```text
c_i(s_t, a_t) = 1[a_i,t != 0] w_i(s_t)
```

where `w_i(s_t)` is a safety weight. The reward used for PPO becomes:

```text
r'_i,t = r_i,t - lambda_i c_i(s_t, a_t)
```

After each rollout, the multiplier is updated by:

```text
lambda_i <- clip(lambda_i + lr (mean(c_i) - target), 0, lambda_max)
```

If the agent intervenes too much in costly states, `lambda_i` increases. If the
agent is below budget, `lambda_i` decreases. This avoids choosing a fixed
penalty value by hand.

Enable it with:

```bash
--adaptive-intervention-budget true
```

### 3. State-Dependent Safety Cost

The recommended cost mode is local and smooth:

```text
w_i(s_t) = sigmoid(k (rho_safe - local_max_rho_i(s_t)))
```

This makes non-idle actions expensive when the local region is safe and cheap
when the local region is stressed.

The default adaptive cost mode is:

```bash
--intervention-budget-cost-mode local_safe
--intervention-budget-rho-threshold 0.90
--intervention-budget-rho-sharpness 25.0
```

Other modes:

- `nonidle`: every non-idle action has cost `1`.
- `global_safe`: uses global max rho instead of local max rho.
- `local_safe`: uses each agent's local max rho; this is the decentralized
  option.

`rho_threshold` is not a hard override here. It is the center of a sigmoid.

## Key Arguments

```bash
--adaptive-intervention-budget true
--intervention-budget-target 0.20
--intervention-budget-lr 0.02
--intervention-budget-init-lambda 0.0
--intervention-budget-max-lambda 10.0
--intervention-budget-warmup-steps 0
--intervention-budget-cost-mode local_safe
--intervention-budget-rho-threshold 0.90
--intervention-budget-rho-sharpness 25.0
```

Interpretation:

- `target`: desired mean cost per agent per environment step.
- `lr`: speed of lambda adaptation.
- `lambda`: learned penalty coefficient, one per agent.
- `cost_mode`: what type of intervention is expensive.
- `rho_sharpness`: how close the smooth cost is to a hard safety threshold.

## W&B Logs

Training budget logs:

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
```

Use these to check:

- If `cost > target`, lambda should increase.
- If `cost < target`, lambda should decrease or stay near zero.
- `intervention_rate_when_budget_costly` should fall if the method works.

Action distribution logs:

```text
train/frac_action_0_agent_i
train/explain/action_nonidle_agent_i
test/explain/frac_action_0_agent_i
test/explain/action_nonidle_agent_i
train_eval/explain/frac_action_0_agent_i
train_eval/explain/action_nonidle_agent_i
```

The `test/...` and `train_eval/...` metrics are the final executed evaluation
actions. They are the correct metrics for comparing against evaluation-only
heuristics.

Gate entropy logs:

```text
train/intervention_gate_entropy_mode_separate
train/intervention_gate_entropy_mult
train/intervention_nonidle_entropy_mult
train/intervention_gate_entropy_agent_i
train/nonidle_action_entropy_agent_i
```

## Recommended First Runs

Flat actor with adaptive local-safe budget:

```bash
sbatch job_jed.sh \
  configs/gine_s0_s1_s2/best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update_s0.toml \
  --adaptive-intervention-budget true \
  --intervention-budget-cost-mode local_safe \
  --intervention-budget-target 0.20 \
  --intervention-budget-lr 0.02 \
  --intervention-budget-rho-threshold 0.90 \
  --intervention-budget-rho-sharpness 25.0
```

Gated actor with corrected entropy and adaptive local-safe budget:

```bash
sbatch job_jed.sh \
  configs/gine_s0_s1_s2/best_00_shared_actor_gnn_gine_a4_concat_flat_critic_gnn_legacy_update_s0.toml \
  --intervention-gate true \
  --intervention-gate-eval-mode hierarchical_greedy \
  --intervention-gate-entropy-mode separate \
  --intervention-gate-entropy-mult 0.2 \
  --intervention-nonidle-entropy-mult 1.0 \
  --adaptive-intervention-budget true \
  --intervention-budget-cost-mode local_safe \
  --intervention-budget-target 0.20 \
  --intervention-budget-lr 0.02
```

## What Would Count As Success?

A successful run should show:

- survival close to the baseline,
- larger `test/explain/frac_action_0_agent_i`,
- lower `train/intervention_rate_when_budget_costly_agent_i`,
- stable lambdas instead of lambdas exploding to the maximum,
- less multi-agent simultaneous intervention when the grid is safe.

If survival collapses while lambdas rise, the budget is too strict or the safety
weight is too broad. Increase `target`, lower `lr`, lower `rho_sharpness`, or
use a warmup.

