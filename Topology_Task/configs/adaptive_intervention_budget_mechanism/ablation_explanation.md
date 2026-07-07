# Adaptive Intervention Budget Mechanism Ablation

This ablation is meant to answer one main question:

Can we make the agents act less often without losing episodic survival, and which part of the intervention-budget mechanism is actually responsible for the improvement?

Here, action `0` means do nothing. A good sparse-control mechanism should keep survival high while increasing the fraction of action `0`, reducing non-idle actions, and keeping the budget cost close to the chosen target.

## Mechanism Being Tested

The adaptive intervention budget adds a sparse-control pressure during training. Instead of only learning to survive, the policy is also pushed to avoid unnecessary topology actions.

The full mechanism has several ingredients:

1. Adaptive Lagrangian budget

   A learned multiplier, lambda, increases when the policy acts too much and decreases when the policy is below the target. This is different from a fixed penalty because the penalty strength adapts during training.

2. State-dependent safety weighting

   The action penalty can be charged only in states that look safe. The idea is to avoid punishing interventions when the grid is close to danger and action may be needed.

3. Local vs global rho weighting

   A global-safe cost uses the maximum rho over the whole grid. A local-safe cost uses the local rho information around each agent. This tests whether a more agent-specific safety signal gives a better sparsity-performance tradeoff.

4. Rho threshold

   The local-safe mechanism needs a threshold, such as rho `0.90`, to decide when intervention should be considered costly. The ablation checks whether `0.90` is special or whether the result is robust to nearby thresholds.

## Existing Reference Runs

These come from `configs/adaptive_intervention_budget_7`.

| Prefix | Meaning | Why it matters |
| --- | --- | --- |
| `aib_00_flat_local_t020_s*` | Adaptive local-safe budget, target `0.20`, rho threshold `0.90` | Main reference for the full flat adaptive-budget mechanism. |
| `aib_01_flat_local_t010_s*` | Adaptive local-safe budget, target `0.10` | Tests a stricter action budget. The policy must act even less. |
| `aib_02_flat_local_t035_s*` | Adaptive local-safe budget, target `0.35` | Tests a looser action budget. The policy is allowed to act more. |
| `aib_03_gate_hgreedy_sep_local_t020_s*` | Intervention gate plus adaptive local-safe budget | Checks whether the gate still helps once sparsity is already in the objective. |
| `aib_04_flat_nonidle_t020_s*` | Adaptive non-idle budget, target `0.20`, no safety weighting | Isolates adaptivity without the state-dependent safe/unsafe weighting. |

## New Mechanism Ablation Runs

These are the 15 runs in this folder, with seeds `0`, `1`, and `2`.

| Prefix | What changes | What it tests |
| --- | --- | --- |
| `aibm_00_fixed_nonidle_p006_s*` | Fixed non-idle penalty `0.06`, no adaptive lambda, no safety weighting | Baseline for a simple fixed sparse-action penalty. This asks whether we needed the adaptive budget at all. |
| `aibm_01_fixed_global_safe_p006_s*` | Fixed penalty `0.06`, charged only when global max rho is below `0.90` | Tests state-dependent safety weighting without adaptivity. If this improves over fixed non-idle, safety weighting itself matters. |
| `aibm_02_adaptive_global_t020_s*` | Adaptive budget, global-safe weighting, target `0.20`, rho threshold `0.90` | Tests whether global safety information is enough, compared with the local-safe AIB reference. |
| `aibm_03_adaptive_local_t020_r085_s*` | Adaptive local-safe budget, target `0.20`, rho threshold `0.85` | Tests a lower safety threshold. Fewer states are considered safe/costly. |
| `aibm_04_adaptive_local_t020_r095_s*` | Adaptive local-safe budget, target `0.20`, rho threshold `0.95` | Tests a higher safety threshold. More states are considered safe/costly. |

## Main Comparisons

### 1. Does adaptivity matter?

Compare:

- `aibm_00_fixed_nonidle_p006_s*`
- `aib_04_flat_nonidle_t020_s*`

Both use non-idle action cost without local/global safety weighting. The difference is fixed penalty versus adaptive lambda.

If `aib_04` gives better survival or better action sparsity at similar survival, then the adaptive Lagrangian mechanism is useful beyond just adding a fixed penalty.

### 2. Does state-dependent safety weighting matter without adaptivity?

Compare:

- `aibm_00_fixed_nonidle_p006_s*`
- `aibm_01_fixed_global_safe_p006_s*`

Both use a fixed coefficient `0.06`. The difference is whether all non-idle actions are penalized, or only actions in globally safe states.

If global-safe fixed penalty performs better, then safety weighting helps even without adaptive lambda.

### 3. Does local rho weighting matter?

Compare:

- `aibm_02_adaptive_global_t020_s*`
- `aib_00_flat_local_t020_s*`

Both are adaptive, both target `0.20`, and both use rho threshold `0.90`. The difference is global-safe versus local-safe cost.

If local-safe performs better, it suggests the agent-specific local safety signal is important. If global-safe is similar, the simpler global signal may be enough.

### 4. Is rho threshold `0.90` special?

Compare:

- `aibm_03_adaptive_local_t020_r085_s*`
- `aib_00_flat_local_t020_s*`
- `aibm_04_adaptive_local_t020_r095_s*`

All are adaptive local-safe budgets with target `0.20`. Only the rho threshold changes.

If performance is stable across `0.85`, `0.90`, and `0.95`, the method is robust. If one threshold clearly dominates, the mechanism is sensitive and the threshold needs tuning.

### 5. How strict should the action budget be?

Compare:

- `aib_01_flat_local_t010_s*`
- `aib_00_flat_local_t020_s*`
- `aib_02_flat_local_t035_s*`

All use adaptive local-safe cost with threshold `0.90`. The target changes from strict `0.10`, to main `0.20`, to loose `0.35`.

This tells us whether survival requires a minimum amount of action, or whether the agents can survive while acting much less.

### 6. Does the intervention gate still help after adding the budget?

Compare:

- `aib_00_flat_local_t020_s*`
- `aib_03_gate_hgreedy_sep_local_t020_s*`

Both use adaptive local-safe budget with target `0.20`. The difference is flat/original actor versus intervention gate.

If the gate still improves action sparsity or survival, then the architecture adds something beyond the loss objective. If not, the budget objective may already solve most of the over-action problem.

## Metrics To Look At

| Metric | What to check |
| --- | --- |
| `test/charts/episodic_survival` | Main performance metric. The sparse mechanism is only useful if survival stays high. |
| Evaluation action-0 fraction | Whether agents actually do nothing more often at inference time. |
| Evaluation non-idle fraction | Direct measure of how often agents intervene. Lower is better if survival stays high. |
| `train/intervention_budget_lambda_mean` | Whether the adaptive multiplier stabilizes or keeps growing. |
| `train/intervention_budget_cost_mean` | Whether the observed action cost approaches the target. |
| `train/intervention_budget_cost_violation_mean` | Whether the policy is above or below the budget target. |
| `train/intervention_budget_safety_weight_mean` | How often actions are considered costly under the safety weighting. |
| `train/intervention_penalty_mean` or `train/intervention_budget_penalty_mean` | Actual reward penalty applied by the mechanism. |

## Expected Interpretation

The strongest result would be:

- survival remains close to the main baseline,
- action `0` fraction increases clearly,
- non-idle action fraction decreases clearly,
- adaptive budget cost is near the target,
- lambda stabilizes instead of exploding,
- the result holds across seeds.

If fixed penalties work as well as adaptive budgets, then the mechanism may be simpler than expected. If adaptive local-safe runs outperform fixed and global-safe variants, then the full adaptive local mechanism is justified.
