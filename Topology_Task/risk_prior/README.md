# Learned Gibbs Risk Prior

This package adds a learned, physics-informed risk prior for topology-control
MAPPO. The goal is to keep the policy learned end-to-end while giving it a
second learned signal: an estimate of how risky each local topology action is
from the current grid state.

The intended final decision rule is:

```text
final_logits(agent, action)
  = actor_logits(agent, action)
  + beta * learned_physics_score(state_graph, agent, action)

learned_physics_score = - predicted_risk / tau
```

Lower predicted post-action risk gives a higher prior score. No handcrafted
action mask is required for the core method: illegal, ambiguous, failing, and
terminal actions are treated as high-risk training labels for the surrogate.

## Roadmap

### Phase 1: Offline Risk Dataset

Collect supervised one-step labels:

```text
state_graph, agent_id, local_action_id -> target_risk
```

The target is usually:

```text
target_risk = max(next_obs.rho)
```

If the simulated action is illegal, ambiguous, fails, or terminates the episode,
the example is kept and assigned `--risk-penalty`. This makes the later model
learn that these actions are dangerous instead of manually masking them.

Implemented in:

```text
risk_prior/collect_dataset.py
```

### Phase 2: Risk Surrogate

Train a frozen action-conditional GNN:

```text
risk_phi(state_graph, agent_id, local_action_id) -> predicted_risk
```

The planned architecture is:

```text
state_graph -> existing GraphEncoder -> graph embedding
agent_id -> learned embedding
local_action_id -> learned embedding
concat -> MLP -> scalar risk
```

The most important validation metric is action ranking quality, not only MSE.
For the Gibbs prior, it matters that low-risk actions receive lower predicted
risk than dangerous actions.

### Phase 3: Soft Gibbs Logit Prior

Load the frozen risk surrogate during MAPPO and adjust actor logits:

```text
prior_score = -risk_phi(state_graph, agent_id, action_id) / tau
final_logits = actor_logits + beta * prior_score
```

This is soft: all actions remain available. The policy can still explore, but
probability mass is shifted away from actions the learned surrogate predicts to
be risky.

### Phase 4: Learned TopK Candidate Restriction

After the soft prior is useful, optionally keep only the K lowest predicted-risk
actions for each agent:

```text
candidate_set = TopK actions with lowest predicted_risk
```

This is stricter than Phase 3. It should only be added after the surrogate is
good enough, because a bad surrogate can hide useful actions from PPO.

### Phase 5: Coordination And Semi-MDP Refinements

Optional follow-ups:

- Add a coordination rule so not every agent applies a non-idle action at the
  same hazard step.
- Extend the existing idle/hazard wrapper into a more faithful semi-MDP update
  with `gamma ** skipped_steps`.
- Train the surrogate on joint actions if local one-agent-at-a-time risk labels
  are not enough.

## Phase 1 In Detail

Phase 1 builds the data needed by Phase 2. It does not modify MAPPO training.

At each environment state, the collector checks:

```text
pre_risk = max(current_obs.rho)
```

If `pre_risk >= --rho-threshold`, the state is considered hazardous and is
labelled. The default threshold is `0.90`.

For every hazardous state, the collector saves the centralized `state_graph`.
Then, for each agent, it samples candidate local topology actions. For each
candidate action, it constructs a joint action where:

```text
controlled agent -> candidate local action
all other agents -> action 0
```

This isolates the one-step risk contribution of one local action. It is the
cleanest first approximation for a multi-agent setup because it avoids the full
combinatorial joint-action space.

The collector simulates the central action for one step from the current
observation and stores:

```text
target_risk = max(simulated_next_obs.rho)
```

If the simulation is illegal, ambiguous, fails, or terminates, it stores:

```text
target_risk = --risk-penalty
```

The default penalty is `2.0`.

## Running Phase 1

Run from `Topology_Task` in the project conda environment:

```bash
python -m risk_prior.collect_dataset \
  --env-id bus14 \
  --seed 0 \
  --max-examples 20000 \
  --actions-per-agent 32
```

Default outputs:

```text
../outputs/risk_prior/<env_id>_seed<seed>_risk_dataset.npz
../outputs/risk_prior/<env_id>_seed<seed>_risk_dataset.meta.json
```

For exhaustive labels on small local action spaces:

```bash
python -m risk_prior.collect_dataset \
  --env-id bus14 \
  --all-actions true
```

For more diverse rollout states, allow occasional random non-idle actions while
walking the environment:

```bash
python -m risk_prior.collect_dataset \
  --env-id bus14 \
  --rollout-nonidle-prob 0.05
```

On JED with Slurm, submit from the repository root:

```bash
sbatch job_risk_prior_jed.sh \
  --env-id bus14 \
  --max-examples 20000 \
  --actions-per-agent 32
```

For a quick smoke test:

```bash
sbatch job_risk_prior_jed.sh \
  --max-examples 200 \
  --max-hazard-states 5 \
  --actions-per-agent 8
```

For a small seed sweep:

```bash
sbatch --array=0-2 job_risk_prior_jed.sh \
  --max-examples 20000 \
  --actions-per-agent 32
```

The Slurm script uses `SEED=$SLURM_ARRAY_TASK_ID` by default for array jobs.

## Phase 1 Outputs

The `.npz` file contains:

```text
node_features       # state graph node tensor
edge_features       # state graph edge tensor
node_mask
edge_mask
agent_index         # integer agent id used by the future model
action_id           # local discrete action id
target_risk         # supervised label
pre_risk            # max rho before the candidate action
reward              # one-step simulated reward
done
is_illegal
is_ambiguous
has_error
simulation_error
episode
env_step
hazard_index
```

The `.meta.json` file contains:

```text
agent_ids
agent_to_index
n_actions_by_agent
action_domains
observation_domains
state_graph_spec
collector settings
sampled exception text
```

## Important Assumptions

- Phase 1 labels one local action at a time while other agents do no-op.
- This is intentionally not a hand-coded action mask.
- The risk label is one-step overload risk, not long-horizon value.
- The first surrogate should be evaluated by ranking quality: if it can put
  risky actions below safer actions, it is useful for a Gibbs prior.
- If local labels are insufficient, a later dataset can include sampled joint
  actions from the current MAPPO policy.

## MAPPO Coordination Diagnostic

The MAPPO training loop logs how many agents choose non-idle actions on the
same environment step. These W&B metrics help decide whether Phase 1's
one-agent-at-a-time labels are close enough to actual execution:

```text
train/non_idle_agents_distribution
train/non_idle_agents_count_0_frac
train/non_idle_agents_count_1_frac
train/non_idle_agents_count_2_frac
...
train/frac_any_non_idle
train/frac_multi_agent_non_idle
train/non_idle_agents_mean
```

If most mass is on `count_0` and `count_1`, unilateral labels are a reasonable
first approximation. If `frac_multi_agent_non_idle` is high, the next dataset
should include sampled joint-action labels from the current MAPPO policy.

To get the same diagnostic from an already saved checkpoint without retraining:

```bash
python -m risk_prior.analyze_checkpoint_actions \
  checkpoint/<checkpoint_name>.tar \
  --eval-episodes 5 \
  --deterministic true \
  --chronic-split test
```

The script prints `count_1_frac` directly, along with the full distribution over
the number of non-idle agents per decision step.
