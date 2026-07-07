# Long-Horizon Action-Space Reduction Plan

This note is the procedure I would follow for the WCCI `bus36_wcci` action-space
problem.

The current symptom is important:

- pure do-nothing survives about `51%` on average;
- simulator-greedy over the current kept actions survives about `50%`;
- MAPPO with the current `256` actions can still learn, but the action set itself
  does not look better than doing nothing under a one-step greedy controller.

My interpretation is that the current reduction is too myopic. It keeps actions
that look good after one simulator step, but a good topology action is often a
prevention or positioning action. It may not win at `t + 1`; it should win over
roughly `t + 3` to `t + 10`.

## Baselines

Use these as the baselines to beat. Do not replace them until a candidate wins on
the same split and with the same evaluation procedure.

| Name | What it is | Current value | Purpose |
| --- | --- | ---: | --- |
| `B0_do_nothing` | No topology action ever | about `51%` average survival | Sanity floor. Any greedy action set below this is suspicious. |
| `B1_h1_mk256_greedy` | Greedy controller over current `wcci_full2048a_90_v3_mk256` | about `50%` average survival | Current one-step action-space baseline. |
| `B2_h1_mk256_MAPPO` | Your current MAPPO config with current `mk256` JSON | compare at `20M` and full-test checkpoint | RL baseline. This is the real baseline to beat. |

The current reduced action space is:

```text
outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk256.json
```

## Candidate Ranking

This is the order I recommend.

| Rank | Candidate | Code needed | Why |
| ---: | --- | --- | --- |
| 1 | `C1_h5_delta_vs_do_nothing` | No | Fastest test of the long-horizon hypothesis. Uses the existing `--outcome-time-step 5`. |
| 2 | `C2_h10_delta_vs_do_nothing` | No | Same idea, longer lookahead. More expensive and slightly riskier because it only sees the final simulated state. |
| 3 | `C3_h5_survival_cost` | Small collector + reducer change | Better than raw rho because it penalizes simulated failure before horizon. This is my best practical next implementation. |
| 4 | `C4_multi_horizon_survival_cost` | Larger collector + reducer change | Best design: protect immediate, medium, and longer horizon at the same time with horizons `[1, 3, 5, 10]`. |
| 5 | `C5_auxiliary_risk_model` | Model change | Learn a risk/action-value signal from the long-horizon dataset. Useful later, not first. |

My recommendation:

1. First run `C1_h5_delta_vs_do_nothing`.
2. If it beats `B1_h1_mk256_greedy` and does not fall below do-nothing, train MAPPO with it.
3. In parallel or next, implement `C3_h5_survival_cost`.
4. Treat `C4_multi_horizon_survival_cost` as the best final version if compute cost is acceptable.

## Metrics To Track

For every candidate, record one row in a local table.

```text
candidate_name
dataset_root
reduced_action_space_json
horizon_or_horizons
selection_metric
selection_method
top_k
min_count
greedy_mean_survival_test
greedy_do_nothing_mean_survival_test
greedy_mean_survival_delta_test
greedy_full_survival_rate_test
mappo_eval_survival_at_20M
mappo_eval_survival_at_30M
best_test_checkpoint_full_eval_survival
action0_fraction
non_idle_fraction
notes
```

Selection rule:

1. A candidate is not good just because greedy improves one or two chronics.
2. A candidate must beat the current action-space greedy baseline on the same
   split.
3. The real winner is selected by MAPPO full-test evaluation from the
   `best_test_...tar` checkpoint, not by the average of the last few training
   points.

## Step 0: Recompute The Current Baselines

Pure do-nothing:

```bash
cd /home/plumet/RL_Marl2grid

ENV_ID=bus36_wcci \
SPLIT=test \
OUTPUT_DIR=outputs/do_nothing_eval/wcci_test \
sbatch Topology_Task/teacher_student/job_do_nothing_eval_jed.sh
```

Greedy evaluation of the current `h=1` reduced action space:

```bash
cd /home/plumet/RL_Marl2grid

REDUCED_ACTION_SPACE=outputs/teacher_student_datasets/wcci_full2048a_90_v3/metadata/reduced_action_space_wcci_full2048a_90_v3_mk256.json \
OUTPUT_DIR=outputs/teacher_student_greedy_eval/wcci_h1_mk256_test \
ENV_ID=bus36_wcci \
SPLIT=test \
DECISION_RHO_THRESHOLD=0.90 \
REQUIRE_IMPROVEMENT=true \
IMPROVEMENT_TOLERANCE=1e-3 \
TIME_STEP=1 \
SIM_WORKERS=71 \
sbatch --time=06:00:00 Topology_Task/teacher_student/job_greedy_reduced_eval_jed.sh
```

The summary file to read is:

```text
outputs/teacher_student_greedy_eval/wcci_h1_mk256_test/greedy_vs_do_nothing_summary.json
```

## Step 1: No-Code Horizon-5 Dataset

This tests the main hypothesis with the existing collector.

The only conceptual change is:

```text
outcome-time-step = 5
```

In the current code this is already supported by:

```text
Topology_Task/teacher_student/collect_bruteforce_action_outcomes.py --outcome-time-step
```

Smoke collection:

```bash
cd /home/plumet/RL_Marl2grid

ENV_ID=bus36_wcci \
OUTPUT_DIR=outputs/teacher_student_datasets/smoke_wcci_full2048a_90_h5_v1 \
COLLECTION_RHO_THRESHOLD=0.90 \
MAX_EPISODES=2 \
OUTCOME_ACTION_SAMPLE_SIZE=64 \
OUTCOME_ACTION_SAMPLE_SIZES=agent_0=all,agent_1=64,agent_2=all,agent_3=all \
OUTCOME_RESAMPLE_ACTIONS_PER_STATE=true \
OUTCOME_ROLLOUT_POLICY=best_simulated \
OUTCOME_SIM_WORKERS=71 \
CHRONIC_SHARD_COUNT=1 \
REDUCE_AFTER=false \
OVERWRITE=false \
sbatch --time=00:45:00 Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh --outcome-time-step 5
```

Full collection:

```bash
cd /home/plumet/RL_Marl2grid

ENV_ID=bus36_wcci \
OUTPUT_DIR=outputs/teacher_student_datasets/wcci_full2048a_90_h5_v1 \
COLLECTION_RHO_THRESHOLD=0.90 \
OUTCOME_ACTION_SAMPLE_SIZE=64 \
OUTCOME_ACTION_SAMPLE_SIZES=agent_0=all,agent_1=2048,agent_2=all,agent_3=all \
OUTCOME_RESAMPLE_ACTIONS_PER_STATE=true \
OUTCOME_ROLLOUT_POLICY=best_simulated \
OUTCOME_SIM_WORKERS=71 \
CHRONIC_SHARD_COUNT=16 \
REDUCE_AFTER=false \
OVERWRITE=false \
sbatch --array=0-15 Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh --outcome-time-step 5
```

Reduce it to `256` actions per agent:

```bash
cd /home/plumet/RL_Marl2grid

DATASET_ROOT=outputs/teacher_student_datasets/wcci_full2048a_90_h5_v1 \
ACTION_REDUCTION_TOP_K=256 \
ACTION_REDUCTION_MIN_COUNT=20 \
ACTION_REDUCTION_METRIC=delta_vs_do_nothing \
ACTION_REDUCTION_METHOD=improvement_rate \
ACTION_REDUCTION_NAME=wcci_h5_improvement_rate_delta_do_nothing_k256_min20 \
ACTION_REDUCTION_REQUIRE_IMPROVEMENT=true \
ACTION_REDUCTION_IMPROVEMENT_TOLERANCE=1e-3 \
sbatch Topology_Task/teacher_student/job_reduce_action_space_jed.sh
```

Expected output:

```text
outputs/teacher_student_datasets/wcci_full2048a_90_h5_v1/metadata/reduced_action_space_wcci_h5_improvement_rate_delta_do_nothing_k256_min20.json
```

Greedy test of this horizon-5 action space:

```bash
cd /home/plumet/RL_Marl2grid

REDUCED_ACTION_SPACE=outputs/teacher_student_datasets/wcci_full2048a_90_h5_v1/metadata/reduced_action_space_wcci_h5_improvement_rate_delta_do_nothing_k256_min20.json \
OUTPUT_DIR=outputs/teacher_student_greedy_eval/wcci_h5_delta_do_nothing_k256_test \
ENV_ID=bus36_wcci \
SPLIT=test \
DECISION_RHO_THRESHOLD=0.90 \
REQUIRE_IMPROVEMENT=true \
IMPROVEMENT_TOLERANCE=1e-3 \
TIME_STEP=5 \
SIM_WORKERS=71 \
sbatch --time=06:00:00 Topology_Task/teacher_student/job_greedy_reduced_eval_jed.sh
```

Interpretation:

- if `greedy_mean_survival_delta_test > 0`, the action set is better than
  do-nothing under the same horizon-5 greedy decision rule;
- if it is close to zero, the action set may still help MAPPO, but it is not a
  strong signal;
- if it is negative, do not spend a 60M MAPPO run on it before inspecting the
  selected actions.

## Step 2: Train MAPPO With Horizon-5 Actions

Duplicate the existing config:

```bash
cd /home/plumet/RL_Marl2grid

cp Topology_Task/configs/wcci_aib/wcci_aib_01_flat_local_t010_72x576_s0.toml \
   Topology_Task/configs/wcci_aib/wcci_aib_01_flat_local_t010_72x576_h5_s0.toml
```

Edit the new TOML:

```toml
name = "wcci_aib_01_flat_local_t010_72x576_h5_s0"
exp_tag = "wcci_aib_01_flat_local_t010_72x576_h5_s0"
reduced_action_space = "outputs/teacher_student_datasets/wcci_full2048a_90_h5_v1/metadata/reduced_action_space_wcci_h5_improvement_rate_delta_do_nothing_k256_min20.json"
total_timesteps = 60000000
```

Launch:

```bash
cd /home/plumet/RL_Marl2grid/Topology_Task

sbatch job_jed.sh configs/wcci_aib/wcci_aib_01_flat_local_t010_72x576_h5_s0.toml
```

Compare against `B2_h1_mk256_MAPPO` at the same training step:

- `20M` because this is where your first run stopped;
- `30M` because the bad 60M run degraded by then;
- final best-test checkpoint using `Topology_Task/full_test_eval`.

## Step 3: No-Code Horizon-10 Variant

Only do this if horizon-5 is promising or ambiguous.

Use the same commands as Step 1, but replace:

```text
wcci_full2048a_90_h5_v1 -> wcci_full2048a_90_h10_v1
--outcome-time-step 5 -> --outcome-time-step 10
TIME_STEP=5 -> TIME_STEP=10
wcci_h5_... -> wcci_h10_...
```

Reason to try it:

- some WCCI failures are delayed, so `t + 10` may catch actions that look fine at
  `t + 5` but create a future overload.

Reason not to trust it alone:

- if you only store final `rho_after_action` at `t + 10`, an action can cause a
  bad transient before `t + 10` and still look acceptable at the final simulated
  state. That is why the survival-aware version is better.

## Step 4: Implement Survival-Aware Horizon Cost

This is the best practical code change after the no-code horizon-5 test.

Problem with the current metric:

```text
delta_vs_do_nothing = rho_after_action - rho_after_do_nothing
```

This ignores whether the simulated trajectory died before the horizon. A low
final rho is meaningless if the simulation already failed.

Add a new metric:

```text
failure_penalty = 10.0

cost_h(action) =
    failure_penalty * int(sim_done_action) + rho_after_action

cost_h(do_nothing) =
    failure_penalty * int(sim_done_do_nothing) + rho_after_do_nothing

delta_long_cost_vs_do_nothing =
    cost_h(action) - cost_h(do_nothing)
```

Lower is better. Negative means the action is better than do-nothing.

Code changes:

1. In `Topology_Task/teacher_student/collect_bruteforce_action_outcomes.py`,
   after the do-nothing simulation, store:

```python
sim_done_do_nothing = bool(do_nothing["sim_done"])
sim_reward_do_nothing = float(do_nothing["sim_reward"])
```

2. In the per-action values, add:

```python
failure_penalty = 10.0
action_long_cost = failure_penalty * float(outcome["sim_done"]) + rho_after_action
noop_long_cost = failure_penalty * float(sim_done_do_nothing) + rho_after_do_nothing
delta_long_cost_vs_do_nothing = action_long_cost - noop_long_cost

values.update(
    {
        "sim_done_do_nothing": sim_done_do_nothing,
        "sim_reward_do_nothing": sim_reward_do_nothing,
        "delta_long_cost_vs_do_nothing": delta_long_cost_vs_do_nothing,
    }
)
```

3. In `Topology_Task/teacher_student/reduce_action_space_from_outcomes.py`, add
   this metric to the parser choices:

```python
choices=[
    "delta_vs_do_nothing",
    "delta_vs_now",
    "rho_after_action",
    "delta_long_cost_vs_do_nothing",
]
```

Then collect horizon-5 again under a new dataset name:

```bash
cd /home/plumet/RL_Marl2grid

ENV_ID=bus36_wcci \
OUTPUT_DIR=outputs/teacher_student_datasets/wcci_full2048a_90_h5_longcost_v1 \
COLLECTION_RHO_THRESHOLD=0.90 \
OUTCOME_ACTION_SAMPLE_SIZE=64 \
OUTCOME_ACTION_SAMPLE_SIZES=agent_0=all,agent_1=2048,agent_2=all,agent_3=all \
OUTCOME_RESAMPLE_ACTIONS_PER_STATE=true \
OUTCOME_ROLLOUT_POLICY=best_simulated \
OUTCOME_SIM_WORKERS=71 \
CHRONIC_SHARD_COUNT=16 \
REDUCE_AFTER=false \
OVERWRITE=false \
sbatch --array=0-15 Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh --outcome-time-step 5
```

Reduce with the new metric:

```bash
cd /home/plumet/RL_Marl2grid

DATASET_ROOT=outputs/teacher_student_datasets/wcci_full2048a_90_h5_longcost_v1 \
ACTION_REDUCTION_TOP_K=256 \
ACTION_REDUCTION_MIN_COUNT=20 \
ACTION_REDUCTION_METRIC=delta_long_cost_vs_do_nothing \
ACTION_REDUCTION_METHOD=improvement_rate \
ACTION_REDUCTION_NAME=wcci_h5_improvement_rate_longcost_k256_min20 \
ACTION_REDUCTION_REQUIRE_IMPROVEMENT=true \
ACTION_REDUCTION_IMPROVEMENT_TOLERANCE=1e-3 \
sbatch Topology_Task/teacher_student/job_reduce_action_space_jed.sh
```

This is my recommended "best practical" action set if it beats the no-code h5
variant.

## Step 5: Implement Multi-Horizon Survival Cost

This is the best design, but it costs more simulations.

Goal:

```text
horizons = [1, 3, 5, 10]
```

For each candidate action and the do-nothing reference, simulate each horizon and
store the worst risk across horizons:

```text
action_any_done = any(sim_done_action_h for h in horizons)
noop_any_done = any(sim_done_do_nothing_h for h in horizons)

action_max_rho = max(rho_after_action_h for h in horizons)
noop_max_rho = max(rho_after_do_nothing_h for h in horizons)

cost_multi(action) =
    10.0 * int(action_any_done) + action_max_rho

cost_multi(do_nothing) =
    10.0 * int(noop_any_done) + noop_max_rho

delta_multi_h_cost_vs_do_nothing =
    cost_multi(action) - cost_multi(do_nothing)
```

Why this is better:

- horizon `1` protects immediate emergencies;
- horizons `3` and `5` capture short-term delayed overloads;
- horizon `10` catches actions that create a later problem;
- `max(rho)` across horizons avoids selecting actions that look good at the final
  horizon only.

Implementation options:

| Option | What to do | Pros | Cons |
| --- | --- | --- | --- |
| Separate datasets | Run h1, h3, h5, h10 separately, then combine offline | No collector redesign | More post-processing and duplicated state matching |
| Single collector extension | Add `--outcome-time-steps 1,3,5,10` and simulate all horizons inside one state | Cleanest data | More code now |

Recommended implementation:

1. Add a new CLI argument to `collect_bruteforce_action_outcomes.py`:

```python
parser.add_argument("--outcome-time-steps", type=str, default=None)
```

2. If it is provided, parse it into a list of ints. Otherwise keep the existing
   `--outcome-time-step` behavior.
3. For each high-rho state, simulate do-nothing and each candidate at every
   horizon.
4. Store per-horizon fields such as:

```text
rho_after_action_h1
rho_after_action_h3
rho_after_action_h5
rho_after_action_h10
rho_after_do_nothing_h1
rho_after_do_nothing_h3
rho_after_do_nothing_h5
rho_after_do_nothing_h10
sim_done_action_h1
sim_done_action_h3
sim_done_action_h5
sim_done_action_h10
sim_done_do_nothing_h1
sim_done_do_nothing_h3
sim_done_do_nothing_h5
sim_done_do_nothing_h10
delta_multi_h_cost_vs_do_nothing
```

5. Add `delta_multi_h_cost_vs_do_nothing` to the reducer metric choices.
6. Reduce with:

```bash
cd /home/plumet/RL_Marl2grid

DATASET_ROOT=outputs/teacher_student_datasets/wcci_full2048a_90_h13510_longcost_v1 \
ACTION_REDUCTION_TOP_K=256 \
ACTION_REDUCTION_MIN_COUNT=20 \
ACTION_REDUCTION_METRIC=delta_multi_h_cost_vs_do_nothing \
ACTION_REDUCTION_METHOD=improvement_rate \
ACTION_REDUCTION_NAME=wcci_h13510_improvement_rate_longcost_k256_min20 \
ACTION_REDUCTION_REQUIRE_IMPROVEMENT=true \
ACTION_REDUCTION_IMPROVEMENT_TOLERANCE=1e-3 \
sbatch Topology_Task/teacher_student/job_reduce_action_space_jed.sh
```

This is the candidate I would call the "best one" architecturally. It is not
automatically the experimental winner; it still has to beat the baselines.

## Step 6: Optional Learned Risk Head

Only do this after you have a better long-horizon action dataset.

Idea:

- train a small auxiliary head to predict whether a candidate action improves
  `delta_long_cost_vs_do_nothing` or `delta_multi_h_cost_vs_do_nothing`;
- use that signal as an auxiliary loss or as a pretraining task for the actor;
- keep evaluation heuristic-free: the policy still chooses actions itself.

This is a model-shape change, not a heuristic threshold. It learns the kind of
thing the heuristic was manually checking.

I would not start here because a bad action set will still limit the policy.
First fix the reduced action space, then teach the model to use it.

## Final Decision Rule

Call a candidate successful only if all of this is true:

1. Its greedy reduced-action evaluation is at least not worse than do-nothing on
   the test split.
2. Its MAPPO curve at `20M` is better than the current `h=1 mk256` MAPPO curve at
   `20M`.
3. Its best-test checkpoint full evaluation is better than the baseline
   checkpoint full evaluation.
4. It does not simply survive by almost never acting, unless that is also paired
   with better survival.

Expected likely winner:

```text
Best quick test:
  C1_h5_delta_vs_do_nothing

Best practical implementation:
  C3_h5_survival_cost

Best final design:
  C4_multi_horizon_survival_cost with horizons [1, 3, 5, 10]

Baseline to beat:
  B2_h1_mk256_MAPPO, using the current wcci_full2048a_90_v3_mk256 action space
```

