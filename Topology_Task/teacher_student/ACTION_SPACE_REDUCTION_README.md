# Bus36 Action-Space Reduction Procedure

This document explains the complete brute-force action-outcome procedure used to
reduce the multi-agent topology action space before MAPPO training.

The goal is to turn a very large discrete action space into a smaller set of
empirically useful actions, while keeping the final environment interface the
same: each agent still selects an integer action, and reduced action `0` still
means do nothing.

## Summary

The procedure has three stages:

1. Collect action-outcome data on risky grid states.
2. Reduce each agent's action space from the collected outcomes.
3. Train MAPPO with the generated reduced-action-space JSON.

For bus36, the original action sizes are approximately:

```text
agent_0: 77
agent_1: 65642
agent_2: 127
agent_3: 1119
total:   66965
```

The main difficulty is `agent_1`, whose local topology action space is much
larger than the others. The current recommended collection therefore evaluates
small action spaces exhaustively and samples the large action space more heavily:

```text
agent_0: all actions
agent_1: 2048 sampled actions per risky state
agent_2: all actions
agent_3: all actions
```

That gives:

```text
77 + 2048 + 127 + 1119 = 3371 action-outcome rows per collected risky state
```

The collector also simulates one extra do-nothing reference action per collected
state. This reference is used to compute `delta_vs_do_nothing`; it is not stored
as a separate outcome row unless action `0` is also part of the sampled action
set.

## Stage 1: Collect Action Outcomes

The collector is:

```text
Topology_Task/teacher_student/collect_bruteforce_action_outcomes.py
```

The JED launcher is:

```text
Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh
```

This stage does not use a trained checkpoint. It instantiates the Grid2Op
environment directly and acts as a brute-force simulator-based teacher.

At each environment step, the collector computes:

```text
global_max_rho = max rho over all power lines
local_max_rho_i = max rho over lines visible to agent i
```

A state is collected only if:

```text
global_max_rho >= COLLECTION_RHO_THRESHOLD
```

The default threshold is:

```text
COLLECTION_RHO_THRESHOLD=0.90
```

So the dataset is not restricted to already overloaded states. It also includes
high-risk states where at least one line is close to overload.

### Recommended Bus36 Collection Command

Run from the repository root on the cluster:

```bash
cd /home/plumet/RL_Marl2grid

OUTPUT_DIR=outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs \
CHRONIC_SHARD_COUNT=15 \
OUTCOME_ACTION_SAMPLE_SIZE=64 \
OUTCOME_ACTION_SAMPLE_SIZES=agent_0=all,agent_1=2048,agent_2=all,agent_3=all \
OUTCOME_RESAMPLE_ACTIONS_PER_STATE=true \
OUTCOME_SIM_WORKERS=71 \
OUTCOME_SIM_START_METHOD=spawn \
ACTION_REDUCTION_TOP_K=208 \
TIMING_EVERY_ENV_STEPS=500 \
OVERWRITE=true \
sbatch --array=0-14 --time=03:30:00 Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh
```

This launches 15 Slurm array jobs. Each job receives a different chronic shard:

```text
part_000
part_001
...
part_014
```

Each part writes to:

```text
outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/parts/part_XXX
```

### Important Collection Parameters

`OUTPUT_DIR`

Base folder where the dataset is written. In array mode, each job writes inside
`OUTPUT_DIR/parts/part_XXX`.

`CHRONIC_SHARD_COUNT`

Number of chronic shards. This should match the number of Slurm array tasks.
With `--array=0-14`, use `CHRONIC_SHARD_COUNT=15`.

`COLLECTION_RHO_THRESHOLD`

Risk threshold for collecting a state. `0.90` means collect states where at
least one line has `rho >= 0.90`.

`OUTCOME_ACTION_SAMPLE_SIZE`

Global fallback number of actions to sample per agent per collected state. This
is overridden by `OUTCOME_ACTION_SAMPLE_SIZES` for agents listed there.

`OUTCOME_ACTION_SAMPLE_SIZES`

Per-agent action sampling rule. Accepted formats:

```text
agent_0=all,agent_1=2048,agent_2=all,agent_3=all
all,2048,all,all
```

The named format is recommended because it is explicit.

`OUTCOME_RESAMPLE_ACTIONS_PER_STATE`

If `true`, a fresh random subset is drawn at every collected risky state. This is
important. If this is `false`, the collector reuses the same sampled action set
for the whole job.

The random stream also includes the chronic shard index, so different Slurm
array parts do not start from the same action-sampling sequence.

`OUTCOME_SIM_WORKERS`

Number of process-owned simulator workers used to evaluate candidate actions.
On JED with `--cpus-per-task=72`, use `71`, leaving one CPU for the main rollout
process.

`OUTCOME_SIM_START_METHOD`

Multiprocessing start method. Use:

```text
spawn
```

`OUTCOME_ROLLOUT_POLICY`

Controls how the environment trajectory advances after action outcomes have
been simulated.

Current useful values:

```text
best_simulated
do_nothing
```

With `best_simulated`, after simulating all candidate unilateral actions at a
risky state, the collector executes the valid action with the lowest simulated
`rho_after_action`. This creates a greedy simulator-teacher trajectory.

With `do_nothing`, the collector never applies the simulated best action during
rollout. This is useful if you want passive trajectories, but it usually gives
less operator-like data.

## What Is Computed Per Candidate Action

For every collected risky state and every sampled action, the collector simulates
a unilateral action:

```text
agent i takes action a
all other agents take action 0
```

This is not a communication mechanism. It is an offline evaluation of the local
effect of each agent's possible intervention at the same physical state.

For each simulated action, the collector stores:

```text
rho_before
rho_after_action
rho_after_do_nothing
delta_vs_now
delta_vs_do_nothing
action_is_valid
action_is_legal
action_is_ambiguous
simulation_exception
sim_done
sim_reward
worst_line_before
worst_line_after_action
worst_line_after_do_nothing
```

The main quantities are:

```text
delta_vs_now = rho_after_action - rho_before
delta_vs_do_nothing = rho_after_action - rho_after_do_nothing
```

Interpretation:

```text
delta < 0  means the action improved the metric
delta = 0  means neutral
delta > 0  means the action worsened the metric
```

The default reduction metric is:

```text
delta_vs_do_nothing
```

This asks whether the action improves the next simulated grid state compared to
doing nothing from the same state.

## What Is Saved

Each part contains:

```text
parts/part_XXX/
  shards/
    shard_00000.npz
    shard_00001.npz
    ...
  state_contexts/
    state_context_00000.npz
    state_context_00001.npz
    ...
  metadata/
    metadata.json
```

### Action-Outcome Shards

The `shards/shard_*.npz` files contain per-agent arrays. Example keys:

```text
obs_agent_0
outcome_agent_0_state_id
outcome_agent_0_episode_id
outcome_agent_0_episode_step
outcome_agent_0_dataset_step
outcome_agent_0_action_id
outcome_agent_0_global_max_rho_before
outcome_agent_0_local_max_rho_before
outcome_agent_0_rho_before
outcome_agent_0_rho_after_action
outcome_agent_0_rho_after_do_nothing
outcome_agent_0_delta_vs_now
outcome_agent_0_delta_vs_do_nothing
outcome_agent_0_label_vs_now
outcome_agent_0_label_vs_do_nothing
outcome_agent_0_action_is_valid
outcome_agent_0_action_is_legal
outcome_agent_0_action_is_ambiguous
outcome_agent_0_simulation_exception
outcome_agent_0_sim_done
outcome_agent_0_sim_reward
outcome_agent_0_worst_line_before
outcome_agent_0_worst_line_after_action
outcome_agent_0_worst_line_after_do_nothing
```

The label encoding is:

```text
-1 improved
 0 neutral
 1 worsened
 2 invalid
```

### State Context Shards

The `state_contexts/state_context_*.npz` files contain one row per collected
risky state. They are joined to action rows with `state_id`.

Example keys:

```text
state_state_id
state_episode_id
state_episode_step
state_dataset_step
state_global_max_rho_before
state_worst_line_before
state_rho_before_vector
state_high_rho_line_mask
state_overloaded_line_mask
state_n_high_rho_lines
state_n_overloaded_lines
state_agent_0_local_max_rho_before
state_agent_0_n_high_rho_lines
state_agent_0_n_overloaded_lines
state_agent_0_has_high_rho_line
state_agent_0_has_overloaded_line
state_agent_0_worst_line_visible
```

Definitions:

```text
state_high_rho_line_mask[l] = true if rho[l] >= COLLECTION_RHO_THRESHOLD
state_overloaded_line_mask[l] = true if rho[l] >= 1.0
```

To get overloaded line ids for one state:

```python
overloaded_line_ids = np.flatnonzero(state_overloaded_line_mask[state_row])
```

### Metadata

Each part metadata stores:

```text
agent_ids
action_sizes
outcome_action_sample_sizes_by_agent
n_env_steps
n_candidate_states
n_outcome_examples
n_unique_actions_by_agent
action_coverage_by_agent
agent_line_domains
line_or_to_subid
line_ex_to_subid
```

The line-domain information is important for explainability.

For an agent with observed substation domain `D_i`, a line `l` is considered
visible to the agent if at least one endpoint substation is in `D_i`:

```text
line l visible to agent i
<=> line_or_to_subid[l] in D_i or line_ex_to_subid[l] in D_i
```

Therefore, boundary lines between two zones are shared. If one endpoint is in
agent 0's domain and the other endpoint is in agent 1's domain, then the same
line belongs to both local line domains. This is intentional: boundary lines are
shared observation and shared responsibility, while actions remain local to each
agent's action space.

## Are There Duplicates?

There are no duplicate action ids within the same agent at the same collected
state when sampling is used, because sampling is without replacement.

There are intentional repetitions:

1. The same action can be evaluated in many different risky states.
2. The same action can appear in several Slurm parts.
3. Action `0` can appear for every agent because each agent has its own action
   space.
4. `state_id` is only unique inside one part. Across parts, `state_id=0` appears
   many times.

For global analysis, use:

```text
(part_id, state_id)
```

as the unique state key.

These repetitions are not a problem. They are required to estimate whether an
action is repeatedly useful across many grid states.

## Stage 2: Reduce The Action Space

The reducer is:

```text
Topology_Task/teacher_student/reduce_action_space_from_outcomes.py
```

The JED launcher is:

```text
Topology_Task/teacher_student/job_reduce_action_space_jed.sh
```

You can launch the reducer either by setting environment variables directly or
by passing a config file to the Slurm wrapper.

Config files live in:

```text
Topology_Task/teacher_student/reduction_configs/
```

Ready-to-run bus36 configs:

```text
bus36_best_per_state_delta_do_nothing_k208.env
bus36_all_improving_delta_do_nothing_k208.env
bus36_improvement_rate_delta_do_nothing_k208.env
```

Each config controls:

```text
DATASET_ROOT
ACTION_REDUCTION_NAME
ACTION_REDUCTION_OUTPUT
ACTION_REDUCTION_TOP_K
ACTION_REDUCTION_MIN_COUNT
ACTION_REDUCTION_METRIC
ACTION_REDUCTION_METHOD
ACTION_REDUCTION_REQUIRE_IMPROVEMENT
ACTION_REDUCTION_IMPROVEMENT_TOLERANCE
```

Launch from the repository root:

```bash
cd /home/plumet/RL_Marl2grid

sbatch Topology_Task/teacher_student/job_reduce_action_space_jed.sh \
  Topology_Task/teacher_student/reduction_configs/bus36_improvement_rate_delta_do_nothing_k208.env
```

This writes:

```text
${DATASET_ROOT}/metadata/reduced_action_space_${ACTION_REDUCTION_NAME}.json
```

For the improvement-rate config above, the output is:

```text
outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_improvement_rate_delta_do_nothing_k208_min20.json
```

The reducer reads all `parts/part_*` datasets and writes the configured JSON
file. If neither `ACTION_REDUCTION_NAME` nor `ACTION_REDUCTION_OUTPUT` is set,
the wrapper generates a descriptive default name:

```text
${DATASET_ROOT}/metadata/reduced_action_space_${METHOD}_${METRIC}_k${TOP_K}_min${MIN_COUNT}.json
```

The output JSON contains, for each agent:

```text
original_action_size
selected_action_size
selected_fraction
selected_action_ids
ranked_actions
stats
```

Action `0` is always included if `include_action_zero=true`.

### Recommended Reduction Command

Run from the repository root:

```bash
cd /home/plumet/RL_Marl2grid

DATASET_ROOT=outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs \
ACTION_REDUCTION_TOP_K=208 \
ACTION_REDUCTION_MIN_COUNT=20 \
ACTION_REDUCTION_METRIC=delta_vs_do_nothing \
ACTION_REDUCTION_METHOD=improvement_rate \
ACTION_REDUCTION_NAME=improvement_rate_delta_do_nothing_k208_min20 \
ACTION_REDUCTION_REQUIRE_IMPROVEMENT=true \
ACTION_REDUCTION_IMPROVEMENT_TOLERANCE=1e-3 \
sbatch Topology_Task/teacher_student/job_reduce_action_space_jed.sh
```

This writes:

```text
outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_improvement_rate_delta_do_nothing_k208_min20.json
```

If you want to keep multiple reduced action spaces from the same dataset, use
the Python reducer directly with a custom output path:

```bash
cd /home/plumet/RL_Marl2grid/Topology_Task

python -u teacher_student/reduce_action_space_from_outcomes.py \
  --dataset outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/parts/part_* \
  --output outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_improvement_rate.json \
  --top-k 208 \
  --min-count 20 \
  --metric delta_vs_do_nothing \
  --selection-method improvement_rate \
  --require-improvement true \
  --improvement-tolerance 1e-3
```

## Reduction Metrics

The reducer can rank actions using one of these metrics:

```text
delta_vs_do_nothing
delta_vs_now
rho_after_action
```

`delta_vs_do_nothing`

This is the recommended default. It measures whether the action is better than
doing nothing at the same state:

```text
delta_vs_do_nothing(a) = rho_after_action(a) - rho_after_do_nothing
```

Lower is better. Negative means the action improves relative to do nothing.

`delta_vs_now`

This measures whether the action lowers the current maximum rho:

```text
delta_vs_now(a) = rho_after_action(a) - rho_before
```

Lower is better. Negative means the action reduces max rho compared to the
current state.

`rho_after_action`

This directly ranks by the resulting simulated max rho after the action. Lower
is better.

The improvement threshold is controlled by:

```text
ACTION_REDUCTION_IMPROVEMENT_TOLERANCE
```

The default is:

```text
1e-3
```

For delta metrics, an action is considered improving if:

```text
metric < -ACTION_REDUCTION_IMPROVEMENT_TOLERANCE
```

## Reduction Methods

There are three reducer methods.

### 1. best_per_state

Command:

```bash
ACTION_REDUCTION_METHOD=best_per_state
```

For each agent and each collected state, this method finds the valid action with
the best metric value. With `ACTION_REDUCTION_REQUIRE_IMPROVEMENT=true`, it only
considers actions satisfying:

```text
metric < -ACTION_REDUCTION_IMPROVEMENT_TOLERANCE
```

Then it gives one count to the winning action:

```text
count[a] += 1
```

Actions are ranked by count.

Interpretation:

```text
How often was this action the best sampled intervention for this agent?
```

This is close to a hard teacher-selection procedure. It is useful when you want
the reduced action set to contain actions that often win outright.

### 2. all_improving

Command:

```bash
ACTION_REDUCTION_METHOD=all_improving
```

For every valid sampled action, if:

```text
metric < -ACTION_REDUCTION_IMPROVEMENT_TOLERANCE
```

then:

```text
count[a] += 1
```

Actions are ranked by count.

Interpretation:

```text
How often did this action improve the grid, even if it was not the single best?
```

This is less aggressive than `best_per_state`. It keeps actions that are
frequently helpful, not only actions that are the best in a particular state.

### 3. improvement_rate

Command:

```bash
ACTION_REDUCTION_METHOD=improvement_rate
ACTION_REDUCTION_MIN_COUNT=20
```

For each action:

```text
seen_count[a] = number of times action a was sampled
improved_count[a] = number of valid simulations where metric < -tolerance
score[a] = improved_count[a] / seen_count[a]
```

Actions are ranked by:

```text
highest score
then highest improved_count
then highest seen_count
then lowest action_id
```

In this mode, `ACTION_REDUCTION_MIN_COUNT` means minimum `seen_count`, not
minimum number of wins.

Interpretation:

```text
When this action is sampled in a risky state, how often does it help?
```

This method is useful when sampling is nonuniform or when one agent has a huge
action space. It avoids favoring actions only because they were sampled more
often, but it requires a reasonable `min-count` to avoid lucky one-shot actions.

Recommended starting value:

```text
ACTION_REDUCTION_MIN_COUNT=20
```

If too many selected actions have low support, increase it to 50 or 100.

## Choosing A Reducer

Recommended first comparison:

```text
best_per_state
all_improving
improvement_rate
```

Use different output filenames so they do not overwrite each other:

```bash
cd /home/plumet/RL_Marl2grid/Topology_Task

python -u teacher_student/reduce_action_space_from_outcomes.py \
  --dataset outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/parts/part_* \
  --output outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_best_per_state.json \
  --top-k 208 \
  --min-count 1 \
  --metric delta_vs_do_nothing \
  --selection-method best_per_state \
  --require-improvement true \
  --improvement-tolerance 1e-3

python -u teacher_student/reduce_action_space_from_outcomes.py \
  --dataset outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/parts/part_* \
  --output outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_all_improving.json \
  --top-k 208 \
  --min-count 5 \
  --metric delta_vs_do_nothing \
  --selection-method all_improving \
  --require-improvement true \
  --improvement-tolerance 1e-3

python -u teacher_student/reduce_action_space_from_outcomes.py \
  --dataset outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/parts/part_* \
  --output outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_improvement_rate.json \
  --top-k 208 \
  --min-count 20 \
  --metric delta_vs_do_nothing \
  --selection-method improvement_rate \
  --require-improvement true \
  --improvement-tolerance 1e-3
```

The best method is empirical. A good reduced space should:

1. keep do-nothing action `0`;
2. include actions from all agents that often reduce risk;
3. avoid selecting only rare lucky actions;
4. improve learning speed or final survival compared to the full action space.

## Stage 3: Train MAPPO With The Reduced Action Space

After reduction, pass the JSON to MAPPO through the environment argument:

```bash
python Topology_Task/main.py \
  --env-id bus36 \
  --action-type topology \
  --reduced-action-space outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_improvement_rate.json
```

With a config-based Slurm job, add the equivalent command-line override to
`job_jed.sh`:

```bash
sbatch job_jed.sh path/to/bus36_config.toml \
  --reduced-action-space outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_improvement_rate.json
```

The environment loads the JSON and maps reduced action ids back to original
Grid2Op action ids. Reduced action `0` maps to original action `0`.

## Optional: Greedy Survival Evaluation

Before training MAPPO, you can test whether the reduced action space contains
enough useful actions to survive chronics under a simulator-greedy controller.

The evaluator is:

```text
Topology_Task/teacher_student/evaluate_greedy_reduced_actions.py
```

The JED launcher is:

```text
Topology_Task/teacher_student/job_greedy_reduced_eval_jed.sh
```

This evaluation compares two policies on the same chronic split:

```text
do_nothing:
  all agents always take action 0

greedy_reduced:
  if global max rho >= threshold:
    simulate every reduced unilateral action
    compare each candidate to simulated do-nothing
    execute the action with the lowest rho_after_action if it improves over do-nothing
  else:
    take action 0
```

The greedy candidate set is unilateral:

```text
agent i takes one reduced action
all other agents take action 0
```

This matches the action-outcome reduction assumption. It does not solve the full
joint combinatorial action space.

Recommended smoke test:

```bash
cd /home/plumet/RL_Marl2grid

REDUCED_ACTION_SPACE=outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_improvement_rate.json \
OUTPUT_DIR=outputs/teacher_student_greedy_eval/smoke_improvement_rate \
MAX_EPISODES=2 \
DECISION_RHO_THRESHOLD=0.90 \
REQUIRE_IMPROVEMENT=true \
IMPROVEMENT_TOLERANCE=1e-3 \
SIM_WORKERS=71 \
sbatch --time=00:45:00 Topology_Task/teacher_student/job_greedy_reduced_eval_jed.sh
```

Full test-split evaluation:

```bash
cd /home/plumet/RL_Marl2grid

REDUCED_ACTION_SPACE=outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_improvement_rate.json \
OUTPUT_DIR=outputs/teacher_student_greedy_eval/test_improvement_rate \
SPLIT=test \
DECISION_RHO_THRESHOLD=0.90 \
REQUIRE_IMPROVEMENT=true \
IMPROVEMENT_TOLERANCE=1e-3 \
SIM_WORKERS=71 \
sbatch --time=06:00:00 Topology_Task/teacher_student/job_greedy_reduced_eval_jed.sh
```

The evaluator writes:

```text
outputs/teacher_student_greedy_eval/.../greedy_vs_do_nothing_summary.json
outputs/teacher_student_greedy_eval/.../greedy_vs_do_nothing_episodes.csv
```

Important summary metrics:

```text
greedy_mean_survival
do_nothing_mean_survival
mean_survival_delta
greedy_full_survival_rate
do_nothing_full_survival_rate
greedy_better_episodes
greedy_equal_episodes
greedy_worse_episodes
```

Interpretation:

If greedy survival is much better than do-nothing, the reduced action space
contains useful emergency actions. If greedy survival is similar to do-nothing,
the reduced action space may be too weak, the reducer may be selecting poor
actions, or the greedy unilateral assumption may be too limited. If greedy is
worse than do-nothing, keep `REQUIRE_IMPROVEMENT=true` and inspect invalid
actions, simulation errors, or whether the metric is selecting actions that
improve one-step rho but create later topology problems.

## Optional: Filter An Existing Dataset To A Higher Rho Threshold

If a dataset was collected with:

```text
COLLECTION_RHO_THRESHOLD=0.90
```

you can derive a stricter dataset from it without rerunning Grid2Op
simulations, provided the original action-outcome shards contain
`global_max_rho_before`. The collector stores this field by default.

The filter is:

```text
Topology_Task/teacher_student/filter_action_outcome_dataset.py
```

The JED launcher is:

```text
Topology_Task/teacher_student/job_filter_action_outcomes_jed.sh
```

The filter keeps only action rows satisfying:

```text
outcome_agent_i_global_max_rho_before > RHO_THRESHOLD
```

and only state-context rows satisfying:

```text
state_global_max_rho_before > RHO_THRESHOLD
```

With `INCLUSIVE=true`, the comparison becomes `>=`.

The filter does not recompute per-line masks such as
`state_high_rho_line_mask`. Those masks remain the masks computed during the
source collection, for example with `COLLECTION_RHO_THRESHOLD=0.90`. The new
threshold is stored separately in metadata as the filter threshold.

This operation is a pure subset selection. It does not resimulate actions, and
it does not change the already-computed values:

```text
rho_after_action
rho_after_do_nothing
delta_vs_now
delta_vs_do_nothing
```

So a `0.95` dataset derived from a `0.90` dataset answers:

```text
Among the states already collected at rho > 0.90, keep only the more severe
states with rho > 0.95.
```

It cannot recover states that were never collected in the original dataset. For
example, deriving `0.95` from `0.90` is valid, but deriving `0.85` from `0.90`
would be incomplete because states with `0.85 < rho <= 0.90` were never
simulated.

Example for WCCI bus36:

```bash
cd /home/plumet/RL_Marl2grid

SOURCE_DATASET=outputs/teacher_student_datasets/wcci_full2048a_90_v3 \
OUTPUT_DIR=outputs/teacher_student_datasets/wcci_full2048a_95_v3 \
RHO_THRESHOLD=0.95 \
INCLUSIVE=false \
OVERWRITE=false \
sbatch --time=04:00:00 Topology_Task/teacher_student/job_filter_action_outcomes_jed.sh
```

After the filter finishes, use the new dataset root exactly like a normal
collected dataset:

```bash
DATASET_ROOT=outputs/teacher_student_datasets/wcci_full2048a_95_v3 \
ACTION_REDUCTION_TOP_K_BY_AGENT=agent_0=77,agent_1=2048,agent_2=127,agent_3=1119 \
ACTION_REDUCTION_MIN_COUNT=20 \
ACTION_REDUCTION_METRIC=delta_vs_do_nothing \
ACTION_REDUCTION_METHOD=improvement_rate \
ACTION_REDUCTION_NAME=wcci_full2048a_95_v3_improvement_rate \
ACTION_REDUCTION_REQUIRE_IMPROVEMENT=true \
ACTION_REDUCTION_IMPROVEMENT_TOLERANCE=1e-3 \
sbatch Topology_Task/teacher_student/job_reduce_action_space_jed.sh
```

The filtered dataset keeps the same layout:

```text
wcci_full2048a_95_v3/
  parts/
    part_000/
      shards/
      state_contexts/
      metadata/
```

## Inspecting Dataset Quality

After collection, inspect each part's metadata:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("Topology_Task/outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs")
for meta_path in sorted(root.glob("parts/part_*/metadata/metadata.json")):
    meta = json.loads(meta_path.read_text())
    print(meta_path.parent.parent.name)
    print("  status:", meta.get("status"))
    print("  candidate_states:", meta.get("n_candidate_states"))
    print("  outcome_examples:", meta.get("n_outcome_examples"))
    print("  coverage:", meta.get("action_coverage_by_agent"))
PY
```

The important checks are:

```text
status == complete
n_candidate_states > 0
n_state_contexts == n_candidate_states
n_outcome_examples == n_candidate_states * sampled_actions_per_state
agent_1 coverage is nontrivial
```

For the recommended bus36 setup:

```text
sampled_actions_per_state = 77 + 2048 + 127 + 1119 = 3371
```

So:

```text
n_outcome_examples should equal n_candidate_states * 3371
```

Small deviations would indicate a collection or storage problem.

## Inspecting A Reduced Action-Space JSON

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("Topology_Task/outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_improvement_rate.json")
payload = json.loads(path.read_text())
print("total:", payload["total_selected_action_size"], "/", payload["total_original_action_size"])
for agent, info in payload["agents"].items():
    print(agent)
    print("  selected:", info["selected_action_size"], "/", info["original_action_size"])
    print("  first ranked:", info["ranked_actions"][:5])
PY
```

For `improvement_rate`, ranked entries look like:

```text
{
  "action_id": 123,
  "score": 0.42,
  "improved_count": 21,
  "seen_count": 50,
  "valid_finite_count": 49
}
```

For `best_per_state` or `all_improving`, ranked entries look like:

```text
{
  "action_id": 123,
  "count": 17
}
```

## Common Pitfalls

Do not use fixed small action subsets for the final dataset.

Use:

```text
OUTCOME_RESAMPLE_ACTIONS_PER_STATE=true
```

Otherwise each part can evaluate the same small action subset repeatedly.

Do not compare `state_id` across parts without the part id.

Use:

```text
(part_id, state_id)
```

Do not trust `improvement_rate` with `min-count=1`.

An action sampled once and improved once would get score `1.0`. Use at least:

```text
ACTION_REDUCTION_MIN_COUNT=20
```

Do not assume boundary lines belong to only one agent.

Boundary lines are visible to both adjacent agents if each agent observes one of
the line endpoints.

Do not assume the dataset is generated by a learned policy.

The brute-force action-outcome dataset is generated by a simulator teacher. The
rollout action is selected using simulated outcomes, not a trained actor.

## Current Recommended Pipeline

1. Collect bus36 action outcomes:

```bash
cd /home/plumet/RL_Marl2grid

OUTPUT_DIR=outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs \
CHRONIC_SHARD_COUNT=15 \
OUTCOME_ACTION_SAMPLE_SIZE=64 \
OUTCOME_ACTION_SAMPLE_SIZES=agent_0=all,agent_1=2048,agent_2=all,agent_3=all \
OUTCOME_RESAMPLE_ACTIONS_PER_STATE=true \
OUTCOME_SIM_WORKERS=71 \
OUTCOME_SIM_START_METHOD=spawn \
ACTION_REDUCTION_TOP_K=208 \
TIMING_EVERY_ENV_STEPS=500 \
OVERWRITE=true \
sbatch --array=0-14 --time=03:30:00 Topology_Task/teacher_student/job_collect_action_outcomes_jed.sh
```

2. Reduce by improvement rate:

```bash
cd /home/plumet/RL_Marl2grid/Topology_Task

python -u teacher_student/reduce_action_space_from_outcomes.py \
  --dataset outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/parts/part_* \
  --output outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_improvement_rate.json \
  --top-k 208 \
  --min-count 20 \
  --metric delta_vs_do_nothing \
  --selection-method improvement_rate \
  --require-improvement true \
  --improvement-tolerance 1e-3
```

3. Train MAPPO with the reduced action space:

```bash
sbatch job_jed.sh path/to/bus36_config.toml \
  --reduced-action-space outputs/teacher_student_datasets/bus36_context_rho090_a1_2048_15jobs/metadata/reduced_action_space_improvement_rate.json
```
