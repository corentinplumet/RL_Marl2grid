# Heuristic Teacher-Student Roadmap

This roadmap describes how to turn the strong `hvg` rho-threshold heuristic
results into a fully learned decentralized policy.

The motivation is:

```text
The global/local rho 0.90 heuristic gives high survival and high action-0 usage,
but it overwrites the action proposed by the trained agent at evaluation time.
The goal is to distill this behavior into a student policy that outputs action
0 directly, without a deployment-time heuristic.
```

This is a teacher-student imitation-learning direction. It is connected to:

- behavior cloning: train a policy on observation/action pairs,
- policy distillation: train a student to reproduce a teacher policy
  [Rusu et al., 2015][rusu-policy-distillation],
- knowledge distillation: transfer a teacher's behavior to another model
  [Hinton et al., 2015][hinton-distillation],
- DAgger / dataset aggregation: fix covariate shift by labeling states visited
  by the student [Ross et al., 2010][ross-dagger].

## Core Idea

The teacher is not just the original MAPPO policy. The teacher is:

```text
trained MAPPO policy + rho-threshold heuristic override
```

For each agent $i$, the trained policy first proposes an action:

$$
\bar a_{i,t}
=
\arg\max_a \pi_i^{base}(a \mid o_{i,t}).
$$

Then the heuristic modifies that action:

$$
a_{i,t}^{T}
=
\begin{cases}
0,
& \text{if the heuristic says the state is safe},\\
\bar a_{i,t},
& \text{otherwise.}
\end{cases}
$$

The dataset contains pairs:

$$
\left(o_{i,t}, a_{i,t}^{T}\right).
$$

The student policy is trained with supervised learning:

$$
\pi_i^{S}(a_i \mid o_i).
$$

The simplest behavior-cloning loss is:

$$
\mathcal L_{\mathrm{BC}}
=
-
\sum_{i,t}
\log
\pi_i^{S}
\left(
a_{i,t}^{T}
\mid
o_{i,t}
\right).
$$

At deployment, the student acts directly:

$$
a_{i,t}^{S}
=
\arg\max_a
\pi_i^{S}(a \mid o_{i,t}),
$$

with no heuristic override.

## Why This Is Interesting

The current heuristic result is strong, but conceptually imperfect:

```text
The policy proposes an action.
The heuristic may replace it by action 0.
The final deployed behavior is partly learned and partly hard-coded.
```

Teacher-student learning tries to convert this into:

```text
The policy directly learns to output action 0 in the states where the heuristic
would have overwritten the action.
```

So the final student can be presented as a learned decentralized policy rather
than a policy plus a hand-coded evaluation filter.

## Global Teacher vs Local Teacher

The most important design choice is whether the teacher uses the global or local
rho heuristic.

### Global Rho Teacher

The global heuristic computes:

$$
\rho_t^{global}
=
\max_{\ell \in \mathcal L}
\rho_{\ell,t}.
$$

It executes:

$$
a_{i,t}^{T,global}
=
\begin{cases}
0,
& \rho_t^{global} < \rho_{safe},\\
\bar a_{i,t},
& \rho_t^{global} \geq \rho_{safe}.
\end{cases}
$$

with:

$$
\rho_{safe}=0.90.
$$

This teacher can be very strong, but it uses privileged global information. If
the student only receives local observations, then the student may not be able
to perfectly reproduce the global teacher.

Use this teacher as:

```text
an upper-bound / privileged-teacher experiment.
```

Do not use it as the main decentralized claim unless the student observation
also contains the information required to reconstruct the global decision.

### Local Rho Teacher

The local heuristic computes:

$$
\rho_{i,t}^{local}
=
\max_{\ell \in \mathcal L_i}
\rho_{\ell,t}.
$$

It executes:

$$
a_{i,t}^{T,local}
=
\begin{cases}
0,
& \rho_{i,t}^{local} < \rho_{safe},\\
\bar a_{i,t},
& \rho_{i,t}^{local} \geq \rho_{safe}.
\end{cases}
$$

This is the recommended teacher for the main experiment because it is compatible
with decentralized execution:

```text
each agent is labeled using information that belongs to its local physical
neighborhood.
```

For a boundary line between two regions, the line can be included in both
agents' local neighborhoods because it touches substations in both zones. This
does not require communication at action time.

## Phase 0: Freeze The Teacher

Goal: select one or more teacher checkpoints and make the teacher deterministic.

Recommended teachers:

```text
hvg_01_eval_rho090_s*       global rho teacher
hvg_04_eval_local_rho090_s* local rho teacher
```

Use deterministic evaluation:

$$
\bar a_{i,t}
=
\arg\max_a \pi_i^{base}(a \mid o_{i,t}).
$$

Then apply the heuristic override to obtain:

$$
a_{i,t}^{T}.
$$

Important: save both actions:

```text
policy_action_before_override = \bar a_{i,t}
teacher_action_after_override = a^T_{i,t}
```

The supervised label should be the final teacher action after override:

$$
y_{i,t}=a_{i,t}^{T}.
$$

## Phase 1: Collect The Teacher Dataset

Goal: run the teacher on chronics and store per-agent training examples.

Each row should correspond to one agent at one environment step:

$$
\left(
i,\ t,\ \mathrm{chronic},\ o_{i,t},\ \bar a_{i,t},\ a_{i,t}^{T}
\right).
$$

Recommended schema:

| Field | Meaning |
| --- | --- |
| `agent_id` | Agent name, e.g. `agent_0`. |
| `agent_index` | Integer agent index. |
| `chronic_id` | Chronic name or index. |
| `episode_id` | Rollout episode index. |
| `episode_step` | Step inside the episode. |
| `local_obs` | Exact local observation used by the actor. |
| `graph_obs` | Graph observation, if the actor uses the GNN path. |
| `flat_obs` | Flat observation, if `gnn_concat_flat=true`. |
| `teacher_policy_action` | Action proposed by the base policy before override. |
| `teacher_action` | Final executed action after heuristic override. |
| `was_overwritten` | `teacher_policy_action != teacher_action`. |
| `teacher_nonidle` | `teacher_action != 0`. |
| `policy_nonidle` | `teacher_policy_action != 0`. |
| `local_max_rho` | Local pre-action max rho for that agent. |
| `global_max_rho` | Global pre-action max rho, useful for analysis. |
| `rho_threshold` | Usually `0.90`. |
| `heuristic_type` | `global_rho` or `local_rho`. |
| `reward` | Environment reward after executing the teacher joint action. |
| `done` | Episode termination flag. |
| `survival_step` | Useful for post-hoc survival analysis. |

For storage, prefer sharded files:

```text
outputs/teacher_student_datasets/
  local_rho090_seed0/
    metadata.json
    shard_00000.npz
    shard_00001.npz
    ...
```

or use HDF5/Zarr if the graph observations become too large for convenient
`.npz` shards.

## Phase 2: Dataset Quality Checks

Before training a student, inspect the dataset. The key risk is class imbalance:
the heuristic teacher will likely output action `0` most of the time.

Compute:

$$
p_i(a=0)
=
\frac{1}{N_i}
\sum_t
\mathbf{1}[a_{i,t}^{T}=0].
$$

Also compute:

$$
p_i(\mathrm{overwrite})
=
\frac{1}{N_i}
\sum_t
\mathbf{1}[\bar a_{i,t}\neq a_{i,t}^{T}].
$$

Required summary metrics:

```text
n_examples_total
n_examples_per_agent
frac_teacher_action_0_agent_i
frac_teacher_nonidle_agent_i
frac_policy_nonidle_agent_i
frac_overwritten_agent_i
frac_overwrite_to_action0_agent_i
teacher_survival_mean
teacher_survival_by_chronic
local_max_rho_histogram_when_teacher_action0
local_max_rho_histogram_when_teacher_nonidle
most_common_teacher_nonidle_actions_agent_i
```

Success criterion:

```text
The dataset contains enough non-idle examples to teach the student when to act,
not only when to do nothing.
```

If the non-idle fraction is extremely low, plain behavior cloning may collapse
to always predicting action `0`.

## Phase 3: Train A Behavior-Cloned Student

The first student should keep the same actor architecture as the baseline flat
actor:

$$
\pi_i^{S}(a_i \mid o_i),
\qquad
a_i \in \{0,\dots,|\mathcal A_i|-1\}.
$$

The default loss is cross-entropy:

$$
\mathcal L_{\mathrm{CE}}
=
-
\frac{1}{N}
\sum_{i,t}
\log
\pi_i^{S}
\left(
a_{i,t}^{T}
\mid
o_{i,t}
\right).
$$

Because action `0` can dominate the dataset, use weighted cross-entropy:

$$
\mathcal L_{\mathrm{WCE}}
=
-
\frac{1}{N}
\sum_{i,t}
w(a_{i,t}^{T})
\log
\pi_i^{S}
\left(
a_{i,t}^{T}
\mid
o_{i,t}
\right).
$$

One simple weighting is:

$$
w(a)
=
\begin{cases}
w_0,
& a=0,\\
w_{nonidle},
& a\neq 0,
\end{cases}
$$

with:

$$
w_{nonidle} > w_0.
$$

This prevents the student from learning the trivial solution:

$$
\pi_i^S(a_i=0 \mid o_i) \approx 1.
$$

## Phase 4: Add An Auxiliary Intervention Loss

A useful auxiliary label is:

$$
z_{i,t}^{T}
=
\mathbf{1}[a_{i,t}^{T}\neq 0].
$$

The student already has:

$$
P_i^S(\mathrm{intervene}\mid o_i)
=
1-\pi_i^S(a_i=0\mid o_i).
$$

Add a binary cross-entropy loss:

$$
\mathcal L_{\mathrm{int}}
=
-
\sum_{i,t}
\left[
z_{i,t}^{T}
\log P_i^S(\mathrm{intervene}\mid o_i)
+
(1-z_{i,t}^{T})
\log
\left(
1-P_i^S(\mathrm{intervene}\mid o_i)
\right)
\right].
$$

The full supervised loss becomes:

$$
\mathcal L_{\mathrm{student}}
=
\mathcal L_{\mathrm{WCE}}
+
\alpha
\mathcal L_{\mathrm{int}}.
$$

This keeps the actor flat, but explicitly teaches:

```text
when to do nothing
when to intervene
which action to use when intervention is needed
```

without introducing a deployment-time heuristic.

## Phase 5: Optional Soft-Label Distillation

If the teacher policy logits are saved, the student can learn from soft labels
instead of only hard action ids.

Let the base policy distribution be:

$$
p_i^{base}(a \mid o_i).
$$

For states where the heuristic does not overwrite, use the base policy
distribution as a soft target.

For states where the heuristic overwrites to no-op, define a heuristic-adjusted
soft target:

$$
q_i^T(a\mid o_i)
=
\begin{cases}
1,
& a=0,\\
0,
& a\neq 0.
\end{cases}
$$

Then add a KL loss:

$$
\mathcal L_{\mathrm{KL}}
=
\sum_{i,t}
D_{\mathrm{KL}}
\left(
q_i^T(\cdot\mid o_i)
\;\|\;
\pi_i^S(\cdot\mid o_i)
\right).
$$

This is closer to policy distillation [Rusu et al.,
2015][rusu-policy-distillation] and knowledge distillation [Hinton et al.,
2015][hinton-distillation].

## Phase 6: Evaluate The Student Without Heuristic

The decisive evaluation is:

```text
student policy
no eval_action_heuristic
deterministic_eval = true
```

The student action is:

$$
a_{i,t}^{S}
=
\arg\max_a
\pi_i^S(a\mid o_{i,t}).
$$

No action should be overwritten.

Compare:

| Method | Evaluation-time heuristic? | Purpose |
| --- | --- | --- |
| Baseline `hvg_00` | No | Original learned policy. |
| Global heuristic teacher `hvg_01` | Yes | Privileged upper-bound teacher. |
| Local heuristic teacher `hvg_04` | Yes | Decentralized-compatible teacher. |
| Student from local teacher | No | Main result. |
| Student from global teacher | No | Tests whether privileged teacher can be approximated locally. |

Metrics:

```text
test/episodic_survival
train_eval/episodic_survival
test/explain/frac_action_0_agent_i
test/explain/action_nonidle_agent_i
test/frac_multi_agent_non_idle
test/explain/delta_max_rho
test/explain/topology_distance_delta
imitation_accuracy_agent_i
imitation_accuracy_nonidle_agent_i
false_noop_rate_agent_i
false_intervention_rate_agent_i
```

Definitions:

$$
\mathrm{accuracy}_i
=
\frac{1}{N_i}
\sum_t
\mathbf{1}
\left[
a_{i,t}^{S}=a_{i,t}^{T}
\right].
$$

False no-op rate:

$$
\mathrm{false\_noop}_i
=
P
\left(
a_{i,t}^{S}=0
\mid
a_{i,t}^{T}\neq 0
\right).
$$

False intervention rate:

$$
\mathrm{false\_intervention}_i
=
P
\left(
a_{i,t}^{S}\neq 0
\mid
a_{i,t}^{T}=0
\right).
$$

Both matter:

```text
false_noop too high        -> student misses necessary interventions
false_intervention too high -> student loses action sparsity
```

## Phase 7: RL Fine-Tuning From The Student

Pure behavior cloning can fail in closed loop because the student visits states
that the teacher dataset did not cover. After supervised pretraining, fine-tune
the student with MAPPO.

Recommended fine-tuning variants:

```text
student initialization + no extra sparse loss
student initialization + sparse fixed penalty p0.010
student initialization + adaptive local-safe budget target 0.20
```

The cleanest objective is:

$$
r'_{i,t}
=
r_{i,t}
-
\lambda_i
\mathbf{1}[a_{i,t}\neq 0]
w_i^{local}(s_t).
$$

This combines:

```text
teacher-student initialization
+ adaptive local sparse control
```

Optional KL regularization can keep the fine-tuned policy close to the student:

$$
\mathcal L_{\mathrm{fine\_tune}}
=
\mathcal L_{\mathrm{PPO}}
+
\beta
D_{\mathrm{KL}}
\left(
\pi_i(\cdot\mid o_i)
\;\|\;
\pi_i^{S,init}(\cdot\mid o_i)
\right).
$$

Use this only if fine-tuning destroys the sparse behavior.

## Phase 8: DAgger If Behavior Cloning Is Not Enough

Behavior cloning trains on states visited by the teacher. During deployment, the
student may make different actions and visit different states. This is the
classic covariate-shift problem addressed by DAgger [Ross et al.,
2010][ross-dagger].

DAgger loop:

```text
1. Train student on the current dataset.
2. Roll out the student without heuristic.
3. At each visited state, query the heuristic teacher for the action it would
   have taken.
4. Add those new labeled states to the dataset.
5. Retrain or fine-tune the student.
```

Mathematically:

$$
\mathcal D_{k+1}
=
\mathcal D_k
\cup
\left\{
\left(
o_{i,t}^{S,k},
a_{i,t}^{T}
\right)
\right\}.
$$

Use DAgger only if:

```text
offline imitation accuracy is high
but closed-loop survival is much worse than the teacher
```

because that pattern means the student is accurate on teacher states but brittle
on its own visited states.

## Phase 9: Optional Learned Override Classifier

A smaller intermediate experiment is to distill only the override decision.

Train a classifier:

$$
m_i(o_i)
=
P(\mathrm{force\ no\text{-}op}\mid o_i).
$$

Label:

$$
y_{i,t}^{override}
=
\mathbf{1}
\left[
a_{i,t}^{T}=0
\;\land\;
\bar a_{i,t}\neq 0
\right].
$$

At inference:

$$
a_{i,t}^{exec}
=
\begin{cases}
0,
& m_i(o_i)>\tau,\\
\bar a_{i,t},
& m_i(o_i)\leq \tau.
\end{cases}
$$

This is less clean than a full student because it still has a two-stage
decision, but it is useful diagnostically:

```text
Can a learned local classifier reproduce the heuristic override?
```

If this works, then full student distillation is likely promising.

## Recommended Implementation Order

1. Add a teacher rollout collector.
2. Collect local-rho teacher dataset first.
3. Add dataset summary script.
4. Train a flat supervised student with weighted cross-entropy.
5. Evaluate the student without heuristic.
6. Add auxiliary intervention loss if the student collapses to action `0`.
7. Fine-tune with MAPPO and sparse/AIB objective.
8. Add DAgger only if closed-loop covariate shift is severe.

## Recommended First Experiments

Use three seeds when moving beyond smoke tests.

| Experiment | Teacher | Student objective | Purpose |
| --- | --- | --- | --- |
| `ts_00_local_bc` | local rho 0.90 | weighted CE | Main decentralized distillation baseline. |
| `ts_01_local_bc_aux` | local rho 0.90 | weighted CE + intervention BCE | Tests whether explicit when-to-act supervision helps. |
| `ts_02_global_bc` | global rho 0.90 | weighted CE | Privileged teacher approximation. |
| `ts_03_local_bc_mappo_sparse` | local rho 0.90 | BC init + sparse p0.010 fine-tune | Tests whether RL recovers survival. |
| `ts_04_local_bc_mappo_aib` | local rho 0.90 | BC init + AIB local-safe fine-tune | Most principled final candidate. |

## Success Criteria

The student is successful if:

```text
1. It runs with eval_action_heuristic = none.
2. Survival remains close to the local heuristic teacher.
3. Action-0 usage remains much higher than the original baseline.
4. Non-idle actions still happen in high-risk local states.
5. Multi-agent simultaneous non-idle actions are reduced or controlled.
```

More formally, the target behavior is:

$$
\mathrm{Survival}(\pi^S)
\approx
\mathrm{Survival}(\pi^T)
$$

and:

$$
P_{\pi^S}(a_i=0)
\gg
P_{\pi^{base}}(a_i=0),
$$

without a deployment-time heuristic.

## Main Risks

### Class Imbalance

If the teacher mostly outputs action `0`, plain behavior cloning can learn a
degenerate student:

$$
\pi_i^S(a_i=0\mid o_i)\approx 1.
$$

Mitigation:

```text
weighted cross-entropy
oversampling non-idle examples
auxiliary intervention loss
```

### Privileged Teacher Information

The global teacher may use information that the decentralized student cannot
observe.

Mitigation:

```text
use local teacher as the main result
report global teacher distillation as privileged-teacher analysis
```

### Closed-Loop Covariate Shift

The student may be accurate on teacher states but poor on states caused by its
own mistakes.

Mitigation:

```text
DAgger
RL fine-tuning
evaluate closed-loop survival, not only imitation accuracy
```

### Loss Of Rare Critical Interventions

The most important examples may be rare overloaded states.

Mitigation:

```text
oversample high-rho states
oversample non-idle teacher actions
report false_noop_rate on teacher-nonidle states
```

## Final Thesis Framing

The clean final story would be:

```text
We first observe that a local rho-threshold heuristic can make the trained
decentralized agents much sparser while preserving survival. Rather than
deploying this heuristic directly, we use it as an expert teacher. We collect
teacher trajectories over chronics and train a decentralized student policy to
imitate the final executed teacher actions. The resulting student removes the
hard-coded evaluation override while preserving the sparse, operator-like
behavior induced by the teacher.
```

If RL fine-tuning with AIB works, the stronger final story is:

```text
Teacher-student imitation provides a sparse and safe initialization, and
adaptive local intervention budgeting then fine-tunes the policy under the true
closed-loop grid-control objective.
```

## References

- [Rusu et al., 2015][rusu-policy-distillation]: policy distillation. Used here
  to frame the transfer from a strong teacher policy to a student policy.
- [Hinton et al., 2015][hinton-distillation]: knowledge distillation. Used here
  to justify optional soft-label imitation.
- [Ross et al., 2010][ross-dagger]: DAgger / dataset aggregation. Used here to
  address behavior-cloning covariate shift.

[rusu-policy-distillation]: https://arxiv.org/abs/1511.06295
[hinton-distillation]: https://arxiv.org/abs/1503.02531
[ross-dagger]: https://arxiv.org/abs/1011.0686
