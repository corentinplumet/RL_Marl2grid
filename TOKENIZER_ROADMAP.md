# Tokenizer Transformer Roadmap

This roadmap describes how to add transformer-based actor-critic encoders for
the WCCI 36 topology-control environment while preserving the current MAPPO
training loop, reduced action-space support, intervention gate, and existing
MLP/GNN baselines.

The goal is not only to make the network deeper than the current
`[256, 256, 256]` MLP. The goal is to change the representation from one flat
vector to a set of meaningful grid tokens, so the model can learn interactions
between production, consumption, line loading, topology, cooldowns,
maintenance, and local agent regions.

Two tokenization families should be implemented:

```text
broad feature-group tokens
grid entity tokens
```

Both should feed the same transformer encoder API so they can be compared
cleanly in WCCI runs.

## Current Code Context

The relevant current implementation is:

```text
Topology_Task/alg/mappo/agent.py
  Actor and Critic select encoders with actor_encoder / critic_encoder.
  Existing choices are mlp and gnn.

Topology_Task/common/gnn.py
  GraphEncoder and GraphAndFlatEncoder for busbar graph observations.

Topology_Task/common/graph.py
  GridGraphBuilder builds fixed-shape busbar graph observations.

Topology_Task/common/utils.py
  get_flat_obs, get_joint_obs, strip_state_graph, tensor casting utilities.

Topology_Task/env/utils.py
  Builds Grid2Op/MARL envs, chooses flat vs graph observations, formats obs.

Topology_Task/env/config.py
  Defines actor_encoder / critic_encoder CLI choices.

Topology_Task/alg/mappo/config.py
  Defines PPO and GNN hyperparameters.

Topology_Task/env/scenario.json
  Defines available observation attribute groups.

Topology_Task/configs/wcci_reduced_mlp/
  Current WCCI 36 reduced-action-space MLP configs.
```

The initial transformer implementation should follow the same pattern as the
GNN path:

```text
flat observation remains available
optional structured observation is added when a structured encoder is enabled
Actor and Critic turn structured observations into one embedding
existing categorical and intervention-gated action heads remain unchanged
```

This keeps the first version low-risk. The transformer becomes a drop-in
encoder, not a rewrite of MAPPO.

## Design Principles

### Keep MLP, GNN, And Transformer Comparable

Do not remove or rewrite the current MLP/GNN behavior. Add a new encoder value:

```toml
actor_encoder = "transformer"
critic_encoder = "transformer"
tokenizer_type = "group"      # or "entity", later "hybrid"
```

Why this matters:

- WCCI 36 is complex, and a transformer may help, but it may also be less
  sample-efficient.
- We need clean ablations against `mlp` and `gnn`.
- Existing checkpoints and configs should not break.

### Separate Tokenization From Attention

The tokenizer should only define what tokens exist and what raw features each
token carries. The transformer should only define how tokens are embedded,
mixed, pooled, and returned as an encoder vector.

Why this matters:

- Broad-group and entity tokenizers can share one encoder implementation.
- Token schema tests can run without training a policy.
- Future tokenizers can be added without touching PPO.

### Start With A Compatible Actor Head

Version 1 should keep the existing action heads:

```text
transformer tokens -> pooled embedding -> existing categorical action head
transformer tokens -> pooled embedding -> existing intervention gate heads
```

Why this matters:

- It isolates representation changes from action-factorization changes.
- Reduced action-space configs continue to work.
- PPO loss, entropy, KL, checkpointing, and evaluation need minimal changes.

Later, add action-conditioned or entity-conditioned heads.

## Target Observation Structure

When transformer encoders are enabled, each agent observation should become:

```python
{
    "flat": np.ndarray[flat_dim],
    "tokens": {
        "features": np.ndarray[n_tokens, token_feature_dim],
        "type_ids": np.ndarray[n_tokens],
        "entity_ids": np.ndarray[n_tokens],
        "entity_type_ids": np.ndarray[n_tokens],
        "token_mask": np.ndarray[n_tokens],
        "feature_mask": np.ndarray[n_tokens, token_feature_dim],
        # optional, phase 2+
        "substation_ids": np.ndarray[n_tokens],
        "line_or_sub_ids": np.ndarray[n_tokens],
        "line_ex_sub_ids": np.ndarray[n_tokens],
    },
    "state_tokens": same schema for centralized critic
}
```

The actor uses `"tokens"` for the local agent region. The critic uses
`"state_tokens"` for the full grid, just as the GNN critic currently uses
`"state_graph"`.

Why this matters:

- Actor stays decentralized at execution time.
- Critic can remain centralized during training.
- `flat` remains present for `transformer_concat_flat = true` ablations and
  debugging.

## Tokenizer Family A: Broad Feature-Group Tokens

Broad feature-group tokenization is closest to the original idea:

```text
one token for all production features
one token for all load features
one token for all line loading features
one token for topology/connectivity features
one token for cooldown/maintenance features
one token for global context
```

### Proposed Group Tokens

For WCCI 36, start with these local actor tokens:

```text
global
generation
load
line_loading
line_status
topology
substation_cooldown
line_cooldown
maintenance
redispatch
curtailment_if_available
local_domain
```

The centralized critic uses the same token types over the full grid instead of
the local observation domain.

### Group Token Raw Features

`global`:

```text
normalized timestep if available
max rho
mean rho
number of overloaded lines
number of disconnected lines
number of substations on cooldown
number of lines on cooldown
agent/domain id for actor tokens
```

`generation`:

```text
gen_p
gen_theta
target_dispatch
actual_dispatch
gen_margin_up
gen_margin_down
gen_p_before_curtail if available
curtailment if available
curtailment_limit if available
summary stats: mean, max, min, std, sum
```

`load`:

```text
load_p
load_theta
summary stats: mean, max, min, std, sum
```

`line_loading`:

```text
rho
timestep_overflow
optional p_or/p_ex/a_or/a_ex/theta_or/theta_ex if added later
summary stats: max, top-k rho values, mean, std, count above thresholds
```

`line_status`:

```text
line_status
count disconnected
fraction disconnected
local line mask summary
```

`topology`:

```text
topo_vect for local assets
bus assignment counts per substation
number of substations with split topology
number of assets on bus 1 / bus 2
```

`substation_cooldown`:

```text
time_before_cooldown_sub
max, mean, count positive
```

`line_cooldown`:

```text
time_before_cooldown_line
max, mean, count positive
```

`maintenance`:

```text
time_next_maintenance
duration_next_maintenance
count lines with maintenance soon
minimum time_next_maintenance among active planned outages
```

`local_domain`:

```text
binary or summary indicators for controlled substations
number of controlled substations
number of observed neighbor substations
number of local lines
```

### Why Broad-Group Tokens Are Useful

Broad tokens are cheap:

- small token count,
- low transformer memory cost,
- simple shape management,
- fast enough for 72 parallel WCCI envs.

They are also a good bridge from the current MLP. The model still sees groups
of features, but attention can learn that line loading should attend to
topology, maintenance, and cooldown context.

### Main Limitation

One large token for all lines or all generators still hides entity structure.
The model must infer which line or substation is important from packed vectors.
This is better than a flat MLP but weaker than entity tokens when the decision
depends on a specific overloaded line and nearby substations.

## Tokenizer Family B: Grid Entity Tokens

Entity tokenization makes each physical grid object a token. This should be the
main target architecture after the broad-group version works.

### Proposed Entity Tokens

For WCCI 36:

```text
CLS/global token
one generator token per generator
one load token per load
one line token per line
one substation token per substation
optional one busbar token per substation busbar
optional one agent-domain token for actor observations
```

The local actor tokenizer should include:

```text
controlled substations
neighbor substations if enabled
lines touching the observed substations
generators attached to included substations
loads attached to included substations
one global/local summary token
```

The centralized critic tokenizer should include all WCCI 36 entities.

### Entity Feature Design

`generator` token:

```text
gen_p
gen_theta
target_dispatch
actual_dispatch
gen_margin_up
gen_margin_down
gen_p_before_curtail if available
curtailment if available
curtailment_limit if available
attached substation id embedding
current bus id embedding
domain mask
```

`load` token:

```text
load_p
load_theta
attached substation id embedding
current bus id embedding
domain mask
```

`line` token:

```text
line_status
rho
timestep_overflow
time_before_cooldown_line
time_next_maintenance
duration_next_maintenance
origin substation id embedding
extremity substation id embedding
origin bus id embedding
extremity bus id embedding
is_local_boundary_line
domain mask
```

`substation` token:

```text
time_before_cooldown_sub
number of attached generators
number of attached loads
number of attached line endpoints
number of assets on bus 1
number of assets on bus 2
is_controlled_by_agent
is_observed_neighbor
```

`busbar` token, optional:

```text
substation id embedding
busbar id embedding
sum gen_p on busbar
sum load_p on busbar
number of line endpoints on busbar
domain mask
```

`CLS/global` token:

```text
max rho
mean rho
number of overloaded lines
number of disconnected lines
number of active cooldowns
number of upcoming maintenance events
agent id for actor observations
```

### Why Entity Tokens Are Useful

Entity tokens match the causal structure of the problem better than flat
vectors:

- A high `rho` belongs to a specific line.
- A topology action belongs to a specific substation.
- A generator/load imbalance belongs to attached substations.
- A line endpoint depends on busbar assignment.
- Maintenance risk is tied to specific lines.

The transformer can learn interactions such as:

```text
overloaded line -> endpoint substations -> local topology cooldown -> nearby load/generation
```

This is exactly the kind of relational reasoning a flat MLP struggles with.

### Main Limitation

Entity tokens create more tokens and need more careful engineering:

- variable counts across local agents,
- padding and token masks,
- entity ID embeddings,
- connection information,
- larger parameter count and slower PPO updates,
- higher risk of overfitting or unstable RL.

For that reason, entity tokenization should come after a simple broad-group
transformer is already training end-to-end.

## Optional Family C: Hybrid Tokens

After both variants run, add:

```text
entity tokens + broad summary tokens
```

This often works well because the transformer receives both detailed local
objects and precomputed global summaries.

The hybrid actor sequence could be:

```text
[CLS]
[global summary]
[line loading summary]
[topology summary]
[generator tokens]
[load tokens]
[line tokens]
[substation tokens]
```

Why this helps:

- summary tokens stabilize learning early,
- entity tokens provide precise localization,
- attention can use summaries as routing/context tokens.

## Transformer Encoder Architecture

Add a new module:

```text
Topology_Task/common/tokenizer.py
Topology_Task/common/token_transformer.py
```

`tokenizer.py` should contain observation builders and token schema utilities.
`token_transformer.py` should contain PyTorch modules.

### Token Builder Interface

Create a builder similar in spirit to `GridGraphBuilder`:

```python
class GridTokenBuilder:
    def __init__(
        self,
        g2op_env,
        observation_domains,
        tokenizer_type: str,
        include_neighbors: bool,
        include_maintenance: bool,
        max_token_feature_dim: int,
    ) -> None:
        ...

    @property
    def specs(self) -> Dict[str, Dict[str, Any]]:
        ...

    def build(self, obs) -> Dict[str, Dict[str, np.ndarray]]:
        ...
```

The return value should mirror the graph builder:

```python
{
    "state": state_tokens,
    "agent_0": local_tokens,
    "agent_1": local_tokens,
    ...
}
```

### Token Tensor Schema

Use fixed-size arrays so rollout buffers can store them:

```text
features: [n_tokens, token_feature_dim], float32
feature_mask: [n_tokens, token_feature_dim], float32
token_mask: [n_tokens], float32
type_ids: [n_tokens], int64
entity_type_ids: [n_tokens], int64
entity_ids: [n_tokens], int64
substation_ids: [n_tokens], int64
```

Why `feature_mask` matters:

- group and entity tokens may not all use the same number of raw features,
- padding should not silently become a real zero-valued feature,
- masked feature inputs make the tokenizer easier to debug.

### Embedding Layer

The transformer should not consume raw padded features directly. Use:

```text
masked raw features -> Linear(token_feature_dim, d_model)
token type embedding -> d_model
entity type embedding -> d_model
entity id embedding -> d_model
substation/bus/endpoint embeddings -> d_model
sum -> LayerNorm -> transformer
```

Configuration:

```toml
transformer_d_model = 128
transformer_n_heads = 4
transformer_layers = 3
transformer_ff_dim = 512
transformer_dropout = 0.05
transformer_activation = "gelu"
transformer_pool = "cls"      # or "mean"
transformer_concat_flat = false
transformer_use_entity_id_embeddings = true
```

Why embeddings matter:

- token order should not be the only identity signal,
- line 17 and line 18 need distinguishable learned IDs,
- a generator attached to substation 4 should carry that location,
- local actor sequences can be padded/reordered without losing identity.

### Attention Mask

Use `src_key_padding_mask` from `token_mask`:

```text
token_mask == 0 means padding token, ignore it in attention
```

Do not use causal masking. This is an observation encoder, not an autoregressive
decoder.

### Pooling

Start with a learned `CLS` token:

```text
CLS output -> actor/critic head
```

Also implement masked mean pooling for ablation:

```text
mean(non-padding token outputs) -> actor/critic head
```

Why:

- `CLS` gives the model one dedicated summary vector.
- mean pooling is sometimes more stable with small RL data.

## Actor And Critic Integration

Modify `Topology_Task/alg/mappo/agent.py`.

### Actor

Add:

```python
elif self.encoder_type == "transformer":
    if getattr(envs, "token_specs", None) is None:
        raise ValueError("actor_encoder=transformer requires token observations.")
    flat_dim = int(np.prod(envs.observation_space[agent_id].shape))
    self.encoder = TokenAndFlatEncoder(
        envs.token_specs[agent_id],
        flat_dim=flat_dim,
        args=args,
        use_flat=getattr(args, "transformer_concat_flat", False),
    )
    actor_input_dim = self.encoder.out_dim
```

Then update `_encode`:

```python
if self.encoder_type == "transformer":
    return self.encoder(x, token_key="tokens")
```

Everything after the encoder can remain unchanged in version 1:

```text
flat categorical actor
intervention-gated actor
do-nothing initialization
training/eval methods
```

### Critic

Add the same path but use `state_tokens`:

```python
if self.encoder_type == "transformer":
    x = self.encoder(x, token_key="state_tokens")
```

The critic should keep a centralized full-grid token sequence when
`decentralized = true`.

### Joint Observation Utility

Modify `Topology_Task/common/utils.py`:

```python
def get_state_token_obs(obs_dict):
    ...

def strip_state_tokens(obs):
    ...

def get_joint_obs(obs_dict, encoder, decentralized=True):
    if encoder == "transformer":
        return {"flat": flat, "state_tokens": get_state_token_obs(obs_dict)}
```

Why this matters:

- Rollout and PPO code already call `get_joint_obs`.
- Keeping this abstraction avoids transformer-specific logic in the training
  loop.

## Environment Integration

Modify `Topology_Task/env/config.py`:

```text
actor_encoder choices: mlp, gnn, transformer
critic_encoder choices: mlp, gnn, transformer
```

Modify `Topology_Task/env/utils.py`:

```python
self.use_graph_obs = any_gnn_enabled(args)
self.use_token_obs = any_transformer_enabled(args)
```

Build token specs:

```python
self.token_builder = None
self.token_specs = None
if self.use_token_obs:
    self.token_builder = GridTokenBuilder(...)
    self.token_specs = self.token_builder.specs
```

Format observations:

```python
if self.use_token_obs:
    tokens = self.token_builder.build(self._obs)
    return {
        agent_id: {
            "flat": gym_obs[agent_id],
            "tokens": tokens[agent_id],
            "state_tokens": tokens["state"],
        }
        for agent_id in self.g2op_ma_env.agents
    }
```

Do not remove the graph path. If both GNN and transformer are enabled, raise a
clear error for version 1:

```text
actor_encoder/critic_encoder cannot mix gnn and transformer until mixed
structured observations are explicitly supported.
```

Why:

- It prevents accidentally returning graph and token dicts with unclear memory
  cost.
- Mixed GNN actor + transformer critic can be added later if needed.

## Normalization Strategy

The current flat observations use running normalization when `norm_obs = true`.
Tokens also need normalization, but not all token values should be treated the
same way.

Phase 1 approach:

```text
Use the same normalized flat observation in parallel.
Build token raw values from Grid2Op observation arrays.
Apply simple static scaling where obvious.
```

Examples:

```text
rho: leave around 0-2
line_status: 0/1
topology bus ids: map bus 1/2 to -1/+1 or 0/1
cooldowns: divide by max configured cooldown if known, else clip/divide by 10
maintenance time: clip and divide by a horizon such as 288 or 2016
power values: divide by thermal/load/generator scale if available, else rely on LayerNorm initially
```

Phase 2 approach:

```text
Add running token normalization with masks.
Maintain stats per token feature column, not per token position.
Optionally maintain stats per token type.
```

Why:

- Transformers are sensitive to scale.
- WCCI contains power, booleans, cooldown counters, and topology IDs in one
  sequence.
- Bad scaling can make attention focus on the largest numerical feature rather
  than the most useful one.

## Connectivity And Topology Encoding

Do not feed one dense connectivity matrix as the only topology representation
for the final design. It is acceptable as a broad-group diagnostic token, but
the entity tokenizer should encode connectivity through entity metadata:

```text
line token has origin/extremity substation ids
line token has origin/extremity current bus ids
substation token has attached asset summaries
busbar token optionally represents busbar-local aggregation
```

Phase 2 optional improvement:

```text
attention bias based on physical adjacency
```

Example relation types:

```text
same substation
line connects substations
asset attached to substation
line endpoint attached to busbar
same agent domain
neighbor domain
```

Why:

- A vanilla transformer can learn connectivity, but it must rediscover graph
  structure from IDs and features.
- A relation bias gives the model a useful prior without forcing pure GNN
  message passing.

## Action Head Roadmap

### Version 1: Pooled Transformer Embedding

Keep:

```text
pooled transformer embedding -> MLP head -> logits over reduced local actions
```

Why:

- Minimal integration risk.
- Compatible with current reduced action-space files.
- Direct comparison against MLP and GNN.

### Version 2: Action-Conditioned Scoring

Later, score each action candidate with token context:

```text
action metadata embedding + pooled state embedding + relevant entity token
  -> scalar logit for that action
```

Needed metadata:

```text
which substation the action changes
which line status change if any
which assets/busbar config the action touches
whether action is no-op
```

Why:

- WCCI reduced action space still has many local topology actions.
- A flat action head does not know which action belongs to which token except
  by memorized action index.
- Entity-conditioned scoring should generalize better across actions touching
  similar substations.

### Version 3: Hierarchical Local Topology Head

If action-conditioned scoring works, consider:

```text
gate: do nothing vs intervene
substation selector
local topology configuration selector
```

This matches the operational decomposition but requires more action-space
metadata and careful log-prob reconstruction for PPO.

## Implementation Phases

### Phase 0: Inspect And Freeze Schemas

Tasks:

1. Add a small inspection script:

   ```text
   Topology_Task/tools/inspect_wcci_token_schema.py
   ```

2. Print WCCI 36 values:

   ```text
   n_sub
   n_line
   n_gen
   n_load
   n_storage
   topology vector length
   per-agent controlled substations
   per-agent local lines
   current reduced action sizes
   available observation attributes
   ```

3. Save one example local and state token schema as JSON.

Why:

- Token tensors must be fixed-shape for rollout storage.
- WCCI local agents have different domain sizes.
- We need to know maximum local token counts before choosing padding sizes.

Deliverables:

```text
docs or logs showing token counts per tokenizer
one sample token spec JSON
```

### Phase 1: Add Config Plumbing

Tasks:

1. Extend encoder choices to include `transformer`.
2. Add tokenizer args:

   ```text
   tokenizer_type = "group" | "entity" | "hybrid"
   tokenizer_include_neighbors = true/false
   tokenizer_include_maintenance = true/false
   tokenizer_max_feature_dim = 64
   tokenizer_entity_order = "type_then_id"
   ```

3. Add transformer args:

   ```text
   transformer_d_model
   transformer_n_heads
   transformer_layers
   transformer_ff_dim
   transformer_dropout
   transformer_activation
   transformer_pool
   transformer_concat_flat
   transformer_use_entity_id_embeddings
   transformer_use_substation_embeddings
   ```

4. Update config TOML parsing through existing `run_from_config.py` flow.

Why:

- This makes every experiment reproducible from TOML.
- It avoids hard-coded architecture choices inside modules.

Validation:

```text
python Topology_Task/run_from_config.py --help-like config parse path if available
unit smoke test that args accept transformer values
```

### Phase 2: Implement Broad Feature-Group Tokenizer

Tasks:

1. Create `GridTokenBuilder` with `tokenizer_type="group"`.
2. Build local actor group tokens from each agent observation domain.
3. Build centralized critic group tokens from the full grid.
4. Return fixed arrays with masks and IDs.
5. Add a shape/unit test that runs `env.reset()` and checks:

   ```text
   features shape
   token_mask shape
   no NaN/Inf
   state token count is fixed
   per-agent token counts are padded to spec
   ```

Why:

- This is the fastest end-to-end proof that transformer observations can pass
  through vectorized MAPPO.
- It gives a cheap first comparison against the MLP.

Expected result:

```text
actor_encoder="transformer", tokenizer_type="group" trains without shape errors
critic_encoder="transformer", tokenizer_type="group" trains without shape errors
```

### Phase 3: Implement Token Transformer Encoder

Tasks:

1. Add `TokenTransformerEncoder`.
2. Add `TokenAndFlatEncoder`.
3. Use `nn.TransformerEncoderLayer(batch_first=True, norm_first=True)` if the
   installed PyTorch version supports it.
4. Support batched and unbatched observations, mirroring `GraphEncoder`.
5. Add masked `CLS` and masked mean pooling.
6. Add parameter-count logging using the existing tools.

Why:

- A reusable encoder lets group/entity/hybrid tokenizers share one model path.
- Masking and unbatched support avoid rollout/eval surprises.

Validation:

```text
synthetic token batch forward pass
env reset -> actor.get_action(obs) smoke test
critic.get_value(joint_obs) smoke test
```

### Phase 4: Wire Actor, Critic, And Rollout Utilities

Tasks:

1. Add `transformer` branches in `Actor` and `Critic`.
2. Add `get_state_token_obs` and `strip_state_tokens` in `common/utils.py`.
3. Extend `get_joint_obs` for transformer critics.
4. Ensure rollout buffers handle nested token dicts through existing
   `zeros_like_with_leading`, `set_nested_at_step`, and tensor casting helpers.
5. Check checkpoint save/load works without special cases.

Why:

- The training loop already supports nested dict observations for GNNs.
- Reusing that machinery keeps the implementation compact.

Validation:

```text
short CPU run on bus14 with transformer group tokens
short CPU run on bus36_wcci with 1-2 envs and tiny timesteps
checkpoint save/load smoke test
```

### Phase 5: Implement Entity Tokenizer

Tasks:

1. Reuse Grid2Op metadata already used by `GridGraphBuilder`:

   ```text
   line_or_to_subid
   line_ex_to_subid
   gen_to_subid
   load_to_subid
   line_or_pos_topo_vect
   line_ex_pos_topo_vect
   gen_pos_topo_vect
   load_pos_topo_vect
   topo_vect
   ```

2. Build entity specs for:

   ```text
   state
   agent_0
   agent_1
   agent_2
   agent_3
   ```

3. Include only local entities for actor tokens.
4. Include all entities for critic tokens.
5. Add token type IDs:

   ```text
   CLS = 0
   generator = 1
   load = 2
   line = 3
   substation = 4
   busbar = 5
   summary = 6
   padding = 7
   ```

6. Add stable entity IDs:

   ```text
   generator id in [0, n_gen)
   load id in [0, n_load)
   line id in [0, n_line)
   substation id in [0, n_sub)
   busbar id in [0, n_sub * n_busbar)
   ```

Why:

- Entity tokens let the model reason about specific overloaded lines and local
  substations.
- Stable IDs make attention outputs interpretable and action-conditioned heads
  possible later.

Validation:

```text
token count per agent is stable across reset/step
line endpoint IDs match Grid2Op metadata
topo bus IDs change after topology actions
padding tokens remain masked
```

### Phase 6: WCCI Configs And Ablations

Create a new config folder:

```text
Topology_Task/configs/wcci_tokenizer/
```

Initial configs:

```text
wcci_group_transformer_128x3_s0.toml
wcci_group_transformer_128x3_s1.toml
wcci_group_transformer_128x3_s2.toml
wcci_entity_transformer_128x3_s0.toml
wcci_entity_transformer_128x3_s1.toml
wcci_entity_transformer_128x3_s2.toml
```

Start conservative:

```toml
actor_encoder = "transformer"
critic_encoder = "transformer"
tokenizer_type = "group" # then "entity"
transformer_d_model = 128
transformer_n_heads = 4
transformer_layers = 3
transformer_ff_dim = 512
transformer_dropout = 0.05
transformer_pool = "cls"
transformer_concat_flat = false
actor_layers = [256]
critic_layers = [256]
actor_lr = 0.00005
critic_lr = 0.00005
max_grad_norm = 0.5
```

Reason for smaller heads:

- the transformer already provides depth and mixing,
- `[256, 256, 256]` after a transformer may be unnecessarily heavy,
- PPO with a large model can become unstable before it becomes better.

Also test:

```toml
transformer_concat_flat = true
```

Why:

- If pure tokens underperform, concat-flat tells us whether the tokenizer lost
  important information.

### Phase 7: Diagnostics And Logging

Add logs:

```text
model/actor_params
model/critic_params
model/token_count_actor_agent_*
model/token_count_state
model/transformer_d_model
model/transformer_layers
train/actor_grad_norm
train/critic_grad_norm
train/approx_kl
train/entropy
train/illegal_action_rate_agent_*
train/frac_any_non_idle
eval/mean_survival
eval/mean_reward
```

Optional transformer-specific logs:

```text
train/token_embedding_norm
train/cls_embedding_norm
train/attention_entropy_layer_*
train/padding_token_attention_mass
```

Why:

- Transformers can fail silently by attending to padding, saturating logits, or
  producing unstable gradients.
- Survival alone is too delayed to debug representation problems.

### Phase 8: Pretraining And Sample Efficiency

If RL from scratch is unstable, use the teacher/student datasets:

```text
Topology_Task/teacher_student/
```

Pretraining options:

1. Behavior cloning from greedy/reduced-action labels.
2. Auxiliary prediction:

   ```text
   predict max rho next step
   predict overloaded line mask
   predict whether do-nothing survives
   predict best local action from outcome dataset
   ```

3. Initialize actor transformer from BC, then run MAPPO.

Why:

- Transformers are data-hungry.
- WCCI topology control has sparse useful interventions.
- A supervised warm start can teach the token encoder grid semantics before PPO
  starts pushing noisy policy gradients through it.

### Phase 9: Better Action Heads

Once entity transformer is stable, add action metadata.

Tasks:

1. Extend reduced action-space metadata with:

   ```text
   action id
   touched substation id
   touched line id if any
   action kind: no-op, bus-change, line-status-change
   asset ids changed
   ```

2. Add an action scorer:

   ```text
   pooled embedding
   relevant substation token
   relevant line token if any
   action embedding
   -> scalar logit
   ```

3. Reconstruct categorical logits for PPO.

Why:

- The current flat head treats action 31 and action 32 as arbitrary categories.
- Action-conditioned scoring lets the policy use local token embeddings to score
  actions that touch those local entities.

## Experiment Matrix

Run in this order:

```text
1. WCCI MLP baseline already available
2. WCCI group transformer, small model
3. WCCI group transformer + concat flat
4. WCCI entity transformer, small model
5. WCCI entity transformer + concat flat
6. WCCI hybrid transformer
7. Entity transformer + intervention gate
8. Entity transformer + action-conditioned head
9. Entity transformer + BC warm start
```

Primary metrics:

```text
test survival
test reward
illegal action rate
non-idle action rate
multi-agent simultaneous action rate
training wall-clock time
parameter count
seed variance
```

Interpretation rules:

```text
group > mlp:
  attention over feature groups helps.

entity > group:
  object-level structure matters.

entity + concat flat > entity:
  tokenizer is missing useful flat information.

entity unstable but BC warm start stable:
  representation is useful but RL from scratch is sample-inefficient.

transformer survival equal to MLP but fewer bad interventions:
  still scientifically useful because the policy is more operator-like.
```

## Risks And Mitigations

### Risk: Transformer Is Too Slow

Mitigation:

```text
start with d_model=128, layers=2-3, heads=4
keep token count small in group tokenizer
profile rollout vs update time separately
try actor transformer with MLP critic or MLP actor with transformer critic only after core support is stable
```

### Risk: PPO Becomes Unstable

Mitigation:

```text
lower actor_lr/critic_lr to 5e-5
use max_grad_norm=0.5
use dropout 0.0-0.05
reduce update_epochs if KL spikes
monitor approx_kl and entropy
keep init_do_nothing_prob/action0 priors available
```

### Risk: Tokenizer Loses Information

Mitigation:

```text
compare transformer_concat_flat=false vs true
add a debug script that reconstructs key raw arrays from tokens
unit test no NaN/Inf and expected counts
start with broad group tokens before entity tokens
```

### Risk: Model Memorizes Entity IDs

Mitigation:

```text
run with and without entity ID embeddings
use substation/line relation features, not only IDs
evaluate across held-out chronic split
avoid oversized d_model early
```

### Risk: Connectivity Is Weakly Represented

Mitigation:

```text
include endpoint substation and busbar embeddings on line tokens
include substation tokens
add relation attention bias after baseline entity model works
compare against existing GINE/GAT graph baselines
```

## Recommended First Implementation Slice

Implement this exact slice first:

```text
1. Add config choices and args for transformer/tokenizer.
2. Add group tokenizer only.
3. Add TokenTransformerEncoder with CLS pooling.
4. Wire Actor/Critic/get_joint_obs/env formatting.
5. Run a tiny bus14 smoke test.
6. Run a tiny bus36_wcci smoke test.
7. Add WCCI group transformer configs.
8. Only then implement entity tokenizer.
```

This sequence gives an early end-to-end signal while avoiding the hardest
entity-schema work until the transformer plumbing is proven.

## Definition Of Done

The tokenizer/transformer work is ready for real experiments when:

```text
MLP and GNN configs still run unchanged.
actor_encoder="transformer" and critic_encoder="transformer" parse from TOML.
Group tokenizer produces fixed-shape local and state token observations.
Entity tokenizer produces fixed-shape local and state token observations.
Actor and critic forward passes work on CPU for reset and rollout batches.
Short MAPPO training runs complete for bus14 and bus36_wcci.
Checkpoint save/load works.
WCCI configs exist for group and entity transformer ablations.
Parameter count and token counts are logged.
No token tensors contain NaN/Inf.
Padding tokens are masked in transformer attention.
```

## Final Recommendation

Build the broad-group tokenizer first because it is the fastest way to test the
transformer path. Then build the entity tokenizer because it is the version most
likely to improve WCCI 36 decisions. Treat the entity transformer as the main
scientific model, and treat broad-group tokens as the low-risk bridge and
ablation.

