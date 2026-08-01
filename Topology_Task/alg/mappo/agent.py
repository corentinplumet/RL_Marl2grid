from typing import Sequence

from torch.distributions import Categorical, Normal

from alg.mappo.action_pooling import (
    CandidateActionAttentionPool,
    CandidateActionMeanPool,
    CandidateAttentionOutput,
)
from common.imports import *
from common.action_metadata import ActionGraphMetadata
from common.gnn import GraphAndFlatEncoder, GraphEncoder
from common.token_transformer import TokenAndFlatEncoder
from common.utils import Linear, get_flat_obs, th_act_fns


def build_mlp_head(
    input_dim: int,
    hidden_layers: List[int],
    output_dim: int,
    act_fn_name: str,
) -> nn.Sequential:
    layers = []
    act_fn = th_act_fns[act_fn_name]
    prev_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.extend([Linear(prev_dim, hidden_dim, act_fn_name), act_fn])
        prev_dim = hidden_dim
    layers.append(Linear(prev_dim, output_dim, "linear"))
    return nn.Sequential(*layers)


class CandidateActionScorer(nn.Module):
    """Score every discrete action from global and touched-node context."""

    def __init__(
        self,
        graph_dim: int,
        node_dim: int,
        hidden_layers: List[int],
        act_fn_name: str,
        metadata: ActionGraphMetadata,
        pool_mode: str = "typed_mean",
        use_action_features: bool = True,
        use_do_nothing_head: bool = True,
        attention_scope: str = "affected",
        attention_heads: int = 1,
        attention_dim: int = 0,
        attention_temperature: float = 1.0,
        attention_query: str = "global_action_features",
        attention_normalizer: str = "softmax",
    ) -> None:
        super().__init__()
        metadata.validate()
        self.n_actions = metadata.n_actions
        self.node_dim = int(node_dim)
        self.pool_mode = str(pool_mode).lower()
        if self.pool_mode not in {"mean", "typed_mean", "typed_attention"}:
            raise ValueError(
                "candidate_action_pool must be 'mean', 'typed_mean', or "
                f"'typed_attention', got {pool_mode!r}."
            )
        self.use_action_features = bool(use_action_features)
        self.use_do_nothing_head = bool(use_do_nothing_head)

        for name in (
            "action_features",
            "busbar_indices",
            "busbar_mask",
            "line_indices",
            "line_mask",
            "load_indices",
            "load_mask",
            "generator_indices",
            "generator_mask",
            "is_do_nothing",
            "original_action_ids",
        ):
            self.register_buffer(name, getattr(metadata, name).clone())

        if self.pool_mode == "typed_attention":
            self.pool = CandidateActionAttentionPool(
                graph_dim=int(graph_dim),
                node_dim=self.node_dim,
                n_actions=self.n_actions,
                action_feature_dim=metadata.action_feature_dim,
                scope=attention_scope,
                heads=attention_heads,
                attention_dim=attention_dim,
                temperature=attention_temperature,
                query_mode=attention_query,
                normalizer=attention_normalizer,
            )
        else:
            self.pool = CandidateActionMeanPool(self.pool_mode)

        local_dim = self.node_dim * (
            4 if self.pool_mode in {"typed_mean", "typed_attention"} else 1
        )
        scorer_input_dim = int(graph_dim) + local_dim
        if self.use_action_features:
            scorer_input_dim += metadata.action_feature_dim
        self.scorer = build_mlp_head(
            scorer_input_dim,
            hidden_layers,
            1,
            act_fn_name,
        )
        self.do_nothing_actor = (
            build_mlp_head(
                int(graph_dim),
                hidden_layers,
                1,
                act_fn_name,
            )
            if self.use_do_nothing_head
            else None
        )
        self.do_nothing_logit_bias = nn.Parameter(th.zeros(()))

    def _typed_metadata(self):
        return (
            ("busbar", self.busbar_indices, self.busbar_mask),
            ("line", self.line_indices, self.line_mask),
            ("load", self.load_indices, self.load_mask),
            ("generator", self.generator_indices, self.generator_mask),
        )

    def _local_context(
        self,
        graph_embedding: th.Tensor,
        node_embeddings: th.Tensor,
        *,
        return_attention: bool = False,
    ) -> CandidateAttentionOutput:
        typed_metadata = self._typed_metadata()
        if self.pool_mode == "typed_attention":
            return self.pool(
                graph_embedding,
                node_embeddings,
                self.action_features,
                typed_metadata,
                return_weights=return_attention,
            )
        return self.pool(node_embeddings, typed_metadata)

    def forward(
        self,
        graph_embedding: th.Tensor,
        node_embeddings: th.Tensor,
        *,
        return_attention: bool = False,
    ):
        unbatched = graph_embedding.dim() == 1
        if unbatched:
            graph_embedding = graph_embedding.unsqueeze(0)
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)
        if graph_embedding.shape[0] != node_embeddings.shape[0]:
            raise ValueError(
                "Graph and node embeddings must have the same batch size."
            )

        attention_output = self._local_context(
            graph_embedding,
            node_embeddings,
            return_attention=return_attention,
        )
        local_context = attention_output.context
        batch_size = int(graph_embedding.shape[0])
        global_context = graph_embedding.unsqueeze(1).expand(
            batch_size, self.n_actions, -1
        )
        scorer_inputs = [global_context, local_context]
        if self.use_action_features:
            scorer_inputs.append(
                self.action_features.unsqueeze(0).expand(batch_size, -1, -1)
            )
        logits = self.scorer(th.cat(scorer_inputs, dim=-1)).squeeze(-1)

        if self.do_nothing_actor is not None:
            do_nothing_logit = self.do_nothing_actor(graph_embedding).squeeze(-1)
            logits = th.where(
                self.is_do_nothing.unsqueeze(0),
                do_nothing_logit.unsqueeze(-1),
                logits,
            )
        logits = logits + (
            self.is_do_nothing.to(dtype=logits.dtype).unsqueeze(0)
            * self.do_nothing_logit_bias
        )
        if unbatched:
            logits = logits.squeeze(0)
            attention_output.context = attention_output.context.squeeze(0)
            if attention_output.weights is not None:
                attention_output.weights = {
                    name: weights.squeeze(0)
                    for name, weights in attention_output.weights.items()
                }
        if return_attention:
            return logits, attention_output
        return logits

    def init_do_nothing_prior(self, init_p0: float) -> None:
        prior_logit = float(
            np.log(init_p0 * (self.n_actions - 1) / (1.0 - init_p0))
        )
        with th.no_grad():
            shared_output = self.scorer[-1]
            shared_output.weight.zero_()
            shared_output.bias.zero_()
            self.do_nothing_logit_bias.zero_()
            if self.do_nothing_actor is None:
                self.do_nothing_logit_bias.fill_(prior_logit)
            else:
                do_nothing_output = self.do_nothing_actor[-1]
                do_nothing_output.weight.zero_()
                do_nothing_output.bias.fill_(prior_logit)


class Actor(nn.Module):
    def __init__(
        self,
        id: int,
        envs: gym.Env,
        args: Dict[str, Any],
        continuous_actions: bool,
        shared_graph_encoder: Optional[GraphEncoder] = None,
    ):
        super().__init__()

        agent_id = f"agent_{id}"
        self.encoder_type = getattr(args, "actor_encoder", "mlp")
        self.actor_action_head = str(
            getattr(args, "actor_action_head", "mlp")
        ).lower()
        if self.actor_action_head not in {"mlp", "candidate_pool"}:
            raise ValueError(
                "actor_action_head must be 'mlp' or 'candidate_pool', got "
                f"{self.actor_action_head!r}."
            )
        self.intervention_gate = bool(getattr(args, "intervention_gate", False))
        self.intervention_gate_eval_mode = str(
            getattr(args, "intervention_gate_eval_mode", "final_action_map")
        )
        self.intervention_gate_entropy_mode = str(
            getattr(args, "intervention_gate_entropy_mode", "coupled")
        )
        if self.intervention_gate_eval_mode not in {
            "final_action_map",
            "hierarchical_greedy",
        }:
            raise ValueError(
                "intervention_gate_eval_mode must be 'final_action_map' or "
                f"'hierarchical_greedy', got {self.intervention_gate_eval_mode!r}."
            )
        if self.intervention_gate_entropy_mode not in {"coupled", "separate"}:
            raise ValueError(
                "intervention_gate_entropy_mode must be 'coupled' or "
                f"'separate', got {self.intervention_gate_entropy_mode!r}."
            )
        self.intervention_gate_entropy_mult = float(
            getattr(args, "intervention_gate_entropy_mult", 1.0)
        )
        self.intervention_nonidle_entropy_mult = float(
            getattr(args, "intervention_nonidle_entropy_mult", 1.0)
        )

        if self.encoder_type == "mlp":
            self.encoder = None
            actor_input_dim = int(np.prod(envs.observation_space[agent_id].shape))
        elif self.encoder_type == "gnn":
            if getattr(envs, "graph_specs", None) is None:
                raise ValueError(
                    "actor_encoder=gnn requires thesis-style graph observations."
                )
            flat_dim = int(np.prod(envs.observation_space[agent_id].shape))
            self.encoder = GraphAndFlatEncoder(
                envs.graph_specs[agent_id],
                flat_dim=flat_dim,
                args=args,
                use_flat=getattr(args, "gnn_concat_flat", False),
                graph_encoder=shared_graph_encoder,
            )
            actor_input_dim = self.encoder.out_dim
        elif self.encoder_type == "transformer":
            if getattr(envs, "token_specs", None) is None:
                raise ValueError(
                    "actor_encoder=transformer requires tokenizer observations."
                )
            flat_dim = int(np.prod(envs.observation_space[agent_id].shape))
            self.encoder = TokenAndFlatEncoder(
                envs.token_specs[agent_id],
                flat_dim=flat_dim,
                args=args,
                use_flat=getattr(args, "transformer_concat_flat", False),
            )
            actor_input_dim = self.encoder.out_dim
        else:
            raise ValueError(
                f"Unsupported actor encoder '{self.encoder_type}'. "
                "Use 'mlp', 'gnn', or 'transformer'."
            )

        actor_layers = args.actor_layers
        if continuous_actions:
            raise ("Redispatching actions are not yet implemented")
        else:
            n_actions = int(envs.action_space[agent_id].n)
            self.n_actions = n_actions
            if self.actor_action_head == "candidate_pool":
                if self.encoder_type != "gnn":
                    raise ValueError(
                        "actor_action_head=candidate_pool requires "
                        "actor_encoder=gnn."
                    )
                if self.intervention_gate:
                    raise ValueError(
                        "actor_action_head=candidate_pool does not yet support "
                        "intervention_gate=true."
                    )
                graph_spec = envs.graph_specs[agent_id]
                metadata = graph_spec.get("action_graph_metadata")
                if not isinstance(metadata, ActionGraphMetadata):
                    raise ValueError(
                        "Candidate-action metadata is missing from the agent "
                        "graph spec. Recreate the environment with "
                        "actor_action_head=candidate_pool."
                    )
                if metadata.n_actions != n_actions:
                    raise ValueError(
                        f"Candidate metadata has {metadata.n_actions} actions, "
                        f"but {agent_id} exposes {n_actions}."
                    )
                self.actor = CandidateActionScorer(
                    graph_dim=actor_input_dim,
                    node_dim=self.encoder.node_out_dim,
                    hidden_layers=actor_layers,
                    act_fn_name=args.actor_act_fn,
                    metadata=metadata,
                    pool_mode=getattr(
                        args, "candidate_action_pool", "typed_mean"
                    ),
                    use_action_features=getattr(
                        args, "candidate_action_use_features", True
                    ),
                    use_do_nothing_head=getattr(
                        args, "candidate_action_do_nothing_head", True
                    ),
                    attention_scope=getattr(
                        args, "candidate_action_attention_scope", "affected"
                    ),
                    attention_heads=getattr(
                        args, "candidate_action_attention_heads", 1
                    ),
                    attention_dim=getattr(
                        args, "candidate_action_attention_dim", 0
                    ),
                    attention_temperature=getattr(
                        args, "candidate_action_attention_temperature", 1.0
                    ),
                    attention_query=getattr(
                        args,
                        "candidate_action_attention_query",
                        "global_action_features",
                    ),
                    attention_normalizer=getattr(
                        args, "candidate_action_attention_normalizer", "softmax"
                    ),
                )
                self.get_action = self.get_discrete_action
                self.get_eval_action = self.get_eval_discrete_action
            elif self.intervention_gate:
                if n_actions <= 1:
                    raise ValueError(
                        "intervention_gate=True requires at least one non-idle action."
                    )
                self.gate_actor = build_mlp_head(
                    actor_input_dim, actor_layers, 2, args.actor_act_fn
                )
                self.nonidle_actor = build_mlp_head(
                    actor_input_dim, actor_layers, n_actions - 1, args.actor_act_fn
                )
                self.get_action = self.get_intervention_gated_action
                self.get_eval_action = self.get_eval_intervention_gated_action
            else:
                # Logit layer: Xavier with linear gain (=1.0). Using the ReLU gain
                # here (the Linear utility's default) amplifies logits and produces
                # an essentially-deterministic random initial policy on bus14.
                self.actor = build_mlp_head(
                    actor_input_dim, actor_layers, n_actions, args.actor_act_fn
                )
                self.get_action = self.get_discrete_action
                self.get_eval_action = self.get_eval_discrete_action

            # Optional: bias initial policy toward action 0 (do-nothing).
            # Random topology changes on bus14 crash the grid at step 0, so a
            # do-nothing prior dramatically accelerates early learning.
            init_p0 = getattr(args, "init_do_nothing_prob", 0.0)
            if init_p0 > 0.0:
                assert 0.0 < init_p0 < 1.0, (
                    f"init_do_nothing_prob must be in (0, 1), got {init_p0}"
                )
                self._init_do_nothing_prior(init_p0)

    def _init_do_nothing_prior(self, init_p0: float) -> None:
        if self.actor_action_head == "candidate_pool":
            self.actor.init_do_nothing_prior(init_p0)
            return
        if self.intervention_gate:
            gate_layer = self.gate_actor[-1]
            nonidle_layer = self.nonidle_actor[-1]
            with th.no_grad():
                gate_layer.weight.zero_()
                gate_layer.bias[0] = float(np.log(init_p0))
                gate_layer.bias[1] = float(np.log(1.0 - init_p0))
                nonidle_layer.weight.zero_()
                nonidle_layer.bias.zero_()
            return

        out_layer = self.actor[-1]
        with th.no_grad():
            out_layer.weight.zero_()
            out_layer.bias.zero_()
            out_layer.bias[0] = float(
                np.log(init_p0 * (self.n_actions - 1) / (1.0 - init_p0))
            )

    def _encode(self, x: th.Tensor) -> th.Tensor:
        if self.encoder_type == "gnn":
            return self.encoder(x, graph_key="graph")
        if self.encoder_type == "transformer":
            return self.encoder(x, token_key="tokens")
        if self.encoder is not None:
            return self.encoder(x)
        return get_flat_obs(x)

    def _actor_logits(self, x: th.Tensor) -> th.Tensor:
        if self.actor_action_head == "candidate_pool":
            graph_embedding, node_embeddings = self.encoder.forward_with_nodes(
                x, graph_key="graph"
            )
            return self.actor(graph_embedding, node_embeddings)
        return self.actor(self._encode(x))

    def get_candidate_attention(
        self,
        x: th.Tensor,
        action_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Return candidate logits and action-specific node attention maps."""
        if self.actor_action_head != "candidate_pool":
            raise ValueError(
                "Candidate attention is available only for candidate_pool actors."
            )
        if self.actor.pool_mode != "typed_attention":
            raise ValueError(
                "Candidate attention requires candidate_action_pool=typed_attention."
            )
        graph_embedding, node_embeddings = self.encoder.forward_with_nodes(
            x,
            graph_key="graph",
        )
        logits, output = self.actor(
            graph_embedding,
            node_embeddings,
            return_attention=True,
        )
        result = {
            "logits": logits,
            "weights": output.weights,
            "node_indices": output.node_indices,
            "eligible_masks": output.eligible_masks,
        }
        if action_ids is None:
            return result

        selected = th.as_tensor(action_ids, dtype=th.long, device=logits.device)
        if selected.dim() != 1:
            raise ValueError("action_ids must be a one-dimensional sequence.")
        if selected.numel() and bool(
            th.any((selected < 0) | (selected >= self.n_actions))
        ):
            raise ValueError(
                f"action_ids must lie in [0, {self.n_actions - 1}]."
            )
        action_dim = 0 if logits.dim() == 1 else 1
        result["logits"] = logits.index_select(action_dim, selected)
        weight_action_dim = 0 if logits.dim() == 1 else 1
        result["weights"] = {
            name: weights.index_select(weight_action_dim, selected)
            for name, weights in output.weights.items()
        }
        result["node_indices"] = {
            name: indices.index_select(0, selected.to(indices.device))
            for name, indices in output.node_indices.items()
        }
        result["eligible_masks"] = {
            name: mask.index_select(0, selected.to(mask.device))
            for name, mask in output.eligible_masks.items()
        }
        return result

    def get_discrete_action(
        self,
        x: th.Tensor,
        action: th.Tensor = None,
        action0_bonus: float = 0.0,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Sample discrete actions and compute log probabilities and entropy.

        Args:
            x: Input observations.
            action: Specific action to take. Defaults to None.
            action0_bonus: Optional logit bonus added to action 0.

        Returns:
            A tuple containing tensors for the sampled discrete actions, the log probability of the sampled actions, and the entropy of the action distribution.
        """
        logits = self._actor_logits(x)
        if action0_bonus != 0.0:
            logits = logits.clone()
            logits[..., 0] = logits[..., 0] + action0_bonus
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy()

    def get_eval_discrete_action(
        self, x: th.Tensor, deterministic: bool = True
    ) -> th.Tensor:
        """Evaluate discrete actions greedily or by sampling.

        Args:
            x: Input observations.
            deterministic: If True, return the highest-logit action; otherwise sample.

        Returns:
            A tensor with selected discrete actions for evaluation.
        """
        if not deterministic:
            return self.get_discrete_action(x)[0]
        logits = self._actor_logits(x)
        return th.argmax(logits, dim=-1)

    def _gated_distributions(
        self,
        x: th.Tensor,
        action0_bonus: float = 0.0,
    ) -> Tuple[Categorical, Categorical]:
        encoded = self._encode(x)
        gate_logits = self.gate_actor(encoded)
        if action0_bonus != 0.0:
            gate_logits = gate_logits.clone()
            gate_logits[..., 0] = gate_logits[..., 0] + action0_bonus
        nonidle_logits = self.nonidle_actor(encoded)
        return Categorical(logits=gate_logits), Categorical(logits=nonidle_logits)

    def _gated_action_log_prob(
        self,
        gate_dist: Categorical,
        nonidle_dist: Categorical,
        action: th.Tensor,
    ) -> th.Tensor:
        action = action.long()
        if th.any((action < 0) | (action >= self.n_actions)):
            raise ValueError(
                f"Gated actor received action outside [0, {self.n_actions - 1}]."
            )

        is_noop = action == 0
        gate_target = th.where(is_noop, th.zeros_like(action), th.ones_like(action))
        gate_logprob = gate_dist.log_prob(gate_target)
        nonidle_target = (action - 1).clamp_min(0)
        nonidle_logprob = nonidle_dist.log_prob(nonidle_target)
        return th.where(is_noop, gate_logprob, gate_logprob + nonidle_logprob)

    def get_intervention_gated_action(
        self,
        x: th.Tensor,
        action: th.Tensor = None,
        action0_bonus: float = 0.0,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Sample or score actions from a do-nothing/intervene hierarchy."""
        gate_dist, nonidle_dist = self._gated_distributions(
            x, action0_bonus=action0_bonus
        )
        if action is None:
            gate = gate_dist.sample()
            nonidle_action = nonidle_dist.sample() + 1
            action = th.where(
                gate == 0,
                th.zeros_like(nonidle_action),
                nonidle_action,
            )
        else:
            action = action.long()

        logprob = self._gated_action_log_prob(gate_dist, nonidle_dist, action)
        gate_entropy = gate_dist.entropy()
        nonidle_entropy = nonidle_dist.entropy()
        if self.intervention_gate_entropy_mode == "separate":
            entropy = (
                self.intervention_gate_entropy_mult * gate_entropy
                + self.intervention_nonidle_entropy_mult * nonidle_entropy
            )
        else:
            entropy = gate_entropy + gate_dist.probs[..., 1] * nonidle_entropy
        return action, logprob, entropy

    def get_intervention_gate_diagnostics(
        self,
        x: th.Tensor,
        action0_bonus: float = 0.0,
    ) -> Dict[str, th.Tensor]:
        """Return rollout-time diagnostics for the learned intervention gate."""
        if not self.intervention_gate:
            return {}
        gate_dist, nonidle_dist = self._gated_distributions(
            x, action0_bonus=action0_bonus
        )
        return {
            "prob_do_nothing": gate_dist.probs[..., 0],
            "prob_intervene": gate_dist.probs[..., 1],
            "gate_entropy": gate_dist.entropy(),
            "nonidle_action_entropy": nonidle_dist.entropy(),
        }

    def get_eval_intervention_gated_action(
        self, x: th.Tensor, deterministic: bool = True
    ) -> th.Tensor:
        """Evaluate the gated policy with the configured deterministic decoder."""
        if not deterministic:
            return self.get_intervention_gated_action(x)[0]

        gate_dist, nonidle_dist = self._gated_distributions(x)
        if self.intervention_gate_eval_mode == "hierarchical_greedy":
            gate = th.argmax(gate_dist.logits, dim=-1)
            nonidle_action = th.argmax(nonidle_dist.logits, dim=-1) + 1
            return th.where(
                gate == 0,
                th.zeros_like(nonidle_action),
                nonidle_action,
            )

        gate_log_probs = th.log_softmax(gate_dist.logits, dim=-1)
        nonidle_log_probs = th.log_softmax(nonidle_dist.logits, dim=-1)
        final_log_probs = th.cat(
            [
                gate_log_probs[..., :1],
                gate_log_probs[..., 1:2] + nonidle_log_probs,
            ],
            dim=-1,
        )
        return th.argmax(final_log_probs, dim=-1)

    def get_continuous_action(
        self, x: th.Tensor, action: th.Tensor = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        raise ("Redispatching actions are not yet implemented")

    def get_eval_continuous_action(self, x: th.Tensor) -> th.Tensor:
        raise ("Redispatching actions are not yet implemented")


class Critic(nn.Module):
    """Neural network-based agent for policy gradient methods, supporting both discrete and continuous action spaces.

    Attributes:
        critic (nn.Sequential): Critic network for value estimation.
        actor (nn.Sequential): Actor network for action selection.
        logstd (nn.Parameter): Log standard deviation for continuous action spaces.
    """

    def __init__(self, envs: gym.Env, args: Dict[str, Any]):
        """
        Initialize the Critic with specified environment, and arguments.

        Args:
            envs: The environment.
            args: Arguments for configuration.
        """
        super().__init__()
        self.encoder_type = getattr(args, "critic_encoder", "mlp")

        critic_layers = args.critic_layers
        if self.encoder_type == "mlp":
            self.encoder = None
            joint_obs_shape = (
                sum(space.shape[0] for space in envs.observation_space.values())
                if args.decentralized
                else envs.observation_space["agent_0"].shape[-1]
            )
            critic_input_dim = int(np.prod((joint_obs_shape,)))
        elif self.encoder_type == "gnn":
            if getattr(envs, "graph_specs", None) is None:
                raise ValueError(
                    "critic_encoder=gnn requires thesis-style graph observations."
                )
            flat_dim = (
                sum(space.shape[0] for space in envs.observation_space.values())
                if args.decentralized
                else envs.observation_space["agent_0"].shape[-1]
            )
            self.encoder = GraphAndFlatEncoder(
                envs.graph_specs["state"],
                flat_dim=flat_dim,
                args=args,
                use_flat=getattr(args, "gnn_concat_flat", False),
            )
            critic_input_dim = self.encoder.out_dim
        elif self.encoder_type == "transformer":
            if getattr(envs, "token_specs", None) is None:
                raise ValueError(
                    "critic_encoder=transformer requires tokenizer observations."
                )
            flat_dim = (
                sum(space.shape[0] for space in envs.observation_space.values())
                if args.decentralized
                else envs.observation_space["agent_0"].shape[-1]
            )
            self.encoder = TokenAndFlatEncoder(
                envs.token_specs["state"],
                flat_dim=flat_dim,
                args=args,
                use_flat=getattr(args, "transformer_concat_flat", False),
            )
            critic_input_dim = self.encoder.out_dim
        else:
            raise ValueError(
                f"Unsupported critic encoder '{self.encoder_type}'. "
                "Use 'mlp', 'gnn', or 'transformer'."
            )
        self.critic = build_mlp_head(
            critic_input_dim, critic_layers, 1, args.critic_act_fn
        )

    def get_value(self, x: th.Tensor) -> th.Tensor:
        """Compute value estimate (critic output) for given observations.

        Args:
            x: Input observations.

        Returns:
            A tensor containing value estimates.
        """
        if self.encoder_type == "gnn":
            x = self.encoder(x, graph_key="state_graph")
        elif self.encoder_type == "transformer":
            x = self.encoder(x, token_key="state_tokens")
        elif self.encoder is not None:
            x = self.encoder(x)
        else:
            x = get_flat_obs(x)
        return self.critic(x)
