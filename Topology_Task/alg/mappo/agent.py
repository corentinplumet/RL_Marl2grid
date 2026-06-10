from torch.distributions import Categorical, Normal

from common.imports import *
from common.gnn import GraphAndFlatEncoder, GraphEncoder
from common.utils import Linear, get_flat_obs, th_act_fns
from risk_prior.surrogate import graph_pre_risk, load_risk_surrogate


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
        self.agent_index = int(id)
        self.encoder_type = getattr(args, "actor_encoder", "mlp")
        self.risk_prior = None
        self.risk_prior_payload = None
        self.risk_prior_beta = float(getattr(args, "risk_prior_beta", 0.0))
        self.risk_prior_tau = float(getattr(args, "risk_prior_tau", 1.0))
        self.risk_prior_clip = float(getattr(args, "risk_prior_clip", 5.0))
        self.risk_prior_graph_spec = None

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
        else:
            raise ValueError(
                f"Unsupported actor encoder '{self.encoder_type}'. Use 'mlp' or 'gnn'."
            )

        actor_layers = args.actor_layers
        if continuous_actions:
            raise ("Redispatching actions are not yet implemented")
        else:
            n_actions = int(envs.action_space[agent_id].n)
            self.n_actions = n_actions
            # Logit layer: Xavier with linear gain (=1.0). Using the ReLU gain
            # here (the Linear utility's default) amplifies logits and produces
            # an essentially-deterministic random initial policy on bus14.
            self.actor = build_mlp_head(
                actor_input_dim, actor_layers, n_actions, args.actor_act_fn
            )
            out_layer = self.actor[-1]
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
                with th.no_grad():
                    out_layer.weight.zero_()
                    out_layer.bias.zero_()
                    out_layer.bias[0] = float(
                        np.log(init_p0 * (n_actions - 1) / (1.0 - init_p0))
                    )

        risk_prior_checkpoint = str(getattr(args, "risk_prior_checkpoint", "") or "")
        if risk_prior_checkpoint:
            self._load_risk_prior(risk_prior_checkpoint, envs)

    def _encode(self, x: th.Tensor) -> th.Tensor:
        if self.encoder_type == "gnn":
            return self.encoder(x, graph_key="graph")
        if self.encoder is not None:
            return self.encoder(x)
        return get_flat_obs(x)

    def set_risk_prior_beta(self, beta: float) -> None:
        self.risk_prior_beta = float(beta)

    def _load_risk_prior(self, checkpoint_path: str, envs: gym.Env) -> None:
        if getattr(envs, "graph_specs", None) is None:
            raise ValueError(
                "--risk-prior-checkpoint requires graph observations. This should "
                "be automatic when the flag is set; check any_gnn_enabled(args)."
            )

        self.risk_prior, self.risk_prior_payload = load_risk_surrogate(
            checkpoint_path,
            map_location="cpu",
        )
        self.risk_prior_graph_spec = self.risk_prior_payload.get(
            "graph_spec",
            self.risk_prior_payload["metadata"]["state_graph_spec"],
        )

        if self.agent_index >= self.risk_prior.n_agents:
            raise ValueError(
                f"Risk-prior checkpoint has {self.risk_prior.n_agents} agents, "
                f"but actor index is {self.agent_index}."
            )
        if self.n_actions > self.risk_prior.action_vocab_size:
            raise ValueError(
                f"Risk-prior checkpoint action vocab is {self.risk_prior.action_vocab_size}, "
                f"but {self.agent_index=} has {self.n_actions} actions."
            )

        env_state_spec = envs.graph_specs["state"]
        for key in ("node_dim", "edge_dim"):
            if int(env_state_spec[key]) != int(self.risk_prior_graph_spec[key]):
                raise ValueError(
                    f"Risk-prior graph spec mismatch for {key}: "
                    f"env={env_state_spec[key]}, checkpoint={self.risk_prior_graph_spec[key]}"
                )

    def _policy_logits(self, x: th.Tensor, action0_bonus: float = 0.0) -> th.Tensor:
        logits = self.actor(self._encode(x))
        if action0_bonus != 0.0:
            logits = logits.clone()
            logits[..., 0] = logits[..., 0] + action0_bonus
        return self._apply_risk_prior(x, logits)

    def _apply_risk_prior(self, x: th.Tensor, logits: th.Tensor) -> th.Tensor:
        if self.risk_prior is None or self.risk_prior_beta == 0.0:
            return logits
        if self.risk_prior_tau <= 0.0:
            raise ValueError("--risk-prior-tau must be strictly positive.")
        if not isinstance(x, dict) or "state_graph" not in x:
            raise ValueError(
                "Risk prior requires observations with a 'state_graph' entry. "
                "Do not strip state_graph from actor observations when the prior is enabled."
            )

        with th.no_grad():
            action_risk = self._predict_action_risk_matrix(x["state_graph"], logits.device)
            prior_score = -action_risk / self.risk_prior_tau
            if self.risk_prior_clip > 0.0:
                prior_score = prior_score.clamp(
                    min=-self.risk_prior_clip,
                    max=self.risk_prior_clip,
                )
        return logits + self.risk_prior_beta * prior_score.to(logits.device)

    def _predict_action_risk_matrix(
        self,
        state_graph: Dict[str, th.Tensor],
        device: th.device,
    ) -> th.Tensor:
        if self.risk_prior is None or self.risk_prior_graph_spec is None:
            raise RuntimeError("Risk prior is not loaded.")

        graph = {
            key: value.to(device=device, dtype=th.float32)
            for key, value in state_graph.items()
        }
        unbatched = graph["node_features"].dim() == 2
        if unbatched:
            graph = {key: value.unsqueeze(0) for key, value in graph.items()}

        pre_risk = graph_pre_risk(graph, self.risk_prior_graph_spec)
        self.risk_prior.eval()
        action_risk = self.risk_prior.predict_local_action_risks(
            state_graph=graph,
            agent_index=self.agent_index,
            n_actions=self.n_actions,
            pre_risk=pre_risk,
        )
        return action_risk.squeeze(0) if unbatched else action_risk

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
        logits = self._policy_logits(x, action0_bonus=action0_bonus)
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
        logits = self._policy_logits(x)
        return th.argmax(logits, dim=-1)

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
        else:
            raise ValueError(
                f"Unsupported critic encoder '{self.encoder_type}'. Use 'mlp' or 'gnn'."
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
        elif self.encoder is not None:
            x = self.encoder(x)
        else:
            x = get_flat_obs(x)
        return self.critic(x)
