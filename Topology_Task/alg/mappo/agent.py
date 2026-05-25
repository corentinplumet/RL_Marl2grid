from torch.distributions import Categorical, Normal

from common.imports import *
from common.utils import Linear, th_act_fns


class Actor(nn.Module):
    def __init__(
        self, id: int, envs: gym.Env, args: Dict[str, Any], continuous_actions: bool
    ):
        super().__init__()

        # Actor network setup
        actor_layers = args.actor_layers
        act_str, act_fn = args.actor_act_fn, th_act_fns[args.actor_act_fn]
        layers = []
        layers.extend(
            [
                Linear(
                    np.prod(envs.observation_space[f"agent_{id}"].shape),
                    actor_layers[0],
                    act_str,
                ),
                act_fn,
            ]
        )
        for idx, embed_dim in enumerate(actor_layers[1:], start=1):
            layers.extend([Linear(actor_layers[idx - 1], embed_dim, act_str), act_fn])

        # Final layer differs for continuous vs. discrete actions
        if continuous_actions:
            raise ("Redispatching actions are not yet implemented")
        else:
            n_actions = int(envs.action_space[f"agent_{id}"].n)
            # Logit layer: Xavier with linear gain (=1.0). Using the ReLU gain
            # here (the Linear utility's default) amplifies logits and produces
            # an essentially-deterministic random initial policy on bus14.
            out_layer = Linear(actor_layers[-1], n_actions, "linear")
            layers.append(out_layer)
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
        self.actor = nn.Sequential(*layers)

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
        logits = self.actor(x)
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
        logits = self.actor(x)
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

        joint_obs_shape = (
            sum(space.shape[0] for space in envs.observation_space.values())
            if args.decentralized
            else envs.observation_space["agent_0"].shape[-1]
        )

        # Critic network setup
        critic_layers = args.critic_layers
        act_str, act_fn = args.critic_act_fn, th_act_fns[args.critic_act_fn]
        layers = []
        layers.extend(
            [Linear(np.prod((joint_obs_shape,)), critic_layers[0], act_str), act_fn]
        )

        for idx, embed_dim in enumerate(critic_layers[1:], start=1):
            layers.extend([Linear(critic_layers[idx - 1], embed_dim, act_str), act_fn])
        layers.append(Linear(critic_layers[-1], 1, "linear"))
        self.critic = nn.Sequential(*layers)

    def get_value(self, x: th.Tensor) -> th.Tensor:
        """Compute value estimate (critic output) for given observations.

        Args:
            x: Input observations.

        Returns:
            A tensor containing value estimates.
        """
        return self.critic(x)
