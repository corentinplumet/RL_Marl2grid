from collections import deque

from common.action_trace import (
    build_action_trace_table,
    collect_unique_action_ids,
    decode_action_ids_safely,
    tensor_scalar_to_float,
    tensor_scalar_to_int,
)
from common.imports import *
from common.logger import Logger
from common.utils import cast_np_to_tensors, stack_agent_obs_by_env
from .utils import MAEnvWrapper
from .wrappers import RecordEpisodeStatistics


class Evaluator:
    """Evaluator class for evaluating a reinforcement learning model deterministically.

    Attributes:
        env (gym.Env): Vectorized environment for evaluation.
        max_steps (int): Maximum number of steps in an episode.
        logger (Logger): Logger for storing evaluation metrics.
        device (th.device): Device to run the model on (e.g., 'cpu' or 'cuda').
    """

    def __init__(
        self,
        args: Dict[str, Any],
        logger: Logger,
        device: th.device,
        chronic_split: Optional[str] = None,
        metric_prefix: Optional[str] = None,
    ) -> None:
        """Initialize the Evaluator with the given arguments, logger, and device.

        Args:
            args: Arguments containing environment configuration.
            logger: Logger for storing evaluation metrics.
            device: Device to run the model on.
        """

        self.env = RecordEpisodeStatistics(
            MAEnvWrapper(args, eval_env=True, chronic_split=chronic_split)
        )  # Initialize synchronized vector environment

        self.max_steps = (
            self.env.g2op_env.chronics_handler.max_episode_duration()
        )  # Get max episode duration
        self.logger = logger  # Logger for evaluation metrics
        self.device = device  # Device for model inference
        self.env_id = args.env_id
        # Fix n° rewards based on the env specs (for simplifying logging ops)
        # The reward returned by the env is an increasing survival reward
        self.reward_tags = [
            "Redispatch Reward",
            "Line Margin Reward",
            "Overload Reward",
        ]
        if args.action_type == "topology":
            self.reward_tags += ["Topology Reward"]
        if args.n1_reward:
            self.reward_tags += ["N1 Reward"]
        if self.env_id == "bus118":
            self.reward_tags.append(
                []
            )  # l2rpn_idf_2023 (bus118) returns an additional 0 reward value; we fix it with an empty metrics
        self.use_heuristic = args.use_heuristic
        self.deterministic_eval = getattr(args, "deterministic_eval", True)
        self.eval_episodes = getattr(args, "eval_episodes", 10)
        self.trace_rollout_actions = bool(
            getattr(args, "trace_rollout_actions", False)
        )
        self.trace_rollout_max_steps = int(
            getattr(args, "trace_rollout_max_steps", 512)
        )
        self.trace_rollout_decode_actions = bool(
            getattr(args, "trace_rollout_decode_actions", True)
        )
        if getattr(args, "split_chronics", False) and getattr(
            args, "eval_all_split_chronics", True
        ):
            split_size = getattr(self.env.env, "chronic_split_size", None)
            if split_size is not None:
                self.eval_episodes = split_size
        self.metric_prefix = metric_prefix
        self.chronic_split = chronic_split
        # if self.use_heuristic: self.env.set_n_rewards(len(self.reward_tags))

    def evaluate(
        self, glob_step: int, actors: Dict, eval_ep: Optional[int] = None
    ) -> float:
        """Evaluate the model over a specified number of episodes.

        Args:
            glob_step: Global step for logging purposes.
            model: Model to be evaluated.
            eval_ep: Number of episodes for evaluation.
        """

        if eval_ep is None:
            eval_ep = self.eval_episodes

        ep_survivals: Deque[float] = deque(
            maxlen=eval_ep
        )  # Queue to store survival rates of episodes
        ep_returns: Deque[float] = deque(
            maxlen=eval_ep
        )  # Queue to store returns of episodes
        ep_rewards = np.zeros(len(self.reward_tags))

        obs, info = self.env.reset()
        obs = cast_np_to_tensors(obs, self.device)
        # if self.use_heuristic: ep_rewards += list(info['rewards'].values())

        action = {}
        agent_ids = list(actors.keys())
        trace_records = []
        trace_episode = 0
        trace_episode_step = 0
        trace_step = 0
        while len(ep_survivals) < eval_ep:
            for agent, model in actors.items():
                action[agent] = model.get_eval_action(
                    obs[agent], deterministic=self.deterministic_eval
                )

            action_ids = {
                agent: tensor_scalar_to_int(action[agent]) for agent in agent_ids
            }
            next_obs, reward, terminations, truncations, info = self.env.step(action)
            done = bool(
                np.logical_or(
                    terminations[agent_ids[0]],
                    truncations[agent_ids[0]],
                )
            )
            if (
                self.logger is not None
                and self.trace_rollout_actions
                and len(trace_records) < self.trace_rollout_max_steps
            ):
                trace_records.append(
                    {
                        "source": self.metric_prefix or "eval",
                        "rollout": 0,
                        "global_step": glob_step,
                        "step": trace_step,
                        "env_idx": 0,
                        "episode": trace_episode,
                        "episode_step": trace_episode_step,
                        "non_idle_agents": sum(
                            action_id != 0 for action_id in action_ids.values()
                        ),
                        "reward_agent_0": tensor_scalar_to_float(
                            reward[agent_ids[0]]
                        ),
                        "done": done,
                        "actions": action_ids,
                    }
                )
            trace_step += 1

            obs = cast_np_to_tensors(next_obs, self.device)
            if not self.use_heuristic:
                ep_rewards += list(info["agent_0"]["rewards"].values())
            # Record rewards for plotting purposes
            if "episode" in info:  # Denote end of an episode
                ep_survivals.append(
                    self.env.g2op_ma_env._cent_env.nb_time_step / self.max_steps
                )
                if not self.use_heuristic:
                    ep_returns.append(ep_rewards)
                obs, _ = self.env.reset()
                obs = cast_np_to_tensors(obs, self.device)
                if not self.use_heuristic:
                    ep_rewards = np.zeros(len(self.reward_tags))
                trace_episode += 1
                trace_episode_step = 0
            else:
                trace_episode_step += 1

        # Calculate average survival rate and return over the evaluated episodes
        avg_survival = sum(ep_survivals) / eval_ep
        avg_return = [sum(r) / eval_ep for r in zip(*ep_returns)]

        # Log the metrics if logger is available
        if self.logger:
            self.logger.store_metrics(
                glob_step,
                avg_survival,
                avg_return,
                self.reward_tags if self.env_id != "bus118" else self.reward_tags[:-1],
                prefix=self.metric_prefix,
            )
            if trace_records:
                decoded_actions = {}
                if self.trace_rollout_decode_actions:
                    action_ids_by_agent = collect_unique_action_ids(
                        trace_records, agent_ids
                    )
                    decoded_actions = decode_action_ids_safely(
                        self.env.decode_action_ids,
                        action_ids_by_agent,
                    )
                trace_key = f"{self.metric_prefix or 'eval'}/rollout_action_trace"
                wb.log(
                    {
                        trace_key: build_action_trace_table(
                            trace_records, agent_ids, decoded_actions
                        ),
                        "charts/global_step": glob_step,
                    },
                    step=glob_step,
                )

        eval_label = self.metric_prefix or "eval"
        print(
            f"{eval_label} at step {glob_step}, survival={avg_survival * 100:.3f}%, return={avg_return}"
        )
        return avg_survival


class CMDPEvaluator(Evaluator):
    """Evaluator class for evaluating a constrained reinforcement learning model deterministically.

    Attributes:
        env (gym.Env): Vectorized environment for evaluation.
        max_steps (int): Maximum number of steps in an episode.
        logger (Logger): Logger for storing evaluation metrics.
        device (th.device): Device to run the model on (e.g., 'cpu' or 'cuda').
    """

    def __init__(
        self,
        args: Dict[str, Any],
        logger: Logger,
        device: th.device,
        chronic_split: Optional[str] = None,
        metric_prefix: Optional[str] = None,
    ) -> None:
        super().__init__(args, logger, device, chronic_split, metric_prefix)

    def evaluate(
        self, glob_step: int, actors: Dict, eval_ep: Optional[int] = None
    ) -> float:
        """Evaluate the model over a specified number of episodes.

        Args:
            glob_step: Global step for logging purposes.
            model: Model to be evaluated.
            eval_ep: Number of episodes for evaluation.
        """

        if eval_ep is None:
            eval_ep = self.eval_episodes

        ep_survivals: Deque[float] = deque(
            maxlen=eval_ep
        )  # Queue to store survival rates of episodes
        ep_returns: Deque[float] = deque(
            maxlen=eval_ep
        )  # Queue to store returns of episodes
        ep_cost_returns: Deque[float] = deque(
            maxlen=eval_ep
        )  # Queue to store cost returns of episodes
        ep_rewards = np.zeros(len(self.reward_tags))
        ep_costs = 0

        obs, info = self.env.reset()
        obs = cast_np_to_tensors(obs, self.device)
        if self.use_heuristic:
            ep_rewards += list(info["rewards"].values())

        action = {}
        while len(ep_survivals) < eval_ep:
            for agent, model in actors.items():
                action[agent] = model.get_eval_action(
                    obs[agent], deterministic=self.deterministic_eval
                )

            next_obs, _, _, _, info = self.env.step(action)

            obs = cast_np_to_tensors(next_obs, self.device)
            ep_rewards += list(info["agent_0"]["rewards"].values())
            ep_costs += info["cost"]

            # Record rewards for plotting purposes
            if "episode" in info:  # Denote end of an episode
                ep_survivals.append(
                    self.env.g2op_ma_env._cent_env.nb_time_step / self.max_steps
                )
                ep_returns.append(ep_rewards)
                ep_cost_returns.append(ep_costs)

                obs, _ = self.env.reset()
                obs = cast_np_to_tensors(obs, self.device)
                ep_rewards = np.zeros(len(self.reward_tags))
                ep_costs = 0

        # Calculate average survival rate and return over the evaluated episodes
        avg_survival = sum(ep_survivals) / eval_ep
        avg_return = [sum(r) / eval_ep for r in zip(*ep_returns)]
        avg_cost_return = [sum(ep_cost_returns) / eval_ep]

        # Log the metrics if logger is available
        if self.logger:
            self.logger.store_metrics(
                glob_step,
                avg_survival,
                avg_return,
                avg_cost_return,
                self.reward_tags if self.env_id != "bus118" else self.reward_tags[:-1],
                prefix=self.metric_prefix,
            )

        eval_label = self.metric_prefix or "eval"
        print(
            f"{eval_label} at step {glob_step}, survival={avg_survival * 100:.3f}%, return={avg_return}"
        )
        return avg_survival
