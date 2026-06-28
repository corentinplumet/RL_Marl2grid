import os
from collections import deque

from common.action_trace import (
    build_action_trace_table,
    collect_unique_action_ids,
    decode_action_ids_safely,
    tensor_scalar_to_float,
    tensor_scalar_to_int,
)
from common.explainability import explain_arrays_from_infos, summarize_explain_arrays
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
        self.eval_action_heuristic = str(
            getattr(args, "eval_action_heuristic", "none")
        )
        if self.eval_action_heuristic not in {
            "none",
            "rho_threshold",
            "local_rho_threshold",
        }:
            raise ValueError(
                "eval_action_heuristic must be 'none', 'rho_threshold', or "
                "'local_rho_threshold', "
                f"got {self.eval_action_heuristic!r}."
            )
        self.eval_action_rho_threshold = float(
            getattr(args, "eval_action_rho_threshold", 0.90)
        )
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
        self.eval_progress_print = bool(getattr(args, "eval_progress_print", False))
        # if self.use_heuristic: self.env.set_n_rewards(len(self.reward_tags))

    def _current_chronic_name(self) -> str:
        """Best-effort label for the currently evaluated Grid2Op chronic."""
        handler = getattr(getattr(self.env, "g2op_env", None), "chronics_handler", None)
        if handler is None:
            return "unknown"

        def label_from(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, (str, bytes, os.PathLike)):
                text = os.fsdecode(value)
                if not text:
                    return None
                return os.path.basename(os.path.normpath(text))
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            return None

        objects = []
        seen = set()
        queue = [handler]
        while queue:
            obj = queue.pop(0)
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            objects.append(obj)
            for attr in ("data", "_data", "real_data", "_real_data"):
                nested = getattr(obj, attr, None)
                if nested is not None and id(nested) not in seen:
                    queue.append(nested)

        # Prefer the deepest active chronic object over the top-level handler,
        # whose get_name/get_id can stay stale with Grid2Op's MA wrapper.
        for obj in reversed(objects):
            for name in ("get_name", "get_id"):
                getter = getattr(obj, name, None)
                if not callable(getter):
                    continue
                try:
                    label = label_from(getter())
                except Exception:
                    continue
                if label is not None:
                    return label
            for name in (
                "name",
                "_name",
                "path",
                "_path",
                "chronics_name",
                "_chronics_name",
                "current_chronics",
                "_current_chronics",
                "_prev_cache_id",
            ):
                label = label_from(getattr(obj, name, None))
                if label is not None:
                    return label
        return "unknown"

    @staticmethod
    def _format_chronic_names(chronic_names: List[str], max_items: int = 12) -> str:
        names = [str(name) for name in chronic_names]
        if not names:
            return "[]"
        if len(names) <= max_items:
            return "[" + ", ".join(names) + "]"
        n_head = max_items // 2
        n_tail = max_items - n_head
        shown = names[:n_head] + [f"... ({len(names)} total)"] + names[-n_tail:]
        return "[" + ", ".join(shown) + "]"

    def _eval_heuristic_decision(
        self, agent_ids: List[str]
    ) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """Return per-agent do-nothing overrides and pre-action max rho values."""
        force_noop = {agent: False for agent in agent_ids}
        max_rhos = {agent: float("nan") for agent in agent_ids}
        if self.eval_action_heuristic == "none":
            return force_noop, max_rhos
        if self.eval_action_heuristic == "rho_threshold":
            max_rho = float(self.env.get_current_max_rho())
            force = (
                np.isfinite(max_rho)
                and max_rho < self.eval_action_rho_threshold
            )
            return (
                {agent: bool(force) for agent in agent_ids},
                {agent: max_rho for agent in agent_ids},
            )
        if self.eval_action_heuristic == "local_rho_threshold":
            local_max_rhos = self.env.get_current_agent_max_rho()
            for agent in agent_ids:
                max_rho = float(local_max_rhos.get(agent, float("nan")))
                max_rhos[agent] = max_rho
                force_noop[agent] = bool(
                    np.isfinite(max_rho)
                    and max_rho < self.eval_action_rho_threshold
                )
            return force_noop, max_rhos
        raise ValueError(f"Unsupported eval action heuristic: {self.eval_action_heuristic}")

    @staticmethod
    def _zero_action_like(action: Any) -> Any:
        if isinstance(action, th.Tensor):
            return th.zeros_like(action)
        return 0

    def _log_per_step_reward_metrics(
        self,
        glob_step: int,
        avg_return_per_step: List[float],
        tags: List[str],
    ) -> None:
        prefix = f"{self.metric_prefix}/" if self.metric_prefix else ""
        record = {
            f"{prefix}{tag} per step": float(value)
            for tag, value in zip(tags, avg_return_per_step)
        }
        record["charts/global_step"] = glob_step
        wb.log(record, step=glob_step)

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

        eval_label = self.metric_prefix or "eval"
        if self.eval_progress_print:
            print(
                f"{eval_label} evaluation start: {eval_ep} chronics, "
                f"max_steps={self.max_steps}",
                flush=True,
            )

        ep_survivals: Deque[float] = deque(
            maxlen=eval_ep
        )  # Queue to store survival rates of episodes
        ep_returns: Deque[float] = deque(
            maxlen=eval_ep
        )  # Queue to store returns of episodes
        ep_returns_per_step: Deque[float] = deque(
            maxlen=eval_ep
        )  # Queue to store episode-average reward components
        ep_rewards = np.zeros(len(self.reward_tags))

        obs, info = self.env.reset()
        obs = cast_np_to_tensors(obs, self.device)
        # if self.use_heuristic: ep_rewards += list(info['rewards'].values())

        action = {}
        agent_ids = list(actors.keys())
        action_nonidle_counts = {agent: 0 for agent in agent_ids}
        heuristic_policy_nonidle_counts = {agent: 0 for agent in agent_ids}
        heuristic_blocked_nonidle_counts = {agent: 0 for agent in agent_ids}
        heuristic_force_noop_counts = {agent: 0 for agent in agent_ids}
        heuristic_any_force_noop_steps = 0
        heuristic_all_force_noop_steps = 0
        heuristic_max_rhos = {agent: [] for agent in agent_ids}
        explain_eval_arrays = None
        n_eval_steps = 0
        trace_records = []
        trace_episode = 0
        trace_episode_step = 0
        trace_step = 0
        episode_chronic_names = []
        while len(ep_survivals) < eval_ep:
            for agent, model in actors.items():
                action[agent] = model.get_eval_action(
                    obs[agent], deterministic=self.deterministic_eval
                )

            policy_action_ids = {
                agent: tensor_scalar_to_int(action[agent]) for agent in agent_ids
            }
            force_noop, heuristic_max_rho = self._eval_heuristic_decision(agent_ids)
            if self.eval_action_heuristic != "none":
                if any(force_noop.values()):
                    heuristic_any_force_noop_steps += 1
                if all(force_noop.values()):
                    heuristic_all_force_noop_steps += 1
                for agent in agent_ids:
                    heuristic_max_rhos[agent].append(heuristic_max_rho[agent])
                    if force_noop[agent]:
                        heuristic_force_noop_counts[agent] += 1
                        heuristic_blocked_nonidle_counts[agent] += int(
                            policy_action_ids[agent] != 0
                        )
                        action[agent] = self._zero_action_like(action[agent])
                    heuristic_policy_nonidle_counts[agent] += int(
                        policy_action_ids[agent] != 0
                    )

            action_ids = {
                agent: tensor_scalar_to_int(action[agent]) for agent in agent_ids
            }
            next_obs, reward, terminations, truncations, info = self.env.step(action)
            for agent, action_id in action_ids.items():
                action_nonidle_counts[agent] += int(action_id != 0)
            n_eval_steps += 1
            step_explain = explain_arrays_from_infos(info)
            if explain_eval_arrays is None:
                explain_eval_arrays = {
                    key: [value] for key, value in step_explain.items()
                }
            else:
                for key, value in step_explain.items():
                    explain_eval_arrays[key].append(value)
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
                episode_length = max(
                    int(self.env.g2op_ma_env._cent_env.nb_time_step), 1
                )
                episode_survival = episode_length / self.max_steps
                chronic_name = self._current_chronic_name()
                episode_chronic_names.append(chronic_name)
                ep_survivals.append(episode_survival)
                if not self.use_heuristic:
                    ep_returns.append(ep_rewards)
                    ep_returns_per_step.append(ep_rewards / episode_length)
                if self.eval_progress_print:
                    completed = len(ep_survivals)
                    running_survival = sum(ep_survivals) / max(completed, 1)
                    print(
                        f"{eval_label} chronic {completed}/{eval_ep}: "
                        f"{chronic_name} survival={episode_survival * 100:.3f}% "
                        f"steps={episode_length}/{self.max_steps} "
                        f"running_mean={running_survival * 100:.3f}%",
                        flush=True,
                    )
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
        avg_return_per_step = [
            sum(r) / eval_ep for r in zip(*ep_returns_per_step)
        ]

        # Log the metrics if logger is available
        if self.logger:
            reward_tags = (
                self.reward_tags if self.env_id != "bus118" else self.reward_tags[:-1]
            )
            self.logger.store_metrics(
                glob_step,
                avg_survival,
                avg_return,
                reward_tags,
                prefix=self.metric_prefix,
            )
            self._log_per_step_reward_metrics(
                glob_step, avg_return_per_step, reward_tags
            )
            if explain_eval_arrays:
                eval_label = self.metric_prefix or "eval"
                record = summarize_explain_arrays(
                    {
                        key: np.concatenate(values)
                        for key, values in explain_eval_arrays.items()
                    },
                    prefix=f"{eval_label}/explain",
                )
                for agent in agent_ids:
                    action_nonidle_frac = action_nonidle_counts[agent] / max(
                        n_eval_steps, 1
                    )
                    record[f"{eval_label}/explain/action_nonidle_{agent}"] = (
                        action_nonidle_frac
                    )
                    record[f"{eval_label}/explain/frac_action_0_{agent}"] = (
                        1.0 - action_nonidle_frac
                    )
                    if getattr(actors[agent], "intervention_gate", False):
                        record[f"{eval_label}/explain/gate_intervened_{agent}"] = (
                            action_nonidle_frac
                        )
                if self.eval_action_heuristic != "none":
                    all_rhos = [
                        value
                        for values in heuristic_max_rhos.values()
                        for value in values
                    ]
                    finite_rhos = np.asarray(all_rhos, dtype=np.float32)
                    finite_rhos = finite_rhos[np.isfinite(finite_rhos)]
                    record[f"{eval_label}/heuristic/rho_threshold"] = (
                        self.eval_action_rho_threshold
                    )
                    record[f"{eval_label}/heuristic/is_local"] = float(
                        self.eval_action_heuristic == "local_rho_threshold"
                    )
                    record[f"{eval_label}/heuristic/force_noop_frac"] = (
                        heuristic_any_force_noop_steps / max(n_eval_steps, 1)
                    )
                    record[f"{eval_label}/heuristic/force_noop_any_agent_frac"] = (
                        heuristic_any_force_noop_steps / max(n_eval_steps, 1)
                    )
                    record[f"{eval_label}/heuristic/force_noop_all_agents_frac"] = (
                        heuristic_all_force_noop_steps / max(n_eval_steps, 1)
                    )
                    record[f"{eval_label}/heuristic/mean_pre_action_max_rho"] = (
                        float(finite_rhos.mean()) if finite_rhos.size else float("nan")
                    )
                    for agent in agent_ids:
                        agent_rhos = np.asarray(
                            heuristic_max_rhos[agent], dtype=np.float32
                        )
                        agent_rhos = agent_rhos[np.isfinite(agent_rhos)]
                        record[
                            f"{eval_label}/heuristic/force_noop_frac_{agent}"
                        ] = heuristic_force_noop_counts[agent] / max(n_eval_steps, 1)
                        record[
                            f"{eval_label}/heuristic/mean_pre_action_max_rho_{agent}"
                        ] = (
                            float(agent_rhos.mean())
                            if agent_rhos.size
                            else float("nan")
                        )
                        record[
                            f"{eval_label}/heuristic/policy_nonidle_{agent}"
                        ] = heuristic_policy_nonidle_counts[agent] / max(n_eval_steps, 1)
                        record[
                            f"{eval_label}/heuristic/blocked_nonidle_{agent}"
                        ] = heuristic_blocked_nonidle_counts[agent] / max(
                            n_eval_steps, 1
                        )
                record["charts/global_step"] = glob_step
                wb.log(record, step=glob_step)
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

        if self.eval_progress_print:
            print(
                f"{eval_label} evaluation finished: {eval_ep} chronics, "
                f"mean_survival={avg_survival * 100:.3f}%",
                flush=True,
            )
        print(
            f"{eval_label} at step {glob_step}, "
            f"chronics={self._format_chronic_names(episode_chronic_names)}, "
            f"survival={avg_survival * 100:.3f}%, return={avg_return}"
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
        ep_returns_per_step: Deque[float] = deque(
            maxlen=eval_ep
        )  # Queue to store episode-average reward components
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
                episode_length = max(
                    int(self.env.g2op_ma_env._cent_env.nb_time_step), 1
                )
                ep_survivals.append(episode_length / self.max_steps)
                ep_returns.append(ep_rewards)
                ep_returns_per_step.append(ep_rewards / episode_length)
                ep_cost_returns.append(ep_costs)

                obs, _ = self.env.reset()
                obs = cast_np_to_tensors(obs, self.device)
                ep_rewards = np.zeros(len(self.reward_tags))
                ep_costs = 0

        # Calculate average survival rate and return over the evaluated episodes
        avg_survival = sum(ep_survivals) / eval_ep
        avg_return = [sum(r) / eval_ep for r in zip(*ep_returns)]
        avg_return_per_step = [
            sum(r) / eval_ep for r in zip(*ep_returns_per_step)
        ]
        avg_cost_return = [sum(ep_cost_returns) / eval_ep]

        # Log the metrics if logger is available
        if self.logger:
            reward_tags = (
                self.reward_tags if self.env_id != "bus118" else self.reward_tags[:-1]
            )
            self.logger.store_metrics(
                glob_step,
                avg_survival,
                avg_return,
                avg_cost_return,
                reward_tags,
                prefix=self.metric_prefix,
            )
            self._log_per_step_reward_metrics(
                glob_step, avg_return_per_step, reward_tags
            )

        eval_label = self.metric_prefix or "eval"
        print(
            f"{eval_label} at step {glob_step}, survival={avg_survival * 100:.3f}%, return={avg_return}"
        )
        return avg_survival
