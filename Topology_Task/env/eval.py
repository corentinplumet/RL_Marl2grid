import csv
import json
import os
from collections import Counter, deque
from pathlib import Path

from common.action_trace import (
    build_action_trace_table,
    collect_unique_action_ids,
    decode_action_ids_safely,
    tensor_scalar_to_float,
    tensor_scalar_to_int,
)
from common.explainability import (
    extract_transition_explain,
    explain_arrays_from_infos,
    summarize_explain_arrays,
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
        self._eval_base_seed = int(getattr(args, "seed", 0))
        self.trace_rollout_actions = bool(
            getattr(args, "trace_rollout_actions", False)
        )
        self.trace_rollout_max_steps = int(
            getattr(args, "trace_rollout_max_steps", 512)
        )
        self.trace_rollout_decode_actions = bool(
            getattr(args, "trace_rollout_decode_actions", True)
        )
        self._split_chronics = bool(getattr(args, "split_chronics", False))
        self._eval_all_split_chronics = bool(
            getattr(args, "eval_all_split_chronics", True)
        )
        if self._split_chronics and self._eval_all_split_chronics:
            split_size = getattr(self.env.env, "chronic_split_size", None)
            if split_size is not None:
                self.eval_episodes = split_size
        self.metric_prefix = metric_prefix
        self.chronic_split = chronic_split
        self.eval_progress_print = bool(getattr(args, "eval_progress_print", False))
        self.full_test_save_action_summary = bool(
            getattr(args, "full_test_save_action_summary", False)
        )
        self.full_test_action_summary_json = str(
            getattr(args, "full_test_action_summary_json", "") or ""
        )
        self.full_test_action_distribution_csv = str(
            getattr(args, "full_test_action_distribution_csv", "") or ""
        )
        self.full_test_episode_summary_csv = str(
            getattr(args, "full_test_episode_summary_csv", "") or ""
        )
        self.full_test_action_trace_csv = str(
            getattr(args, "full_test_action_trace_csv", "") or ""
        )
        self.full_test_decode_action_distribution = bool(
            getattr(args, "full_test_decode_action_distribution", False)
        )
        self.last_action_artifacts: Dict[str, str] = {}
        self.last_action_summary: Dict[str, Any] = {}
        # if self.use_heuristic: self.env.set_n_rewards(len(self.reward_tags))

    def _should_randomize_eval_chronics(self, eval_ep: int) -> bool:
        """Return whether this eval should draw a fresh random chronic subset."""
        if self.env_id == "bus14":
            return False
        if not self._split_chronics:
            return False
        split_size = getattr(self.env.env, "chronic_split_size", None)
        if split_size is not None and int(eval_ep) >= int(split_size):
            return False
        return True

    def _randomize_eval_chronics(self, glob_step: int, eval_ep: int) -> None:
        if not self._should_randomize_eval_chronics(eval_ep):
            return
        prefix_offset = 0 if self.metric_prefix in {None, "test"} else 1_000_003
        split_offset = 0 if self.chronic_split in {None, "test"} else 2_000_003
        shuffle_seed = (
            self._eval_base_seed
            + int(glob_step)
            + prefix_offset
            + split_offset
        )
        self.env.reshuffle_chronics(seed=shuffle_seed)
        if self.eval_progress_print:
            eval_label = self.metric_prefix or "eval"
            print(
                f"{eval_label} random chronic subset: "
                f"seed={shuffle_seed} episodes={eval_ep}",
                flush=True,
            )

    def _current_chronic_name(self) -> str:
        """Best-effort label for the currently evaluated Grid2Op chronic."""
        value = self._current_chronic_field("chronic_name")
        if value != "unknown":
            return value

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

    def _current_chronic_field(self, field: str) -> str:
        getter = getattr(self.env, "get_current_chronic_info", None)
        if not callable(getter):
            return "unknown"
        try:
            info = getter()
        except Exception:
            return "unknown"
        value = info.get(field) if isinstance(info, dict) else None
        return "unknown" if value is None else str(value)

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

    @staticmethod
    def _chronic_field_from_info(info: Any, field: str) -> Optional[str]:
        if isinstance(info, dict) and "final_info" in info:
            info = info["final_info"]
        if not isinstance(info, dict):
            return None
        value = info.get(field)
        return None if value is None else str(value)

    @classmethod
    def _chronic_name_from_info(cls, info: Any) -> Optional[str]:
        return cls._chronic_field_from_info(info, "chronic_name")

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

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if isinstance(value, (list, tuple, set)):
            return "|".join(str(item) for item in value)
        if value is None:
            return ""
        return value

    @staticmethod
    def _safe_counter_dict(counter: Counter) -> Dict[str, int]:
        return {str(key): int(value) for key, value in sorted(counter.items())}

    @staticmethod
    def _most_common_key(counter: Counter) -> int:
        if not counter:
            return -1
        return int(counter.most_common(1)[0][0])

    def _action_trace_columns(self, agent_ids: List[str]) -> List[str]:
        columns = [
            "global_step",
            "eval_label",
            "step",
            "episode",
            "episode_step",
            "chronic_name",
            "chronic_path",
            "chronic_fingerprint",
            "chronic_datetime",
            "chronic_reset_count",
            "pre_max_rho",
            "pre_worst_line",
            "pre_worst_line_name",
            "pre_worst_line_or_subid",
            "pre_worst_line_ex_subid",
            "pre_worst_line_agents",
            "post_max_rho",
            "post_worst_line",
            "non_idle_agents",
            "reward_agent_0",
            "done",
        ]
        for agent in agent_ids:
            columns.extend(
                [
                    f"policy_action_id_{agent}",
                    f"action_id_{agent}",
                    f"action_nonidle_{agent}",
                    f"heuristic_force_noop_{agent}",
                    f"heuristic_pre_max_rho_{agent}",
                    f"local_max_rho_{agent}",
                    f"local_worst_line_{agent}",
                    f"local_worst_line_name_{agent}",
                    f"local_worst_line_or_subid_{agent}",
                    f"local_worst_line_ex_subid_{agent}",
                    f"local_worst_line_agents_{agent}",
                ]
            )
        return columns

    def _get_rho_summary(self) -> Dict[str, Any]:
        getter = getattr(self.env, "get_current_rho_summary", None)
        if not callable(getter):
            return {}
        try:
            return getter()
        except Exception:
            return {}

    def _get_agent_rho_summary(self) -> Dict[str, Dict[str, Any]]:
        getter = getattr(self.env, "get_current_agent_rho_summary", None)
        if not callable(getter):
            return {}
        try:
            return getter()
        except Exception:
            return {}

    def _write_action_artifacts(
        self,
        *,
        glob_step: int,
        eval_label: str,
        agent_ids: List[str],
        eval_ep: int,
        n_eval_steps: int,
        action_counts: Dict[str, Counter],
        policy_action_counts: Dict[str, Counter],
        heuristic_force_noop_counts: Dict[str, int],
        heuristic_blocked_nonidle_counts: Dict[str, int],
        worst_line_counts: Counter,
        episode_rows: List[Dict[str, Any]],
    ) -> None:
        if not self.full_test_save_action_summary:
            return

        self.last_action_artifacts = dict(self.last_action_artifacts)
        self.last_action_summary = {}

        decoded_actions = {}
        if self.full_test_decode_action_distribution:
            action_ids_by_agent = {
                agent: sorted(int(action_id) for action_id in counts.keys())
                for agent, counts in action_counts.items()
            }
            decoded_actions = decode_action_ids_safely(
                self.env.decode_action_ids,
                action_ids_by_agent,
            )

        distribution_rows = []
        for agent in agent_ids:
            total = sum(action_counts[agent].values())
            policy_total = sum(policy_action_counts[agent].values())
            all_action_ids = sorted(
                set(action_counts[agent].keys())
                | set(policy_action_counts[agent].keys())
            )
            for action_id in all_action_ids:
                count = int(action_counts[agent].get(action_id, 0))
                policy_count = int(policy_action_counts[agent].get(action_id, 0))
                row = {
                    "agent_id": agent,
                    "action_id": int(action_id),
                    "is_action0": int(action_id == 0),
                    "count": count,
                    "fraction": count / max(total, 1),
                    "policy_count": policy_count,
                    "policy_fraction": policy_count / max(policy_total, 1),
                }
                if decoded_actions:
                    row["decoded_action"] = decoded_actions.get(agent, {}).get(
                        int(action_id), ""
                    )
                distribution_rows.append(row)

        if self.full_test_action_distribution_csv:
            path = Path(self.full_test_action_distribution_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            columns = [
                "agent_id",
                "action_id",
                "is_action0",
                "count",
                "fraction",
                "policy_count",
                "policy_fraction",
            ]
            if decoded_actions:
                columns.append("decoded_action")
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(distribution_rows)
            self.last_action_artifacts["action_distribution_csv"] = str(path)

        if self.full_test_episode_summary_csv:
            path = Path(self.full_test_episode_summary_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            if episode_rows:
                columns = list(episode_rows[0].keys())
            else:
                columns = [
                    "episode",
                    "chronic_name",
                    "chronic_fingerprint",
                    "survival",
                    "steps",
                ]
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(episode_rows)
            self.last_action_artifacts["episode_summary_csv"] = str(path)

        summary = {
            "global_step": int(glob_step),
            "eval_label": eval_label,
            "eval_episodes": int(eval_ep),
            "n_eval_steps": int(n_eval_steps),
            "action_distribution_csv": self.last_action_artifacts.get(
                "action_distribution_csv"
            ),
            "episode_summary_csv": self.last_action_artifacts.get(
                "episode_summary_csv"
            ),
            "action_trace_csv": self.last_action_artifacts.get("action_trace_csv"),
            "agents": {},
            "worst_line_counts": self._safe_counter_dict(worst_line_counts),
            "most_common_worst_line": self._most_common_key(worst_line_counts),
        }
        for agent in agent_ids:
            total = sum(action_counts[agent].values())
            action0_count = int(action_counts[agent].get(0, 0))
            nonidle_count = int(total - action0_count)
            summary["agents"][agent] = {
                "n_steps": int(total),
                "action0_count": action0_count,
                "action0_fraction": action0_count / max(total, 1),
                "nonidle_count": nonidle_count,
                "nonidle_fraction": nonidle_count / max(total, 1),
                "heuristic_force_noop_count": int(
                    heuristic_force_noop_counts.get(agent, 0)
                ),
                "heuristic_blocked_nonidle_count": int(
                    heuristic_blocked_nonidle_counts.get(agent, 0)
                ),
                "action_counts": self._safe_counter_dict(action_counts[agent]),
                "policy_action_counts": self._safe_counter_dict(
                    policy_action_counts[agent]
                ),
            }

        if self.full_test_action_summary_json:
            path = Path(self.full_test_action_summary_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
                f.write("\n")
            self.last_action_artifacts["action_summary_json"] = str(path)
        self.last_action_summary = summary

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
        self._randomize_eval_chronics(glob_step, eval_ep)
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
        episode_chronic_paths = []
        episode_chronic_fingerprints = []
        save_action_logs = bool(
            self.full_test_save_action_summary or self.full_test_action_trace_csv
        )
        action_counts = {agent: Counter() for agent in agent_ids}
        policy_action_counts = {agent: Counter() for agent in agent_ids}
        episode_action_counts = {agent: Counter() for agent in agent_ids}
        episode_policy_action_counts = {agent: Counter() for agent in agent_ids}
        worst_line_counts = Counter()
        episode_worst_line_counts = Counter()
        episode_max_pre_rho = float("-inf")
        episode_rows: List[Dict[str, Any]] = []
        trace_file = None
        trace_writer = None
        if self.full_test_action_trace_csv:
            trace_path = Path(self.full_test_action_trace_csv)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_file = trace_path.open("w", newline="", encoding="utf-8")
            trace_writer = csv.DictWriter(
                trace_file,
                fieldnames=self._action_trace_columns(agent_ids),
            )
            trace_writer.writeheader()
            self.last_action_artifacts["action_trace_csv"] = str(trace_path)

        try:
            while len(ep_survivals) < eval_ep:
                pre_rho_summary = self._get_rho_summary() if save_action_logs else {}
                pre_agent_rho_summary = (
                    self._get_agent_rho_summary() if save_action_logs else {}
                )
                chronic_name_before = self._current_chronic_name()
                chronic_path_before = self._current_chronic_field("chronic_path")
                chronic_fingerprint_before = self._current_chronic_field(
                    "chronic_fingerprint"
                )
                chronic_datetime_before = self._current_chronic_field(
                    "chronic_datetime"
                )
                chronic_reset_count_before = self._current_chronic_field(
                    "chronic_reset_count"
                )

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
                    if save_action_logs:
                        action_counts[agent][int(action_id)] += 1
                        policy_action_counts[agent][int(policy_action_ids[agent])] += 1
                        episode_action_counts[agent][int(action_id)] += 1
                        episode_policy_action_counts[agent][
                            int(policy_action_ids[agent])
                        ] += 1
                if save_action_logs:
                    worst_line = int(pre_rho_summary.get("worst_line", -1))
                    if worst_line >= 0:
                        worst_line_counts[worst_line] += 1
                        episode_worst_line_counts[worst_line] += 1
                    pre_max_rho = float(pre_rho_summary.get("max_rho", np.nan))
                    if np.isfinite(pre_max_rho):
                        episode_max_pre_rho = max(episode_max_pre_rho, pre_max_rho)
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
                transition_explain = extract_transition_explain(info) or {}
                post_explain = transition_explain.get("post", {})
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
                if trace_writer is not None:
                    row = {
                        "global_step": int(glob_step),
                        "eval_label": eval_label,
                        "step": int(trace_step),
                        "episode": int(trace_episode),
                        "episode_step": int(trace_episode_step),
                        "chronic_name": chronic_name_before,
                        "chronic_path": chronic_path_before,
                        "chronic_fingerprint": chronic_fingerprint_before,
                        "chronic_datetime": chronic_datetime_before,
                        "chronic_reset_count": chronic_reset_count_before,
                        "pre_max_rho": pre_rho_summary.get("max_rho", np.nan),
                        "pre_worst_line": pre_rho_summary.get("worst_line", -1),
                        "pre_worst_line_name": pre_rho_summary.get("line_name", ""),
                        "pre_worst_line_or_subid": pre_rho_summary.get(
                            "line_or_subid", -1
                        ),
                        "pre_worst_line_ex_subid": pre_rho_summary.get(
                            "line_ex_subid", -1
                        ),
                        "pre_worst_line_agents": self._csv_value(
                            pre_rho_summary.get("line_agents", [])
                        ),
                        "post_max_rho": post_explain.get("max_rho", np.nan),
                        "post_worst_line": post_explain.get("worst_line", -1),
                        "non_idle_agents": sum(
                            action_id != 0 for action_id in action_ids.values()
                        ),
                        "reward_agent_0": tensor_scalar_to_float(reward[agent_ids[0]]),
                        "done": done,
                    }
                    for agent in agent_ids:
                        local_summary = pre_agent_rho_summary.get(agent, {})
                        row.update(
                            {
                                f"policy_action_id_{agent}": int(
                                    policy_action_ids[agent]
                                ),
                                f"action_id_{agent}": int(action_ids[agent]),
                                f"action_nonidle_{agent}": int(
                                    action_ids[agent] != 0
                                ),
                                f"heuristic_force_noop_{agent}": int(
                                    force_noop.get(agent, False)
                                ),
                                f"heuristic_pre_max_rho_{agent}": heuristic_max_rho.get(
                                    agent, np.nan
                                ),
                                f"local_max_rho_{agent}": local_summary.get(
                                    "max_rho", np.nan
                                ),
                                f"local_worst_line_{agent}": local_summary.get(
                                    "worst_line", -1
                                ),
                                f"local_worst_line_name_{agent}": local_summary.get(
                                    "line_name", ""
                                ),
                                f"local_worst_line_or_subid_{agent}": local_summary.get(
                                    "line_or_subid", -1
                                ),
                                f"local_worst_line_ex_subid_{agent}": local_summary.get(
                                    "line_ex_subid", -1
                                ),
                                f"local_worst_line_agents_{agent}": self._csv_value(
                                    local_summary.get("line_agents", [])
                                ),
                            }
                        )
                    trace_writer.writerow(row)
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
                    chronic_name = (
                        self._chronic_name_from_info(info)
                        or self._current_chronic_name()
                    )
                    chronic_path = (
                        self._chronic_field_from_info(info, "chronic_path")
                        or self._current_chronic_field("chronic_path")
                    )
                    chronic_seed = (
                        self._chronic_field_from_info(info, "chronic_seed")
                        or self._current_chronic_field("chronic_seed")
                    )
                    chronic_index = (
                        self._chronic_field_from_info(info, "chronic_index")
                        or self._current_chronic_field("chronic_index")
                    )
                    chronic_order_position = (
                        self._chronic_field_from_info(info, "chronic_order_position")
                        or self._current_chronic_field("chronic_order_position")
                    )
                    chronic_fingerprint = (
                        self._chronic_field_from_info(info, "chronic_fingerprint")
                        or "unknown"
                    )
                    if save_action_logs:
                        episode_row = {
                            "episode": int(trace_episode),
                            "chronic_name": chronic_name,
                            "chronic_path": chronic_path,
                            "chronic_seed": chronic_seed,
                            "chronic_index": chronic_index,
                            "chronic_order_position": chronic_order_position,
                            "chronic_fingerprint": chronic_fingerprint,
                            "steps": int(episode_length),
                            "max_steps": int(self.max_steps),
                            "survival": float(episode_survival),
                            "max_pre_action_rho": (
                                float(episode_max_pre_rho)
                                if np.isfinite(episode_max_pre_rho)
                                else float("nan")
                            ),
                            "most_common_pre_worst_line": self._most_common_key(
                                episode_worst_line_counts
                            ),
                        }
                        for agent in agent_ids:
                            total = sum(episode_action_counts[agent].values())
                            action0 = int(episode_action_counts[agent].get(0, 0))
                            episode_row[f"action0_count_{agent}"] = action0
                            episode_row[f"action0_frac_{agent}"] = action0 / max(
                                total, 1
                            )
                            episode_row[f"nonidle_count_{agent}"] = int(
                                total - action0
                            )
                            episode_row[f"policy_nonidle_count_{agent}"] = int(
                                sum(
                                    count
                                    for action_id, count in episode_policy_action_counts[
                                        agent
                                    ].items()
                                    if int(action_id) != 0
                                )
                            )
                        episode_rows.append(episode_row)
                    episode_chronic_names.append(chronic_name)
                    episode_chronic_paths.append(chronic_path)
                    episode_chronic_fingerprints.append(chronic_fingerprint)
                    ep_survivals.append(episode_survival)
                    if not self.use_heuristic:
                        ep_returns.append(ep_rewards)
                        ep_returns_per_step.append(ep_rewards / episode_length)
                    if self.eval_progress_print:
                        completed = len(ep_survivals)
                        running_survival = sum(ep_survivals) / max(completed, 1)
                        print(
                            f"{eval_label} chronic {completed}/{eval_ep}: "
                            f"{chronic_name} path={chronic_path} "
                            f"idx={chronic_index} order={chronic_order_position} "
                            f"seed={chronic_seed} "
                            f"survival={episode_survival * 100:.3f}% "
                            f"fingerprint={chronic_fingerprint} "
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
                    episode_action_counts = {agent: Counter() for agent in agent_ids}
                    episode_policy_action_counts = {
                        agent: Counter() for agent in agent_ids
                    }
                    episode_worst_line_counts = Counter()
                    episode_max_pre_rho = float("-inf")
                else:
                    trace_episode_step += 1
        finally:
            if trace_file is not None:
                trace_file.close()

        # Calculate average survival rate and return over the evaluated episodes
        avg_survival = sum(ep_survivals) / eval_ep
        avg_return = [sum(r) / eval_ep for r in zip(*ep_returns)]
        avg_return_per_step = [
            sum(r) / eval_ep for r in zip(*ep_returns_per_step)
        ]

        self._write_action_artifacts(
            glob_step=glob_step,
            eval_label=eval_label,
            agent_ids=agent_ids,
            eval_ep=eval_ep,
            n_eval_steps=n_eval_steps,
            action_counts=action_counts,
            policy_action_counts=policy_action_counts,
            heuristic_force_noop_counts=heuristic_force_noop_counts,
            heuristic_blocked_nonidle_counts=heuristic_blocked_nonidle_counts,
            worst_line_counts=worst_line_counts,
            episode_rows=episode_rows,
        )

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
            f"paths={self._format_chronic_names(episode_chronic_paths)}, "
            f"fingerprints={self._format_chronic_names(episode_chronic_fingerprints)}, "
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
