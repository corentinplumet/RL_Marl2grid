import os
import re
import json
import hashlib
from collections import defaultdict
from packaging import version

from gymnasium.spaces import Discrete, Box

import grid2op
from grid2op.Chronics import MultifolderWithCache, Multifolder
from grid2op.gym_compat import (
    GymEnv,
    BoxGymObsSpace,
    DiscreteActSpace,
)  # if we import gymnasium, GymEnv will convert to Gymnasium!
from grid2op.multi_agent import MultiAgentEnv
from grid2op.Reward import CombinedReward
from lightsim2grid import LightSimBackend
from ray.rllib.env.multi_agent_env import MultiAgentEnv as MAEnv

from common.imports import *
from .reward import (
    LineMarginReward,
    RedispRewardv1,
    N1ContingencyRewardv1,
    FlatRewardv1,
    DistanceRewardv1,
    OverloadReward,
)

# Get the directory of the current module
ENV_DIR = os.path.dirname(__file__)

MIN_GLOP_VERSION = version.parse("1.10.4.dev1")
if version.parse(grid2op.__version__) < MIN_GLOP_VERSION:
    raise RuntimeError(
        f"Please upgrade to grid2op >= {MIN_GLOP_VERSION}."
        "You might need to install it from github "
        "`pip install git+https://github.com/rte-france/grid2op.git@dev_1.10.4`"
    )

RHO_SAFETY_THRESHOLD = 0.90
CHRONIC_SPLITS = ("train", "test", "validation")


def _fraction_arg(value: float, name: str) -> float:
    fraction = float(value)
    if fraction > 1.0:
        fraction /= 100.0
    if fraction < 0.0 or fraction >= 1.0:
        raise ValueError(f"{name} must be in [0, 1) or [0, 100). Got {value}.")
    return fraction


def _chronic_split_fractions(args: Dict[str, Any]) -> Tuple[float, float]:
    test_fraction = _fraction_arg(
        getattr(args, "test_chronics_pct", 0.2), "test_chronics_pct"
    )
    validation_fraction = _fraction_arg(
        getattr(args, "validation_chronics_pct", 0.1), "validation_chronics_pct"
    )
    if test_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "test_chronics_pct + validation_chronics_pct must leave at least one train split fraction."
        )
    return test_fraction, validation_fraction


def _chronic_split_seed(args: Dict[str, Any]) -> int:
    seed = getattr(args, "chronic_split_seed", None)
    return getattr(args, "seed", 0) if seed is None else seed


def _collect_chronic_paths_from(value: Any) -> List[str]:
    if value is None or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Dict):
        iterable = value.keys()
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        iterable = value
    else:
        return []

    paths = []
    for item in iterable:
        if item is None:
            continue
        paths.append(str(item))
    return paths


def _get_available_chronics(chronics_handler: Any) -> List[str]:
    """Best-effort extraction of chronic ids/paths before applying a Grid2Op filter."""
    candidates = []
    objects = []
    seen_objects = set()
    queue = [chronics_handler]
    while queue:
        obj = queue.pop(0)
        if obj is None or id(obj) in seen_objects:
            continue
        objects.append(obj)
        seen_objects.add(id(obj))
        for attr in ("real_data", "_real_data", "data", "_data"):
            nested = getattr(obj, attr, None)
            if nested is not None and id(nested) not in seen_objects:
                queue.append(nested)

    chronic_attr_groups = (
        ("subpaths", "_subpaths", "paths", "_paths"),
        ("available_chronics", "_available_chronics"),
        ("chronics_list", "_chronics_list"),
        ("chronics", "_chronics"),
        ("names_chronics_to_backend", "_names_chronics_to_backend"),
    )
    for chronic_attrs in chronic_attr_groups:
        candidates.clear()
        for obj in objects:
            for attr in chronic_attrs:
                candidates.extend(
                    _collect_chronic_paths_from(getattr(obj, attr, None))
                )
        seen = set()
        unique = []
        for path in candidates:
            if path not in seen:
                unique.append(path)
                seen.add(path)
        if unique:
            return sorted(unique)
    return []


def _count_chronics(n_chronics: int, fraction: float, split_name: str) -> int:
    if fraction == 0.0:
        return 0
    count = max(1, int(round(n_chronics * fraction)))
    if count >= n_chronics:
        raise ValueError(
            f"{split_name} split would consume all {n_chronics} chronics. Reduce its percentage."
        )
    return count


def _build_chronic_splits(
    chronics: List[str], args: Dict[str, Any]
) -> Dict[str, List[str]]:
    n_chronics = len(chronics)
    test_fraction, validation_fraction = _chronic_split_fractions(args)
    n_test = _count_chronics(n_chronics, test_fraction, "test")
    n_validation = _count_chronics(
        n_chronics, validation_fraction, "validation"
    )
    if n_test + n_validation >= n_chronics:
        raise ValueError(
            f"Not enough chronics ({n_chronics}) for non-empty train/test/validation splits."
        )

    rng = np.random.default_rng(_chronic_split_seed(args))
    shuffled = [chronics[idx] for idx in rng.permutation(n_chronics)]
    return {
        "test": shuffled[:n_test],
        "validation": shuffled[n_test : n_test + n_validation],
        "train": shuffled[n_test + n_validation :],
    }


def _chronic_key_variants(chronic_path: Any) -> set:
    text = str(chronic_path)
    norm = os.path.normpath(text)
    return {text, norm, os.path.basename(norm)}


def _stable_chronic_hash(chronic_path: Any, seed: int) -> float:
    text = f"{seed}:{chronic_path}".encode("utf-8")
    digest = hashlib.sha256(text).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def _hash_chronic_split(chronic_path: Any, args: Dict[str, Any]) -> str:
    test_fraction, validation_fraction = _chronic_split_fractions(args)
    value = _stable_chronic_hash(chronic_path, _chronic_split_seed(args))
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def _resolve_chronic_split(
    args: Dict[str, Any], eval_env: bool, chronic_split: Optional[str]
) -> Optional[str]:
    if not getattr(args, "split_chronics", False):
        return None
    split_name = chronic_split or ("test" if eval_env else "train")
    if split_name not in CHRONIC_SPLITS:
        raise ValueError(
            f"Invalid chronic split: {split_name}. Choose from {CHRONIC_SPLITS}."
        )
    test_fraction, validation_fraction = _chronic_split_fractions(args)
    if split_name == "test" and test_fraction == 0.0:
        raise ValueError("The test chronic split is empty because test_chronics_pct=0.")
    if split_name == "validation" and validation_fraction == 0.0:
        raise ValueError(
            "The validation chronic split is empty because validation_chronics_pct=0."
        )
    return split_name


def _apply_chronic_split(
    chronics_handler: Any, args: Dict[str, Any], split_name: str
) -> Dict[str, Any]:
    if not hasattr(chronics_handler, "set_filter"):
        raise AttributeError("Grid2Op chronics handler does not expose set_filter().")

    available_chronics = _get_available_chronics(chronics_handler)
    if available_chronics:
        splits = _build_chronic_splits(available_chronics, args)
        selected_chronics = splits[split_name]
        if not selected_chronics:
            raise ValueError(f"Chronic split '{split_name}' is empty.")
        selected_keys = set()
        for chronic in selected_chronics:
            selected_keys.update(_chronic_key_variants(chronic))
        chronics_handler.set_filter(
            lambda chronic_path, keys=selected_keys: bool(
                keys.intersection(_chronic_key_variants(chronic_path))
            )
        )
        return {
            "split": split_name,
            "selected": len(selected_chronics),
            "total": len(available_chronics),
            "exact": True,
        }

    chronics_handler.set_filter(
        lambda chronic_path: _hash_chronic_split(chronic_path, args) == split_name
    )
    return {
        "split": split_name,
        "selected": None,
        "total": None,
        "exact": False,
    }


def load_config(file_path: str) -> Dict:
    """Load configuration from a JSON file.

    Args:
        file_path: Path to the JSON configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    # Get the directory of the current module (__file__ contains the path of the current file)
    with open(f"{ENV_DIR}/{file_path}", "r") as file:
        config = json.load(file)
    return config


class MAEnvWrapper(MAEnv):
    def __init__(
        self,
        args: Dict[str, Any],
        resume_run: bool = False,
        idx: int = 0,
        generate_class: bool = False,
        async_vec_env: bool = False,
        action_space=None,
        eval_env: bool = False,
        chronic_split: Optional[str] = None,
    ) -> Any:
        """Create and configure a grid2op environment.

        Args:
            args: Arguments containing environment configuration parameters.
            idx: Index of the environment instance.
            resume_run: Whether to resume a previous run.
            generate_class: Whether to generate classes for asynchronous environments.
            async_vec_env: Whether the environment is asynchronous.
            action_space: A previously generated action space for the agents (to share between processes)
            eval_env: Whether this environment is used for policy evaluation.
            chronic_split: Optional split override when --split-chronics is enabled.

        Returns:
            A configured grid2op environment wrapped in a GymEnv.
        """
        super().__init__()

        config = load_config(args.env_config_path)  # Load environment configuration
        env_id = args.env_id
        env_type = args.action_type.lower()

        env_config = config["environments"]
        assert env_id in env_config.keys(), (
            f"Invalid environment ID: {env_id}. Available IDs are: {env_config.keys()}"
        )

        env_types = ["topology", "redispatch"]
        assert env_type in env_types, (
            f"Invalid environment type: {env_type}. Available IDs are: {env_types}"
        )

        # GRID2OP
        # Create a grid2op environment with specified backend and reward structure
        # Separate rewards for the eval env for logging
        rewards = {}
        if eval_env:
            rewards["redispatchReward"] = RedispRewardv1()
            rewards["lineMarginReward"] = LineMarginReward()
            rewards["overloadReward"] = OverloadReward(
                constrained=args.constraints_type != 0
            )
            if env_type == "topology":
                rewards["topologyReward"] = DistanceRewardv1()

        if args.n1_reward:
            rewards["n1ContingencyReward"] = N1ContingencyRewardv1(
                l_ids=list(range(env_config[env_id]["n_line"])), normalize=True
            )

        # With vec envs, infos return an array of dicts (one for each env) containing the rewards
        self.g2op_env = grid2op.make(
            env_config[env_id]["grid2op_id"],
            reward_class=CombinedReward,
            backend=LightSimBackend(),
            other_rewards=rewards,
            chronics_class=Multifolder if args.optimize_mem else MultifolderWithCache,
        )

        ###
        # print(self.g2op_env.chronics_handler.max_episode_duration())

        self.g2op_env.seed(args.seed + idx)
        self.g2op_env.chronics_handler.seed(args.seed + idx)
        self.chronic_split = _resolve_chronic_split(args, eval_env, chronic_split)
        self.chronic_split_size = None
        self.chronic_split_total = None
        if self.chronic_split is not None:
            split_summary = _apply_chronic_split(
                self.g2op_env.chronics_handler, args, self.chronic_split
            )
            self.chronic_split_size = split_summary["selected"]
            self.chronic_split_total = split_summary["total"]
            if idx == 0 or eval_env:
                if split_summary["exact"]:
                    print(
                        f"Chronic split '{self.chronic_split}': "
                        f"{self.chronic_split_size}/{self.chronic_split_total} chronics"
                    )
                else:
                    print(
                        f"Chronic split '{self.chronic_split}': using hash filter "
                        "(exact split size unavailable from Grid2Op handler)"
                    )
        self.g2op_env.chronics_handler.shuffle()

        if args.optimize_mem:
            self.g2op_env.chronics_handler.set_chunk_size(
                100
            )  # Instead of loading all episode data, get chunks of 100

        # Assign a filter (e.g., use only chronics that have "december" in their name) to reduce memory footprint
        # self.g2op_env.chronics_handler.set_filter(lambda x: re.match(".*0000.*", x) is not None)
        # Create the cache; otherwise it'll only load the first scenario
        # TODO seed setup does not work with this - if we print the seeds, they are different (but agents' obs across envs are still the same)
        self.g2op_env.chronics_handler.reset()

        # print(self.g2op_env.chronics_handler.max_episode_duration())

        cr = (
            self.g2op_env.get_reward_instance()
        )  # Initialize the combined reward instance
        # Per step (cumulative) positive reward for staying alive; reaches 1 at the end of the episode
        # cr.addReward("IncreasingFlatReward",
        # IncreasingFlatRewardv1(per_timestep=1/g2op_env.chronics_handler.max_episode_duration()),
        #            1.0)
        cr.addReward("FlatReward", FlatRewardv1(per_timestep=1), 1.0)
        if not eval_env:
            topology_reward_weight = getattr(args, "topology_reward_weight", 0.0)
            if env_type == "topology" and topology_reward_weight > 0.0:
                # 0 if topology is the original one, negative when topology changes.
                cr.addReward(
                    "TopologyReward", DistanceRewardv1(), topology_reward_weight
                )
            cr.addReward(
                "redispatchReward", RedispRewardv1(), 1.0
            )  # Custom one, see common.rewards
            line_margin_reward_weight = getattr(args, "line_margin_reward_weight", 0.0)
            if line_margin_reward_weight > 0.0:
                cr.addReward(
                    "LineMarginReward", LineMarginReward(), line_margin_reward_weight
                )
        if args.constraints_type != 2:
            cr.addReward(
                "overloadReward",
                OverloadReward(constrained=args.constraints_type != 0),
                1.0,
            )  # Custom one, see common.rewards

        cr.initialize(self.g2op_env)  # Finalize the reward setup

        if generate_class:
            self.g2op_env.generate_classes()
            print("Class generated offline for AsyncVecEnv execution")
            quit()

        # MARL
        agent_ids = range(
            len(config["environments"][env_id]["agent_stations"])
            if args.difficulty == 0
            else self.g2op_env.n_sub
        )
        if args.difficulty == 0:
            self.action_domains = {
                f"agent_{idx}": config["environments"][env_id]["agent_stations"][idx]
                for idx in agent_ids
            }
        elif args.difficulty == 1:
            self.action_domains = {f"agent_{idx}": [idx] for idx in agent_ids}
        else:
            raise NotImplementedError("There are only 2 difficulty levels!")

        if args.decentralized:
            self.observation_domains = self.action_domains
        else:
            self.observation_domains = {
                f"agent_{idx}": list(range(self.g2op_env.n_sub)) for idx in agent_ids
            }

        self.g2op_ma_env = MultiAgentEnv(
            self.g2op_env,
            action_domains=self.action_domains,
            observation_domains=self.observation_domains,
        )

        self._agent_ids = set(self.g2op_ma_env.agents)
        self._agent_ids = self.g2op_ma_env.agents
        self.g2op_ma_env.seed(args.seed + idx)

        # Prepare action and observation spaces
        state_attrs = config["state_attrs"]
        obs_attrs = state_attrs["default"]
        if env_config[env_id]["maintenance"]:
            obs_attrs += state_attrs["maintenance"]

        if env_type == "topology":
            obs_attrs += state_attrs["topology"]
            obs_attrs += state_attrs["redispatch"]
            if env_config[env_id]["renewable"]:
                obs_attrs += state_attrs["curtailment"]
            if env_config[env_id]["battery"]:
                obs_attrs += state_attrs["storage"]
        else:
            raise NotImplementedError(
                "Redispatching environments are not implemented yet!"
            )

        # MARL spaces
        self._aux_observation_space = {
            agent_id: BoxGymObsSpace(
                self.g2op_ma_env.observation_spaces[agent_id],
                attr_to_keep=obs_attrs,
            )
            for agent_id in self.g2op_ma_env.agents
        }

        # to avoid "weird" pickle issues
        self.observation_space = {
            agent_id: Box(
                low=self._aux_observation_space[agent_id].low,
                high=self._aux_observation_space[agent_id].high,
                dtype=self._aux_observation_space[agent_id].dtype,
            )
            for agent_id in self.g2op_ma_env.agents
        }

        # raise alert or alarm is not supported by ALL_ATTR_FOR_DISCRETE nor ATTR_DISCRETE
        act_attr_to_keep = ["change_bus", "change_line_status"]
        if env_type == "topology":
            self._conv_action_space = {
                agent_id: DiscreteActSpace(
                    self.g2op_ma_env.action_spaces[agent_id],
                    attr_to_keep=act_attr_to_keep,
                )
                for agent_id in self.g2op_ma_env.agents
            }

            # to avoid "weird" pickle issues
            self.action_space = {
                agent_id: Discrete(n=self._conv_action_space[agent_id].n)
                for agent_id in self.g2op_ma_env.agents
            }
        else:
            raise NotImplementedError("Make the implementation in this case")

        self.constraints_type = args.constraints_type

        self.eval_env = eval_env
        self.norm_obs = args.norm_obs
        if self.norm_obs:
            self.epsilon = 1e-8
            self.obs_stats = defaultdict(
                lambda: {
                    "count": 0.0,
                    "mean": None,
                    "var": None,
                }
            )

        self.use_heuristic = args.use_heuristic

    @property
    def _risk_overflow(self) -> bool:
        """Check if the maximum rho value exceeds the safety threshold."""
        return self.g2op_ma_env._cent_env.current_obs.rho.max() >= RHO_SAFETY_THRESHOLD

    @property
    def _obs(self) -> np.ndarray:
        """Get the current observation from the Grid2Op environment."""
        return self.g2op_ma_env._cent_env.current_obs

    def _get_idle_actions(self) -> List:
        """Return an empty list of actions if risk overflow, otherwise return a default action."""
        if self._risk_overflow:
            return []
        return {
            agent_id: self._conv_action_space[agent_id].from_gym(0)
            for agent_id in self._agent_ids
        }

    def apply_idle_actions(self) -> Tuple[float, bool, Dict]:
        """Apply heuristic actions until a risky situation or episode end.

        Returns:
            A tuple containing the cumulative heuristic reward, a boolean indicating
            if the episode is done, and additional info.
        """
        use_heuristic, heuristic_reward = True, 0
        done, info = False, {}

        while use_heuristic:
            g2o_actions = self._get_idle_actions()
            if not g2o_actions:
                break

            obs, reward, done, info = self.g2op_ma_env.step(g2o_actions)

            heuristic_reward += reward["agent_0"]

            if (
                done["agent_0"] or self._risk_overflow
            ):  # Resume the agent if in a risky situation
                use_heuristic = False
                break

        return obs, heuristic_reward, done, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        done = True  # It could happen that a heuristic episode ends in the reset step
        while done:
            obs = (
                self.g2op_ma_env.reset()
            )  # reset the underlying multi agent environment

            if self.use_heuristic and not self._risk_overflow:
                obs, _, done, _ = self.apply_idle_actions()

            done = (
                done["agent_0"] if isinstance(done, Dict) else False
            )  # Manage the case when self._risk_overflow directly after reset

        return self._format_obs(obs), {}

    def seed(self, seed):
        return self.g2op_ma_env.seed(seed)

    def _format_obs(self, grid2op_obs):
        gym_obs = {
            agent_id: self._aux_observation_space[agent_id].to_gym(
                grid2op_obs[agent_id]
            )
            for agent_id in self.g2op_ma_env.agents
        }
        if self.norm_obs:
            if not self.eval_env:
                self._update_stats(gym_obs)
            gym_obs = self._normalize(gym_obs)

        # return the proper dictionnary
        return gym_obs

    def _update_stats(self, obs):
        for agent_id, ob in obs.items():
            stats = self.obs_stats[agent_id]
            if stats["mean"] is None:
                stats["mean"] = ob.astype(np.float64).copy()
                stats["var"] = np.zeros_like(ob, dtype=np.float64)
                stats["count"] = 1.0
                continue
            stats["count"] += 1
            delta = ob - stats["mean"]
            stats["mean"] += delta / stats["count"]
            delta2 = ob - stats["mean"]
            stats["var"] += delta * delta2

    def _normalize(self, obs):
        norm_obs = {}
        for agent_id, ob in obs.items():
            stats = self.obs_stats[agent_id]
            var = stats["var"] / stats["count"]
            var = np.maximum(var, self.epsilon)  # avoid sqrt of negative or zero
            std = np.sqrt(var)
            normed = (ob - stats["mean"]) / std
            normed = np.where(
                np.isfinite(normed), normed, 0.0
            )  # replace NaN/inf with 0
            norm_obs[agent_id] = normed
        return norm_obs

    def get_obs_stats(self):
        if not self.norm_obs:
            return {}
        return {
            agent_id: {
                "count": s["count"],
                "mean": None if s["mean"] is None else s["mean"].copy(),
                "var": None if s["var"] is None else s["var"].copy(),
            }
            for agent_id, s in self.obs_stats.items()
        }

    def set_obs_stats(self, stats):
        if not self.norm_obs or not stats:
            return
        self.obs_stats.clear()
        for agent_id, s in stats.items():
            self.obs_stats[agent_id] = {
                "count": s["count"],
                "mean": None if s["mean"] is None else s["mean"].copy(),
                "var": None if s["var"] is None else s["var"].copy(),
            }

    def _get_grid2op_act(self, actions):
        return {
            agent_id: self._conv_action_space[agent_id].from_gym(
                actions[agent_id] if actions else 0
            )
            for agent_id in self.g2op_ma_env.agents
        }

    def step(self, actions):
        # convert the action to grid2op
        grid2op_act = self._get_grid2op_act(actions)

        # do a step in the underlying multi agent environment
        obs, r, done, info = self.g2op_ma_env.step(grid2op_act)

        if self.use_heuristic and not done["agent_0"] and not self._risk_overflow:
            obs, heuristic_reward, done, info = self.apply_idle_actions()
            r = {k: v + heuristic_reward for k, v in r.items()}

        self._get_cost(done, info)

        # Retrieve the observation in the proper form
        gym_obs = self._format_obs(obs)

        truncateds = {
            k: False for k in self.g2op_ma_env.agents
        }  # TODO truncation is not used in g2o

        return gym_obs, r, done, truncateds, info

    def _get_cost(self, done, info):
        if (
            self.constraints_type == 1
        ):  # TODO add check on n° steps (if it's done but the grid survived for the entire episode, it's not a constraint violation)
            if done["agent_0"]:
                info["cost"] = 1
            else:
                info["cost"] = 0
        elif self.constraints_type == 2:
            n_disconnections = sum(~self.g2op_ma_env._cent_env.current_obs.line_status)
            n_overloads = 0  # If it's game over and the grid is disconnected (i..e, n_disconnections = n_lines), then we cannot get the thermal limit and we don't have to compute n_overloads

            if not done["agent_0"]:
                ampere_flows = np.abs(
                    self.g2op_ma_env._cent_env.backend.get_line_flow()
                )
                thermal_limits = np.abs(self.g2op_ma_env._cent_env.get_thermal_limit())
                margin = thermal_limits - ampere_flows
                n_overloads = len(margin[margin < 0])

            info["cost"] = n_disconnections + n_overloads
        else:
            return
