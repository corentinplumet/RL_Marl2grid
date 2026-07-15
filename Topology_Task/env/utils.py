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
from common.explainability import EXPLAIN_INFO_KEY
from common.graph import GridGraphBuilder
from common.tokenizer import GridTokenBuilder
from common.utils import any_gnn_enabled, any_transformer_enabled
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
CHRONIC_SPLITS = ("train", "test")


def _stable_int_seed(*parts: Any) -> int:
    text = ":".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(text).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _chronic_seed_path_key(root: Any, path: Any) -> str:
    try:
        return os.path.normpath(os.path.relpath(str(path), str(root)))
    except Exception:
        return os.path.normpath(str(path))


class PathSeededMultifolder(Multifolder):
    """Multifolder with per-scenario deterministic chronic seeds.

    Grid2Op's WCCI chronic generator samples maintenance during episode
    initialization. The default Multifolder draws the chronic seed from a
    stream, so the same scenario can get different maintenance depending on
    whether train, test, or all is evaluated first. This class makes the seed a
    stable function of the base Grid2Op seed, scenario index, and path.
    """

    def seed(self, seed: int):
        self._path_seed_base = int(seed)
        return super().seed(seed)

    def _seed_base(self) -> int:
        base_seed = getattr(self, "_path_seed_base", None)
        if base_seed is None:
            base_seed = self.seed_used if self.seed_used is not None else 0
            self._path_seed_base = int(base_seed)
        return int(base_seed)

    def _seed_for_path(self, path: Any, index: Any = None) -> int:
        base_seed = self._seed_base()
        scenario_key = _chronic_seed_path_key(self.path, path)
        return _stable_int_seed(base_seed, index, scenario_key)

    def _set_current_chronic_metadata(self) -> None:
        if self._order is None or len(self._order) == 0:
            return
        order_position = int(self._prev_cache_id % len(self._order))
        id_scenario = int(self._order[order_position])
        this_path = self.subpaths[id_scenario]
        seed_chronics = self._seed_for_path(this_path, id_scenario)
        self.current_chronic_seed = int(seed_chronics)
        self.current_chronic_index = id_scenario
        self.current_chronic_order_position = order_position
        self.current_chronic_path = str(this_path)

    def next_chronics(self):
        super().next_chronics()
        self._set_current_chronic_metadata()

    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):
        self._order_backend_loads = order_backend_loads
        self._order_backend_prods = order_backend_prods
        self._order_backend_lines = order_backend_lines
        self._order_backend_subs = order_backend_subs
        self._names_chronics_to_backend = names_chronics_to_backend

        self.n_gen = len(order_backend_prods)
        self.n_load = len(order_backend_loads)
        self.n_line = len(order_backend_lines)

        if self._order is None:
            self.reset()

        id_scenario = self._order[self._prev_cache_id]
        this_path = self.subpaths[id_scenario]
        seed_chronics = self._seed_for_path(this_path, id_scenario)
        self._set_current_chronic_metadata()
        self.data = self._get_nex_data(this_path)
        self.data.seed(seed_chronics)

        self.data.initialize(
            order_backend_loads,
            order_backend_prods,
            order_backend_lines,
            order_backend_subs,
            names_chronics_to_backend=names_chronics_to_backend,
        )
        self.start_datetime = self.data.start_datetime
        self.current_datetime = self.data.current_datetime
        if self.action_space is not None:
            self.data.action_space = self.action_space
        self._max_iter = self.data.max_iter


class PathSeededMultifolderWithCache(MultifolderWithCache):
    """Cached Multifolder variant with stable per-scenario chronic seeds."""

    def _seed_base(self) -> int:
        base_seed = getattr(self, "_path_seed_base", None)
        if base_seed is None:
            base_seed = self.seed_used if self.seed_used is not None else 0
            self._path_seed_base = int(base_seed)
        return int(base_seed)

    def _seed_for_path(self, path: Any, index: Any = None) -> int:
        base_seed = self._seed_base()
        scenario_key = _chronic_seed_path_key(self.path, path)
        return _stable_int_seed(base_seed, index, scenario_key)

    def _set_current_chronic_metadata(self) -> None:
        if self._order is None or len(self._order) == 0:
            return
        order_position = int(self._prev_cache_id % len(self._order))
        id_scenario = int(self._order[order_position])
        this_path = self.subpaths[id_scenario]
        seed_chronics = self._seed_for_path(this_path, id_scenario)
        self.current_chronic_seed = int(seed_chronics)
        self.current_chronic_index = id_scenario
        self.current_chronic_order_position = order_position
        self.current_chronic_path = str(this_path)

    def _refresh_cached_seeds(self) -> None:
        self._cached_seeds = np.empty(len(self.subpaths), dtype=np.int64)
        for i, path in enumerate(self.subpaths):
            self._cached_seeds[i] = self._seed_for_path(path, i)

    def reset(self):
        self._refresh_cached_seeds()
        return super().reset()

    def next_chronics(self):
        super().next_chronics()
        self._set_current_chronic_metadata()

    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):
        super().initialize(
            order_backend_loads,
            order_backend_prods,
            order_backend_lines,
            order_backend_subs,
            names_chronics_to_backend=names_chronics_to_backend,
        )
        self._set_current_chronic_metadata()

    def seed(self, seed: int):
        self._path_seed_base = int(seed)
        res = Multifolder.seed(self, seed)
        self._refresh_cached_seeds()
        for i, data in enumerate(self._cached_data or []):
            if data is None:
                continue
            data.seed(self._cached_seeds[i])
            data.regenerate_with_new_seed()
        return res


def _fraction_arg(value: float, name: str) -> float:
    fraction = float(value)
    if fraction > 1.0:
        fraction /= 100.0
    if fraction < 0.0 or fraction >= 1.0:
        raise ValueError(f"{name} must be in [0, 1) or [0, 100). Got {value}.")
    return fraction


def _chronic_split_fraction(args: Dict[str, Any]) -> float:
    test_fraction = _fraction_arg(
        getattr(args, "test_chronics_pct", 0.2), "test_chronics_pct"
    )
    if test_fraction >= 1.0:
        raise ValueError(
            "test_chronics_pct must leave at least one train split fraction."
        )
    return test_fraction


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
    test_fraction = _chronic_split_fraction(args)
    n_test = _count_chronics(n_chronics, test_fraction, "test")
    if n_test >= n_chronics:
        raise ValueError(
            f"Not enough chronics ({n_chronics}) for non-empty train/test splits."
        )

    rng = np.random.default_rng(_chronic_split_seed(args))
    shuffled = [chronics[idx] for idx in rng.permutation(n_chronics)]
    return {
        "test": shuffled[:n_test],
        "train": shuffled[n_test:],
    }


def _shuffle_chronic_order(
    chronics: List[str],
    args: Dict[str, Any],
    split_name: str,
    env_index: int,
) -> Tuple[List[str], int]:
    order_seed = _stable_int_seed(
        _chronic_split_seed(args),
        getattr(args, "seed", 0),
        split_name,
        "order",
        int(env_index),
    )
    if not chronics:
        return [], order_seed
    rng = np.random.default_rng(order_seed)
    order = rng.permutation(len(chronics))
    return [chronics[int(idx)] for idx in order], order_seed


def _chronic_key_variants(chronic_path: Any) -> set:
    text = str(chronic_path)
    norm = os.path.normpath(text)
    return {text, norm, os.path.basename(norm)}


def _stable_chronic_hash(chronic_path: Any, seed: int) -> float:
    text = f"{seed}:{chronic_path}".encode("utf-8")
    digest = hashlib.sha256(text).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def _hash_chronic_split(chronic_path: Any, args: Dict[str, Any]) -> str:
    test_fraction = _chronic_split_fraction(args)
    value = _stable_chronic_hash(chronic_path, _chronic_split_seed(args))
    if value < test_fraction:
        return "test"
    return "train"


def _chronic_shard_count(args: Dict[str, Any]) -> int:
    count = int(getattr(args, "chronic_shard_count", 1) or 1)
    if count <= 0:
        raise ValueError(f"chronic_shard_count must be positive. Got {count}.")
    return count


def _chronic_shard_index(args: Dict[str, Any]) -> int:
    count = _chronic_shard_count(args)
    index = int(getattr(args, "chronic_shard_index", 0) or 0)
    if index < 0 or index >= count:
        raise ValueError(
            f"chronic_shard_index must be in [0, {count}). Got {index}."
        )
    return index


def _chronic_shard_hash(chronic_path: Any, args: Dict[str, Any]) -> int:
    seed = _chronic_split_seed(args)
    text = f"{seed}:shard:{chronic_path}".encode("utf-8")
    digest = hashlib.sha256(text).hexdigest()
    return int(digest[:16], 16)


def _hash_chronic_shard(chronic_path: Any, args: Dict[str, Any]) -> int:
    return _chronic_shard_hash(chronic_path, args) % _chronic_shard_count(args)


def _current_chronic_subpaths(chronics_handler: Any) -> List[str]:
    for attr in ("subpaths", "_subpaths", "paths", "_paths"):
        value = getattr(chronics_handler, attr, None)
        paths = _collect_chronic_paths_from(value)
        if paths:
            return paths
    return []


def _set_chronics_handler_order(
    chronics_handler: Any,
    ordered_chronics: List[str],
    reset_position: bool = False,
) -> bool:
    if not ordered_chronics:
        return False
    subpaths = _current_chronic_subpaths(chronics_handler)
    if not subpaths:
        return False

    key_to_index: Dict[str, int] = {}
    for idx, path in enumerate(subpaths):
        for key in _chronic_key_variants(path):
            key_to_index.setdefault(key, idx)

    order = []
    seen = set()
    for chronic in ordered_chronics:
        chronic_index = None
        for key in _chronic_key_variants(chronic):
            if key in key_to_index:
                chronic_index = key_to_index[key]
                break
        if chronic_index is None or chronic_index in seen:
            continue
        order.append(int(chronic_index))
        seen.add(chronic_index)

    if not order:
        return False

    order_array = np.asarray(order, dtype=int)
    shuffle_method = getattr(chronics_handler, "shuffle", None)
    if callable(shuffle_method):
        shuffle_method(lambda _order, order_array=order_array: order_array.copy())
    else:
        chronics_handler._order = order_array
    prev_cache_id = getattr(chronics_handler, "_prev_cache_id", None)
    if reset_position and hasattr(chronics_handler, "_prev_cache_id"):
        chronics_handler._prev_cache_id = -1
    elif prev_cache_id is not None:
        chronics_handler._prev_cache_id = int(prev_cache_id) % len(order)
    if not reset_position and prev_cache_id is not None:
        metadata_method = getattr(
            chronics_handler, "_set_current_chronic_metadata", None
        )
        if callable(metadata_method):
            metadata_method()
    return True


def _shuffle_existing_chronics_handler_order(
    chronics_handler: Any,
    seed: int,
) -> bool:
    order = getattr(chronics_handler, "_order", None)
    if order is None:
        return False
    order = np.asarray(order, dtype=int)
    if order.size <= 1:
        return False
    rng = np.random.default_rng(int(seed))
    chronics_handler._order = order[rng.permutation(order.size)]
    prev_cache_id = getattr(chronics_handler, "_prev_cache_id", None)
    if prev_cache_id is not None:
        chronics_handler._prev_cache_id = int(prev_cache_id) % int(order.size)
    metadata_method = getattr(chronics_handler, "_set_current_chronic_metadata", None)
    if callable(metadata_method):
        metadata_method()
    return True


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
    test_fraction = _chronic_split_fraction(args)
    if split_name == "test" and test_fraction == 0.0:
        raise ValueError("The test chronic split is empty because test_chronics_pct=0.")
    return split_name


def _apply_chronic_split(
    chronics_handler: Any,
    args: Dict[str, Any],
    split_name: str,
    env_index: int = 0,
) -> Dict[str, Any]:
    if not hasattr(chronics_handler, "set_filter"):
        raise AttributeError("Grid2Op chronics handler does not expose set_filter().")

    available_chronics = _get_available_chronics(chronics_handler)
    shard_count = _chronic_shard_count(args)
    shard_index = _chronic_shard_index(args)
    if available_chronics:
        splits = _build_chronic_splits(available_chronics, args)
        selected_chronics = splits[split_name]
        if shard_count > 1:
            selected_chronics = [
                chronic
                for idx, chronic in enumerate(selected_chronics)
                if idx % shard_count == shard_index
            ]
        if not selected_chronics:
            raise ValueError(
                f"Chronic split '{split_name}' shard "
                f"{shard_index}/{shard_count} is empty."
            )
        ordered_chronics, order_seed = _shuffle_chronic_order(
            selected_chronics, args, split_name, env_index
        )
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
            "shard_index": shard_index,
            "shard_count": shard_count,
            "exact": True,
            "order_seed": order_seed,
            "ordered_chronics": ordered_chronics,
        }

    chronics_handler.set_filter(
        lambda chronic_path: (
            _hash_chronic_split(chronic_path, args) == split_name
            and _hash_chronic_shard(chronic_path, args) == shard_index
        )
    )
    return {
        "split": split_name,
        "selected": None,
        "total": None,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "exact": False,
        "order_seed": None,
        "ordered_chronics": None,
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


def _resolve_optional_task_path(path: str) -> Optional[str]:
    path = os.path.expanduser(str(path or "").strip())
    if not path:
        return None
    if os.path.isabs(path):
        return path

    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(os.path.dirname(ENV_DIR), path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _load_reduced_action_id_mapping(
    path: str,
    agent_ids,
    original_action_sizes: Dict[str, int],
) -> Tuple[Dict[str, List[int]], str]:
    resolved_path = _resolve_optional_task_path(path)
    if resolved_path is None:
        return {}, ""
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"Reduced action-space file does not exist: {resolved_path}"
        )

    with open(resolved_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    agents_payload = payload.get("agents", {})
    if not isinstance(agents_payload, dict):
        raise ValueError(
            f"Reduced action-space file {resolved_path} is missing an 'agents' object."
        )

    mapping: Dict[str, List[int]] = {}
    for agent_id in agent_ids:
        agent_payload = agents_payload.get(agent_id)
        if not isinstance(agent_payload, dict):
            raise ValueError(
                f"Reduced action-space file {resolved_path} has no entry for {agent_id}."
            )
        selected = agent_payload.get("selected_action_ids")
        if not selected:
            raise ValueError(
                f"Reduced action-space file {resolved_path} has no selected actions "
                f"for {agent_id}."
            )

        deduped = []
        seen = set()
        for action_id in selected:
            original_action_id = int(action_id)
            if original_action_id in seen:
                continue
            original_size = int(original_action_sizes[agent_id])
            if original_action_id < 0 or original_action_id >= original_size:
                raise ValueError(
                    f"Reduced action id {original_action_id} for {agent_id} is "
                    f"outside the original action space [0, {original_size})."
                )
            deduped.append(original_action_id)
            seen.add(original_action_id)

        if 0 not in seen:
            deduped.insert(0, 0)
        elif deduped[0] != 0:
            deduped = [0] + [action_id for action_id in deduped if action_id != 0]
        mapping[agent_id] = deduped

    return mapping, resolved_path


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
        self.use_graph_obs = any_gnn_enabled(args)
        self.use_token_obs = any_transformer_enabled(args)
        if self.use_graph_obs and self.use_token_obs:
            raise ValueError(
                "Mixing GNN and transformer structured observations is not "
                "supported yet. Use only mlp/gnn or mlp/transformer encoders."
            )

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

        # Maintenance chronics sample outage schedules from the chronic RNG.
        # Seed them by scenario identity so train/test/all filters see the same
        # realization for the same chronic instead of depending on iteration order.
        deterministic_maintenance = bool(env_config[env_id].get("maintenance", False))
        if deterministic_maintenance:
            chronics_class = (
                PathSeededMultifolder
                if args.optimize_mem
                else PathSeededMultifolderWithCache
            )
        else:
            chronics_class = Multifolder if args.optimize_mem else MultifolderWithCache

        # With vec envs, infos return an array of dicts (one for each env) containing the rewards
        self.g2op_env = grid2op.make(
            env_config[env_id]["grid2op_id"],
            reward_class=CombinedReward,
            backend=LightSimBackend(),
            other_rewards=rewards,
            chronics_class=chronics_class,
        )

        ###
        # print(self.g2op_env.chronics_handler.max_episode_duration())

        self.g2op_env.seed(args.seed + idx)
        self.g2op_env.chronics_handler.seed(args.seed + idx)
        self.chronic_split = _resolve_chronic_split(args, eval_env, chronic_split)
        self.chronic_split_size = None
        self.chronic_split_total = None
        self.chronic_split_order = None
        split_summary = None
        if self.chronic_split is not None:
            split_summary = _apply_chronic_split(
                self.g2op_env.chronics_handler,
                args,
                self.chronic_split,
                env_index=idx,
            )
            self.chronic_split_size = split_summary["selected"]
            self.chronic_split_total = split_summary["total"]
            if split_summary.get("ordered_chronics"):
                self.chronic_split_order = list(split_summary["ordered_chronics"])
            if idx == 0 or eval_env:
                shard_text = (
                    f" shard {split_summary['shard_index']}/"
                    f"{split_summary['shard_count']}"
                    if split_summary["shard_count"] > 1
                    else ""
                )
                if split_summary["exact"]:
                    print(
                        f"Chronic split '{self.chronic_split}'{shard_text}: "
                        f"{self.chronic_split_size}/{self.chronic_split_total} chronics"
                    )
                    print(
                        f"Chronic split '{self.chronic_split}' order shuffled "
                        f"with seed {split_summary['order_seed']}",
                        flush=True,
                    )
                    head = ", ".join(
                        os.path.basename(os.path.normpath(chronic))
                        for chronic in split_summary["ordered_chronics"][:8]
                    )
                    print(
                        f"Chronic split '{self.chronic_split}' order head: {head}",
                        flush=True,
                    )
                else:
                    print(
                        f"Chronic split '{self.chronic_split}'{shard_text}: "
                        "using hash filter "
                        "(exact split size unavailable from Grid2Op handler)"
                    )

        if args.optimize_mem:
            self.g2op_env.chronics_handler.set_chunk_size(
                100
            )  # Instead of loading all episode data, get chunks of 100

        # Assign a filter (e.g., use only chronics that have "december" in their name) to reduce memory footprint
        # self.g2op_env.chronics_handler.set_filter(lambda x: re.match(".*0000.*", x) is not None)
        # Create the cache; otherwise it'll only load the first scenario
        # TODO seed setup does not work with this - if we print the seeds, they are different (but agents' obs across envs are still the same)
        self.g2op_env.chronics_handler.reset()
        if split_summary and split_summary.get("ordered_chronics"):
            _set_chronics_handler_order(
                self.g2op_env.chronics_handler,
                split_summary["ordered_chronics"],
                reset_position=True,
            )
        else:
            order_seed = _stable_int_seed(
                getattr(args, "seed", 0),
                env_id,
                "order",
                idx,
            )
            _shuffle_existing_chronics_handler_order(
                self.g2op_env.chronics_handler,
                order_seed,
            )

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
        if self.chronic_split_order:
            successes = self._set_active_chronic_order(
                self.chronic_split_order, reset_position=True
            )
            if successes == 0:
                raise RuntimeError(
                    "Could not apply the shuffled chronic order to the active "
                    "training/evaluation handler."
                )

        # Prepare action and observation spaces
        state_attrs = config["state_attrs"]
        if env_type != "topology":
            raise NotImplementedError(
                "Redispatching environments are not implemented yet!"
            )
        custom_obs_attrs = getattr(args, "obs_attrs", None)
        if custom_obs_attrs:
            obs_attrs = list(dict.fromkeys(str(attr) for attr in custom_obs_attrs))
            known_obs_attrs = {
                attr for attrs in state_attrs.values() for attr in attrs
            }
            unknown_obs_attrs = [
                attr for attr in obs_attrs if attr not in known_obs_attrs
            ]
            if unknown_obs_attrs:
                raise ValueError(
                    "Unknown obs_attrs entries: "
                    f"{unknown_obs_attrs}. Known attributes: {sorted(known_obs_attrs)}"
                )
        else:
            obs_attrs = list(state_attrs["default"])
            if env_config[env_id]["maintenance"]:
                obs_attrs += state_attrs["maintenance"]

            obs_attrs += state_attrs["topology"]
            if not self.use_graph_obs:
                obs_attrs += state_attrs["redispatch"]
                if env_config[env_id]["renewable"]:
                    obs_attrs += state_attrs["curtailment"]
                if env_config[env_id]["battery"]:
                    obs_attrs += state_attrs["storage"]

        # MARL spaces
        self._aux_observation_space = {
            agent_id: BoxGymObsSpace(
                self.g2op_ma_env.observation_spaces[agent_id],
                attr_to_keep=obs_attrs,
            )
            for agent_id in self.g2op_ma_env.agents
        }
        flat_dims = {
            agent_id: int(np.prod(self._aux_observation_space[agent_id].low.shape))
            for agent_id in self.g2op_ma_env.agents
        }

        self.obs_norm_masks = {}
        for agent_id, flat_dim in flat_dims.items():
            self.obs_norm_masks[agent_id] = np.ones(flat_dim, dtype=bool)

        self.graph_builder = None
        self.graph_specs = None
        if self.use_graph_obs:
            self.graph_builder = GridGraphBuilder(
                self.g2op_env,
                self.observation_domains,
                include_neighbors=getattr(args, "gnn_include_neighbors", False),
                include_maintenance=env_config[env_id]["maintenance"],
            )
            self.graph_specs = self.graph_builder.specs
        self.token_builder = None
        self.token_specs = None
        if self.use_token_obs:
            self.token_builder = GridTokenBuilder(
                self.g2op_env,
                self.observation_domains,
                tokenizer_type=getattr(args, "tokenizer_type", "group"),
                include_neighbors=getattr(args, "tokenizer_include_neighbors", True),
                include_maintenance=getattr(
                    args,
                    "tokenizer_include_maintenance",
                    env_config[env_id]["maintenance"],
                ),
                max_token_feature_dim=getattr(
                    args, "tokenizer_max_feature_dim", 128
                ),
                include_busbar_tokens=getattr(
                    args, "tokenizer_include_busbar_tokens", False
                ),
            )
            self.token_specs = self.token_builder.specs
        self.agent_line_domains = self._make_agent_line_domains()

        # to avoid "weird" pickle issues
        self.observation_space = {
            agent_id: Box(
                low=np.full(
                    flat_dims[agent_id],
                    -np.inf,
                    dtype=np.float32,
                ),
                high=np.full(
                    flat_dims[agent_id],
                    np.inf,
                    dtype=np.float32,
                ),
                dtype=np.float32,
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
            self._original_action_space_sizes = {
                agent_id: int(self._conv_action_space[agent_id].n)
                for agent_id in self.g2op_ma_env.agents
            }
            (
                self._reduced_action_id_mapping,
                self._reduced_action_space_path,
            ) = _load_reduced_action_id_mapping(
                getattr(args, "reduced_action_space", ""),
                self.g2op_ma_env.agents,
                self._original_action_space_sizes,
            )

            # to avoid "weird" pickle issues
            self.action_space = {
                agent_id: Discrete(
                    n=len(self._reduced_action_id_mapping[agent_id])
                    if self._reduced_action_id_mapping
                    else self._conv_action_space[agent_id].n
                )
                for agent_id in self.g2op_ma_env.agents
            }
            if self._reduced_action_id_mapping:
                print(
                    "Loaded reduced action space from "
                    f"{self._reduced_action_space_path}: "
                    + ", ".join(
                        f"{agent_id}={self.action_space[agent_id].n}/"
                        f"{self._original_action_space_sizes[agent_id]}"
                        for agent_id in self.g2op_ma_env.agents
                    ),
                    flush=True,
                )
        else:
            raise NotImplementedError("Make the implementation in this case")

        self.constraints_type = args.constraints_type

        self.eval_env = eval_env
        self.current_chronic_name = "unknown"
        self.current_chronic_fingerprint = "unknown"
        self.current_chronic_datetime = "unknown"
        self.current_chronic_reset_count = 0
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

    def reshuffle_chronics(self, seed: Optional[int] = None) -> None:
        """Best-effort reshuffle of the active chronic handler.

        This is used by non-bus14 evaluation to draw a fresh random subset at
        each eval call while keeping the train/test split filter unchanged.
        """
        targets = [
            getattr(getattr(self, "g2op_env", None), "chronics_handler", None),
            getattr(
                getattr(self.g2op_ma_env, "_cent_env", None),
                "chronics_handler",
                None,
            ),
        ]
        errors = []
        shuffled = False
        seen = set()
        for handler in targets:
            if handler is None or id(handler) in seen:
                continue
            seen.add(id(handler))
            try:
                if seed is not None:
                    seed_method = getattr(handler, "seed", None)
                    if callable(seed_method) and not isinstance(
                        handler,
                        (PathSeededMultifolder, PathSeededMultifolderWithCache),
                    ):
                        seed_method(int(seed))
                shuffle_method = getattr(handler, "shuffle", None)
                if seed is None and callable(shuffle_method):
                    shuffle_method()
                    shuffled = True
                reset_method = getattr(handler, "reset", None)
                if callable(reset_method):
                    reset_method()
                if seed is not None:
                    shuffled = (
                        _shuffle_existing_chronics_handler_order(handler, int(seed))
                        or shuffled
                    )
            except Exception as exc:
                errors.append(f"{type(handler).__name__}: {exc}")

        if not shuffled:
            detail = "; ".join(errors[-4:]) if errors else "no shuffle method found"
            raise RuntimeError(f"Could not reshuffle chronics: {detail}")

    def _set_active_chronic_order(
        self, ordered_chronics: List[str], *, reset_position: bool
    ) -> int:
        """Apply a chronic order to every live handler backing this env."""
        targets = []
        if hasattr(self, "g2op_ma_env"):
            targets.extend(self._current_chronic_handlers())
        else:
            targets.append(
                getattr(getattr(self, "g2op_env", None), "chronics_handler", None)
            )

        successes = 0
        seen = set()
        for handler in targets:
            if handler is None or id(handler) in seen:
                continue
            seen.add(id(handler))
            if _set_chronics_handler_order(
                handler, ordered_chronics, reset_position=reset_position
            ):
                successes += 1
        return successes

    def set_chronic_window(self, start: int, count: int) -> Dict[str, int]:
        """Rotate the active chronic order so the next reset starts a window.

        This is used by validation when evaluating a small fixed-size subset of
        a train/test split: eval 0 gets chronics 0..N-1, eval 1 gets N..2N-1,
        and so on, wrapping around the split when needed.
        """
        if not self.chronic_split_order:
            raise RuntimeError("No explicit chronic split order is available.")

        split_size = len(self.chronic_split_order)
        if split_size <= 0:
            raise RuntimeError("The active chronic split is empty.")
        count = min(max(int(count), 1), split_size)
        start = int(start) % split_size
        ordered = self.chronic_split_order[start:] + self.chronic_split_order[:start]

        successes = self._set_active_chronic_order(ordered, reset_position=True)
        if successes == 0:
            raise RuntimeError("Could not set the active chronic evaluation window.")

        end = start + count
        wrapped = end > split_size

        return {
            "start": start,
            "count": count,
            "split_size": split_size,
            "end_exclusive": end % split_size if wrapped else end,
            "wrapped": int(wrapped),
        }

    def set_chronic_id(self, chronic_id: int) -> None:
        """Best-effort request for Grid2Op to use a specific chronic on reset."""
        chronic_id = int(chronic_id)
        cent_env = getattr(self.g2op_ma_env, "_cent_env", None)
        targets = [
            cent_env,
            getattr(cent_env, "chronics_handler", None),
            getattr(self, "g2op_env", None),
            getattr(getattr(self, "g2op_env", None), "chronics_handler", None),
        ]
        errors = []
        successes = []
        seen = set()
        for target in targets:
            if target is None or id(target) in seen:
                continue
            seen.add(id(target))
            for method_name in ("set_id", "tell_id"):
                method = getattr(target, method_name, None)
                if not callable(method):
                    continue
                try:
                    method(chronic_id)
                    successes.append(f"{type(target).__name__}.{method_name}")
                    break
                except Exception as exc:
                    errors.append(f"{type(target).__name__}.{method_name}: {exc}")
        if not successes:
            detail = "; ".join(errors[-4:]) if errors else "no set_id/tell_id method found"
            raise RuntimeError(f"Could not set chronic id {chronic_id}: {detail}")

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

        self.current_chronic_reset_count += 1
        self.current_chronic_name = self.get_current_chronic_name()
        self.current_chronic_fingerprint = self.get_current_chronic_fingerprint()
        self.current_chronic_datetime = self.get_current_chronic_datetime()
        return self._format_obs(obs), self.get_current_chronic_info()

    @staticmethod
    def _chronic_label_from(value: Any) -> Optional[str]:
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

    def _current_chronic_handlers(self) -> List[Any]:
        """Return chronic handlers in the same priority order as current_obs."""
        candidates = [
            getattr(
                getattr(self.g2op_ma_env, "_cent_env", None),
                "chronics_handler",
                None,
            ),
            getattr(getattr(self, "g2op_env", None), "chronics_handler", None),
        ]
        handlers = []
        seen = set()
        for handler in candidates:
            if handler is None or id(handler) in seen:
                continue
            handlers.append(handler)
            seen.add(id(handler))
        return handlers

    def _current_chronic_objects(self) -> List[Any]:
        object_groups = []
        for handler in self._current_chronic_handlers():
            object_groups.append(self._chronic_objects_from_handler(handler))

        objects = []
        for group in reversed(object_groups):
            objects.extend(group)
        return objects

    @staticmethod
    def _chronic_objects_from_handler(handler: Any) -> List[Any]:
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
        return objects

    def _current_chronic_attr(self, *names: str) -> Any:
        for obj in reversed(self._current_chronic_objects()):
            for name in names:
                try:
                    value = getattr(obj, name)
                except Exception:
                    continue
                if value is not None:
                    return value
        return None

    @staticmethod
    def _format_chronic_attr(value: Any) -> Any:
        if value is None:
            return "unknown"
        if isinstance(value, (int, np.integer)):
            return int(value)
        return str(value)

    def get_current_chronic_name(self) -> str:
        """Best-effort label for the current Grid2Op chronic.

        This is instrumentation only. Some Grid2Op / multi-agent wrapper
        combinations expose a stale top-level handler label, so we inspect nested
        handler data first and fall back conservatively.
        """
        objects = self._current_chronic_objects()

        for obj in reversed(objects):
            for name in ("get_name", "get_id"):
                getter = getattr(obj, name, None)
                if not callable(getter):
                    continue
                try:
                    label = self._chronic_label_from(getter())
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
                label = self._chronic_label_from(getattr(obj, name, None))
                if label is not None:
                    return label
        return "unknown"

    def get_current_chronic_path(self) -> str:
        """Best-effort full path/id for the current Grid2Op chronic."""
        objects = self._current_chronic_objects()

        for obj in reversed(objects):
            for name in ("current_chronic_path", "_current_chronic_path"):
                value = getattr(obj, name, None)
                if value is not None:
                    return str(value)
        for obj in reversed(objects):
            getter = getattr(obj, "get_id", None)
            if callable(getter):
                try:
                    value = getter()
                except Exception:
                    value = None
                if value is not None:
                    return str(value)
            for name in (
                "path",
                "_path",
                "chronics_name",
                "_chronics_name",
                "current_chronics",
                "_current_chronics",
            ):
                value = getattr(obj, name, None)
                if value is not None:
                    return str(value)
        return "unknown"

    def get_current_chronic_seed(self) -> Any:
        return self._format_chronic_attr(
            self._current_chronic_attr("current_chronic_seed", "_current_chronic_seed")
        )

    def get_current_chronic_index(self) -> Any:
        return self._format_chronic_attr(
            self._current_chronic_attr("current_chronic_index", "_current_chronic_index")
        )

    def get_current_chronic_order_position(self) -> Any:
        return self._format_chronic_attr(
            self._current_chronic_attr(
                "current_chronic_order_position", "_current_chronic_order_position"
            )
        )

    @staticmethod
    def _hash_obs_value(hasher: "hashlib._Hash", name: str, value: Any) -> bool:
        if value is None:
            return False
        try:
            array = np.asarray(value)
        except Exception:
            hasher.update(name.encode("utf-8"))
            hasher.update(repr(value).encode("utf-8"))
            return True

        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        try:
            if array.dtype.kind in ("f", "c"):
                numeric = np.nan_to_num(
                    np.round(array.astype(np.float64), 6),
                    nan=0.0,
                    posinf=1e9,
                    neginf=-1e9,
                )
                hasher.update(numeric.tobytes())
            elif array.dtype.kind in ("b", "i", "u"):
                hasher.update(array.astype(np.int64).tobytes())
            else:
                hasher.update(repr(value).encode("utf-8"))
        except Exception:
            hasher.update(repr(value).encode("utf-8"))
        return True

    def get_current_chronic_fingerprint(self) -> str:
        """Fingerprint the current exogenous grid state seen by the agent.

        The Grid2Op handler label can be stale with some wrapper combinations.
        This hash is based on observation values such as calendar, loads, and
        productions, so it changes when the underlying chronic/time series
        changes even if the public chronic name does not.
        """
        try:
            obs = self._obs
        except Exception:
            return "unknown"

        hasher = hashlib.sha256()
        used_any = False
        for name in (
            "year",
            "month",
            "day",
            "hour_of_day",
            "minute_of_hour",
            "day_of_week",
            "load_p",
            "load_q",
            "gen_p",
            "gen_q",
            "gen_v",
            "prod_p",
            "prod_q",
            "prod_v",
            "rho",
        ):
            used_any = self._hash_obs_value(
                hasher, name, getattr(obs, name, None)
            ) or used_any
        return hasher.hexdigest()[:12] if used_any else "unknown"

    def get_current_chronic_datetime(self) -> str:
        try:
            obs = self._obs
        except Exception:
            return "unknown"

        getter = getattr(obs, "get_time_stamp", None)
        if callable(getter):
            try:
                return str(getter())
            except Exception:
                pass
        parts = []
        for name in ("year", "month", "day", "hour_of_day", "minute_of_hour"):
            value = getattr(obs, name, None)
            if value is not None:
                parts.append(f"{name}={value}")
        return ",".join(parts) if parts else "unknown"

    def get_current_chronic_info(self) -> Dict[str, Any]:
        return {
            "chronic_name": self.current_chronic_name,
            "chronic_path": self.get_current_chronic_path(),
            "chronic_seed": self.get_current_chronic_seed(),
            "chronic_index": self.get_current_chronic_index(),
            "chronic_order_position": self.get_current_chronic_order_position(),
            "chronic_fingerprint": self.current_chronic_fingerprint,
            "chronic_datetime": self.current_chronic_datetime,
            "chronic_reset_count": self.current_chronic_reset_count,
        }

    def seed(self, seed):
        return self.g2op_ma_env.seed(seed)

    def close(self):
        if hasattr(self, "g2op_env"):
            self.g2op_env.close()

    def _format_obs(self, grid2op_obs):
        flat_obs = {
            agent_id: self._aux_observation_space[agent_id].to_gym(
                grid2op_obs[agent_id]
            )
            for agent_id in self.g2op_ma_env.agents
        }

        gym_obs = flat_obs

        if self.norm_obs:
            if not self.eval_env:
                self._update_stats(gym_obs)
            gym_obs = self._normalize(gym_obs)

        if self.use_graph_obs:
            graphs = self.graph_builder.build(self._obs)
            return {
                agent_id: {
                    "flat": gym_obs[agent_id],
                    "graph": graphs[agent_id],
                    "state_graph": graphs["state"],
                }
                for agent_id in self.g2op_ma_env.agents
            }

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

        # return the proper dictionary
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
            normed = ob.astype(np.float32).copy()
            mask = self.obs_norm_masks.get(
                agent_id, np.ones_like(normed, dtype=bool)
            )
            normed[mask] = (ob[mask] - stats["mean"][mask]) / std[mask]
            normed = np.where(np.isfinite(normed), normed, 0.0)
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

    def get_current_max_rho(self) -> float:
        """Return the current Grid2Op max rho before the next action is applied."""
        rho = getattr(self._obs, "rho", None)
        if rho is None:
            return float("nan")
        rho = np.asarray(rho, dtype=np.float32)
        if rho.size == 0 or not np.isfinite(rho).any():
            return float("nan")
        return float(np.nanmax(rho))

    def _make_agent_line_domains(self) -> Dict[str, np.ndarray]:
        """Return line ids each agent can use for local rho heuristics."""
        line_or = np.asarray(
            getattr(self.g2op_env, "line_or_to_subid"), dtype=np.int64
        )
        line_ex = np.asarray(
            getattr(self.g2op_env, "line_ex_to_subid"), dtype=np.int64
        )
        line_domains = {}
        for agent_id, domain_nodes in self.observation_domains.items():
            domain_nodes = np.asarray(domain_nodes, dtype=np.int64)
            mask = np.isin(line_or, domain_nodes) | np.isin(line_ex, domain_nodes)
            line_domains[agent_id] = np.nonzero(mask)[0].astype(np.int64)
        return line_domains

    def get_current_agent_max_rho(self) -> Dict[str, float]:
        """Return each agent's local pre-action max rho."""
        rho = getattr(self._obs, "rho", None)
        if rho is None:
            return {agent_id: float("nan") for agent_id in self._agent_ids}
        rho = np.asarray(rho, dtype=np.float32)
        result = {}
        for agent_id in self._agent_ids:
            line_ids = np.asarray(
                self.agent_line_domains.get(agent_id, []), dtype=np.int64
            )
            if line_ids.size == 0 or rho.size == 0:
                result[agent_id] = float("nan")
                continue
            local_rho = rho[line_ids]
            finite = np.isfinite(local_rho)
            result[agent_id] = (
                float(local_rho[finite].max()) if finite.any() else float("nan")
            )
        return result

    def _line_context(self, line_id: int) -> Dict[str, Any]:
        """Return stable metadata for a Grid2Op line id."""
        line_id = int(line_id)
        if line_id < 0:
            return {
                "line_id": -1,
                "line_name": "",
                "line_or_subid": -1,
                "line_ex_subid": -1,
                "line_agents": [],
            }

        names = getattr(self.g2op_env, "name_line", None)
        line_name = ""
        try:
            if names is not None and line_id < len(names):
                line_name = str(names[line_id])
        except Exception:
            line_name = ""

        def line_endpoint(attr: str) -> int:
            values = getattr(self.g2op_env, attr, None)
            try:
                if values is not None and line_id < len(values):
                    return int(values[line_id])
            except Exception:
                pass
            return -1

        line_agents = []
        for agent_id, line_ids in self.agent_line_domains.items():
            if line_id in set(np.asarray(line_ids, dtype=np.int64).tolist()):
                line_agents.append(agent_id)

        return {
            "line_id": line_id,
            "line_name": line_name,
            "line_or_subid": line_endpoint("line_or_to_subid"),
            "line_ex_subid": line_endpoint("line_ex_to_subid"),
            "line_agents": line_agents,
        }

    def _rho_summary_for_line_ids(self, line_ids: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Return max rho plus line metadata over all lines or a subset."""
        rho = getattr(self._obs, "rho", None)
        rho = np.asarray([] if rho is None else rho, dtype=np.float32)
        if line_ids is None:
            candidate_ids = np.arange(rho.size, dtype=np.int64)
        else:
            candidate_ids = np.asarray(line_ids, dtype=np.int64)
            candidate_ids = candidate_ids[
                (candidate_ids >= 0) & (candidate_ids < rho.size)
            ]

        if rho.size == 0 or candidate_ids.size == 0:
            max_rho = float("nan")
            worst_line = -1
        else:
            candidate_rho = rho[candidate_ids]
            finite = np.isfinite(candidate_rho)
            if finite.any():
                finite_ids = candidate_ids[finite]
                finite_rho = candidate_rho[finite]
                local_idx = int(np.argmax(finite_rho))
                worst_line = int(finite_ids[local_idx])
                max_rho = float(finite_rho[local_idx])
            else:
                max_rho = float("nan")
                worst_line = -1

        summary = {"max_rho": max_rho, "worst_line": worst_line}
        summary.update(self._line_context(worst_line))
        return summary

    def get_current_rho_summary(self) -> Dict[str, Any]:
        """Return current global max rho and the corresponding line metadata."""
        return self._rho_summary_for_line_ids()

    def get_current_agent_rho_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return each agent's local max rho and corresponding line metadata."""
        return {
            agent_id: self._rho_summary_for_line_ids(
                self.agent_line_domains.get(agent_id, np.asarray([], dtype=np.int64))
            )
            for agent_id in self._agent_ids
        }

    def get_explainability_state(self) -> Dict[str, Any]:
        """Return lightweight physical diagnostics for the current Grid2Op state."""
        obs = self._obs
        max_rho, worst_line = self._rho_summary_from_obs(obs)
        topo_vect = getattr(obs, "topo_vect", None)
        topo_vect = np.asarray([] if topo_vect is None else topo_vect)
        topology_distance = (
            float(np.sum(topo_vect != 1)) if topo_vect.size > 0 else float("nan")
        )
        return {
            "max_rho": max_rho,
            "worst_line": worst_line,
            "topology_distance": topology_distance,
        }

    @staticmethod
    def _rho_summary_from_obs(obs: Any) -> Tuple[float, int]:
        """Return max rho and its line id for a Grid2Op observation-like object."""
        rho = getattr(obs, "rho", None)
        rho = np.asarray([] if rho is None else rho, dtype=np.float32)
        finite_rho = np.isfinite(rho)
        if rho.size > 0 and finite_rho.any():
            masked_rho = np.where(finite_rho, rho, -np.inf)
            worst_line = int(np.argmax(masked_rho))
            max_rho = float(masked_rho[worst_line])
        else:
            worst_line = -1
            max_rho = float("nan")
        return max_rho, worst_line

    @staticmethod
    def _action_id_to_int(action_id: Any) -> int:
        if isinstance(action_id, th.Tensor):
            action_id = action_id.detach().cpu().item()
        elif isinstance(action_id, np.ndarray):
            action_id = action_id.reshape(-1)[0].item()
        return int(action_id)

    def _map_action_id_for_agent(self, agent_id: str, action_id: Any) -> int:
        action_id = self._action_id_to_int(action_id)
        mapping = getattr(self, "_reduced_action_id_mapping", {}).get(agent_id)
        if not mapping:
            return action_id
        if action_id < 0 or action_id >= len(mapping):
            raise ValueError(
                f"Reduced action id {action_id} for {agent_id} is outside "
                f"[0, {len(mapping)})."
            )
        return int(mapping[action_id])

    def _get_grid2op_act(self, actions):
        actions = actions or {}
        return {
            agent_id: self._conv_action_space[agent_id].from_gym(
                self._map_action_id_for_agent(agent_id, actions.get(agent_id, 0))
            )
            for agent_id in self.g2op_ma_env.agents
        }

    def _build_global_action_for_simulation(
        self, actions: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        local_actions = self._get_grid2op_act(actions)
        cent_env = self.g2op_ma_env._cent_env
        proposed_action = cent_env.action_space({})

        for agent_id in self.g2op_ma_env.agent_order:
            proposed_action += self.g2op_ma_env._local_action_to_global(
                local_actions[agent_id]
            )

        validation = {
            "action_is_ambiguous": False,
            "action_is_legal": False,
            "action_is_valid": False,
            "validation_error": "",
        }
        try:
            ambiguous, ambiguous_reason = proposed_action.is_ambiguous()
            validation["action_is_ambiguous"] = bool(ambiguous)
            if ambiguous:
                validation["validation_error"] = str(ambiguous_reason)[:500]
                return proposed_action, validation

            proposed_action.get_topological_impact(
                cent_env.get_current_line_status(),
                _store_in_cache=True,
                _read_from_cache=False,
            )
            is_legal, legal_reason = cent_env._game_rules(
                action=proposed_action,
                env=cent_env,
            )
            validation["action_is_legal"] = bool(is_legal)
            validation["action_is_valid"] = bool(is_legal)
            if not is_legal:
                validation["validation_error"] = str(legal_reason)[:500]
        except Exception as exc:
            validation["validation_error"] = f"{type(exc).__name__}: {exc}"[:500]
        return proposed_action, validation

    def simulate_joint_action_outcome(
        self,
        actions: Dict[str, Any],
        *,
        time_step: int = 1,
    ) -> Dict[str, Any]:
        """Simulate a multi-agent action without advancing the environment."""
        rho_before, worst_line_before = self._rho_summary_from_obs(self._obs)
        result = self._empty_action_outcome(rho_before, worst_line_before)

        global_action, validation = self._build_global_action_for_simulation(actions)
        result.update(validation)
        if not validation["action_is_valid"]:
            return result

        return self._simulate_prebuilt_action_outcome(
            obs=self._obs,
            global_action=global_action,
            result=result,
            time_step=time_step,
        )

    def _empty_action_outcome(
        self,
        rho_before: float,
        worst_line_before: int,
    ) -> Dict[str, Any]:
        return {
            "rho_before": rho_before,
            "worst_line_before": worst_line_before,
            "rho_after": float("nan"),
            "worst_line_after": -1,
            "sim_reward": float("nan"),
            "sim_done": False,
            "simulation_succeeded": False,
            "simulation_exception": False,
            "simulation_error": "",
            "action_is_ambiguous": False,
            "action_is_legal": False,
            "action_is_valid": False,
            "validation_error": "",
        }

    def _simulate_prebuilt_action_outcome(
        self,
        *,
        obs: Any,
        global_action: Any,
        result: Dict[str, Any],
        time_step: int,
    ) -> Dict[str, Any]:
        try:
            try:
                sim_obs, sim_reward, sim_done, sim_info = obs.simulate(
                    global_action,
                    time_step=time_step,
                )
            except TypeError:
                sim_obs, sim_reward, sim_done, sim_info = obs.simulate(
                    global_action
                )
            rho_after, worst_line_after = self._rho_summary_from_obs(sim_obs)
            result["rho_after"] = rho_after
            result["worst_line_after"] = worst_line_after
            result["sim_reward"] = float(sim_reward)
            result["sim_done"] = bool(sim_done)
            if isinstance(sim_info, dict):
                exception = sim_info.get("exception", None)
                result["simulation_exception"] = bool(exception)
                if exception:
                    result["simulation_error"] = str(exception)[:500]
            result["simulation_succeeded"] = not result["simulation_exception"]
            result["action_is_valid"] = bool(result["simulation_succeeded"])
        except Exception as exc:
            result["simulation_exception"] = True
            result["simulation_error"] = f"{type(exc).__name__}: {exc}"[:500]
            result["action_is_valid"] = False
        return result

    def simulate_action_outcome(
        self,
        agent_id: str,
        action_id: Any,
        *,
        time_step: int = 1,
    ) -> Dict[str, Any]:
        """Simulate one agent's action while all other agents do nothing."""
        if agent_id not in self.g2op_ma_env.agents:
            raise KeyError(f"Unknown agent_id {agent_id!r}.")
        actions = {other_agent: 0 for other_agent in self.g2op_ma_env.agents}
        actions[agent_id] = self._action_id_to_int(action_id)
        return self.simulate_joint_action_outcome(actions, time_step=time_step)

    def simulate_action_outcomes(
        self,
        requests: List[Dict[str, Any]],
        *,
        time_step: int = 1,
        num_workers: int = 1,
    ) -> List[Dict[str, Any]]:
        """Simulate many unilateral agent/action requests at the current state."""
        if int(num_workers) > 1:
            # Grid2Op / LightSim simulation is not thread-safe on copied
            # observations. Parallel brute-force collection uses process-owned
            # environment replicas in collect_bruteforce_action_outcomes.py.
            pass
        return [
            self.simulate_action_outcome(
                str(request["agent_id"]),
                request["action_id"],
                time_step=time_step,
            )
            for request in requests
        ]

    def decode_action(self, agent_id: str, action_id: int, max_chars: int = 600) -> str:
        """Return a compact human-readable Grid2Op action description."""
        action_id = int(action_id)
        if agent_id not in self._conv_action_space:
            return f"unknown agent {agent_id}"
        exposed_size = int(self.action_space[agent_id].n)
        if action_id < 0 or action_id >= exposed_size:
            return f"invalid action id {action_id}"
        original_action_id = self._map_action_id_for_agent(agent_id, action_id)
        if original_action_id == 0:
            text = "DO-NOTHING"
        else:
            try:
                action = self._conv_action_space[agent_id].from_gym(original_action_id)
                text = " | ".join(
                    line.strip() for line in str(action).splitlines() if line.strip()
                )
                text = re.sub(r"\s+", " ", text).strip()
                if not text:
                    text = str(action).strip() or f"action {original_action_id}"
            except Exception as exc:
                return f"decode error: {type(exc).__name__}: {exc}"
        if getattr(self, "_reduced_action_id_mapping", None):
            text = f"reduced {action_id} -> original {original_action_id}: {text}"
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        return text

    def decode_action_ids(
        self, action_ids_by_agent: Dict[str, List[int]]
    ) -> Dict[str, Dict[int, str]]:
        return {
            agent_id: {
                int(action_id): self.decode_action(agent_id, int(action_id))
                for action_id in action_ids
            }
            for agent_id, action_ids in action_ids_by_agent.items()
        }

    def step(self, actions):
        explain_pre = self.get_explainability_state()
        # convert the action to grid2op
        grid2op_act = self._get_grid2op_act(actions)

        # do a step in the underlying multi agent environment
        obs, r, done, info = self.g2op_ma_env.step(grid2op_act)

        if self.use_heuristic and not done["agent_0"] and not self._risk_overflow:
            obs, heuristic_reward, done, info = self.apply_idle_actions()
            r = {k: v + heuristic_reward for k, v in r.items()}

        self._get_cost(done, info)
        if isinstance(info, dict):
            info.update(self.get_current_chronic_info())
            info[EXPLAIN_INFO_KEY] = {
                "pre": explain_pre,
                "post": self.get_explainability_state(),
            }

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
