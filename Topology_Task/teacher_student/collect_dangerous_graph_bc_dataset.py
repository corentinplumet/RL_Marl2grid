#!/usr/bin/env python3
"""Collect paired graph-BC labels for one or more WCCI action spaces."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

import numpy as np
import torch as th

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from common.utils import cast_np_to_tensors, set_random_seed, str2bool
from alg.mappo.config import get_alg_args
from env.eval import Evaluator
from env.config import get_env_args
from env.utils import _load_reduced_action_id_mapping
from full_test_eval.evaluate_checkpoint import (
    _as_namespace,
    _build_actors,
    _build_zero_shot_transfer_actors,
    _configure_obs_normalization,
    _configure_legacy_connected_feature,
    _extract_obs_stats,
    _load_checkpoint,
    _merge_missing_defaults,
    _repo_relative,
    _resolve_checkpoint_path,
    _resolve_device,
)
from teacher_student.dangerous_graph_bc import (
    DangerousGraphBCWriter,
    best_action_labels,
    choose_concerned_agents,
    copy_actor_observation,
    write_json,
)
from teacher_student.dataset import (
    export_metadata_file,
    metadata_dir,
    metadata_path,
    shards_dir,
)


def _current_chronic_info(evaluator: Evaluator) -> Dict[str, str]:
    getter = getattr(evaluator.env.env, "get_current_chronic_info", None)
    info = getter() if callable(getter) else {}
    return {
        key: str(info.get(key, "unknown"))
        for key in ("chronic_name", "chronic_fingerprint", "chronic_datetime")
    }


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {path} is not empty. Use a new path or "
                "pass --overwrite true."
            )
        allowed_files = {"metadata.json", "summary.json"}
        for child in path.iterdir():
            if child.name in allowed_files:
                child.unlink()
            elif child.is_dir() and child.name in {"shards", "metadata"}:
                shutil.rmtree(child)
            else:
                raise FileExistsError(f"Refusing to delete unexpected path: {child}")
    path.mkdir(parents=True, exist_ok=True)
    shards_dir(path).mkdir(parents=True, exist_ok=True)
    metadata_dir(path).mkdir(parents=True, exist_ok=True)


def _task_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (TASK_DIR / path).resolve()
    return path


def _cli_values(value: Any) -> List[str]:
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, list):
        result: List[str] = []
        for item in value:
            result.extend(_cli_values(item))
        return result
    return [str(value)]


def _args_from_config(config_path: Path) -> Namespace:
    """Parse one normal training TOML through the project argument parsers."""
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    config_args = dict(config.get("args", {}))
    if not config_args:
        raise ValueError(f"Config has no [args] section: {config_path}")

    argv = [str(Path(__file__).name)]
    for key, value in config_args.items():
        if value == "":
            continue
        argv.append("--" + key.replace("_", "-"))
        argv.extend(_cli_values(value))
    argv.extend(str(value) for value in config.get("run", {}).get("extra_args", []))

    main_parser = argparse.ArgumentParser(add_help=False)
    main_parser.add_argument("--alg", default="MAPPO")
    main_parser.add_argument("--seed", type=int, default=0)
    main_parser.add_argument("--cuda", type=str2bool, default=False)
    main_parser.add_argument("--th-deterministic", type=str2bool, default=False)
    main_parser.add_argument("--n-threads", type=int, default=4)
    main_parser.add_argument("--track", type=str2bool, default=False)
    main_parser.add_argument("--checkpoint", type=str2bool, default=False)
    main_parser.add_argument("--verbose", type=str2bool, default=False)

    original_argv = sys.argv
    try:
        sys.argv = argv
        main_args = main_parser.parse_known_args()[0]
        return Namespace(
            **vars(main_args),
            **vars(get_env_args()),
            **vars(get_alg_args()),
        )
    finally:
        sys.argv = original_argv


def _parse_action_space_specs(
    values: List[str], checkpoint_action_space: str
) -> Dict[str, str]:
    if not values:
        if not checkpoint_action_space:
            raise ValueError(
                "The checkpoint has no reduced action space. Pass at least one "
                "--label-action-space LABEL=PATH."
            )
        return {"checkpoint": checkpoint_action_space}

    parsed: Dict[str, str] = {}
    for value in values:
        label, separator, path = str(value).partition("=")
        label = label.strip()
        path = path.strip()
        if not separator or not label or not path:
            raise ValueError(
                "--label-action-space must use LABEL=PATH, for example "
                "mk64=outputs/.../reduced_action_space_..._mk64.json"
            )
        if label in parsed:
            raise ValueError(f"Duplicate action-space label: {label!r}.")
        resolved = _task_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Action-space file does not exist: {resolved}")
        parsed[label] = str(resolved)
    return parsed


def _declared_action_count(path: str) -> int:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    total = 0
    for agent_payload in payload.get("agents", {}).values():
        selected = [int(value) for value in agent_payload.get("selected_action_ids", [])]
        total += len(set(selected).union({0}))
    if total <= 0:
        raise ValueError(f"No selected actions found in {path}.")
    return total


def _prepare_dataset_dirs(
    output_root: Path,
    labels: List[str],
    overwrite: bool,
) -> Dict[str, Path]:
    if len(labels) == 1:
        _prepare_output_dir(output_root, overwrite)
        return {labels[0]: output_root}

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output_root} is not empty. Use a new path or "
                "pass --overwrite true."
            )
        expected = set(labels).union({"metadata", "metadata_exports"})
        unexpected = [child for child in output_root.iterdir() if child.name not in expected]
        if unexpected:
            raise FileExistsError(
                "Refusing to delete unexpected multi-space output paths: "
                + ", ".join(str(path) for path in unexpected)
            )
        for child in output_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_root.mkdir(parents=True, exist_ok=True)
    result = {label: output_root / label for label in labels}
    for path in result.values():
        _prepare_output_dir(path, overwrite=False)
    return result


def _space_indices(
    *,
    action_space_specs: Mapping[str, str],
    agent_ids: List[str],
    original_action_sizes: Dict[str, int],
    simulation_mapping: Dict[str, List[int]],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, List[int]]]]:
    indices: Dict[str, Dict[str, np.ndarray]] = {}
    mappings: Dict[str, Dict[str, List[int]]] = {}
    for label, path in action_space_specs.items():
        mapping, _ = _load_reduced_action_id_mapping(
            path, agent_ids, original_action_sizes
        )
        mappings[label] = mapping
        indices[label] = {}
        for agent in agent_ids:
            reverse = {
                int(original_id): exposed_id
                for exposed_id, original_id in enumerate(simulation_mapping[agent])
            }
            missing = [
                original_id
                for original_id in mapping[agent]
                if int(original_id) not in reverse
            ]
            if missing:
                raise ValueError(
                    f"Action space {label!r} is not a subset of the simulation "
                    f"space for {agent}; missing original ids {missing[:8]}."
                )
            indices[label][agent] = np.asarray(
                [reverse[int(original_id)] for original_id in mapping[agent]],
                dtype=np.int64,
            )
    return indices, mappings


def _subset_outcomes(
    annotated: List[Dict[str, Any]],
    agent: str,
    indices: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    max_to_target = {
        int(max_action): target_action
        for target_action, max_action in enumerate(indices.tolist())
    }
    return [
        {**outcome, "action_id": max_to_target[int(outcome["action_id"])]}
        for outcome in annotated
        if str(outcome["agent_id"]) == agent
        and int(outcome["action_id"]) in max_to_target
    ]


def _policy_logits(
    actors: Dict[str, Any],
    obs: Dict[str, Any],
    device: th.device,
) -> Dict[str, np.ndarray]:
    obs_tensors = cast_np_to_tensors(obs, device)
    logits: Dict[str, np.ndarray] = {}
    with th.no_grad():
        for agent, actor in actors.items():
            agent_logits = actor._actor_logits(obs_tensors[agent])
            logits[agent] = (
                agent_logits.reshape(-1, agent_logits.shape[-1])[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=True)
            )
    return logits


def _metadata_payload(
    *,
    status: str,
    cli: Namespace,
    checkpoint_path: Path | None,
    checkpoint_step: int | None,
    source_config: Path | None,
    has_policy_logits: bool,
    args: Namespace,
    obs_norm_mode: str,
    action_space_label: str,
    reduced_action_space: str,
    action_id_mapping: Dict[str, List[int]],
    agent_ids: List[str],
    action_sizes: Dict[str, int],
    simulation_action_space_label: str,
    simulation_reduced_action_space: str,
    simulation_action_sizes: Dict[str, int],
    rollout_action_space_label: str,
    writer: DangerousGraphBCWriter,
    env_steps: int,
    completed_episodes: int,
    skipped_safe_states: int,
    skipped_capped_dangerous_states: int,
    skipped_gap_dangerous_states: int,
    unique_fingerprints: set[str],
    metrics: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    return {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector": "teacher_student.collect_dangerous_graph_bc_dataset",
        "dataset_mode": "dangerous_graph_bc",
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_global_step": (
            int(checkpoint_step) if checkpoint_step is not None else None
        ),
        "source_config": str(source_config) if source_config is not None else None,
        "has_policy_logits": bool(has_policy_logits),
        "reference_logits_kind": (
            "checkpoint_policy" if has_policy_logits else "unavailable_zero_placeholder"
        ),
        "checkpoint_exp_tag": str(getattr(args, "exp_tag", "")),
        "env_id": str(getattr(args, "env_id", "")),
        "action_space_label": action_space_label,
        "reduced_action_space": reduced_action_space,
        "action_id_mapping": action_id_mapping,
        "simulation_action_space_label": simulation_action_space_label,
        "simulation_reduced_action_space": simulation_reduced_action_space,
        "simulation_action_sizes": simulation_action_sizes,
        "actor_encoder": str(getattr(args, "actor_encoder", "")),
        "actor_action_head": str(getattr(args, "actor_action_head", "")),
        "gnn_graph_type": str(getattr(args, "gnn_graph_type", "")),
        "gnn_physical_scaling": bool(getattr(args, "gnn_physical_scaling", False)),
        "gnn_running_norm": bool(getattr(args, "gnn_running_norm", False)),
        "obs_normalization": obs_norm_mode,
        "split": cli.split,
        "split_chronics": bool(cli.split_chronics),
        "eval_all_split_chronics": bool(cli.eval_all_split_chronics),
        "chronic_sample_seed": int(cli.chronic_sample_seed),
        "max_dangerous_states": cli.max_dangerous_states,
        "max_dangerous_states_per_episode": cli.max_dangerous_states_per_episode,
        "min_dangerous_query_gap": int(cli.min_dangerous_query_gap),
        "danger_rho_threshold": float(cli.danger_rho_threshold),
        "local_rho_threshold": float(cli.local_rho_threshold),
        "min_improvement": float(cli.min_improvement),
        "outcome_time_step": int(cli.outcome_time_step),
        "rollout_policy": cli.rollout_policy,
        "rollout_action_space_label": rollout_action_space_label,
        "agent_ids": agent_ids,
        "action_sizes": action_sizes,
        "paired_action_space_collection": True,
        "label_rule": (
            "best valid non-terminal unilateral action if it rescues terminal "
            "do-nothing or lowers next-step max rho by min_improvement; else action 0"
        ),
        "negative_rule": "unconcerned agents and non-improving concerned agents use action 0",
        "n_env_steps": int(env_steps),
        "n_dangerous_states": int(writer.total_states),
        "n_skipped_safe_states": int(skipped_safe_states),
        "n_skipped_capped_dangerous_states": int(
            skipped_capped_dangerous_states
        ),
        "n_skipped_gap_dangerous_states": int(skipped_gap_dangerous_states),
        "n_completed_episodes": int(completed_episodes),
        "n_unique_chronic_fingerprints": int(len(unique_fingerprints)),
        "n_shards": int(len(writer.paths)),
        "shards": [str(path) for path in writer.paths],
        "agent_summary": metrics,
        "layout": {
            "version": 2,
            "shards_dir": "shards",
            "metadata_file": "metadata/metadata.json",
        },
    }


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        help=(
            "Optional policy checkpoint for DAgger-style state visitation and "
            "reference logits. Use --config for checkpoint-free collection."
        ),
    )
    source.add_argument(
        "--config",
        type=Path,
        help=(
            "Training TOML that defines the WCCI environment and graph-observation "
            "schema. No model weights are loaded in this mode."
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=TASK_DIR / "checkpoint")
    parser.add_argument(
        "--label-action-space",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Reduced action space for which labels are saved. Repeat the option "
            "to collect paired datasets, e.g. mk64=...mk64.json and "
            "mk256=...mk256.json. The largest supplied space is simulated."
        ),
    )
    parser.add_argument(
        "--rollout-action-space",
        default=None,
        help=(
            "Action-space label used by checkpoint and best_simulated rollouts. "
            "Defaults to the checkpoint's matching space when supplied, otherwise "
            "to the largest space."
        ),
    )
    parser.add_argument("--transfer-action-head-source-agent", default="agent_0")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--split-chronics", type=str2bool, default=True)
    parser.add_argument("--eval-all-split-chronics", type=str2bool, default=False)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--chronic-sample-seed", type=int, default=0)
    parser.add_argument("--max-dangerous-states", type=int, default=None)
    parser.add_argument(
        "--max-dangerous-states-per-episode", type=int, default=None
    )
    parser.add_argument("--min-dangerous-query-gap", type=int, default=1)
    parser.add_argument("--danger-rho-threshold", type=float, default=0.90)
    parser.add_argument("--local-rho-threshold", type=float, default=None)
    parser.add_argument("--min-improvement", type=float, default=0.001)
    parser.add_argument("--outcome-time-step", type=int, default=1)
    parser.add_argument(
        "--rollout-policy",
        choices=["auto", "checkpoint", "best_simulated", "do_nothing"],
        default="auto",
    )
    parser.add_argument(
        "--obs-normalization",
        choices=["auto", "disable", "require"],
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--compress", type=str2bool, default=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.local_rho_threshold is None:
        cli.local_rho_threshold = cli.danger_rho_threshold
    if cli.shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    if cli.min_improvement < 0:
        raise ValueError("--min-improvement must be non-negative.")
    for name in ("max_dangerous_states", "max_dangerous_states_per_episode"):
        value = getattr(cli, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if cli.min_dangerous_query_gap <= 0:
        raise ValueError("--min-dangerous-query-gap must be positive.")

    checkpoint_path: Path | None = None
    source_config: Path | None = None
    first_record: Dict[str, Any] | None = None
    if cli.checkpoint:
        checkpoint_dir = cli.checkpoint_dir.expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = (TASK_DIR / checkpoint_dir).resolve()
        checkpoint_path = _resolve_checkpoint_path(cli.checkpoint, checkpoint_dir)
        first_record = _load_checkpoint(checkpoint_path, th.device("cpu"))
        args = _merge_missing_defaults(_as_namespace(first_record["args"]))
        args = _configure_legacy_connected_feature(args, first_record)
        cli.rollout_policy = (
            "checkpoint" if cli.rollout_policy == "auto" else cli.rollout_policy
        )
    else:
        source_config = cli.config.expanduser()
        if not source_config.is_absolute():
            source_config = (TASK_DIR / source_config).resolve()
        if not source_config.exists():
            raise FileNotFoundError(f"Config does not exist: {source_config}")
        args = _args_from_config(source_config)
        cli.rollout_policy = (
            "best_simulated"
            if cli.rollout_policy == "auto"
            else cli.rollout_policy
        )
        if cli.rollout_policy == "checkpoint":
            raise ValueError(
                "--rollout-policy checkpoint requires --checkpoint. With --config, "
                "use best_simulated or do_nothing."
            )
    checkpoint_action_space = str(getattr(args, "reduced_action_space", "") or "")
    action_space_specs = _parse_action_space_specs(
        cli.label_action_space, checkpoint_action_space
    )
    simulation_label = max(
        action_space_specs,
        key=lambda label: _declared_action_count(action_space_specs[label]),
    )
    simulation_action_space = action_space_specs[simulation_label]
    checkpoint_space_matches = [
        label
        for label, path in action_space_specs.items()
        if checkpoint_path is not None
        and checkpoint_action_space
        and _task_path(path) == _task_path(checkpoint_action_space)
    ]
    rollout_action_space = (
        cli.rollout_action_space
        or (checkpoint_space_matches[0] if checkpoint_space_matches else simulation_label)
    )
    if rollout_action_space not in action_space_specs:
        raise ValueError(
            f"Unknown --rollout-action-space {rollout_action_space!r}; choose one "
            f"of {sorted(action_space_specs)}."
        )

    output_dir = cli.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (TASK_DIR / output_dir).resolve()
    dataset_dirs = _prepare_dataset_dirs(
        output_dir, list(action_space_specs), cli.overwrite
    )

    args.reduced_action_space = simulation_action_space
    args.track = False
    args.split_chronics = bool(cli.split_chronics)
    args.eval_all_split_chronics = bool(cli.eval_all_split_chronics)
    args.deterministic_eval = True
    args.eval_action_heuristic = "none"
    args.trace_rollout_actions = False
    args.eval_progress_print = False
    if cli.max_episodes is not None:
        args.eval_episodes = int(cli.max_episodes)
    if cli.n_threads is not None:
        args.n_threads = int(cli.n_threads)
    args.n_threads = max(int(getattr(args, "n_threads", 1)), 1)

    required = {
        "WCCI no-maintenance environment": str(getattr(args, "env_id", ""))
        == "bus36_wcci_nomaint",
        "graph actor": str(getattr(args, "actor_encoder", "")) == "gnn",
        "candidate action head": str(getattr(args, "actor_action_head", ""))
        == "candidate_pool",
        "reduced action space": bool(str(getattr(args, "reduced_action_space", ""))),
    }
    missing = [name for name, ok in required.items() if not ok]
    if missing:
        raise ValueError(
            "Collection source is incompatible with graph BC collection: "
            + ", ".join(missing)
        )

    obs_stats = _extract_obs_stats(first_record) if first_record is not None else {}
    if first_record is not None:
        args, obs_stats, obs_norm_mode = _configure_obs_normalization(
            args, obs_stats, cli.obs_normalization
        )
    else:
        if cli.obs_normalization == "require":
            raise ValueError(
                "--obs-normalization require needs checkpoint normalization "
                "statistics. Use disable or auto for checkpoint-free collection."
            )
        args.norm_obs = False
        args.gnn_running_norm = False
        obs_stats = {}
        obs_norm_mode = "disabled_checkpoint_free"
    set_random_seed(int(getattr(args, "seed", 0)))
    device = _resolve_device(args, cli.device)
    record = (
        _load_checkpoint(checkpoint_path, device)
        if checkpoint_path is not None
        else None
    )
    evaluator = Evaluator(
        args,
        logger=None,
        device=device,
        chronic_split=cli.split,
        metric_prefix="dangerous_graph_bc",
    )
    if obs_stats:
        evaluator.env.env.set_obs_stats(obs_stats)
    actors: Dict[str, Any] = {}
    if checkpoint_path is not None:
        same_action_space = bool(
            checkpoint_action_space
            and _task_path(checkpoint_action_space) == Path(simulation_action_space)
        )
        if same_action_space:
            actors = _build_actors(record, args, evaluator, device)
        else:
            transferable = {
                "share_actor_gnn": bool(getattr(args, "share_actor_gnn", False)),
                "share_candidate_scorer": bool(
                    getattr(args, "share_candidate_scorer", False)
                ),
                "gnn_concat_flat=false": not bool(
                    getattr(args, "gnn_concat_flat", False)
                ),
            }
            missing_transfer = [name for name, ok in transferable.items() if not ok]
            if missing_transfer:
                raise ValueError(
                    "Retargeting the checkpoint to the largest action space requires "
                    "a transferable shared actor; missing: "
                    + ", ".join(missing_transfer)
                )
            actors, _ = _build_zero_shot_transfer_actors(
                checkpoint_path,
                args,
                evaluator,
                device,
                cli.transfer_action_head_source_agent,
            )
        agent_ids = sorted(actors)
    else:
        agent_ids = [
            f"agent_{index}"
            for index in range(len(evaluator.env.env.observation_space.keys()))
        ]
    simulation_action_sizes = {
        agent: int(evaluator.env.env.action_space[agent].n) for agent in agent_ids
    }
    original_action_sizes = dict(evaluator.env.env._original_action_space_sizes)
    simulation_mapping = {
        agent: list(evaluator.env.env._reduced_action_id_mapping[agent])
        for agent in agent_ids
    }
    space_indices, action_id_mappings = _space_indices(
        action_space_specs=action_space_specs,
        agent_ids=agent_ids,
        original_action_sizes=original_action_sizes,
        simulation_mapping=simulation_mapping,
    )
    action_sizes = {
        label: {
            agent: int(len(space_indices[label][agent])) for agent in agent_ids
        }
        for label in action_space_specs
    }

    writers = {
        label: DangerousGraphBCWriter(
            shards_dir(dataset_dirs[label]), agent_ids, cli.shard_size, cli.compress
        )
        for label in action_space_specs
    }
    metrics = {
        label: {
            agent: {
                "dangerous_examples": 0,
                "concerned": 0,
                "nonidle_targets": 0,
            }
            for agent in agent_ids
        }
        for label in action_space_specs
    }
    target_episodes = int(cli.max_episodes or evaluator.eval_episodes)
    if cli.split_chronics and not cli.eval_all_split_chronics:
        evaluator.env.reshuffle_chronics(seed=cli.chronic_sample_seed)
    obs, _ = evaluator.env.reset()
    env_steps = 0
    completed_episodes = 0
    episode_step = 0
    skipped_safe_states = 0
    skipped_capped_dangerous_states = 0
    skipped_gap_dangerous_states = 0
    episode_dangerous_states = 0
    last_dangerous_query_step = -int(cli.min_dangerous_query_gap)
    unique_fingerprints: set[str] = set()
    started = time.perf_counter()
    completed_normally = False

    print("========== Dangerous-state graph BC collection ==========")
    if checkpoint_path is not None:
        print(f"Source checkpoint: {_repo_relative(checkpoint_path)}")
    else:
        print(f"Source config (no weights): {_repo_relative(source_config)}")
    print(f"Output: {output_dir}")
    print(f"Split / episodes: {cli.split} / {target_episodes}")
    print(
        "Sampling: "
        f"seed={cli.chronic_sample_seed} max_states={cli.max_dangerous_states} "
        f"per_episode={cli.max_dangerous_states_per_episode} "
        f"min_gap={cli.min_dangerous_query_gap}"
    )
    print(f"Danger/local rho: {cli.danger_rho_threshold} / {cli.local_rho_threshold}")
    print(f"Min improvement: {cli.min_improvement}")
    print(
        f"Rollout policy/action space: {cli.rollout_policy}/{rollout_action_space}"
    )
    print(f"Simulation action space: {simulation_label} {simulation_action_sizes}")
    for label in action_space_specs:
        print(f"Label action space: {label} {action_sizes[label]}")
    print("===========================================================")

    try:
        while completed_episodes < target_episodes:
            if cli.max_env_steps is not None and env_steps >= cli.max_env_steps:
                break
            dangerous_states = next(iter(writers.values())).total_states
            if (
                cli.max_dangerous_states is not None
                and dangerous_states >= cli.max_dangerous_states
            ):
                break

            chronic = _current_chronic_info(evaluator)
            unique_fingerprints.add(chronic["chronic_fingerprint"])
            global_rho = float(evaluator.env.get_current_max_rho())
            local_rhos = {
                agent: float(value)
                for agent, value in evaluator.env.get_current_agent_max_rho().items()
            }
            if actors:
                policy_logits = _policy_logits(actors, obs, device)
                rollout_policy_actions = {}
                for agent in agent_ids:
                    rollout_indices = space_indices[rollout_action_space][agent]
                    restricted_logits = policy_logits[agent][rollout_indices]
                    restricted_action = int(np.argmax(restricted_logits))
                    rollout_policy_actions[agent] = int(
                        rollout_indices[restricted_action]
                    )
            else:
                # The writer keeps fixed fields for compatibility with the BC
                # trainer. These are explicitly marked as unavailable in metadata
                # and must never be used for a preservation/distillation loss.
                policy_logits = {
                    agent: np.zeros(simulation_action_sizes[agent], dtype=np.float32)
                    for agent in agent_ids
                }
                rollout_policy_actions = {agent: 0 for agent in agent_ids}

            is_dangerous = bool(
                np.isfinite(global_rho)
                and global_rho >= cli.danger_rho_threshold
            )
            episode_cap_reached = bool(
                cli.max_dangerous_states_per_episode is not None
                and episode_dangerous_states
                >= cli.max_dangerous_states_per_episode
            )
            gap_satisfied = bool(
                episode_step - last_dangerous_query_step
                >= cli.min_dangerous_query_gap
            )

            rollout_actions: Dict[str, Any]
            if is_dangerous and not episode_cap_reached and gap_satisfied:
                concerned, used_fallback = choose_concerned_agents(
                    local_rhos, cli.local_rho_threshold
                )
                do_nothing = evaluator.env.simulate_action_outcomes(
                    [{"agent_id": agent_ids[0], "action_id": 0}],
                    time_step=cli.outcome_time_step,
                )[0]
                requests = [
                    {"agent_id": agent, "action_id": action_id}
                    for agent in concerned
                    for action_id in range(1, simulation_action_sizes[agent])
                ]
                outcomes = evaluator.env.simulate_action_outcomes(
                    requests, time_step=cli.outcome_time_step
                )
                annotated = [
                    {
                        **outcome,
                        "agent_id": request["agent_id"],
                        "action_id": request["action_id"],
                    }
                    for request, outcome in zip(requests, outcomes)
                ]
                labels_by_space: Dict[str, Dict[str, Dict[str, Any]]] = {}
                for space_label in action_space_specs:
                    target_outcomes = {
                        agent: _subset_outcomes(
                            annotated, agent, space_indices[space_label][agent]
                        )
                        for agent in agent_ids
                    }
                    labels = best_action_labels(
                        agent_ids=agent_ids,
                        concerned_agents=concerned,
                        outcomes=[
                            outcome
                            for agent_outcomes in target_outcomes.values()
                            for outcome in agent_outcomes
                        ],
                        do_nothing_outcome=do_nothing,
                        min_improvement=cli.min_improvement,
                    )
                    labels_by_space[space_label] = labels
                    state_id = writers[space_label].total_states
                    agent_values = {}
                    for agent in agent_ids:
                        target = labels[agent]
                        is_concerned = agent in concerned
                        metrics[space_label][agent]["dangerous_examples"] += 1
                        metrics[space_label][agent]["concerned"] += int(
                            is_concerned
                        )
                        metrics[space_label][agent]["nonidle_targets"] += int(
                            target["target_is_nonidle"]
                        )
                        indices = space_indices[space_label][agent]
                        restricted_logits = policy_logits[agent][indices]
                        agent_values[agent] = {
                            "obs": copy_actor_observation(obs[agent]),
                            "policy_action": int(np.argmax(restricted_logits)),
                            "policy_logits": restricted_logits,
                            "concerned": is_concerned,
                            "local_max_rho": local_rhos.get(agent, float("nan")),
                            **target,
                        }
                    flushed = writers[space_label].append(
                        row={
                            "state_id": state_id,
                            "episode_id": completed_episodes,
                            "episode_step": episode_step,
                            "dataset_step": env_steps,
                            "global_max_rho": global_rho,
                            "concerned_fallback": used_fallback,
                            **chronic,
                        },
                        agent_values=agent_values,
                    )
                    if flushed is not None:
                        print(
                            f"wrote {space_label}: {flushed} "
                            f"states={writers[space_label].total_states}",
                            flush=True,
                        )
                episode_dangerous_states += 1
                last_dangerous_query_step = episode_step

                if cli.rollout_policy == "best_simulated":
                    rollout_labels = labels_by_space[rollout_action_space]
                    candidates = [
                        (
                            float(rollout_labels[agent]["best_rho_after"]),
                            agent,
                            int(rollout_labels[agent]["target_action"]),
                        )
                        for agent in concerned
                        if rollout_labels[agent]["target_is_nonidle"]
                    ]
                    rollout_actions = {agent: 0 for agent in agent_ids}
                    if candidates:
                        _, selected_agent, selected_action = min(candidates)
                        rollout_actions[selected_agent] = int(
                            space_indices[rollout_action_space][selected_agent][
                                selected_action
                            ]
                        )
                elif cli.rollout_policy == "do_nothing":
                    rollout_actions = {agent: 0 for agent in agent_ids}
                else:
                    rollout_actions = rollout_policy_actions
            else:
                if not is_dangerous:
                    skipped_safe_states += 1
                elif episode_cap_reached:
                    skipped_capped_dangerous_states += 1
                else:
                    skipped_gap_dangerous_states += 1
                rollout_actions = (
                    rollout_policy_actions
                    if cli.rollout_policy == "checkpoint"
                    else {agent: 0 for agent in agent_ids}
                )

            next_obs, _, terminations, truncations, _ = evaluator.env.step(rollout_actions)
            done = bool(terminations[agent_ids[0]] or truncations[agent_ids[0]])
            env_steps += 1
            episode_step += 1
            if done:
                completed_episodes += 1
                obs, _ = evaluator.env.reset()
                episode_step = 0
                episode_dangerous_states = 0
                last_dangerous_query_step = -int(cli.min_dangerous_query_gap)
            else:
                obs = next_obs

            if env_steps == 1 or env_steps % max(cli.progress_every, 1) == 0:
                elapsed = time.perf_counter() - started
                dangerous_states = next(iter(writers.values())).total_states
                print(
                    f"steps={env_steps} episodes={completed_episodes}/{target_episodes} "
                    f"dangerous={dangerous_states} elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )
        completed_normally = True
    finally:
        metadata_by_space = {}
        for space_label, writer in writers.items():
            final_shard = writer.flush()
            if final_shard is not None:
                print(
                    f"wrote {space_label}: {final_shard} "
                    f"states={writer.total_states}",
                    flush=True,
                )
            payload = _metadata_payload(
                status="complete" if completed_normally else "interrupted",
                cli=cli,
                checkpoint_path=checkpoint_path,
                checkpoint_step=(
                    int(record.get("global_step", 0))
                    if record is not None
                    else None
                ),
                source_config=source_config,
                has_policy_logits=bool(actors),
                args=args,
                obs_norm_mode=obs_norm_mode,
                action_space_label=space_label,
                reduced_action_space=action_space_specs[space_label],
                action_id_mapping=action_id_mappings[space_label],
                agent_ids=agent_ids,
                action_sizes=action_sizes[space_label],
                simulation_action_space_label=simulation_label,
                simulation_reduced_action_space=simulation_action_space,
                simulation_action_sizes=simulation_action_sizes,
                rollout_action_space_label=rollout_action_space,
                writer=writer,
                env_steps=env_steps,
                completed_episodes=completed_episodes,
                skipped_safe_states=skipped_safe_states,
                skipped_capped_dangerous_states=skipped_capped_dangerous_states,
                skipped_gap_dangerous_states=skipped_gap_dangerous_states,
                unique_fingerprints=unique_fingerprints,
                metrics=metrics[space_label],
            )
            meta_path = metadata_path(dataset_dirs[space_label])
            write_json(meta_path, payload)
            export_metadata_file(dataset_dirs[space_label], meta_path, "metadata")
            metadata_by_space[space_label] = str(meta_path)

        if len(writers) > 1:
            write_json(
                output_dir / "metadata" / "metadata.json",
                {
                    "status": "complete" if completed_normally else "interrupted",
                    "dataset_mode": "dangerous_graph_bc_multi_action_space",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "simulation_action_space_label": simulation_label,
                    "simulation_reduced_action_space": simulation_action_space,
                    "rollout_action_space_label": rollout_action_space,
                    "checkpoint": (
                        str(checkpoint_path) if checkpoint_path is not None else None
                    ),
                    "source_config": (
                        str(source_config) if source_config is not None else None
                    ),
                    "has_policy_logits": bool(actors),
                    "paired_action_spaces": metadata_by_space,
                    "target_episodes": target_episodes,
                    "chronic_sample_seed": cli.chronic_sample_seed,
                    "max_dangerous_states": cli.max_dangerous_states,
                    "max_dangerous_states_per_episode": (
                        cli.max_dangerous_states_per_episode
                    ),
                    "min_dangerous_query_gap": cli.min_dangerous_query_gap,
                    "n_dangerous_states": next(iter(writers.values())).total_states,
                },
            )
        evaluator.env.close()

    print("========== Collection complete ==========")
    for space_label, writer in writers.items():
        print(
            f"{space_label}: dangerous_states={writer.total_states} "
            f"shards={len(writer.paths)} "
            f"metadata={metadata_path(dataset_dirs[space_label])}"
        )


if __name__ == "__main__":
    main()
