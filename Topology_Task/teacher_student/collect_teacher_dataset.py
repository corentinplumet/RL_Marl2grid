#!/usr/bin/env python3
"""Collect teacher-student imitation data from a heuristic-augmented policy."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from argparse import Namespace
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from common.action_trace import tensor_scalar_to_float, tensor_scalar_to_int
from common.utils import cast_np_to_tensors, set_random_seed, str2bool
from env.eval import Evaluator
from full_test_eval.evaluate_checkpoint import (
    _as_namespace,
    _build_actors,
    _configure_obs_normalization,
    _extract_obs_stats,
    _find_checkpoint_by_model_step,
    _load_checkpoint,
    _merge_missing_defaults,
    _repo_relative,
    _resolve_checkpoint_path,
    _resolve_device,
)
from teacher_student.dataset import (
    METADATA_DIR_NAME,
    SHARDS_DIR_NAME,
    metadata_dir,
    metadata_path,
    shards_dir,
)


def _safe_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(TASK_DIR))
    except ValueError:
        return str(path.resolve())


def _str_or_unknown(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _current_chronic_info(evaluator: Evaluator) -> Dict[str, str]:
    env = evaluator.env.env
    getter = getattr(env, "get_current_chronic_info", None)
    if callable(getter):
        info = getter()
    else:
        info = {}
    return {
        "chronic_name": _str_or_unknown(info.get("chronic_name")),
        "chronic_fingerprint": _str_or_unknown(info.get("chronic_fingerprint")),
        "chronic_datetime": _str_or_unknown(info.get("chronic_datetime")),
    }


def _teacher_override_decision(
    evaluator: Evaluator,
    agent_ids: List[str],
) -> Tuple[Dict[str, bool], Dict[str, float], float]:
    """Mirror Evaluator's eval-action heuristic and return diagnostics."""
    heuristic = str(getattr(evaluator, "eval_action_heuristic", "none"))
    threshold = float(getattr(evaluator, "eval_action_rho_threshold", 0.90))
    force_noop = {agent: False for agent in agent_ids}
    local_max_rhos = {agent: float("nan") for agent in agent_ids}
    global_max_rho = float("nan")

    try:
        global_max_rho = float(evaluator.env.get_current_max_rho())
    except Exception:
        global_max_rho = float("nan")

    try:
        local_max = evaluator.env.get_current_agent_max_rho()
        for agent in agent_ids:
            local_max_rhos[agent] = float(local_max.get(agent, float("nan")))
    except Exception:
        pass

    if heuristic == "none":
        return force_noop, local_max_rhos, global_max_rho

    if heuristic == "rho_threshold":
        force = np.isfinite(global_max_rho) and global_max_rho < threshold
        return {agent: bool(force) for agent in agent_ids}, local_max_rhos, global_max_rho

    if heuristic == "local_rho_threshold":
        for agent in agent_ids:
            max_rho = local_max_rhos[agent]
            force_noop[agent] = bool(np.isfinite(max_rho) and max_rho < threshold)
        return force_noop, local_max_rhos, global_max_rho

    raise ValueError(f"Unsupported eval action heuristic: {heuristic!r}.")


def _flat_obs_for_storage(obs: Dict[str, Any], agent_id: str) -> np.ndarray:
    value = obs[agent_id]
    if isinstance(value, dict):
        raise NotImplementedError(
            "Phase 1 teacher-student collection currently supports flat MLP "
            "observations only. The selected checkpoint uses nested graph "
            f"observations for {agent_id}."
        )
    return np.asarray(value, dtype=np.float32).copy()


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {_safe_path(path)} already exists and is not empty. "
                "Use a new --output-dir or pass --overwrite true."
            )
        for child in path.iterdir():
            if child.name == "metadata.json" or child.name == "summary.json":
                child.unlink()
            elif child.name.startswith("shard_") and child.suffix == ".npz":
                child.unlink()
            elif child.is_dir() and child.name in {"tmp", SHARDS_DIR_NAME, METADATA_DIR_NAME}:
                shutil.rmtree(child)
            else:
                raise FileExistsError(
                    f"Refusing to delete unexpected file in output dir: {child}"
                )
    path.mkdir(parents=True, exist_ok=True)
    shards_dir(path).mkdir(parents=True, exist_ok=True)
    metadata_dir(path).mkdir(parents=True, exist_ok=True)


class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        agent_ids: List[str],
        shard_size: int,
        compress: bool,
    ) -> None:
        self.output_dir = output_dir
        self.agent_ids = agent_ids
        self.shard_size = int(shard_size)
        self.compress = bool(compress)
        self.shard_idx = 0
        self.rows: List[Dict[str, Any]] = []
        self.agent_rows: Dict[str, Dict[str, List[Any]]] = {
            agent: defaultdict(list) for agent in agent_ids
        }
        self.paths: List[Path] = []

    def __len__(self) -> int:
        return len(self.rows)

    def append(
        self,
        *,
        row: Dict[str, Any],
        agent_values: Dict[str, Dict[str, Any]],
    ) -> Optional[Path]:
        self.rows.append(row)
        for agent, values in agent_values.items():
            for key, value in values.items():
                self.agent_rows[agent][key].append(value)
        if len(self.rows) >= self.shard_size:
            return self.flush()
        return None

    def flush(self) -> Optional[Path]:
        if not self.rows:
            return None

        arrays: Dict[str, Any] = {
            "episode_id": np.asarray([r["episode_id"] for r in self.rows], dtype=np.int64),
            "episode_step": np.asarray(
                [r["episode_step"] for r in self.rows], dtype=np.int32
            ),
            "dataset_step": np.asarray(
                [r["dataset_step"] for r in self.rows], dtype=np.int64
            ),
            "global_max_rho": np.asarray(
                [r["global_max_rho"] for r in self.rows], dtype=np.float32
            ),
            "reward_agent_0": np.asarray(
                [r["reward_agent_0"] for r in self.rows], dtype=np.float32
            ),
            "done": np.asarray([r["done"] for r in self.rows], dtype=bool),
            "done_episode_length": np.asarray(
                [r["done_episode_length"] for r in self.rows], dtype=np.int32
            ),
            "chronic_name": np.asarray(
                [r["chronic_name"] for r in self.rows], dtype=str
            ),
            "chronic_fingerprint": np.asarray(
                [r["chronic_fingerprint"] for r in self.rows], dtype=str
            ),
            "chronic_datetime": np.asarray(
                [r["chronic_datetime"] for r in self.rows], dtype=str
            ),
        }

        for agent in self.agent_ids:
            values = self.agent_rows[agent]
            arrays[f"obs_{agent}"] = np.stack(values["obs"]).astype(np.float32)
            arrays[f"policy_action_{agent}"] = np.asarray(
                values["policy_action"], dtype=np.int32
            )
            arrays[f"teacher_action_{agent}"] = np.asarray(
                values["teacher_action"], dtype=np.int32
            )
            arrays[f"force_noop_{agent}"] = np.asarray(
                values["force_noop"], dtype=bool
            )
            arrays[f"was_overwritten_{agent}"] = np.asarray(
                values["was_overwritten"], dtype=bool
            )
            arrays[f"local_max_rho_{agent}"] = np.asarray(
                values["local_max_rho"], dtype=np.float32
            )

        path = self.output_dir / f"shard_{self.shard_idx:05d}.npz"
        if self.compress:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)

        self.paths.append(path)
        self.shard_idx += 1
        self.rows = []
        self.agent_rows = {agent: defaultdict(list) for agent in self.agent_ids}
        return path


def _agent_metric_template() -> Dict[str, int]:
    return {
        "n": 0,
        "policy_action_0": 0,
        "policy_nonidle": 0,
        "teacher_action_0": 0,
        "teacher_nonidle": 0,
        "force_noop": 0,
        "was_overwritten": 0,
        "overwrite_to_action0": 0,
    }


def _metadata(
    *,
    status: str,
    cli: Namespace,
    checkpoint_path: Path,
    checkpoint_step: int,
    args: Namespace,
    obs_norm_mode: str,
    obs_stats_available: bool,
    agent_ids: List[str],
    action_sizes: Dict[str, int],
    obs_shapes: Dict[str, List[int]],
    writer: ShardWriter,
    env_steps: int,
    completed_episodes: int,
    unique_fingerprints: set,
    episode_lengths: List[int],
    agent_metrics: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    agent_summary = {}
    for agent, metrics in agent_metrics.items():
        n = max(metrics["n"], 1)
        agent_summary[agent] = {
            **metrics,
            "policy_action_0_frac": metrics["policy_action_0"] / n,
            "policy_nonidle_frac": metrics["policy_nonidle"] / n,
            "teacher_action_0_frac": metrics["teacher_action_0"] / n,
            "teacher_nonidle_frac": metrics["teacher_nonidle"] / n,
            "force_noop_frac": metrics["force_noop"] / n,
            "was_overwritten_frac": metrics["was_overwritten"] / n,
            "overwrite_to_action0_frac": metrics["overwrite_to_action0"] / n,
        }

    return {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector": "teacher_student.collect_teacher_dataset",
        "phase": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint_step),
        "exp_tag": getattr(args, "exp_tag", ""),
        "env_id": getattr(args, "env_id", ""),
        "seed": int(getattr(args, "seed", 0)),
        "split": cli.split,
        "split_chronics": bool(args.split_chronics),
        "eval_all_split_chronics": bool(args.eval_all_split_chronics),
        "deterministic_eval": bool(args.deterministic_eval),
        "eval_action_heuristic": getattr(args, "eval_action_heuristic", "none"),
        "eval_action_rho_threshold": float(
            getattr(args, "eval_action_rho_threshold", 0.90)
        ),
        "obs_normalization": obs_norm_mode,
        "obs_stats_available": bool(obs_stats_available),
        "norm_obs_effective": bool(getattr(args, "norm_obs", False)),
        "agent_ids": agent_ids,
        "action_sizes": action_sizes,
        "obs_shapes": obs_shapes,
        "shard_size": int(cli.shard_size),
        "compress": bool(cli.compress),
        "layout": {
            "version": 2,
            "shards_dir": SHARDS_DIR_NAME,
            "metadata_dir": METADATA_DIR_NAME,
            "metadata_file": f"{METADATA_DIR_NAME}/metadata.json",
            "summary_file": f"{METADATA_DIR_NAME}/summary.json",
        },
        "max_episodes_requested": cli.max_episodes,
        "max_env_steps_requested": cli.max_env_steps,
        "n_shards": int(len(writer.paths)),
        "shards": [_safe_path(path) for path in writer.paths],
        "n_env_steps": int(env_steps),
        "n_agent_examples": int(env_steps * len(agent_ids)),
        "n_completed_episodes": int(completed_episodes),
        "n_unique_chronic_fingerprints": int(len(unique_fingerprints)),
        "episode_length_mean": (
            float(np.mean(episode_lengths)) if episode_lengths else None
        ),
        "episode_length_min": int(min(episode_lengths)) if episode_lengths else None,
        "episode_length_max": int(max(episode_lengths)) if episode_lengths else None,
        "agent_summary": agent_summary,
    }


def _write_metadata(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Collect teacher-student imitation data from a checkpoint."
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--checkpoint", type=str, help="Exact checkpoint path or stem.")
    selector.add_argument(
        "--model", type=str, help="Model/run-name substring used to find a checkpoint."
    )
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument(
        "--step-policy",
        type=str,
        default="exact",
        choices=["exact", "before", "after", "nearest"],
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=TASK_DIR / "checkpoint",
        help="Directory containing .tar checkpoints.",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--split-chronics", type=str2bool, default=True)
    parser.add_argument("--eval-all-split-chronics", type=str2bool, default=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--deterministic-eval", type=str2bool, default=True)
    parser.add_argument(
        "--eval-action-heuristic",
        type=str,
        default="local_rho_threshold",
        choices=["none", "rho_threshold", "local_rho_threshold"],
    )
    parser.add_argument("--eval-action-rho-threshold", type=float, default=0.90)
    parser.add_argument(
        "--intervention-gate-eval-mode",
        type=str,
        default="checkpoint",
        choices=["checkpoint", "final_action_map", "hierarchical_greedy"],
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument(
        "--obs-normalization",
        type=str,
        default="require",
        choices=["auto", "disable", "require"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--compress", type=str2bool, default=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.max_episodes is not None and cli.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive when provided.")
    if cli.max_env_steps is not None and cli.max_env_steps <= 0:
        raise ValueError("--max-env-steps must be positive when provided.")
    if cli.shard_size <= 0:
        raise ValueError("--shard-size must be positive.")

    checkpoint_dir = cli.checkpoint_dir.expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (TASK_DIR / checkpoint_dir).resolve()
    output_dir = cli.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (TASK_DIR / output_dir).resolve()
    _prepare_output_dir(output_dir, cli.overwrite)

    if cli.checkpoint:
        checkpoint_path = _resolve_checkpoint_path(cli.checkpoint, checkpoint_dir)
    else:
        checkpoint_path = _find_checkpoint_by_model_step(
            cli.model, cli.step, checkpoint_dir, cli.step_policy
        )

    first_record = th.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = _merge_missing_defaults(_as_namespace(first_record["args"]))
    args.track = False
    args.split_chronics = bool(cli.split_chronics)
    args.eval_all_split_chronics = bool(cli.eval_all_split_chronics)
    args.deterministic_eval = bool(cli.deterministic_eval)
    args.trace_rollout_actions = False
    args.eval_progress_print = False
    args.eval_action_heuristic = cli.eval_action_heuristic
    args.eval_action_rho_threshold = float(cli.eval_action_rho_threshold)
    if cli.intervention_gate_eval_mode != "checkpoint":
        args.intervention_gate_eval_mode = cli.intervention_gate_eval_mode
    if cli.n_threads is not None:
        args.n_threads = int(cli.n_threads)
    args.n_threads = max(int(getattr(args, "n_threads", 1)), 1)

    obs_stats = _extract_obs_stats(first_record)
    args, obs_stats, obs_norm_mode = _configure_obs_normalization(
        args, obs_stats, cli.obs_normalization
    )
    set_random_seed(getattr(args, "seed", 0))
    device = _resolve_device(args, cli.device)
    record = _load_checkpoint(checkpoint_path, device)
    checkpoint_step = int(record.get("global_step", 0))

    evaluator = Evaluator(
        args,
        logger=None,
        device=device,
        chronic_split=cli.split,
        metric_prefix=f"teacher_dataset_{cli.split}",
    )
    if obs_stats:
        evaluator.env.env.set_obs_stats(obs_stats)
    actors = _build_actors(record, args, evaluator, device)
    agent_ids = list(actors.keys())

    obs_space = evaluator.env.env.observation_space
    action_space = evaluator.env.env.action_space
    obs_shapes = {agent: list(obs_space[agent].shape) for agent in agent_ids}
    action_sizes = {agent: int(action_space[agent].n) for agent in agent_ids}
    shard_output_dir = shards_dir(output_dir)
    writer = ShardWriter(shard_output_dir, agent_ids, cli.shard_size, cli.compress)
    metadata_file = metadata_path(output_dir)

    target_episodes = cli.max_episodes or int(evaluator.eval_episodes)
    print("========== Teacher dataset collection ==========")
    print(f"Checkpoint: {_repo_relative(checkpoint_path)}")
    print(f"Checkpoint global_step: {checkpoint_step}")
    print(f"Output dir: {_safe_path(output_dir)}")
    print(f"Shard dir: {_safe_path(shard_output_dir)}")
    print(f"Metadata dir: {_safe_path(metadata_dir(output_dir))}")
    print(f"Split: {cli.split}")
    print(f"Target episodes: {target_episodes}")
    print(f"Max env steps: {cli.max_env_steps or 'none'}")
    print(f"Eval heuristic: {args.eval_action_heuristic}")
    print(f"Eval rho threshold: {args.eval_action_rho_threshold}")
    print(f"Obs normalization: {obs_norm_mode}")
    print(f"Device: {device}")
    print("================================================")

    obs, _ = evaluator.env.reset()
    env_steps = 0
    completed_episodes = 0
    episode_step = 0
    unique_fingerprints = set()
    episode_lengths: List[int] = []
    agent_metrics = {agent: _agent_metric_template() for agent in agent_ids}

    while completed_episodes < target_episodes:
        if cli.max_env_steps is not None and env_steps >= cli.max_env_steps:
            print(
                f"Reached --max-env-steps={cli.max_env_steps}; stopping early.",
                flush=True,
            )
            break

        for agent in agent_ids:
            _flat_obs_for_storage(obs, agent)

        chronic_info = _current_chronic_info(evaluator)
        unique_fingerprints.add(chronic_info["chronic_fingerprint"])
        obs_tensors = cast_np_to_tensors(obs, device)

        policy_actions = {}
        with th.no_grad():
            for agent, actor in actors.items():
                policy_actions[agent] = actor.get_eval_action(
                    obs_tensors[agent],
                    deterministic=args.deterministic_eval,
                )

        force_noop, local_max_rhos, global_max_rho = _teacher_override_decision(
            evaluator, agent_ids
        )
        teacher_actions = {}
        for agent in agent_ids:
            teacher_actions[agent] = (
                evaluator._zero_action_like(policy_actions[agent])
                if force_noop[agent]
                else policy_actions[agent]
            )

        policy_action_ids = {
            agent: tensor_scalar_to_int(policy_actions[agent]) for agent in agent_ids
        }
        teacher_action_ids = {
            agent: tensor_scalar_to_int(teacher_actions[agent]) for agent in agent_ids
        }

        next_obs, reward, terminations, truncations, info = evaluator.env.step(
            teacher_actions
        )
        done = bool(
            np.logical_or(
                terminations[agent_ids[0]],
                truncations[agent_ids[0]],
            )
        )
        done_episode_length = 0
        if done:
            done_episode_length = max(
                int(evaluator.env.g2op_ma_env._cent_env.nb_time_step), 1
            )

        row = {
            "episode_id": completed_episodes,
            "episode_step": episode_step,
            "dataset_step": env_steps,
            "global_max_rho": global_max_rho,
            "reward_agent_0": tensor_scalar_to_float(reward[agent_ids[0]]),
            "done": done,
            "done_episode_length": done_episode_length,
            **chronic_info,
        }
        agent_values = {}
        for agent in agent_ids:
            policy_action = policy_action_ids[agent]
            teacher_action = teacher_action_ids[agent]
            was_overwritten = policy_action != teacher_action
            metrics = agent_metrics[agent]
            metrics["n"] += 1
            metrics["policy_action_0"] += int(policy_action == 0)
            metrics["policy_nonidle"] += int(policy_action != 0)
            metrics["teacher_action_0"] += int(teacher_action == 0)
            metrics["teacher_nonidle"] += int(teacher_action != 0)
            metrics["force_noop"] += int(force_noop[agent])
            metrics["was_overwritten"] += int(was_overwritten)
            metrics["overwrite_to_action0"] += int(
                was_overwritten and teacher_action == 0
            )
            agent_values[agent] = {
                "obs": _flat_obs_for_storage(obs, agent),
                "policy_action": policy_action,
                "teacher_action": teacher_action,
                "force_noop": force_noop[agent],
                "was_overwritten": was_overwritten,
                "local_max_rho": local_max_rhos[agent],
            }

        flushed = writer.append(row=row, agent_values=agent_values)
        env_steps += 1
        episode_step += 1

        if flushed is not None:
            _write_metadata(
                metadata_file,
                _metadata(
                    status="running",
                    cli=cli,
                    checkpoint_path=checkpoint_path,
                    checkpoint_step=checkpoint_step,
                    args=args,
                    obs_norm_mode=obs_norm_mode,
                    obs_stats_available=bool(obs_stats),
                    agent_ids=agent_ids,
                    action_sizes=action_sizes,
                    obs_shapes=obs_shapes,
                    writer=writer,
                    env_steps=env_steps,
                    completed_episodes=completed_episodes,
                    unique_fingerprints=unique_fingerprints,
                    episode_lengths=episode_lengths,
                    agent_metrics=agent_metrics,
                ),
            )
            print(f"wrote {_safe_path(flushed)} at env_step={env_steps}", flush=True)

        if done:
            completed_episodes += 1
            episode_lengths.append(done_episode_length)
            if completed_episodes % max(cli.progress_every, 1) == 0:
                print(
                    "progress: "
                    f"episodes={completed_episodes}/{target_episodes} "
                    f"env_steps={env_steps} "
                    f"last_len={done_episode_length} "
                    f"unique_fingerprints={len(unique_fingerprints)}",
                    flush=True,
                )
            obs, _ = evaluator.env.reset()
            episode_step = 0
        else:
            obs = next_obs

    final_shard = writer.flush()
    if final_shard is not None:
        print(f"wrote {_safe_path(final_shard)} at env_step={env_steps}", flush=True)

    final_metadata = _metadata(
        status="complete",
        cli=cli,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
        args=args,
        obs_norm_mode=obs_norm_mode,
        obs_stats_available=bool(obs_stats),
        agent_ids=agent_ids,
        action_sizes=action_sizes,
        obs_shapes=obs_shapes,
        writer=writer,
        env_steps=env_steps,
        completed_episodes=completed_episodes,
        unique_fingerprints=unique_fingerprints,
        episode_lengths=episode_lengths,
        agent_metrics=agent_metrics,
    )
    _write_metadata(metadata_file, final_metadata)
    evaluator.env.close()

    print("========== Teacher dataset complete ==========")
    print(f"Output dir: {_safe_path(output_dir)}")
    print(f"Shard dir: {_safe_path(shard_output_dir)}")
    print(f"Metadata dir: {_safe_path(metadata_dir(output_dir))}")
    print(f"Shards: {len(writer.paths)}")
    print(f"Env steps: {env_steps}")
    print(f"Agent examples: {env_steps * len(agent_ids)}")
    print(f"Completed episodes: {completed_episodes}")
    print(f"Unique fingerprints: {len(unique_fingerprints)}")
    for agent in agent_ids:
        summary = final_metadata["agent_summary"][agent]
        print(
            f"{agent}: teacher_action0={summary['teacher_action_0_frac']:.4f} "
            f"teacher_nonidle={summary['teacher_nonidle_frac']:.4f} "
            f"overwritten={summary['was_overwritten_frac']:.4f}"
        )
    print(f"Metadata: {_safe_path(metadata_file)}")


if __name__ == "__main__":
    main()
