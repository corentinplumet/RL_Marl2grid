#!/usr/bin/env python3
"""Collect teacher-student imitation data from a heuristic-augmented policy."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
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
    METADATA_EXPORT_DIR_NAME,
    SHARDS_DIR_NAME,
    export_metadata_file,
    metadata_dir,
    metadata_export_dir,
    metadata_path,
    shards_dir,
)


OUTCOME_LABEL_IMPROVED = -1
OUTCOME_LABEL_NEUTRAL = 0
OUTCOME_LABEL_WORSENED = 1
OUTCOME_LABEL_INVALID = 2


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


def _policy_logits_for_storage(actor: Any, obs_tensor: th.Tensor, agent_id: str) -> np.ndarray:
    if bool(getattr(actor, "intervention_gate", False)):
        raise NotImplementedError(
            "Soft-label collection currently supports flat non-gated actors only. "
            f"The selected checkpoint uses intervention_gate=True for {agent_id}."
        )
    if getattr(actor, "encoder_type", "mlp") != "mlp":
        raise NotImplementedError(
            "Soft-label collection currently supports actor_encoder='mlp' only. "
            f"The selected checkpoint uses actor_encoder={actor.encoder_type!r} "
            f"for {agent_id}."
        )
    logits = actor.actor(actor._encode(obs_tensor))
    if logits.ndim == 0:
        raise ValueError(f"Actor logits for {agent_id} are scalar; expected actions.")
    if logits.ndim > 1:
        if logits.shape[0] != 1:
            raise ValueError(
                f"Actor logits for {agent_id} have unexpected shape "
                f"{tuple(logits.shape)}."
            )
        logits = logits[0]
    return logits.detach().cpu().numpy().astype(np.float32, copy=True)


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
            if "policy_logits" in values:
                arrays[f"policy_logits_{agent}"] = np.stack(
                    values["policy_logits"]
                ).astype(np.float32)

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


class ActionOutcomeShardWriter:
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
        self.rows_in_shard = 0
        self.agent_rows: Dict[str, Dict[str, List[Any]]] = {
            agent: defaultdict(list) for agent in agent_ids
        }
        self.paths: List[Path] = []
        self.total_rows = 0

    def __len__(self) -> int:
        return self.rows_in_shard

    def append(self, *, agent_id: str, values: Dict[str, Any]) -> Optional[Path]:
        rows = self.agent_rows[agent_id]
        for key, value in values.items():
            rows[key].append(value)
        self.rows_in_shard += 1
        self.total_rows += 1
        if self.rows_in_shard >= self.shard_size:
            return self.flush()
        return None

    def flush(self) -> Optional[Path]:
        if self.rows_in_shard == 0:
            return None

        arrays: Dict[str, Any] = {}
        for agent in self.agent_ids:
            values = self.agent_rows[agent]
            if not values:
                continue
            prefix = f"outcome_{agent}"
            arrays[f"{prefix}_episode_id"] = np.asarray(
                values["episode_id"], dtype=np.int64
            )
            arrays[f"{prefix}_episode_step"] = np.asarray(
                values["episode_step"], dtype=np.int32
            )
            arrays[f"{prefix}_dataset_step"] = np.asarray(
                values["dataset_step"], dtype=np.int64
            )
            arrays[f"{prefix}_action_id"] = np.asarray(
                values["action_id"], dtype=np.int32
            )
            arrays[f"{prefix}_global_max_rho_before"] = np.asarray(
                values["global_max_rho_before"], dtype=np.float32
            )
            arrays[f"{prefix}_local_max_rho_before"] = np.asarray(
                values["local_max_rho_before"], dtype=np.float32
            )
            arrays[f"{prefix}_rho_before"] = np.asarray(
                values["rho_before"], dtype=np.float32
            )
            arrays[f"{prefix}_rho_after_action"] = np.asarray(
                values["rho_after_action"], dtype=np.float32
            )
            arrays[f"{prefix}_rho_after_do_nothing"] = np.asarray(
                values["rho_after_do_nothing"], dtype=np.float32
            )
            arrays[f"{prefix}_delta_vs_now"] = np.asarray(
                values["delta_vs_now"], dtype=np.float32
            )
            arrays[f"{prefix}_delta_vs_do_nothing"] = np.asarray(
                values["delta_vs_do_nothing"], dtype=np.float32
            )
            arrays[f"{prefix}_label_vs_now"] = np.asarray(
                values["label_vs_now"], dtype=np.int8
            )
            arrays[f"{prefix}_label_vs_do_nothing"] = np.asarray(
                values["label_vs_do_nothing"], dtype=np.int8
            )
            arrays[f"{prefix}_action_is_valid"] = np.asarray(
                values["action_is_valid"], dtype=bool
            )
            arrays[f"{prefix}_action_is_legal"] = np.asarray(
                values["action_is_legal"], dtype=bool
            )
            arrays[f"{prefix}_action_is_ambiguous"] = np.asarray(
                values["action_is_ambiguous"], dtype=bool
            )
            arrays[f"{prefix}_simulation_exception"] = np.asarray(
                values["simulation_exception"], dtype=bool
            )
            arrays[f"{prefix}_sim_done"] = np.asarray(
                values["sim_done"], dtype=bool
            )
            arrays[f"{prefix}_sim_reward"] = np.asarray(
                values["sim_reward"], dtype=np.float32
            )
            arrays[f"{prefix}_worst_line_before"] = np.asarray(
                values["worst_line_before"], dtype=np.int32
            )
            arrays[f"{prefix}_worst_line_after_action"] = np.asarray(
                values["worst_line_after_action"], dtype=np.int32
            )
            arrays[f"{prefix}_worst_line_after_do_nothing"] = np.asarray(
                values["worst_line_after_do_nothing"], dtype=np.int32
            )
            arrays[f"{prefix}_chronic_name"] = np.asarray(
                values["chronic_name"], dtype=str
            )
            arrays[f"{prefix}_chronic_fingerprint"] = np.asarray(
                values["chronic_fingerprint"], dtype=str
            )
            arrays[f"{prefix}_chronic_datetime"] = np.asarray(
                values["chronic_datetime"], dtype=str
            )
            arrays[f"{prefix}_validation_error"] = np.asarray(
                values["validation_error"], dtype=str
            )
            arrays[f"{prefix}_simulation_error"] = np.asarray(
                values["simulation_error"], dtype=str
            )
            arrays[f"obs_{agent}"] = np.stack(values["obs"]).astype(np.float32)

        path = self.output_dir / f"shard_{self.shard_idx:05d}.npz"
        if self.compress:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)

        self.paths.append(path)
        self.shard_idx += 1
        self.rows_in_shard = 0
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
        "save_policy_logits": bool(cli.save_policy_logits),
        "soft_labels_available": bool(cli.save_policy_logits),
        "layout": {
            "version": 2,
            "shards_dir": SHARDS_DIR_NAME,
            "metadata_dir": METADATA_DIR_NAME,
            "metadata_export_dir": f"../{METADATA_EXPORT_DIR_NAME}",
            "metadata_file": f"{METADATA_DIR_NAME}/metadata.json",
            "summary_file": f"{METADATA_DIR_NAME}/summary.json",
            "metadata_export_file": f"../{METADATA_EXPORT_DIR_NAME}/{cli.output_dir.name}_metadata.json",
            "summary_export_file": f"../{METADATA_EXPORT_DIR_NAME}/{cli.output_dir.name}_summary.json",
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


def _outcome_agent_metric_template() -> Dict[str, int]:
    return {
        "n": 0,
        "valid": 0,
        "invalid": 0,
        "improved_vs_now": 0,
        "neutral_vs_now": 0,
        "worsened_vs_now": 0,
        "improved_vs_do_nothing": 0,
        "neutral_vs_do_nothing": 0,
        "worsened_vs_do_nothing": 0,
        "sim_done": 0,
        "simulation_exception": 0,
    }


def _outcome_label(delta: float, *, valid: bool, tolerance: float) -> int:
    if not valid or not np.isfinite(delta):
        return OUTCOME_LABEL_INVALID
    if delta < -tolerance:
        return OUTCOME_LABEL_IMPROVED
    if delta > tolerance:
        return OUTCOME_LABEL_WORSENED
    return OUTCOME_LABEL_NEUTRAL


def _candidate_action_ids(
    *,
    action_size: int,
    include_action_zero: bool,
    sample_size: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    start = 0 if include_action_zero else 1
    action_ids = np.arange(start, int(action_size), dtype=np.int32)
    if sample_size is not None and int(sample_size) < action_ids.size:
        action_ids = rng.choice(
            action_ids,
            size=int(sample_size),
            replace=False,
        ).astype(np.int32)
        action_ids.sort()
    return action_ids


def _update_outcome_metrics(metrics: Dict[str, int], values: Dict[str, Any]) -> None:
    metrics["n"] += 1
    valid = bool(values["action_is_valid"])
    metrics["valid"] += int(valid)
    metrics["invalid"] += int(not valid)
    metrics["sim_done"] += int(values["sim_done"])
    metrics["simulation_exception"] += int(values["simulation_exception"])

    label_vs_now = int(values["label_vs_now"])
    label_vs_do_nothing = int(values["label_vs_do_nothing"])
    metrics["improved_vs_now"] += int(label_vs_now == OUTCOME_LABEL_IMPROVED)
    metrics["neutral_vs_now"] += int(label_vs_now == OUTCOME_LABEL_NEUTRAL)
    metrics["worsened_vs_now"] += int(label_vs_now == OUTCOME_LABEL_WORSENED)
    metrics["improved_vs_do_nothing"] += int(
        label_vs_do_nothing == OUTCOME_LABEL_IMPROVED
    )
    metrics["neutral_vs_do_nothing"] += int(
        label_vs_do_nothing == OUTCOME_LABEL_NEUTRAL
    )
    metrics["worsened_vs_do_nothing"] += int(
        label_vs_do_nothing == OUTCOME_LABEL_WORSENED
    )


def _outcome_metadata(
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
    writer: ActionOutcomeShardWriter,
    env_steps: int,
    candidate_states: int,
    skipped_states: int,
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
            "valid_frac": metrics["valid"] / n,
            "invalid_frac": metrics["invalid"] / n,
            "improved_vs_now_frac": metrics["improved_vs_now"] / n,
            "neutral_vs_now_frac": metrics["neutral_vs_now"] / n,
            "worsened_vs_now_frac": metrics["worsened_vs_now"] / n,
            "improved_vs_do_nothing_frac": metrics["improved_vs_do_nothing"] / n,
            "neutral_vs_do_nothing_frac": metrics["neutral_vs_do_nothing"] / n,
            "worsened_vs_do_nothing_frac": metrics["worsened_vs_do_nothing"] / n,
            "sim_done_frac": metrics["sim_done"] / n,
            "simulation_exception_frac": metrics["simulation_exception"] / n,
        }

    return {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector": "teacher_student.collect_teacher_dataset",
        "dataset_mode": "action_outcomes",
        "phase": 2,
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
        "collection_rho_threshold": float(cli.collection_rho_threshold),
        "outcome_delta_tolerance": float(cli.outcome_delta_tolerance),
        "outcome_time_step": int(cli.outcome_time_step),
        "outcome_rollout_policy": cli.outcome_rollout_policy,
        "outcome_action_sample_size": cli.outcome_action_sample_size,
        "outcome_include_action_zero": bool(cli.outcome_include_action_zero),
        "obs_normalization": obs_norm_mode,
        "obs_stats_available": bool(obs_stats_available),
        "norm_obs_effective": bool(getattr(args, "norm_obs", False)),
        "agent_ids": agent_ids,
        "action_sizes": action_sizes,
        "obs_shapes": obs_shapes,
        "label_encoding": {
            "improved": OUTCOME_LABEL_IMPROVED,
            "neutral": OUTCOME_LABEL_NEUTRAL,
            "worsened": OUTCOME_LABEL_WORSENED,
            "invalid": OUTCOME_LABEL_INVALID,
        },
        "shard_size": int(cli.shard_size),
        "compress": bool(cli.compress),
        "layout": {
            "version": 2,
            "shards_dir": SHARDS_DIR_NAME,
            "metadata_dir": METADATA_DIR_NAME,
            "metadata_export_dir": f"../{METADATA_EXPORT_DIR_NAME}",
            "metadata_file": f"{METADATA_DIR_NAME}/metadata.json",
            "summary_file": f"{METADATA_DIR_NAME}/summary.json",
            "metadata_export_file": f"../{METADATA_EXPORT_DIR_NAME}/{cli.output_dir.name}_metadata.json",
            "summary_export_file": f"../{METADATA_EXPORT_DIR_NAME}/{cli.output_dir.name}_summary.json",
        },
        "max_episodes_requested": cli.max_episodes,
        "max_env_steps_requested": cli.max_env_steps,
        "n_shards": int(len(writer.paths)),
        "shards": [_safe_path(path) for path in writer.paths],
        "n_env_steps": int(env_steps),
        "n_candidate_states": int(candidate_states),
        "n_skipped_states_below_threshold": int(skipped_states),
        "n_outcome_examples": int(writer.total_rows),
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


def _write_dataset_metadata(dataset_dir: Path, path: Path, payload: Dict[str, Any]) -> Path:
    _write_metadata(path, payload)
    export_metadata_file(dataset_dir, path, "metadata")
    return path


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "unknown"
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m{sec:04.1f}s"


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def _teacher_actions_for_step(
    *,
    evaluator: Evaluator,
    actors: Dict[str, Any],
    obs: Dict[str, Any],
    agent_ids: List[str],
    device: th.device,
    deterministic_eval: bool,
    save_policy_logits: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, int], Dict[str, np.ndarray]]:
    obs_tensors = cast_np_to_tensors(obs, device)
    policy_actions = {}
    policy_logits = {}
    with th.no_grad():
        for agent, actor in actors.items():
            if save_policy_logits:
                policy_logits[agent] = _policy_logits_for_storage(
                    actor, obs_tensors[agent], agent
                )
            policy_actions[agent] = actor.get_eval_action(
                obs_tensors[agent],
                deterministic=deterministic_eval,
            )

    force_noop, _, _ = _teacher_override_decision(evaluator, agent_ids)
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
    return teacher_actions, policy_action_ids, teacher_action_ids, policy_logits


def _collect_action_outcome_dataset(
    *,
    cli: Namespace,
    checkpoint_path: Path,
    checkpoint_step: int,
    args: Namespace,
    obs_norm_mode: str,
    obs_stats_available: bool,
    evaluator: Evaluator,
    actors: Dict[str, Any],
    agent_ids: List[str],
    action_sizes: Dict[str, int],
    obs_shapes: Dict[str, List[int]],
    output_dir: Path,
    shard_output_dir: Path,
    metadata_file: Path,
    device: th.device,
) -> None:
    writer = ActionOutcomeShardWriter(
        shard_output_dir,
        agent_ids,
        cli.shard_size,
        cli.compress,
    )
    target_episodes = cli.max_episodes or int(evaluator.eval_episodes)
    rng = np.random.default_rng(int(getattr(args, "seed", 0)))
    candidate_ids_by_agent = {
        agent: _candidate_action_ids(
            action_size=action_sizes[agent],
            include_action_zero=cli.outcome_include_action_zero,
            sample_size=cli.outcome_action_sample_size,
            rng=rng,
        )
        for agent in agent_ids
    }

    print("========== Action-outcome dataset collection ==========")
    print(f"Checkpoint: {_repo_relative(checkpoint_path)}")
    print(f"Checkpoint global_step: {checkpoint_step}")
    print(f"Output dir: {_safe_path(output_dir)}")
    print(f"Shard dir: {_safe_path(shard_output_dir)}")
    print(f"Metadata dir: {_safe_path(metadata_dir(output_dir))}")
    print(f"Metadata export dir: {_safe_path(metadata_export_dir(output_dir))}")
    print(f"Split: {cli.split}")
    print(f"Target episodes: {target_episodes}")
    print(f"Max env steps: {cli.max_env_steps or 'none'}")
    print(f"Collection rho threshold: {cli.collection_rho_threshold}")
    print(f"Outcome rollout policy: {cli.outcome_rollout_policy}")
    print(f"Outcome time step: {cli.outcome_time_step}")
    print(f"Outcome delta tolerance: {cli.outcome_delta_tolerance}")
    print(f"Include action 0: {cli.outcome_include_action_zero}")
    print(f"Action sample size: {cli.outcome_action_sample_size or 'all'}")
    print(f"Timing every env steps: {cli.timing_every_env_steps}")
    for agent in agent_ids:
        print(
            f"{agent}: evaluating {len(candidate_ids_by_agent[agent])}/"
            f"{action_sizes[agent]} actions per collected state"
        )
    print(f"Obs normalization: {obs_norm_mode}")
    print(f"Device: {device}")
    print("=======================================================")

    obs, _ = evaluator.env.reset()
    env_steps = 0
    completed_episodes = 0
    episode_step = 0
    candidate_states = 0
    skipped_states = 0
    unique_fingerprints = set()
    episode_lengths: List[int] = []
    agent_metrics = {agent: _outcome_agent_metric_template() for agent in agent_ids}
    run_start_time = time.perf_counter()
    timed_env_steps = 0
    timed_sim_actions = 0
    timed_wall_seconds = 0.0
    timed_env_step_seconds = 0.0
    timed_collection_seconds = 0.0
    timed_sim_action_seconds = 0.0

    while completed_episodes < target_episodes:
        step_start_time = time.perf_counter()
        collection_seconds = 0.0
        sim_action_seconds = 0.0
        sim_action_count = 0
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
        try:
            global_max_rho = float(evaluator.env.get_current_max_rho())
        except Exception:
            global_max_rho = float("nan")
        try:
            local_max_rhos = evaluator.env.get_current_agent_max_rho()
        except Exception:
            local_max_rhos = {agent: float("nan") for agent in agent_ids}

        should_collect = bool(
            np.isfinite(global_max_rho)
            and global_max_rho >= float(cli.collection_rho_threshold)
        )
        best_rollout_action = {agent: 0 for agent in agent_ids}
        best_rollout_rho = float("inf")

        if should_collect:
            candidate_states += 1
            collection_start_time = time.perf_counter()
            sim_start_time = time.perf_counter()
            do_nothing = evaluator.env.simulate_action_outcomes(
                [{"agent_id": agent_ids[0], "action_id": 0}],
                time_step=cli.outcome_time_step,
            )[0]
            rho_after_do_nothing = float(do_nothing["rho_after"])
            worst_line_after_do_nothing = int(do_nothing["worst_line_after"])

            requests = [
                {"agent_id": agent, "action_id": int(action_id)}
                for agent in agent_ids
                for action_id in candidate_ids_by_agent[agent]
            ]
            outcomes = evaluator.env.simulate_action_outcomes(
                requests,
                time_step=cli.outcome_time_step,
            )
            sim_action_seconds = time.perf_counter() - sim_start_time
            sim_action_count = len(requests) + 1

            for request, outcome in zip(requests, outcomes):
                agent = str(request["agent_id"])
                action_id = int(request["action_id"])
                valid = bool(outcome["action_is_valid"])
                rho_before = float(outcome["rho_before"])
                rho_after_action = float(outcome["rho_after"])
                delta_vs_now = rho_after_action - rho_before
                delta_vs_do_nothing = rho_after_action - rho_after_do_nothing
                label_vs_now = _outcome_label(
                    delta_vs_now,
                    valid=valid,
                    tolerance=cli.outcome_delta_tolerance,
                )
                label_vs_do_nothing = _outcome_label(
                    delta_vs_do_nothing,
                    valid=valid,
                    tolerance=cli.outcome_delta_tolerance,
                )
                values = {
                    "episode_id": completed_episodes,
                    "episode_step": episode_step,
                    "dataset_step": env_steps,
                    "action_id": action_id,
                    "global_max_rho_before": global_max_rho,
                    "local_max_rho_before": float(
                        local_max_rhos.get(agent, float("nan"))
                    ),
                    "rho_before": rho_before,
                    "rho_after_action": rho_after_action,
                    "rho_after_do_nothing": rho_after_do_nothing,
                    "delta_vs_now": delta_vs_now,
                    "delta_vs_do_nothing": delta_vs_do_nothing,
                    "label_vs_now": label_vs_now,
                    "label_vs_do_nothing": label_vs_do_nothing,
                    "action_is_valid": valid,
                    "action_is_legal": bool(outcome["action_is_legal"]),
                    "action_is_ambiguous": bool(outcome["action_is_ambiguous"]),
                    "simulation_exception": bool(outcome["simulation_exception"]),
                    "sim_done": bool(outcome["sim_done"]),
                    "sim_reward": float(outcome["sim_reward"]),
                    "worst_line_before": int(outcome["worst_line_before"]),
                    "worst_line_after_action": int(outcome["worst_line_after"]),
                    "worst_line_after_do_nothing": worst_line_after_do_nothing,
                    "validation_error": str(outcome["validation_error"]),
                    "simulation_error": str(outcome["simulation_error"]),
                    "obs": _flat_obs_for_storage(obs, agent),
                    **chronic_info,
                }
                _update_outcome_metrics(agent_metrics[agent], values)
                flushed = writer.append(agent_id=agent, values=values)
                if (
                    cli.outcome_rollout_policy == "best_simulated"
                    and valid
                    and np.isfinite(rho_after_action)
                    and rho_after_action < best_rollout_rho
                ):
                    best_rollout_rho = rho_after_action
                    best_rollout_action = {other_agent: 0 for other_agent in agent_ids}
                    best_rollout_action[agent] = action_id

                if flushed is not None:
                    _write_dataset_metadata(
                        output_dir,
                        metadata_file,
                        _outcome_metadata(
                            status="running",
                            cli=cli,
                            checkpoint_path=checkpoint_path,
                            checkpoint_step=checkpoint_step,
                            args=args,
                            obs_norm_mode=obs_norm_mode,
                            obs_stats_available=obs_stats_available,
                            agent_ids=agent_ids,
                            action_sizes=action_sizes,
                            obs_shapes=obs_shapes,
                            writer=writer,
                            env_steps=env_steps,
                            candidate_states=candidate_states,
                            skipped_states=skipped_states,
                            completed_episodes=completed_episodes,
                            unique_fingerprints=unique_fingerprints,
                            episode_lengths=episode_lengths,
                            agent_metrics=agent_metrics,
                        ),
                    )
                    print(
                        f"wrote {_safe_path(flushed)} at env_step={env_steps} "
                        f"outcome_examples={writer.total_rows}",
                        flush=True,
                    )
            collection_seconds = time.perf_counter() - collection_start_time
        else:
            skipped_states += 1

        if cli.outcome_rollout_policy == "teacher":
            rollout_actions, _, _, _ = _teacher_actions_for_step(
                evaluator=evaluator,
                actors=actors,
                obs=obs,
                agent_ids=agent_ids,
                device=device,
                deterministic_eval=args.deterministic_eval,
            )
        elif cli.outcome_rollout_policy == "best_simulated":
            rollout_actions = best_rollout_action
        elif cli.outcome_rollout_policy == "do_nothing":
            rollout_actions = {agent: 0 for agent in agent_ids}
        else:
            raise ValueError(
                f"Unsupported outcome rollout policy: {cli.outcome_rollout_policy!r}"
            )

        env_step_start_time = time.perf_counter()
        next_obs, reward, terminations, truncations, info = evaluator.env.step(
            rollout_actions
        )
        env_step_seconds = time.perf_counter() - env_step_start_time
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

        env_steps += 1
        episode_step += 1
        wall_step_seconds = time.perf_counter() - step_start_time
        timed_env_steps += 1
        timed_sim_actions += sim_action_count
        timed_wall_seconds += wall_step_seconds
        timed_env_step_seconds += env_step_seconds
        timed_collection_seconds += collection_seconds
        timed_sim_action_seconds += sim_action_seconds

        timing_every = max(int(cli.timing_every_env_steps), 1)
        if env_steps == 1 or env_steps % timing_every == 0:
            elapsed = time.perf_counter() - run_start_time
            avg_wall_step = _safe_div(timed_wall_seconds, timed_env_steps)
            avg_env_step = _safe_div(timed_env_step_seconds, timed_env_steps)
            avg_collect = _safe_div(timed_collection_seconds, timed_env_steps)
            avg_sim_action = _safe_div(
                timed_sim_action_seconds,
                timed_sim_actions,
            )
            last_sim_action = _safe_div(sim_action_seconds, sim_action_count)
            if cli.max_env_steps is not None:
                eta_seconds = avg_wall_step * max(cli.max_env_steps - env_steps, 0)
            elif completed_episodes > 0:
                eta_seconds = (
                    _safe_div(elapsed, completed_episodes)
                    * max(target_episodes - completed_episodes, 0)
                )
            else:
                eta_seconds = float("nan")
            print(
                "timing: "
                f"env_steps={env_steps} "
                f"episodes={completed_episodes}/{target_episodes} "
                f"last_step={wall_step_seconds:.3f}s "
                f"avg_step={avg_wall_step:.3f}s "
                f"last_env_step={env_step_seconds:.3f}s "
                f"avg_env_step={avg_env_step:.3f}s "
                f"last_collect={collection_seconds:.3f}s "
                f"avg_collect={avg_collect:.3f}s "
                f"last_sim_actions={sim_action_count} "
                f"last_sec_per_sim_action={last_sim_action:.6f}s "
                f"avg_sec_per_sim_action={avg_sim_action:.6f}s "
                f"elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(eta_seconds)}",
                flush=True,
            )

        if done:
            completed_episodes += 1
            episode_lengths.append(done_episode_length)
            if completed_episodes % max(cli.progress_every, 1) == 0:
                print(
                    "progress: "
                    f"episodes={completed_episodes}/{target_episodes} "
                    f"env_steps={env_steps} "
                    f"candidate_states={candidate_states} "
                    f"outcome_examples={writer.total_rows} "
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
        print(
            f"wrote {_safe_path(final_shard)} at env_step={env_steps} "
            f"outcome_examples={writer.total_rows}",
            flush=True,
        )

    final_metadata = _outcome_metadata(
        status="complete",
        cli=cli,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
        args=args,
        obs_norm_mode=obs_norm_mode,
        obs_stats_available=obs_stats_available,
        agent_ids=agent_ids,
        action_sizes=action_sizes,
        obs_shapes=obs_shapes,
        writer=writer,
        env_steps=env_steps,
        candidate_states=candidate_states,
        skipped_states=skipped_states,
        completed_episodes=completed_episodes,
        unique_fingerprints=unique_fingerprints,
        episode_lengths=episode_lengths,
        agent_metrics=agent_metrics,
    )
    _write_dataset_metadata(output_dir, metadata_file, final_metadata)
    evaluator.env.close()

    print("========== Action-outcome dataset complete ==========")
    print(f"Output dir: {_safe_path(output_dir)}")
    print(f"Shard dir: {_safe_path(shard_output_dir)}")
    print(f"Metadata dir: {_safe_path(metadata_dir(output_dir))}")
    print(f"Metadata export dir: {_safe_path(metadata_export_dir(output_dir))}")
    print(f"Shards: {len(writer.paths)}")
    print(f"Env steps: {env_steps}")
    print(f"Candidate states: {candidate_states}")
    print(f"Skipped states below threshold: {skipped_states}")
    print(f"Outcome examples: {writer.total_rows}")
    print(f"Completed episodes: {completed_episodes}")
    print(f"Unique fingerprints: {len(unique_fingerprints)}")
    for agent in agent_ids:
        summary = final_metadata["agent_summary"][agent]
        print(
            f"{agent}: valid={summary['valid_frac']:.4f} "
            f"improved_now={summary['improved_vs_now_frac']:.4f} "
            f"improved_vs_noop={summary['improved_vs_do_nothing_frac']:.4f}"
        )
    print(f"Metadata: {_safe_path(metadata_file)}")
    export_path = metadata_export_dir(output_dir) / f"{output_dir.name}_metadata.json"
    print(f"Metadata export: {_safe_path(export_path)}")


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
    parser.add_argument(
        "--dataset-mode",
        type=str,
        default="imitation",
        choices=["imitation", "action_outcomes"],
        help=(
            "imitation keeps the existing behavior-cloning dataset. "
            "action_outcomes simulates candidate actions at high-rho states and "
            "stores rho before/after labels."
        ),
    )
    parser.add_argument("--deterministic-eval", type=str2bool, default=True)
    parser.add_argument(
        "--eval-action-heuristic",
        type=str,
        default="local_rho_threshold",
        choices=["none", "rho_threshold", "local_rho_threshold"],
    )
    parser.add_argument("--eval-action-rho-threshold", type=float, default=0.90)
    parser.add_argument(
        "--collection-rho-threshold",
        type=float,
        default=None,
        help=(
            "Only collect action-outcome rows when global max rho is at least "
            "this value. Defaults to --eval-action-rho-threshold."
        ),
    )
    parser.add_argument(
        "--outcome-delta-tolerance",
        type=float,
        default=1e-3,
        help=(
            "Tolerance used to label deltas as improved, neutral, or worsened "
            "in action_outcomes mode."
        ),
    )
    parser.add_argument(
        "--outcome-time-step",
        type=int,
        default=1,
        help="Grid2Op simulate(..., time_step=...) value for action_outcomes mode.",
    )
    parser.add_argument(
        "--outcome-rollout-policy",
        type=str,
        default="best_simulated",
        choices=["best_simulated", "teacher", "do_nothing"],
        help=(
            "Action used to advance the real environment after logging simulated "
            "outcomes. best_simulated executes the valid unilateral action with "
            "the lowest simulated rho_after_action."
        ),
    )
    parser.add_argument(
        "--outcome-action-sample-size",
        type=int,
        default=None,
        help=(
            "Optional number of action ids to sample per agent at each collected "
            "state. Omit to evaluate every action id."
        ),
    )
    parser.add_argument(
        "--outcome-include-action-zero",
        type=str2bool,
        default=True,
        help="Whether action_outcomes mode stores action id 0 rows.",
    )
    parser.add_argument(
        "--timing-every-env-steps",
        type=int,
        default=100,
        help=(
            "Print action-outcome timing stats every N environment steps. "
            "Use 1 for a short calibration run."
        ),
    )
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
    parser.add_argument(
        "--save-policy-logits",
        type=str2bool,
        default=False,
        help=(
            "Save base-policy action logits for optional soft-label "
            "distillation. This can substantially increase dataset size."
        ),
    )
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
    if cli.collection_rho_threshold is None:
        cli.collection_rho_threshold = float(cli.eval_action_rho_threshold)
    if cli.outcome_delta_tolerance < 0.0:
        raise ValueError("--outcome-delta-tolerance must be non-negative.")
    if cli.outcome_time_step < 0:
        raise ValueError("--outcome-time-step must be non-negative.")
    if (
        cli.outcome_action_sample_size is not None
        and cli.outcome_action_sample_size <= 0
    ):
        raise ValueError("--outcome-action-sample-size must be positive when provided.")
    if cli.timing_every_env_steps <= 0:
        raise ValueError("--timing-every-env-steps must be positive.")

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
    metadata_file = metadata_path(output_dir)

    if cli.dataset_mode == "action_outcomes":
        _collect_action_outcome_dataset(
            cli=cli,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
            args=args,
            obs_norm_mode=obs_norm_mode,
            obs_stats_available=bool(obs_stats),
            evaluator=evaluator,
            actors=actors,
            agent_ids=agent_ids,
            action_sizes=action_sizes,
            obs_shapes=obs_shapes,
            output_dir=output_dir,
            shard_output_dir=shard_output_dir,
            metadata_file=metadata_file,
            device=device,
        )
        return

    writer = ShardWriter(shard_output_dir, agent_ids, cli.shard_size, cli.compress)

    target_episodes = cli.max_episodes or int(evaluator.eval_episodes)
    print("========== Teacher dataset collection ==========")
    print(f"Checkpoint: {_repo_relative(checkpoint_path)}")
    print(f"Checkpoint global_step: {checkpoint_step}")
    print(f"Output dir: {_safe_path(output_dir)}")
    print(f"Shard dir: {_safe_path(shard_output_dir)}")
    print(f"Metadata dir: {_safe_path(metadata_dir(output_dir))}")
    print(f"Metadata export dir: {_safe_path(metadata_export_dir(output_dir))}")
    print(f"Split: {cli.split}")
    print(f"Target episodes: {target_episodes}")
    print(f"Max env steps: {cli.max_env_steps or 'none'}")
    print(f"Eval heuristic: {args.eval_action_heuristic}")
    print(f"Eval rho threshold: {args.eval_action_rho_threshold}")
    print(f"Save policy logits: {cli.save_policy_logits}")
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
        policy_logits = {}
        with th.no_grad():
            for agent, actor in actors.items():
                if cli.save_policy_logits:
                    policy_logits[agent] = _policy_logits_for_storage(
                        actor, obs_tensors[agent], agent
                    )
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
            if cli.save_policy_logits:
                agent_values[agent]["policy_logits"] = policy_logits[agent]

        flushed = writer.append(row=row, agent_values=agent_values)
        env_steps += 1
        episode_step += 1

        if flushed is not None:
            _write_dataset_metadata(
                output_dir,
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
    _write_dataset_metadata(output_dir, metadata_file, final_metadata)
    evaluator.env.close()

    print("========== Teacher dataset complete ==========")
    print(f"Output dir: {_safe_path(output_dir)}")
    print(f"Shard dir: {_safe_path(shard_output_dir)}")
    print(f"Metadata dir: {_safe_path(metadata_dir(output_dir))}")
    print(f"Metadata export dir: {_safe_path(metadata_export_dir(output_dir))}")
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
    export_path = metadata_export_dir(output_dir) / f"{output_dir.name}_metadata.json"
    print(f"Metadata export: {_safe_path(export_path)}")


if __name__ == "__main__":
    main()
