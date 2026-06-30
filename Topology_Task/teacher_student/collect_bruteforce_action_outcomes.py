#!/usr/bin/env python3
"""Brute-force action-outcome collection for reducing a Grid2Op action space."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
from argparse import Namespace
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from common.utils import set_random_seed, str2bool
from env.config import get_env_args
from env.utils import MAEnvWrapper
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
    return "unknown" if value is None else str(value)


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


def _flat_obs_for_storage(obs: Dict[str, Any], agent_id: str) -> np.ndarray:
    value = obs[agent_id]
    if isinstance(value, dict):
        value = value.get("flat", value)
    return np.asarray(value, dtype=np.float32).copy()


def _current_chronic_info(env: MAEnvWrapper) -> Dict[str, str]:
    getter = getattr(env, "get_current_chronic_info", None)
    info = getter() if callable(getter) else {}
    return {
        "chronic_name": _str_or_unknown(info.get("chronic_name")),
        "chronic_fingerprint": _str_or_unknown(info.get("chronic_fingerprint")),
        "chronic_datetime": _str_or_unknown(info.get("chronic_datetime")),
    }


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {_safe_path(path)} already exists and is not empty. "
                "Use a new --output-dir or pass --overwrite true."
            )
        for child in path.iterdir():
            if child.name in {"metadata.json", "summary.json"}:
                child.unlink()
            elif child.name.startswith("shard_") and child.suffix == ".npz":
                child.unlink()
            elif child.is_dir() and child.name in {
                "tmp",
                SHARDS_DIR_NAME,
                METADATA_DIR_NAME,
            }:
                shutil.rmtree(child)
            else:
                raise FileExistsError(
                    f"Refusing to delete unexpected file in output dir: {child}"
                )
    path.mkdir(parents=True, exist_ok=True)
    shards_dir(path).mkdir(parents=True, exist_ok=True)
    metadata_dir(path).mkdir(parents=True, exist_ok=True)


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


def _outcome_label(delta: float, *, valid: bool, tolerance: float) -> int:
    if not valid or not np.isfinite(delta):
        return OUTCOME_LABEL_INVALID
    if delta < -tolerance:
        return OUTCOME_LABEL_IMPROVED
    if delta > tolerance:
        return OUTCOME_LABEL_WORSENED
    return OUTCOME_LABEL_NEUTRAL


class BruteForceOutcomeWriter:
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
        self.total_rows = 0
        self.agent_rows: Dict[str, Dict[str, List[Any]]] = {
            agent: defaultdict(list) for agent in agent_ids
        }
        self.paths: List[Path] = []

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
            int64_keys = {"episode_id", "dataset_step"}
            int32_keys = {
                "episode_step",
                "action_id",
                "worst_line_before",
                "worst_line_after_action",
                "worst_line_after_do_nothing",
            }
            int8_keys = {"label_vs_now", "label_vs_do_nothing"}
            bool_keys = {
                "action_is_valid",
                "action_is_legal",
                "action_is_ambiguous",
                "simulation_exception",
                "sim_done",
            }
            str_keys = {
                "chronic_name",
                "chronic_fingerprint",
                "chronic_datetime",
                "validation_error",
                "simulation_error",
            }
            for key, items in values.items():
                if key == "obs":
                    arrays[f"obs_{agent}"] = np.stack(items).astype(np.float32)
                elif key in int64_keys:
                    arrays[f"{prefix}_{key}"] = np.asarray(items, dtype=np.int64)
                elif key in int32_keys:
                    arrays[f"{prefix}_{key}"] = np.asarray(items, dtype=np.int32)
                elif key in int8_keys:
                    arrays[f"{prefix}_{key}"] = np.asarray(items, dtype=np.int8)
                elif key in bool_keys:
                    arrays[f"{prefix}_{key}"] = np.asarray(items, dtype=bool)
                elif key in str_keys:
                    arrays[f"{prefix}_{key}"] = np.asarray(items, dtype=str)
                else:
                    arrays[f"{prefix}_{key}"] = np.asarray(items, dtype=np.float32)

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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_metadata(dataset_dir: Path, payload: Dict[str, Any]) -> Path:
    path = metadata_path(dataset_dir)
    _write_json(path, payload)
    export_metadata_file(dataset_dir, path, "metadata")
    return path


def _metadata(
    *,
    status: str,
    cli: Namespace,
    env_args: Namespace,
    agent_ids: List[str],
    action_sizes: Dict[str, int],
    obs_shapes: Dict[str, List[int]],
    writer: BruteForceOutcomeWriter,
    env_steps: int,
    candidate_states: int,
    skipped_states: int,
    completed_episodes: int,
    unique_fingerprints: set,
    episode_lengths: List[int],
    covered_action_ids_by_agent: Dict[str, set],
) -> Dict[str, Any]:
    unique_actions_by_agent = {
        agent: int(len(covered_action_ids_by_agent.get(agent, set())))
        for agent in agent_ids
    }
    action_coverage_by_agent = {
        agent: (
            unique_actions_by_agent[agent] / int(action_sizes[agent])
            if int(action_sizes[agent]) > 0
            else float("nan")
        )
        for agent in agent_ids
    }
    return {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector": "teacher_student.collect_bruteforce_action_outcomes",
        "dataset_mode": "action_outcomes",
        "collection_method": "bruteforce_teacher",
        "phase": 0,
        "env_id": env_args.env_id,
        "seed": int(env_args.seed),
        "split": cli.split,
        "split_chronics": bool(env_args.split_chronics),
        "chronic_shard_index": int(cli.chronic_shard_index),
        "chronic_shard_count": int(cli.chronic_shard_count),
        "collection_rho_threshold": float(cli.collection_rho_threshold),
        "outcome_delta_tolerance": float(cli.outcome_delta_tolerance),
        "outcome_time_step": int(cli.outcome_time_step),
        "outcome_rollout_policy": cli.outcome_rollout_policy,
        "outcome_action_sample_size": cli.outcome_action_sample_size,
        "outcome_include_action_zero": bool(cli.outcome_include_action_zero),
        "outcome_resample_actions_per_state": bool(
            cli.outcome_resample_actions_per_state
        ),
        "outcome_sim_workers": int(cli.outcome_sim_workers),
        "outcome_sim_backend": (
            "process_pool" if int(cli.outcome_sim_workers) > 1 else "sequential"
        ),
        "outcome_sim_start_method": cli.outcome_sim_start_method,
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
        },
        "max_episodes_requested": cli.max_episodes,
        "max_env_steps_requested": cli.max_env_steps,
        "n_shards": int(len(writer.paths)),
        "shards": [_safe_path(path) for path in writer.paths],
        "n_env_steps": int(env_steps),
        "n_candidate_states": int(candidate_states),
        "n_skipped_states_below_threshold": int(skipped_states),
        "n_outcome_examples": int(writer.total_rows),
        "n_unique_actions_by_agent": unique_actions_by_agent,
        "action_coverage_by_agent": action_coverage_by_agent,
        "n_completed_episodes": int(completed_episodes),
        "n_unique_chronic_fingerprints": int(len(unique_fingerprints)),
        "episode_length_mean": (
            float(np.mean(episode_lengths)) if episode_lengths else None
        ),
        "episode_length_min": int(min(episode_lengths)) if episode_lengths else None,
        "episode_length_max": int(max(episode_lengths)) if episode_lengths else None,
    }


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Brute-force simulate Grid2Op topology actions to build a reduced "
            "action set before RL training."
        )
    )
    parser.add_argument("--env-id", type=str, default="bus36")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--split-chronics", type=str2bool, default=True)
    parser.add_argument("--test-chronics-pct", type=float, default=0.2)
    parser.add_argument("--chronic-split-seed", type=int, default=None)
    parser.add_argument("--chronic-shard-count", type=int, default=1)
    parser.add_argument("--chronic-shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, default=0, choices=[0, 1])
    parser.add_argument("--decentralized", type=str2bool, default=True)
    parser.add_argument("--optimize-mem", type=str2bool, default=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--collection-rho-threshold", type=float, default=0.90)
    parser.add_argument("--outcome-delta-tolerance", type=float, default=1e-3)
    parser.add_argument("--outcome-time-step", type=int, default=1)
    parser.add_argument(
        "--outcome-rollout-policy",
        type=str,
        default="best_simulated",
        choices=["best_simulated", "do_nothing"],
    )
    parser.add_argument("--outcome-action-sample-size", type=int, default=None)
    parser.add_argument("--outcome-include-action-zero", type=str2bool, default=True)
    parser.add_argument(
        "--outcome-resample-actions-per-state",
        type=str2bool,
        default=True,
        help=(
            "If true, draw a fresh random candidate-action subset at every "
            "collected high-rho state. If false, reuse one fixed subset for the "
            "whole job."
        ),
    )
    parser.add_argument(
        "--outcome-sim-workers",
        type=int,
        default=1,
        help=(
            "Number of process-owned Grid2Op environment replicas used to simulate "
            "candidate actions at one collected state. The default 1 preserves the "
            "original sequential path."
        ),
    )
    parser.add_argument(
        "--outcome-sim-start-method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method for --outcome-sim-workers > 1.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--compress", type=str2bool, default=True)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--timing-every-env-steps", type=int, default=100)
    parser.add_argument("--reduce-after", type=str2bool, default=True)
    parser.add_argument("--action-reduction-top-k", type=int, default=208)
    parser.add_argument("--action-reduction-min-count", type=int, default=1)
    parser.add_argument(
        "--action-reduction-metric",
        type=str,
        default="delta_vs_do_nothing",
        choices=["delta_vs_do_nothing", "delta_vs_now", "rho_after_action"],
    )
    parser.add_argument(
        "--action-reduction-method",
        type=str,
        default="best_per_state",
        choices=["best_per_state", "all_improving"],
    )
    parser.add_argument(
        "--action-reduction-require-improvement",
        type=str2bool,
        default=True,
    )
    return parser.parse_args()


def _build_env_args(cli: Namespace) -> Namespace:
    env_args = get_env_args([])
    env_args.env_id = cli.env_id
    env_args.action_type = "topology"
    env_args.reduced_action_space = ""
    env_args.difficulty = cli.difficulty
    env_args.decentralized = cli.decentralized
    env_args.optimize_mem = cli.optimize_mem
    env_args.split_chronics = bool(cli.split_chronics and cli.split != "all")
    env_args.test_chronics_pct = cli.test_chronics_pct
    env_args.chronic_split_seed = cli.chronic_split_seed
    env_args.chronic_shard_count = cli.chronic_shard_count
    env_args.chronic_shard_index = cli.chronic_shard_index
    env_args.seed = int(cli.seed)
    env_args.norm_obs = False
    env_args.use_heuristic = False
    env_args.actor_encoder = "mlp"
    env_args.critic_encoder = "mlp"
    env_args.track = False
    env_args.n1_reward = bool(getattr(env_args, "n1_reward", False))
    return env_args


def _run_reducer(cli: Namespace, output_dir: Path) -> None:
    reducer = TASK_DIR / "teacher_student" / "reduce_action_space_from_outcomes.py"
    cmd = [
        sys.executable,
        str(reducer),
        "--dataset",
        str(output_dir),
        "--top-k",
        str(cli.action_reduction_top_k),
        "--min-count",
        str(cli.action_reduction_min_count),
        "--metric",
        cli.action_reduction_metric,
        "--selection-method",
        cli.action_reduction_method,
        "--require-improvement",
        "true" if cli.action_reduction_require_improvement else "false",
    ]
    print("Running reducer:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(TASK_DIR), check=True)


@contextlib.contextmanager
def _suppress_grid2op_cleanup_stderr():
    """Hide Grid2Op double-close noise emitted from BaseEnv.__del__.

    Some Grid2Op observation environments raise "closed already" from their
    destructor after the parent env has been closed. That exception is printed as
    "Exception ignored in ..." on stderr during process cleanup, even though the
    collection already finished correctly.
    """
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stderr(devnull):
            yield


def _simulation_worker_main(remote, env_args: Namespace, chronic_split: Optional[str]) -> None:
    env = None
    exit_code = 0
    try:
        set_random_seed(int(env_args.seed))
        env = MAEnvWrapper(env_args, eval_env=True, chronic_split=chronic_split)
        agent_ids = list(env.g2op_ma_env.agents)

        while True:
            cmd, data = remote.recv()
            if cmd == "close":
                remote.send({"ok": True})
                break

            if cmd == "reset":
                env.reset()
                remote.send(
                    {
                        "ok": True,
                        "max_rho": float(env.get_current_max_rho()),
                    }
                )
                continue

            if cmd == "simulate":
                indexed_requests = list(data["indexed_requests"])
                requests = [request for _, request in indexed_requests]
                outcomes = env.simulate_action_outcomes(
                    requests,
                    time_step=int(data.get("time_step", 1)),
                    num_workers=1,
                )
                remote.send(
                    {
                        "ok": True,
                        "outcomes": [
                            (int(idx), outcome)
                            for (idx, _), outcome in zip(indexed_requests, outcomes)
                        ],
                    }
                )
                continue

            if cmd == "sync":
                done = False
                for actions in data["actions"]:
                    _, _, terminations, truncations, _ = env.step(actions)
                    done = bool(
                        terminations[agent_ids[0]] or truncations[agent_ids[0]]
                    )
                    if done:
                        break
                remote.send(
                    {
                        "ok": True,
                        "done": done,
                        "max_rho": (
                            float("nan") if done else float(env.get_current_max_rho())
                        ),
                    }
                )
                continue

            raise NotImplementedError(f"Unknown worker command: {cmd}")
    except Exception:
        exit_code = 1
        try:
            remote.send({"ok": False, "error": traceback.format_exc()})
        except Exception:
            pass
    finally:
        if env is not None:
            with _suppress_grid2op_cleanup_stderr():
                try:
                    env.close()
                except Exception:
                    pass
                env = None
                gc.collect()
        try:
            remote.close()
        except Exception:
            pass
        # Grid2Op can raise from BaseEnv.__del__ during interpreter shutdown
        # after internal observation envs have already been closed. Worker
        # results have already been sent through the pipe, so bypass Python's
        # destructor sweep to avoid turning cleanup noise into failed jobs.
        os._exit(exit_code)


class SimulationWorkerPool:
    def __init__(
        self,
        *,
        env_args: Namespace,
        chronic_split: Optional[str],
        num_workers: int,
        start_method: str,
        sync_tolerance: float = 1e-5,
    ) -> None:
        self.num_workers = max(0, int(num_workers))
        self.sync_tolerance = float(sync_tolerance)
        self.ctx = mp.get_context(start_method)
        self.remotes = []
        self.processes = []

        for _ in range(self.num_workers):
            parent_remote, child_remote = self.ctx.Pipe()
            process = self.ctx.Process(
                target=_simulation_worker_main,
                args=(child_remote, env_args, chronic_split),
            )
            process.daemon = True
            process.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(process)

    def _recv(self, remote) -> Dict[str, Any]:
        response = remote.recv()
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "Unknown simulation worker error"))
        return response

    def reset(self, reference_max_rho: float) -> None:
        for remote in self.remotes:
            remote.send(("reset", {}))
        responses = [self._recv(remote) for remote in self.remotes]
        self._check_rhos(
            [float(response["max_rho"]) for response in responses],
            reference_max_rho,
            context="reset",
        )

    def simulate(
        self,
        requests: List[Dict[str, Any]],
        *,
        time_step: int,
    ) -> List[Dict[str, Any]]:
        if not requests:
            return []
        worker_count = min(self.num_workers, len(requests))
        chunks = [[] for _ in range(worker_count)]
        for idx, request in enumerate(requests):
            chunks[idx % worker_count].append((idx, request))

        active_remotes = self.remotes[:worker_count]
        for remote, chunk in zip(active_remotes, chunks):
            remote.send(
                (
                    "simulate",
                    {
                        "indexed_requests": chunk,
                        "time_step": int(time_step),
                    },
                )
            )

        outcomes: List[Optional[Dict[str, Any]]] = [None] * len(requests)
        for remote in active_remotes:
            response = self._recv(remote)
            for idx, outcome in response["outcomes"]:
                outcomes[int(idx)] = outcome

        if any(outcome is None for outcome in outcomes):
            raise RuntimeError("Internal error: missing worker simulation outcome.")
        return [outcome for outcome in outcomes if outcome is not None]

    def sync(
        self,
        pending_actions: List[Dict[str, int]],
        *,
        reference_max_rho: float,
    ) -> None:
        if not pending_actions:
            self._check_rhos(
                [float(reference_max_rho)] * self.num_workers,
                float(reference_max_rho),
                context="sync",
            )
            return
        for remote in self.remotes:
            remote.send(("sync", {"actions": pending_actions}))
        responses = [self._recv(remote) for remote in self.remotes]
        worker_dones = [bool(response["done"]) for response in responses]
        if any(worker_dones):
            raise RuntimeError(
                "Simulation worker reached done while replaying pending actions "
                f"before a collector candidate state: {worker_dones[:8]}..."
            )
        self._check_rhos(
            [float(response["max_rho"]) for response in responses],
            float(reference_max_rho),
            context="sync",
        )

    def step(
        self,
        actions: Dict[str, int],
        *,
        main_done: bool,
        reference_max_rho: Optional[float],
    ) -> None:
        if main_done:
            raise RuntimeError(
                "SimulationWorkerPool.step is no longer used for terminal steps."
            )
        self.sync([actions], reference_max_rho=float(reference_max_rho))

    def _check_rhos(
        self,
        worker_rhos: List[float],
        reference_max_rho: float,
        *,
        context: str,
    ) -> None:
        reference = float(reference_max_rho)
        for idx, value in enumerate(worker_rhos):
            if np.isfinite(reference) or np.isfinite(value):
                if not np.isclose(
                    value,
                    reference,
                    rtol=0.0,
                    atol=self.sync_tolerance,
                    equal_nan=True,
                ):
                    raise RuntimeError(
                        "Simulation worker desynchronized from collector during "
                        f"{context}: worker={idx} rho={value}, main rho={reference}."
                    )

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", {}))
            except Exception:
                pass
        for remote in self.remotes:
            try:
                self._recv(remote)
            except Exception:
                pass
            try:
                remote.close()
            except Exception:
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def main() -> None:
    cli = parse_args()
    if cli.max_episodes is not None and cli.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive when provided.")
    if cli.max_env_steps is not None and cli.max_env_steps <= 0:
        raise ValueError("--max-env-steps must be positive when provided.")
    if cli.outcome_action_sample_size is not None and cli.outcome_action_sample_size <= 0:
        raise ValueError("--outcome-action-sample-size must be positive when provided.")
    if cli.outcome_sim_workers <= 0:
        raise ValueError("--outcome-sim-workers must be positive.")
    if cli.shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    if cli.timing_every_env_steps <= 0:
        raise ValueError("--timing-every-env-steps must be positive.")
    if cli.chronic_shard_count <= 0:
        raise ValueError("--chronic-shard-count must be positive.")
    if cli.chronic_shard_index < 0 or cli.chronic_shard_index >= cli.chronic_shard_count:
        raise ValueError(
            "--chronic-shard-index must be in "
            f"[0, {cli.chronic_shard_count})."
        )

    output_dir = cli.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (TASK_DIR / output_dir).resolve()
    _prepare_output_dir(output_dir, cli.overwrite)

    env_args = _build_env_args(cli)
    set_random_seed(env_args.seed)
    chronic_split = None if cli.split == "all" else cli.split
    env = MAEnvWrapper(env_args, eval_env=True, chronic_split=chronic_split)
    agent_ids = list(env.g2op_ma_env.agents)
    action_sizes = {agent: int(env.action_space[agent].n) for agent in agent_ids}
    obs_shapes = {agent: list(env.observation_space[agent].shape) for agent in agent_ids}
    writer = BruteForceOutcomeWriter(
        shards_dir(output_dir),
        agent_ids,
        cli.shard_size,
        cli.compress,
    )
    rng = np.random.default_rng(env_args.seed)
    fixed_candidate_ids_by_agent = {
        agent: _candidate_action_ids(
            action_size=action_sizes[agent],
            include_action_zero=cli.outcome_include_action_zero,
            sample_size=cli.outcome_action_sample_size,
            rng=rng,
        )
        for agent in agent_ids
    }

    target_episodes = cli.max_episodes
    if target_episodes is None:
        split_size = getattr(env, "chronic_split_size", None)
        target_episodes = int(split_size) if split_size is not None else 1

    print("========== Brute-force action-outcome collection ==========")
    print(f"Env id: {env_args.env_id}")
    print(f"Split: {cli.split}")
    print(
        f"Chronic shard: {cli.chronic_shard_index}/"
        f"{cli.chronic_shard_count}"
    )
    print(f"Output dir: {_safe_path(output_dir)}")
    print(f"Target episodes: {target_episodes}")
    print(f"Max env steps: {cli.max_env_steps or 'none'}")
    print(f"Collection rho threshold: {cli.collection_rho_threshold}")
    print(f"Rollout policy: {cli.outcome_rollout_policy}")
    print(f"Action sample size: {cli.outcome_action_sample_size or 'all'}")
    print(
        "Resample actions per collected state: "
        f"{cli.outcome_resample_actions_per_state}"
    )
    print(
        "Simulation workers: "
        f"{cli.outcome_sim_workers} "
        f"({'process_pool' if cli.outcome_sim_workers > 1 else 'sequential'})"
    )
    if cli.outcome_sim_workers > 1:
        print(f"Simulation start method: {cli.outcome_sim_start_method}")
    print(f"Reduce after: {cli.reduce_after}")
    for agent in agent_ids:
        print(
            f"{agent}: evaluating {len(fixed_candidate_ids_by_agent[agent])}/"
            f"{action_sizes[agent]} actions per collected state"
        )
    print("===========================================================")

    obs, _ = env.reset()
    sim_pool = None
    if cli.outcome_sim_workers > 1:
        print(
            "Starting process simulation workers: "
            f"{cli.outcome_sim_workers}",
            flush=True,
        )
        sim_pool = SimulationWorkerPool(
            env_args=env_args,
            chronic_split=chronic_split,
            num_workers=cli.outcome_sim_workers,
            start_method=cli.outcome_sim_start_method,
        )
        sim_pool.reset(float(env.get_current_max_rho()))
    env_steps = 0
    completed_episodes = 0
    episode_step = 0
    candidate_states = 0
    skipped_states = 0
    unique_fingerprints = set()
    episode_lengths: List[int] = []
    covered_action_ids_by_agent = {agent: set() for agent in agent_ids}
    pending_worker_actions: List[Dict[str, int]] = []
    run_start_time = time.perf_counter()
    timed_env_steps = 0
    timed_sim_actions = 0
    timed_wall_seconds = 0.0
    timed_env_step_seconds = 0.0
    timed_collection_seconds = 0.0
    timed_sim_action_seconds = 0.0

    while completed_episodes < int(target_episodes):
        step_start_time = time.perf_counter()
        collection_seconds = 0.0
        sim_action_seconds = 0.0
        sim_action_count = 0

        if cli.max_env_steps is not None and env_steps >= cli.max_env_steps:
            print(f"Reached --max-env-steps={cli.max_env_steps}; stopping early.")
            break

        chronic_info = _current_chronic_info(env)
        unique_fingerprints.add(chronic_info["chronic_fingerprint"])
        global_max_rho = float(env.get_current_max_rho())
        local_max_rhos = env.get_current_agent_max_rho()
        should_collect = bool(
            np.isfinite(global_max_rho)
            and global_max_rho >= float(cli.collection_rho_threshold)
        )
        best_rollout_action = {agent: 0 for agent in agent_ids}
        best_rollout_rho = float("inf")

        if should_collect:
            candidate_states += 1
            if sim_pool is not None:
                sync_start_time = time.perf_counter()
                sim_pool.sync(
                    pending_worker_actions,
                    reference_max_rho=global_max_rho,
                )
                pending_worker_actions = []
                collection_seconds += time.perf_counter() - sync_start_time
            collection_start_time = time.perf_counter()
            sim_start_time = time.perf_counter()
            do_nothing = env.simulate_action_outcomes(
                [{"agent_id": agent_ids[0], "action_id": 0}],
                time_step=cli.outcome_time_step,
                num_workers=1,
            )[0]
            rho_after_do_nothing = float(do_nothing["rho_after"])
            worst_line_after_do_nothing = int(do_nothing["worst_line_after"])
            if cli.outcome_resample_actions_per_state:
                candidate_ids_by_agent = {
                    agent: _candidate_action_ids(
                        action_size=action_sizes[agent],
                        include_action_zero=cli.outcome_include_action_zero,
                        sample_size=cli.outcome_action_sample_size,
                        rng=rng,
                    )
                    for agent in agent_ids
                }
            else:
                candidate_ids_by_agent = fixed_candidate_ids_by_agent
            for agent, action_ids in candidate_ids_by_agent.items():
                covered_action_ids_by_agent[agent].update(
                    int(action_id) for action_id in action_ids
                )
            requests = [
                {"agent_id": agent, "action_id": int(action_id)}
                for agent in agent_ids
                for action_id in candidate_ids_by_agent[agent]
            ]
            if sim_pool is None:
                outcomes = env.simulate_action_outcomes(
                    requests,
                    time_step=cli.outcome_time_step,
                    num_workers=1,
                )
            else:
                outcomes = sim_pool.simulate(
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
                values = {
                    "episode_id": completed_episodes,
                    "episode_step": episode_step,
                    "dataset_step": env_steps,
                    "action_id": action_id,
                    "global_max_rho_before": global_max_rho,
                    "local_max_rho_before": float(local_max_rhos.get(agent, np.nan)),
                    "rho_before": rho_before,
                    "rho_after_action": rho_after_action,
                    "rho_after_do_nothing": rho_after_do_nothing,
                    "delta_vs_now": delta_vs_now,
                    "delta_vs_do_nothing": delta_vs_do_nothing,
                    "label_vs_now": _outcome_label(
                        delta_vs_now,
                        valid=valid,
                        tolerance=cli.outcome_delta_tolerance,
                    ),
                    "label_vs_do_nothing": _outcome_label(
                        delta_vs_do_nothing,
                        valid=valid,
                        tolerance=cli.outcome_delta_tolerance,
                    ),
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
                    _write_metadata(
                        output_dir,
                        _metadata(
                            status="running",
                            cli=cli,
                            env_args=env_args,
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
                            covered_action_ids_by_agent=covered_action_ids_by_agent,
                        ),
                    )
                    print(f"wrote {_safe_path(flushed)} at env_step={env_steps}")
            collection_seconds += time.perf_counter() - collection_start_time
        else:
            skipped_states += 1

        rollout_actions = (
            best_rollout_action
            if cli.outcome_rollout_policy == "best_simulated"
            else {agent: 0 for agent in agent_ids}
        )
        env_step_start_time = time.perf_counter()
        next_obs, reward, terminations, truncations, info = env.step(rollout_actions)
        env_step_seconds = time.perf_counter() - env_step_start_time
        done = bool(terminations[agent_ids[0]] or truncations[agent_ids[0]])
        done_episode_length = 0
        if done:
            done_episode_length = max(int(env.g2op_ma_env._cent_env.nb_time_step), 1)

        env_steps += 1
        episode_step += 1
        wall_step_seconds = time.perf_counter() - step_start_time
        timed_env_steps += 1
        timed_sim_actions += sim_action_count
        timed_wall_seconds += wall_step_seconds
        timed_env_step_seconds += env_step_seconds
        timed_collection_seconds += collection_seconds
        timed_sim_action_seconds += sim_action_seconds

        if env_steps == 1 or env_steps % max(cli.timing_every_env_steps, 1) == 0:
            elapsed = time.perf_counter() - run_start_time
            avg_wall_step = _safe_div(timed_wall_seconds, timed_env_steps)
            avg_sim_action = _safe_div(timed_sim_action_seconds, timed_sim_actions)
            last_sim_action = _safe_div(sim_action_seconds, sim_action_count)
            eta_seconds = (
                avg_wall_step * max(cli.max_env_steps - env_steps, 0)
                if cli.max_env_steps is not None
                else float("nan")
            )
            print(
                "timing: "
                f"env_steps={env_steps} "
                f"episodes={completed_episodes}/{target_episodes} "
                f"last_step={wall_step_seconds:.3f}s "
                f"avg_step={avg_wall_step:.3f}s "
                f"last_env_step={env_step_seconds:.3f}s "
                f"last_collect={collection_seconds:.3f}s "
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
                    f"last_len={done_episode_length}",
                    flush=True,
                )
            obs, _ = env.reset()
            pending_worker_actions = []
            if sim_pool is not None:
                sim_pool.reset(float(env.get_current_max_rho()))
            episode_step = 0
        else:
            if sim_pool is not None:
                pending_worker_actions.append(dict(rollout_actions))
            obs = next_obs

    final_shard = writer.flush()
    if final_shard is not None:
        print(f"wrote {_safe_path(final_shard)} at env_step={env_steps}")
    final_metadata = _metadata(
        status="complete",
        cli=cli,
        env_args=env_args,
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
        covered_action_ids_by_agent=covered_action_ids_by_agent,
    )
    metadata_file = _write_metadata(output_dir, final_metadata)
    if sim_pool is not None:
        sim_pool.close()
    with _suppress_grid2op_cleanup_stderr():
        try:
            env.close()
        except Exception:
            pass
        env = None
        gc.collect()

    print("========== Brute-force collection complete ==========")
    print(f"Output dir: {_safe_path(output_dir)}")
    print(f"Metadata: {_safe_path(metadata_file)}")
    print(f"Shards: {len(writer.paths)}")
    print(f"Env steps: {env_steps}")
    print(f"Candidate states: {candidate_states}")
    print(f"Outcome examples: {writer.total_rows}")
    print(f"Completed episodes: {completed_episodes}")
    for agent in agent_ids:
        covered = len(covered_action_ids_by_agent.get(agent, set()))
        total = int(action_sizes[agent])
        frac = covered / total if total > 0 else float("nan")
        print(
            f"{agent}: unique_actions_tested={covered}/{total} "
            f"({frac:.4%})"
        )
    print("====================================================")

    if cli.reduce_after:
        _run_reducer(cli, output_dir)


if __name__ == "__main__":
    main()
