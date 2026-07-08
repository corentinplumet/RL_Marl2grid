#!/usr/bin/env python3
"""Evaluate the do-nothing policy on a chosen Grid2Op environment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from common.utils import set_random_seed, str2bool
from env.config import get_env_args
from env.utils import MAEnvWrapper
from teacher_student.collect_bruteforce_action_outcomes import (
    _format_duration,
    _suppress_grid2op_cleanup_stderr,
)


def _resolve_task_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (TASK_DIR / path).resolve()


def _safe_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(TASK_DIR))
    except ValueError:
        return str(path.resolve())


def _build_env_args(cli: Namespace) -> Namespace:
    env_args = get_env_args([])
    env_args.env_id = cli.env_id
    env_args.action_type = cli.action_type
    env_args.reduced_action_space = ""
    env_args.difficulty = cli.difficulty
    env_args.decentralized = cli.decentralized
    env_args.optimize_mem = cli.optimize_mem
    env_args.split_chronics = bool(cli.split_chronics and cli.split != "all")
    env_args.test_chronics_pct = cli.test_chronics_pct
    env_args.chronic_split_seed = cli.chronic_split_seed
    env_args.chronic_shard_count = cli.chronic_shard_count
    env_args.chronic_shard_index = cli.chronic_shard_index
    env_args.env_config_path = cli.env_config_path
    env_args.seed = int(cli.seed)
    env_args.norm_obs = False
    env_args.use_heuristic = False
    env_args.actor_encoder = "mlp"
    env_args.critic_encoder = "mlp"
    env_args.track = False
    env_args.n1_reward = False
    return env_args


def _current_chronic_info(env: MAEnvWrapper) -> Dict[str, str]:
    getter = getattr(env, "get_current_chronic_info", None)
    info = getter() if callable(getter) else {}
    return {
        "chronic_name": str(info.get("chronic_name", "unknown")),
        "chronic_path": str(info.get("chronic_path", "unknown")),
        "chronic_fingerprint": str(info.get("chronic_fingerprint", "unknown")),
        "chronic_datetime": str(info.get("chronic_datetime", "unknown")),
    }


def _max_episode_duration(env: MAEnvWrapper) -> int:
    try:
        duration = env.g2op_ma_env._cent_env.chronics_handler.max_episode_duration()
        return max(int(duration), 1)
    except Exception:
        return 1


def _reward_scalar(reward: Any) -> float:
    if isinstance(reward, dict):
        values = []
        for value in reward.values():
            try:
                values.append(float(value))
            except Exception:
                pass
        return float(np.mean(values)) if values else float("nan")
    try:
        return float(reward)
    except Exception:
        return float("nan")


def _run_do_nothing_episode(
    *,
    env: MAEnvWrapper,
    agent_ids: List[str],
    max_env_steps: Optional[int],
) -> Dict[str, Any]:
    _, _ = env.reset()
    chronic_info = _current_chronic_info(env)
    max_steps = _max_episode_duration(env)
    steps = 0
    total_reward = 0.0
    done = False
    max_rho_values: List[float] = []

    while not done:
        if max_env_steps is not None and steps >= int(max_env_steps):
            break
        try:
            max_rho = float(env.get_current_max_rho())
        except Exception:
            max_rho = float("nan")
        if np.isfinite(max_rho):
            max_rho_values.append(max_rho)

        actions = {agent: 0 for agent in agent_ids}
        _, reward, terminations, truncations, _ = env.step(actions)
        total_reward += _reward_scalar(reward)
        done = bool(terminations[agent_ids[0]] or truncations[agent_ids[0]])
        steps += 1

    max_rho_array = np.asarray(max_rho_values, dtype=np.float32)
    return {
        **chronic_info,
        "steps": int(steps),
        "max_steps": int(max_steps),
        "survival": float(steps / max(max_steps, 1)),
        "full_survival": bool(steps >= max_steps),
        "return": float(total_reward),
        "initial_max_rho": (
            float(max_rho_array[0]) if max_rho_array.size else float("nan")
        ),
        "mean_max_rho": (
            float(max_rho_array.mean()) if max_rho_array.size else float("nan")
        ),
        "peak_max_rho": (
            float(max_rho_array.max()) if max_rho_array.size else float("nan")
        ),
    }


def _write_outputs(
    *,
    output_dir: Path,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "do_nothing_summary.json"
    csv_path = output_dir / "do_nothing_episodes.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    fieldnames = [
        "episode",
        "chronic_name",
        "chronic_path",
        "chronic_fingerprint",
        "chronic_datetime",
        "steps",
        "max_steps",
        "survival",
        "full_survival",
        "return",
        "initial_max_rho",
        "mean_max_rho",
        "peak_max_rho",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return json_path, csv_path


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the pure do-nothing policy on a selected chronic split."
    )
    parser.add_argument("--env-id", type=str, default="bus36")
    parser.add_argument(
        "--action-type",
        type=str,
        default="topology",
        choices=["topology", "redispatch"],
    )
    parser.add_argument("--env-config-path", type=str, default="scenario.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--split-chronics", type=str2bool, default=True)
    parser.add_argument("--test-chronics-pct", type=float, default=0.2)
    parser.add_argument("--chronic-split-seed", type=int, default=None)
    parser.add_argument("--chronic-shard-count", type=int, default=1)
    parser.add_argument("--chronic-shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, default=0, choices=[0, 1])
    parser.add_argument("--decentralized", type=str2bool, default=True)
    parser.add_argument("--optimize-mem", type=str2bool, default=True)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Number of chronics/episodes to evaluate. Defaults to the split size.",
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=None,
        help="Optional cap on env steps per chronic for smoke tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/do_nothing_eval"),
    )
    parser.add_argument("--progress", type=str2bool, default=True)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.max_episodes is not None and cli.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive when provided.")
    if cli.max_env_steps is not None and cli.max_env_steps <= 0:
        raise ValueError("--max-env-steps must be positive when provided.")

    output_dir = _resolve_task_path(cli.output_dir)
    env_args = _build_env_args(cli)
    set_random_seed(env_args.seed)
    chronic_split = None if cli.split == "all" else cli.split

    env = MAEnvWrapper(env_args, eval_env=True, chronic_split=chronic_split)
    agent_ids = list(env.g2op_ma_env.agents)
    target_episodes = cli.max_episodes
    if target_episodes is None:
        split_size = getattr(env, "chronic_split_size", None)
        target_episodes = int(split_size) if split_size is not None else 1

    print("========== Do-Nothing Evaluation ==========")
    print(f"Env id: {env_args.env_id}")
    print(f"Split: {cli.split}")
    print(f"Split chronics: {env_args.split_chronics}")
    print(f"Test chronics pct: {env_args.test_chronics_pct}")
    print(f"Chronic split seed: {env_args.chronic_split_seed}")
    print(f"Chronic shard: {env_args.chronic_shard_index}/{env_args.chronic_shard_count}")
    print(f"Episodes: {target_episodes}")
    print(f"Max env steps: {cli.max_env_steps if cli.max_env_steps else 'none'}")
    print(f"Output dir: {_safe_path(output_dir)}")
    print("===========================================")

    rows: List[Dict[str, Any]] = []
    start_time = time.perf_counter()
    try:
        for episode in range(int(target_episodes)):
            result = _run_do_nothing_episode(
                env=env,
                agent_ids=agent_ids,
                max_env_steps=cli.max_env_steps,
            )
            row = {"episode": episode, **result}
            rows.append(row)
            if cli.progress:
                elapsed = time.perf_counter() - start_time
                done_eps = episode + 1
                eta = elapsed / done_eps * (int(target_episodes) - done_eps)
                print(
                    "episode "
                    f"{done_eps}/{target_episodes}: "
                    f"survival={100 * row['survival']:.2f}% "
                    f"steps={row['steps']}/{row['max_steps']} "
                    f"full={row['full_survival']} "
                    f"rho_peak={row['peak_max_rho']:.4f} "
                    f"name={row['chronic_name']} "
                    f"path={row['chronic_path']} "
                    f"fp={str(row['chronic_fingerprint'])[:8]} "
                    f"elapsed={_format_duration(elapsed)} "
                    f"eta={_format_duration(eta)}",
                    flush=True,
                )
    finally:
        with _suppress_grid2op_cleanup_stderr():
            try:
                env.close()
            except Exception:
                pass
        gc.collect()

    survivals = np.asarray([row["survival"] for row in rows], dtype=float)
    full_survivals = np.asarray([row["full_survival"] for row in rows], dtype=bool)
    steps = np.asarray([row["steps"] for row in rows], dtype=float)
    unique_fingerprints = {
        str(row["chronic_fingerprint"])
        for row in rows
        if str(row["chronic_fingerprint"]) != "unknown"
    }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "env_id": env_args.env_id,
        "split": cli.split,
        "split_chronics": bool(env_args.split_chronics),
        "test_chronics_pct": float(env_args.test_chronics_pct),
        "chronic_split_seed": env_args.chronic_split_seed,
        "chronic_shard_count": int(env_args.chronic_shard_count),
        "chronic_shard_index": int(env_args.chronic_shard_index),
        "n_episodes": len(rows),
        "n_unique_fingerprints": len(unique_fingerprints),
        "max_env_steps": cli.max_env_steps,
        "mean_survival": float(survivals.mean()) if rows else float("nan"),
        "median_survival": float(np.median(survivals)) if rows else float("nan"),
        "full_survival_rate": float(full_survivals.mean()) if rows else float("nan"),
        "mean_steps": float(steps.mean()) if rows else float("nan"),
        "episodes": rows,
    }
    json_path, csv_path = _write_outputs(
        output_dir=output_dir,
        rows=rows,
        summary=summary,
    )

    print("========== Do-Nothing Evaluation Complete ==========")
    print(f"Episodes: {len(rows)}")
    print(f"Unique fingerprints: {summary['n_unique_fingerprints']}")
    print(f"Mean survival: {100 * summary['mean_survival']:.3f}%")
    print(f"Median survival: {100 * summary['median_survival']:.3f}%")
    print(f"Full survival rate: {100 * summary['full_survival_rate']:.3f}%")
    print(f"Mean steps: {summary['mean_steps']:.1f}")
    print(f"JSON: {_safe_path(json_path)}")
    print(f"CSV: {_safe_path(csv_path)}")
    print("====================================================")


if __name__ == "__main__":
    main()
