#!/usr/bin/env python3
"""Evaluate a reduced action space with a simulator-greedy policy."""

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
    SimulationWorkerPool,
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
    env_args.action_type = "topology"
    env_args.reduced_action_space = str(_resolve_task_path(cli.reduced_action_space))
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


def _current_chronic_info(env: MAEnvWrapper) -> Dict[str, str]:
    getter = getattr(env, "get_current_chronic_info", None)
    info = getter() if callable(getter) else {}
    return {
        "chronic_name": str(info.get("chronic_name", "unknown")),
        "chronic_fingerprint": str(info.get("chronic_fingerprint", "unknown")),
        "chronic_datetime": str(info.get("chronic_datetime", "unknown")),
        "chronic_reset_count": str(info.get("chronic_reset_count", "unknown")),
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


def _candidate_requests(env: MAEnvWrapper, agent_ids: List[str]) -> List[Dict[str, Any]]:
    requests = []
    for agent in agent_ids:
        for action_id in range(int(env.action_space[agent].n)):
            if int(action_id) == 0:
                continue
            requests.append({"agent_id": agent, "action_id": int(action_id)})
    return requests


def _choose_greedy_action(
    *,
    env: MAEnvWrapper,
    agent_ids: List[str],
    requests: List[Dict[str, Any]],
    sim_pool: Optional[SimulationWorkerPool],
    pending_worker_actions: List[Dict[str, int]],
    time_step: int,
    improvement_tolerance: float,
    require_improvement: bool,
) -> Tuple[Dict[str, int], Dict[str, Any], List[Dict[str, int]]]:
    if sim_pool is not None:
        sim_pool.sync(
            pending_worker_actions,
            reference_max_rho=float(env.get_current_max_rho()),
        )
        pending_worker_actions = []

    do_nothing_outcome = env.simulate_action_outcomes(
        [{"agent_id": agent_ids[0], "action_id": 0}],
        time_step=time_step,
        num_workers=1,
    )[0]
    rho_after_do_nothing = float(do_nothing_outcome["rho_after"])

    if sim_pool is None:
        outcomes = env.simulate_action_outcomes(
            requests,
            time_step=time_step,
            num_workers=1,
        )
    else:
        outcomes = sim_pool.simulate(requests, time_step=time_step)

    best_request: Optional[Dict[str, Any]] = None
    best_outcome: Optional[Dict[str, Any]] = None
    best_rho = float("inf")
    valid_count = 0
    improved_count = 0
    for request, outcome in zip(requests, outcomes):
        valid = bool(outcome["action_is_valid"])
        rho_after = float(outcome["rho_after"])
        if not valid or not np.isfinite(rho_after):
            continue
        valid_count += 1
        if rho_after < rho_after_do_nothing - float(improvement_tolerance):
            improved_count += 1
        if rho_after < best_rho:
            best_rho = rho_after
            best_request = request
            best_outcome = outcome

    fallback = {agent: 0 for agent in agent_ids}
    if best_request is None or best_outcome is None:
        return fallback, {
            "selected_agent": "none",
            "selected_action": 0,
            "selected": False,
            "rho_after_selected": float("nan"),
            "rho_after_do_nothing": rho_after_do_nothing,
            "delta_vs_do_nothing": float("nan"),
            "valid_candidates": valid_count,
            "improving_candidates": improved_count,
        }, pending_worker_actions

    delta_vs_do_nothing = best_rho - rho_after_do_nothing
    should_act = (not require_improvement) or (
        delta_vs_do_nothing < -float(improvement_tolerance)
    )
    if not should_act:
        return fallback, {
            "selected_agent": "none",
            "selected_action": 0,
            "selected": False,
            "rho_after_selected": best_rho,
            "rho_after_do_nothing": rho_after_do_nothing,
            "delta_vs_do_nothing": delta_vs_do_nothing,
            "valid_candidates": valid_count,
            "improving_candidates": improved_count,
        }, pending_worker_actions

    action = {agent: 0 for agent in agent_ids}
    selected_agent = str(best_request["agent_id"])
    selected_action = int(best_request["action_id"])
    action[selected_agent] = selected_action
    return action, {
        "selected_agent": selected_agent,
        "selected_action": selected_action,
        "selected": True,
        "rho_after_selected": best_rho,
        "rho_after_do_nothing": rho_after_do_nothing,
        "delta_vs_do_nothing": delta_vs_do_nothing,
        "valid_candidates": valid_count,
        "improving_candidates": improved_count,
    }, pending_worker_actions


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
    while not done:
        if max_env_steps is not None and steps >= int(max_env_steps):
            break
        actions = {agent: 0 for agent in agent_ids}
        _, reward, terminations, truncations, _ = env.step(actions)
        total_reward += _reward_scalar(reward)
        done = bool(terminations[agent_ids[0]] or truncations[agent_ids[0]])
        steps += 1
    return {
        **chronic_info,
        "steps": int(steps),
        "max_steps": int(max_steps),
        "survival": float(steps / max(max_steps, 1)),
        "full_survival": bool(steps >= max_steps),
        "return": float(total_reward),
    }


def _run_greedy_episode(
    *,
    env: MAEnvWrapper,
    agent_ids: List[str],
    requests: List[Dict[str, Any]],
    sim_pool: Optional[SimulationWorkerPool],
    decision_rho_threshold: float,
    time_step: int,
    improvement_tolerance: float,
    require_improvement: bool,
    max_env_steps: Optional[int],
) -> Dict[str, Any]:
    _, _ = env.reset()
    if sim_pool is not None:
        sim_pool.reset(float(env.get_current_max_rho()))

    chronic_info = _current_chronic_info(env)
    max_steps = _max_episode_duration(env)
    steps = 0
    total_reward = 0.0
    done = False
    decision_states = 0
    simulated_actions = 0
    nonidle_actions = 0
    improving_selected_actions = 0
    cumulative_delta = 0.0
    selected_by_agent = {agent: 0 for agent in agent_ids}
    pending_worker_actions: List[Dict[str, int]] = []

    while not done:
        if max_env_steps is not None and steps >= int(max_env_steps):
            break
        global_max_rho = float(env.get_current_max_rho())
        if np.isfinite(global_max_rho) and global_max_rho >= decision_rho_threshold:
            decision_states += 1
            actions, decision, pending_worker_actions = _choose_greedy_action(
                env=env,
                agent_ids=agent_ids,
                requests=requests,
                sim_pool=sim_pool,
                pending_worker_actions=pending_worker_actions,
                time_step=time_step,
                improvement_tolerance=improvement_tolerance,
                require_improvement=require_improvement,
            )
            simulated_actions += len(requests) + 1
            if bool(decision["selected"]):
                nonidle_actions += 1
                improving_selected_actions += int(
                    decision["delta_vs_do_nothing"] < -float(improvement_tolerance)
                )
                cumulative_delta += float(decision["delta_vs_do_nothing"])
                selected_by_agent[str(decision["selected_agent"])] += 1
        else:
            actions = {agent: 0 for agent in agent_ids}

        _, reward, terminations, truncations, _ = env.step(actions)
        if sim_pool is not None:
            pending_worker_actions.append(dict(actions))
        total_reward += _reward_scalar(reward)
        done = bool(terminations[agent_ids[0]] or truncations[agent_ids[0]])
        steps += 1

    return {
        **chronic_info,
        "steps": int(steps),
        "max_steps": int(max_steps),
        "survival": float(steps / max(max_steps, 1)),
        "full_survival": bool(steps >= max_steps),
        "return": float(total_reward),
        "decision_states": int(decision_states),
        "simulated_actions": int(simulated_actions),
        "nonidle_actions": int(nonidle_actions),
        "improving_selected_actions": int(improving_selected_actions),
        "mean_selected_delta_vs_do_nothing": (
            float(cumulative_delta / nonidle_actions)
            if nonidle_actions > 0
            else float("nan")
        ),
        "selected_by_agent": selected_by_agent,
    }


def _write_outputs(
    *,
    output_dir: Path,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "greedy_vs_do_nothing_summary.json"
    csv_path = output_dir / "greedy_vs_do_nothing_episodes.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    fieldnames = [
        "episode",
        "greedy_chronic_name",
        "do_nothing_chronic_name",
        "greedy_chronic_fingerprint",
        "do_nothing_chronic_fingerprint",
        "greedy_chronic_datetime",
        "do_nothing_chronic_datetime",
        "greedy_chronic_reset_count",
        "do_nothing_chronic_reset_count",
        "same_chronic_fingerprint",
        "greedy_steps",
        "do_nothing_steps",
        "max_steps",
        "greedy_survival",
        "do_nothing_survival",
        "greedy_full_survival",
        "do_nothing_full_survival",
        "survival_delta",
        "greedy_nonidle_actions",
        "greedy_decision_states",
        "greedy_simulated_actions",
        "greedy_mean_selected_delta_vs_do_nothing",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return json_path, csv_path


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare do-nothing against a simulator-greedy policy constrained "
            "to a reduced topology action space."
        )
    )
    parser.add_argument("--env-id", type=str, default="bus36")
    parser.add_argument("--reduced-action-space", type=Path, required=True)
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
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--decision-rho-threshold", type=float, default=0.90)
    parser.add_argument("--improvement-tolerance", type=float, default=1e-3)
    parser.add_argument("--require-improvement", type=str2bool, default=True)
    parser.add_argument("--time-step", type=int, default=1)
    parser.add_argument("--sim-workers", type=int, default=1)
    parser.add_argument(
        "--sim-start-method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/teacher_student_greedy_eval"),
    )
    parser.add_argument("--progress", type=str2bool, default=True)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.sim_workers <= 0:
        raise ValueError("--sim-workers must be positive.")
    if cli.max_episodes is not None and cli.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive when provided.")

    output_dir = _resolve_task_path(cli.output_dir)
    env_args = _build_env_args(cli)
    set_random_seed(env_args.seed)
    chronic_split = None if cli.split == "all" else cli.split

    greedy_env = MAEnvWrapper(env_args, eval_env=True, chronic_split=chronic_split)
    do_nothing_env = MAEnvWrapper(env_args, eval_env=True, chronic_split=chronic_split)
    agent_ids = list(greedy_env.g2op_ma_env.agents)
    reduced_action_sizes = {
        agent: int(greedy_env.action_space[agent].n) for agent in agent_ids
    }
    requests = _candidate_requests(greedy_env, agent_ids)
    target_episodes = cli.max_episodes
    if target_episodes is None:
        split_size = getattr(greedy_env, "chronic_split_size", None)
        target_episodes = int(split_size) if split_size is not None else 1

    sim_pool = None
    if cli.sim_workers > 1:
        sim_pool = SimulationWorkerPool(
            env_args=env_args,
            chronic_split=chronic_split,
            num_workers=cli.sim_workers,
            start_method=cli.sim_start_method,
        )

    print("========== Greedy Reduced Action Evaluation ==========")
    print(f"Env id: {env_args.env_id}")
    print(f"Split: {cli.split}")
    print(f"Reduced action space: {_safe_path(Path(env_args.reduced_action_space))}")
    print(f"Output dir: {_safe_path(output_dir)}")
    print(f"Episodes: {target_episodes}")
    print(f"Decision rho threshold: {cli.decision_rho_threshold}")
    print(f"Require improvement: {cli.require_improvement}")
    print(f"Improvement tolerance: {cli.improvement_tolerance}")
    print(f"Simulator workers: {cli.sim_workers}")
    print(
        "Reduced action sizes: "
        + ", ".join(f"{agent}={reduced_action_sizes[agent]}" for agent in agent_ids)
    )
    print(f"Candidate unilateral actions per decision: {len(requests)}")
    print("======================================================")

    rows: List[Dict[str, Any]] = []
    start_time = time.perf_counter()
    try:
        for episode in range(int(target_episodes)):
            greedy_result = _run_greedy_episode(
                env=greedy_env,
                agent_ids=agent_ids,
                requests=requests,
                sim_pool=sim_pool,
                decision_rho_threshold=float(cli.decision_rho_threshold),
                time_step=int(cli.time_step),
                improvement_tolerance=float(cli.improvement_tolerance),
                require_improvement=bool(cli.require_improvement),
                max_env_steps=cli.max_env_steps,
            )
            do_nothing_result = _run_do_nothing_episode(
                env=do_nothing_env,
                agent_ids=agent_ids,
                max_env_steps=cli.max_env_steps,
            )
            row = {
                "episode": episode,
                "greedy_chronic_name": greedy_result["chronic_name"],
                "do_nothing_chronic_name": do_nothing_result["chronic_name"],
                "greedy_chronic_fingerprint": greedy_result["chronic_fingerprint"],
                "do_nothing_chronic_fingerprint": do_nothing_result[
                    "chronic_fingerprint"
                ],
                "greedy_chronic_datetime": greedy_result["chronic_datetime"],
                "do_nothing_chronic_datetime": do_nothing_result["chronic_datetime"],
                "greedy_chronic_reset_count": greedy_result["chronic_reset_count"],
                "do_nothing_chronic_reset_count": do_nothing_result[
                    "chronic_reset_count"
                ],
                "same_chronic_fingerprint": greedy_result["chronic_fingerprint"]
                == do_nothing_result["chronic_fingerprint"],
                "greedy_steps": greedy_result["steps"],
                "do_nothing_steps": do_nothing_result["steps"],
                "max_steps": greedy_result["max_steps"],
                "greedy_survival": greedy_result["survival"],
                "do_nothing_survival": do_nothing_result["survival"],
                "greedy_full_survival": greedy_result["full_survival"],
                "do_nothing_full_survival": do_nothing_result["full_survival"],
                "survival_delta": greedy_result["survival"]
                - do_nothing_result["survival"],
                "greedy_nonidle_actions": greedy_result["nonidle_actions"],
                "greedy_decision_states": greedy_result["decision_states"],
                "greedy_simulated_actions": greedy_result["simulated_actions"],
                "greedy_mean_selected_delta_vs_do_nothing": greedy_result[
                    "mean_selected_delta_vs_do_nothing"
                ],
                "greedy_selected_by_agent": greedy_result["selected_by_agent"],
            }
            rows.append(row)
            if cli.progress:
                elapsed = time.perf_counter() - start_time
                done_eps = episode + 1
                eta = elapsed / done_eps * (int(target_episodes) - done_eps)
                print(
                    "episode "
                    f"{done_eps}/{target_episodes}: "
                    f"greedy={100 * row['greedy_survival']:.2f}% "
                    f"do_nothing={100 * row['do_nothing_survival']:.2f}% "
                    f"delta={100 * row['survival_delta']:.2f}% "
                    f"nonidle={row['greedy_nonidle_actions']} "
                    f"same_chronic={row['same_chronic_fingerprint']} "
                    f"fp={str(row['greedy_chronic_fingerprint'])[:8]}/"
                    f"{str(row['do_nothing_chronic_fingerprint'])[:8]} "
                    f"date={row['greedy_chronic_datetime']} "
                    f"elapsed={_format_duration(elapsed)} "
                    f"eta={_format_duration(eta)}",
                    flush=True,
                )
    finally:
        if sim_pool is not None:
            sim_pool.close()
        for env in (greedy_env, do_nothing_env):
            with _suppress_grid2op_cleanup_stderr():
                try:
                    env.close()
                except Exception:
                    pass
        gc.collect()

    greedy_survivals = np.asarray([row["greedy_survival"] for row in rows], dtype=float)
    noop_survivals = np.asarray(
        [row["do_nothing_survival"] for row in rows], dtype=float
    )
    greedy_full = np.asarray([row["greedy_full_survival"] for row in rows], dtype=bool)
    noop_full = np.asarray(
        [row["do_nothing_full_survival"] for row in rows], dtype=bool
    )
    deltas = greedy_survivals - noop_survivals
    same_fingerprints = np.asarray(
        [bool(row["same_chronic_fingerprint"]) for row in rows], dtype=bool
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "env_id": env_args.env_id,
        "split": cli.split,
        "reduced_action_space": _safe_path(Path(env_args.reduced_action_space)),
        "n_episodes": len(rows),
        "decision_rho_threshold": float(cli.decision_rho_threshold),
        "require_improvement": bool(cli.require_improvement),
        "improvement_tolerance": float(cli.improvement_tolerance),
        "candidate_unilateral_actions_per_decision": int(len(requests)),
        "agent_ids": agent_ids,
        "reduced_action_sizes": reduced_action_sizes,
        "greedy_mean_survival": float(greedy_survivals.mean()) if rows else float("nan"),
        "do_nothing_mean_survival": float(noop_survivals.mean()) if rows else float("nan"),
        "mean_survival_delta": float(deltas.mean()) if rows else float("nan"),
        "greedy_full_survival_rate": float(greedy_full.mean()) if rows else float("nan"),
        "do_nothing_full_survival_rate": float(noop_full.mean()) if rows else float("nan"),
        "greedy_better_episodes": int((deltas > 0).sum()) if rows else 0,
        "greedy_equal_episodes": int(np.isclose(deltas, 0.0).sum()) if rows else 0,
        "greedy_worse_episodes": int((deltas < 0).sum()) if rows else 0,
        "same_chronic_fingerprint_rate": (
            float(same_fingerprints.mean()) if rows else float("nan")
        ),
        "episodes": rows,
    }
    json_path, csv_path = _write_outputs(
        output_dir=output_dir,
        rows=rows,
        summary=summary,
    )

    print("========== Greedy Evaluation Complete ==========")
    print(f"Episodes: {len(rows)}")
    print(f"Greedy mean survival: {100 * summary['greedy_mean_survival']:.3f}%")
    print(f"Do-nothing mean survival: {100 * summary['do_nothing_mean_survival']:.3f}%")
    print(f"Mean survival delta: {100 * summary['mean_survival_delta']:.3f}%")
    print(f"Greedy full survival rate: {100 * summary['greedy_full_survival_rate']:.3f}%")
    print(
        "Do-nothing full survival rate: "
        f"{100 * summary['do_nothing_full_survival_rate']:.3f}%"
    )
    print(
        "Same chronic fingerprint rate: "
        f"{100 * summary['same_chronic_fingerprint_rate']:.3f}%"
    )
    print(f"JSON: {_safe_path(json_path)}")
    print(f"CSV: {_safe_path(csv_path)}")
    print("================================================")


if __name__ == "__main__":
    main()
