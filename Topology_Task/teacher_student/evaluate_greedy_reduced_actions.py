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
        "chronic_path": str(info.get("chronic_path", "unknown")),
        "chronic_seed": str(info.get("chronic_seed", "unknown")),
        "chronic_index": str(info.get("chronic_index", "unknown")),
        "chronic_order_position": str(
            info.get("chronic_order_position", "unknown")
        ),
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


def _sample_candidate_requests(
    requests: List[Dict[str, Any]],
    sample_size: Optional[int],
    sample_seed: Optional[int],
) -> List[Dict[str, Any]]:
    if sample_size is None:
        return list(requests)
    sample_size = int(sample_size)
    if sample_size <= 0:
        raise ValueError("--candidate-sample-size must be positive when provided.")
    if sample_size >= len(requests):
        return list(requests)
    rng = np.random.default_rng(0 if sample_seed is None else int(sample_seed))
    sampled_indices = sorted(
        int(index)
        for index in rng.choice(len(requests), size=sample_size, replace=False)
    )
    return [requests[index] for index in sampled_indices]


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
    chronic_id: Optional[int],
) -> Dict[str, Any]:
    if chronic_id is not None:
        env.set_chronic_id(int(chronic_id))
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
    chronic_id: Optional[int],
) -> Dict[str, Any]:
    if chronic_id is not None:
        env.set_chronic_id(int(chronic_id))
    _, _ = env.reset()
    if sim_pool is not None:
        sim_pool.reset(float(env.get_current_max_rho()), chronic_id=chronic_id)

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
        "requested_chronic_id",
        "greedy_chronic_name",
        "do_nothing_chronic_name",
        "greedy_chronic_path",
        "do_nothing_chronic_path",
        "greedy_chronic_seed",
        "do_nothing_chronic_seed",
        "greedy_chronic_index",
        "do_nothing_chronic_index",
        "greedy_chronic_order_position",
        "do_nothing_chronic_order_position",
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
    parser.add_argument(
        "--chronic-sample-mode",
        type=str,
        default="sequential",
        choices=["sequential", "random"],
        help=(
            "sequential evaluates the first N chronics in Grid2Op order. random "
            "samples chronic ids from the selected split before each episode."
        ),
    )
    parser.add_argument("--chronic-sample-seed", type=int, default=None)
    parser.add_argument(
        "--chronic-sample-replacement",
        type=str2bool,
        default=False,
        help="Sample random chronic ids with replacement.",
    )
    parser.add_argument("--decision-rho-threshold", type=float, default=0.90)
    parser.add_argument("--improvement-tolerance", type=float, default=1e-3)
    parser.add_argument("--require-improvement", type=str2bool, default=True)
    parser.add_argument(
        "--candidate-sample-size",
        type=int,
        default=None,
        help=(
            "Optional fixed random subset size of unilateral candidate actions "
            "to simulate at each greedy decision. Defaults to all candidates."
        ),
    )
    parser.add_argument(
        "--candidate-sample-seed",
        type=int,
        default=None,
        help="Seed for --candidate-sample-size. Defaults to --seed.",
    )
    parser.add_argument(
        "--compare-do-nothing",
        type=str2bool,
        default=True,
        help=(
            "Also replay a full do-nothing episode on the same chronic. Set false "
            "when a separate do-nothing evaluation already exists."
        ),
    )
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
    do_nothing_env = (
        MAEnvWrapper(env_args, eval_env=True, chronic_split=chronic_split)
        if bool(cli.compare_do_nothing)
        else None
    )
    agent_ids = list(greedy_env.g2op_ma_env.agents)
    reduced_action_sizes = {
        agent: int(greedy_env.action_space[agent].n) for agent in agent_ids
    }
    all_requests = _candidate_requests(greedy_env, agent_ids)
    candidate_sample_seed = (
        int(cli.candidate_sample_seed)
        if cli.candidate_sample_seed is not None
        else int(env_args.seed)
    )
    requests = _sample_candidate_requests(
        all_requests, cli.candidate_sample_size, candidate_sample_seed
    )
    target_episodes = cli.max_episodes
    split_size = getattr(greedy_env, "chronic_split_size", None)
    if target_episodes is None:
        target_episodes = int(split_size) if split_size is not None else 1
    chronic_ids: List[Optional[int]] = [None] * int(target_episodes)
    chronic_sample_seed = (
        int(cli.chronic_sample_seed)
        if cli.chronic_sample_seed is not None
        else int(env_args.seed)
    )
    if cli.chronic_sample_mode == "random":
        if split_size is None:
            raise ValueError(
                "--chronic-sample-mode=random requires an exact chronic split size."
            )
        if (
            not bool(cli.chronic_sample_replacement)
            and int(target_episodes) > int(split_size)
        ):
            raise ValueError(
                "Cannot sample more chronics than the split size without "
                "--chronic-sample-replacement true."
            )
        rng = np.random.default_rng(chronic_sample_seed)
        chronic_ids = [
            int(value)
            for value in rng.choice(
                int(split_size),
                size=int(target_episodes),
                replace=bool(cli.chronic_sample_replacement),
            )
        ]

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
    print(f"Chronic sample mode: {cli.chronic_sample_mode}")
    if cli.chronic_sample_mode == "random":
        print(f"Random chronic sample seed: {chronic_sample_seed}")
        print(
            "Random chronic ids preview: "
            + ", ".join(str(value) for value in chronic_ids[:20])
        )
    print(f"Decision rho threshold: {cli.decision_rho_threshold}")
    print(f"Require improvement: {cli.require_improvement}")
    print(f"Improvement tolerance: {cli.improvement_tolerance}")
    print(f"Compare do-nothing episode replay: {cli.compare_do_nothing}")
    print(
        "Candidate sample size: "
        f"{cli.candidate_sample_size if cli.candidate_sample_size is not None else 'all'}"
    )
    print(f"Candidate sample seed: {candidate_sample_seed}")
    print(f"Simulator workers: {cli.sim_workers}")
    print(
        "Reduced action sizes: "
        + ", ".join(f"{agent}={reduced_action_sizes[agent]}" for agent in agent_ids)
    )
    print(f"Candidate unilateral actions total: {len(all_requests)}")
    print(f"Candidate unilateral actions per decision: {len(requests)}")
    print("======================================================")

    rows: List[Dict[str, Any]] = []
    start_time = time.perf_counter()
    try:
        for episode in range(int(target_episodes)):
            requested_chronic_id = chronic_ids[episode]
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
                chronic_id=requested_chronic_id,
            )
            do_nothing_result: Optional[Dict[str, Any]] = None
            if do_nothing_env is not None:
                do_nothing_result = _run_do_nothing_episode(
                    env=do_nothing_env,
                    agent_ids=agent_ids,
                    max_env_steps=cli.max_env_steps,
                    chronic_id=requested_chronic_id,
                )
            row = {
                "episode": episode,
                "requested_chronic_id": (
                    "" if requested_chronic_id is None else int(requested_chronic_id)
                ),
                "greedy_chronic_name": greedy_result["chronic_name"],
                "do_nothing_chronic_name": (
                    do_nothing_result["chronic_name"]
                    if do_nothing_result is not None
                    else "skipped"
                ),
                "greedy_chronic_path": greedy_result["chronic_path"],
                "do_nothing_chronic_path": (
                    do_nothing_result["chronic_path"]
                    if do_nothing_result is not None
                    else "skipped"
                ),
                "greedy_chronic_seed": greedy_result["chronic_seed"],
                "do_nothing_chronic_seed": (
                    do_nothing_result["chronic_seed"]
                    if do_nothing_result is not None
                    else "skipped"
                ),
                "greedy_chronic_index": greedy_result["chronic_index"],
                "do_nothing_chronic_index": (
                    do_nothing_result["chronic_index"]
                    if do_nothing_result is not None
                    else "skipped"
                ),
                "greedy_chronic_order_position": greedy_result[
                    "chronic_order_position"
                ],
                "do_nothing_chronic_order_position": (
                    do_nothing_result["chronic_order_position"]
                    if do_nothing_result is not None
                    else "skipped"
                ),
                "greedy_chronic_fingerprint": greedy_result["chronic_fingerprint"],
                "do_nothing_chronic_fingerprint": (
                    do_nothing_result["chronic_fingerprint"]
                    if do_nothing_result is not None
                    else "skipped"
                ),
                "greedy_chronic_datetime": greedy_result["chronic_datetime"],
                "do_nothing_chronic_datetime": (
                    do_nothing_result["chronic_datetime"]
                    if do_nothing_result is not None
                    else "skipped"
                ),
                "greedy_chronic_reset_count": greedy_result["chronic_reset_count"],
                "do_nothing_chronic_reset_count": (
                    do_nothing_result["chronic_reset_count"]
                    if do_nothing_result is not None
                    else "skipped"
                ),
                "same_chronic_fingerprint": (
                    greedy_result["chronic_fingerprint"]
                    == do_nothing_result["chronic_fingerprint"]
                    if do_nothing_result is not None
                    else False
                ),
                "greedy_steps": greedy_result["steps"],
                "do_nothing_steps": (
                    do_nothing_result["steps"]
                    if do_nothing_result is not None
                    else float("nan")
                ),
                "max_steps": greedy_result["max_steps"],
                "greedy_survival": greedy_result["survival"],
                "do_nothing_survival": (
                    do_nothing_result["survival"]
                    if do_nothing_result is not None
                    else float("nan")
                ),
                "greedy_full_survival": greedy_result["full_survival"],
                "do_nothing_full_survival": (
                    do_nothing_result["full_survival"]
                    if do_nothing_result is not None
                    else False
                ),
                "survival_delta": (
                    greedy_result["survival"] - do_nothing_result["survival"]
                    if do_nothing_result is not None
                    else float("nan")
                ),
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
                if do_nothing_result is not None:
                    do_nothing_text = (
                        f"do_nothing={100 * row['do_nothing_survival']:.2f}% "
                        f"delta={100 * row['survival_delta']:.2f}% "
                        f"do_nothing_steps={row['do_nothing_steps']}/{row['max_steps']} "
                        f"same_chronic={row['same_chronic_fingerprint']} "
                        f"dn_name={row['do_nothing_chronic_name']} "
                        f"dn_path={row['do_nothing_chronic_path']} "
                        f"dn_idx={row['do_nothing_chronic_index']} "
                        f"dn_order={row['do_nothing_chronic_order_position']} "
                        f"dn_seed={row['do_nothing_chronic_seed']} "
                        f"fp={str(row['greedy_chronic_fingerprint'])[:8]}/"
                        f"{str(row['do_nothing_chronic_fingerprint'])[:8]} "
                    )
                else:
                    do_nothing_text = (
                        "do_nothing=skipped delta=skipped "
                        f"fp={str(row['greedy_chronic_fingerprint'])[:8]} "
                    )
                print(
                    "episode "
                    f"{done_eps}/{target_episodes}: "
                    f"greedy={100 * row['greedy_survival']:.2f}% "
                    f"greedy_steps={row['greedy_steps']}/{row['max_steps']} "
                    f"nonidle={row['greedy_nonidle_actions']} "
                    f"requested_id={row['requested_chronic_id']} "
                    f"g_name={row['greedy_chronic_name']} "
                    f"g_path={row['greedy_chronic_path']} "
                    f"g_idx={row['greedy_chronic_index']} "
                    f"g_order={row['greedy_chronic_order_position']} "
                    f"g_seed={row['greedy_chronic_seed']} "
                    f"{do_nothing_text}"
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
        "chronic_sample_mode": cli.chronic_sample_mode,
        "chronic_sample_seed": int(chronic_sample_seed),
        "chronic_sample_replacement": bool(cli.chronic_sample_replacement),
        "requested_chronic_ids": [
            None if chronic_id is None else int(chronic_id) for chronic_id in chronic_ids
        ],
        "decision_rho_threshold": float(cli.decision_rho_threshold),
        "require_improvement": bool(cli.require_improvement),
        "improvement_tolerance": float(cli.improvement_tolerance),
        "compare_do_nothing": bool(cli.compare_do_nothing),
        "candidate_unilateral_actions_total": int(len(all_requests)),
        "candidate_unilateral_actions_per_decision": int(len(requests)),
        "candidate_sample_size": (
            None
            if cli.candidate_sample_size is None
            else int(cli.candidate_sample_size)
        ),
        "candidate_sample_seed": int(candidate_sample_seed),
        "agent_ids": agent_ids,
        "reduced_action_sizes": reduced_action_sizes,
        "greedy_mean_survival": float(greedy_survivals.mean()) if rows else float("nan"),
        "do_nothing_mean_survival": (
            float(noop_survivals.mean())
            if rows and bool(cli.compare_do_nothing)
            else float("nan")
        ),
        "mean_survival_delta": (
            float(deltas.mean())
            if rows and bool(cli.compare_do_nothing)
            else float("nan")
        ),
        "greedy_full_survival_rate": float(greedy_full.mean()) if rows else float("nan"),
        "do_nothing_full_survival_rate": (
            float(noop_full.mean())
            if rows and bool(cli.compare_do_nothing)
            else float("nan")
        ),
        "greedy_better_episodes": (
            int((deltas > 0).sum()) if rows and bool(cli.compare_do_nothing) else 0
        ),
        "greedy_equal_episodes": (
            int(np.isclose(deltas, 0.0).sum())
            if rows and bool(cli.compare_do_nothing)
            else 0
        ),
        "greedy_worse_episodes": (
            int((deltas < 0).sum()) if rows and bool(cli.compare_do_nothing) else 0
        ),
        "same_chronic_fingerprint_rate": (
            float(same_fingerprints.mean())
            if rows and bool(cli.compare_do_nothing)
            else float("nan")
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
    if bool(cli.compare_do_nothing):
        print(f"Do-nothing mean survival: {100 * summary['do_nothing_mean_survival']:.3f}%")
        print(f"Mean survival delta: {100 * summary['mean_survival_delta']:.3f}%")
    else:
        print("Do-nothing mean survival: skipped")
        print("Mean survival delta: skipped")
    print(f"Greedy full survival rate: {100 * summary['greedy_full_survival_rate']:.3f}%")
    if bool(cli.compare_do_nothing):
        print(
            "Do-nothing full survival rate: "
            f"{100 * summary['do_nothing_full_survival_rate']:.3f}%"
        )
        print(
            "Same chronic fingerprint rate: "
            f"{100 * summary['same_chronic_fingerprint_rate']:.3f}%"
        )
    else:
        print("Do-nothing full survival rate: skipped")
        print("Same chronic fingerprint rate: skipped")
    print(f"JSON: {_safe_path(json_path)}")
    print(f"CSV: {_safe_path(csv_path)}")
    print("================================================")


if __name__ == "__main__":
    main()
