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


def _parse_csv_tokens(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    text = str(value).replace("\n", ",").replace(" ", ",")
    return [token.strip() for token in text.split(",") if token.strip()]


def _parse_int_tokens(value: Optional[str]) -> List[int]:
    return [int(token) for token in _parse_csv_tokens(value)]


def _chronic_basename(path: Any) -> str:
    return Path(str(path).rstrip("/")).name


def _get_candidate_chronic_paths(env: MAEnvWrapper) -> List[str]:
    candidates = [str(path) for path in (getattr(env, "chronic_split_order", None) or [])]
    if candidates:
        return candidates
    handlers = getattr(env, "_current_chronic_handlers", lambda: [])()
    seen = set()
    for handler in handlers:
        for attr in ("subpaths", "_subpaths", "paths", "_paths"):
            value = getattr(handler, attr, None)
            if value is None:
                continue
            for path in list(value):
                path = str(path)
                if path not in seen:
                    candidates.append(path)
                    seen.add(path)
    return candidates


def _resolve_target_chronic_path(
    env: MAEnvWrapper,
    *,
    chronic_name: Optional[str],
    chronic_path: Optional[str],
) -> str:
    if chronic_path:
        return str(chronic_path)
    if not chronic_name:
        raise ValueError(
            "Target fingerprint mode needs --target-chronic-name, "
            "--target-chronic-names, or --target-chronic-paths."
        )
    matches = [
        path
        for path in _get_candidate_chronic_paths(env)
        if _chronic_basename(path) == str(chronic_name)
    ]
    if not matches:
        preview = ", ".join(
            _chronic_basename(path) for path in _get_candidate_chronic_paths(env)[:12]
        )
        raise ValueError(
            f"Could not find chronic name {chronic_name!r} in the active split. "
            f"First available names: {preview}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Chronic name {chronic_name!r} is ambiguous. Use "
            "--target-chronic-paths with one exact path per fingerprint."
        )
    return str(matches[0])


def _ordered_fingerprint_search_paths(
    env: MAEnvWrapper,
    chronic_name_hints: List[Optional[str]],
) -> List[str]:
    candidates = _get_candidate_chronic_paths(env)
    if not chronic_name_hints:
        return candidates

    ordered = []
    seen = set()
    hint_names = {str(name) for name in chronic_name_hints if name}
    for path in candidates:
        if _chronic_basename(path) in hint_names and path not in seen:
            ordered.append(path)
            seen.add(path)
    for path in candidates:
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def _resolve_target_paths_by_fingerprint(
    env: MAEnvWrapper,
    *,
    fingerprints: List[str],
    chronic_name_hints: List[Optional[str]],
    env_idx: int,
    progress: bool,
) -> Dict[str, Dict[str, Any]]:
    targets = {str(fingerprint) for fingerprint in fingerprints if fingerprint}
    if not targets:
        return {}

    candidate_paths = _get_candidate_chronic_paths(env)
    if not candidate_paths:
        raise RuntimeError("No candidate chronic paths are available to search.")
    path_to_chronic_id = {str(path): int(idx) for idx, path in enumerate(candidate_paths)}
    search_paths = _ordered_fingerprint_search_paths(env, chronic_name_hints)

    resolved: Dict[str, Dict[str, Any]] = {}
    samples = []
    try:
        _set_env_chronic_sequence(env, candidate_paths)
        for searched, path in enumerate(search_paths, start=1):
            chronic_id = int(path_to_chronic_id[str(path)])
            env.set_chronic_id(chronic_id)
            _, info = env.reset()
            fingerprint = str(info.get("chronic_fingerprint", "unknown"))
            if len(samples) < 8:
                samples.append(f"{_chronic_basename(path)}->{fingerprint}")
            if fingerprint in targets and fingerprint not in resolved:
                resolved[fingerprint] = {
                    "path": str(path),
                    "chronic_id": int(chronic_id),
                }
                if progress:
                    print(
                        "Resolved target fingerprint "
                        f"{fingerprint} on env_idx={env_idx}: "
                        f"id={chronic_id} {_chronic_basename(path)}",
                        flush=True,
                    )
                if set(resolved) == targets:
                    return resolved
            if progress and searched % 250 == 0:
                print(
                    "Searching target fingerprints "
                    f"env_idx={env_idx}: {searched}/{len(search_paths)} paths, "
                    f"resolved={len(resolved)}/{len(targets)}",
                    flush=True,
                )
    finally:
        _set_env_chronic_sequence(env, candidate_paths)

    missing = sorted(targets.difference(resolved))
    raise RuntimeError(
        "Could not resolve target fingerprint(s) in the active split for "
        f"env_idx={env_idx}: {', '.join(missing)}. "
        f"Searched {len(search_paths)} paths. "
        "Sample path->fingerprint values: " + ", ".join(samples)
    )


def _set_env_chronic_sequence(env: MAEnvWrapper, chronic_paths: List[str]) -> None:
    setter = getattr(env, "_set_active_chronic_order", None)
    if not callable(setter):
        raise RuntimeError("Environment does not support setting a chronic sequence.")
    successes = setter([str(path) for path in chronic_paths], reset_position=True)
    if successes == 0:
        raise RuntimeError(
            "Could not set the active chronic sequence to: "
            + ", ".join(str(path) for path in chronic_paths)
        )


def _validate_target_fingerprint(
    chronic_info: Dict[str, str],
    expected_fingerprint: Optional[str],
    *,
    strict: bool,
    context: str,
) -> None:
    if not expected_fingerprint:
        return
    actual = str(chronic_info.get("chronic_fingerprint", "unknown"))
    if actual == str(expected_fingerprint):
        return
    message = (
        f"{context} target fingerprint mismatch: expected "
        f"{expected_fingerprint}, got {actual} for "
        f"{chronic_info.get('chronic_name', 'unknown')} "
        f"path={chronic_info.get('chronic_path', 'unknown')} "
        f"seed={chronic_info.get('chronic_seed', 'unknown')} "
        f"idx={chronic_info.get('chronic_index', 'unknown')}."
    )
    if strict:
        raise RuntimeError(message)
    print("WARNING:", message, flush=True)


def _repeat_or_validate(values: List[Any], n_items: int, name: str) -> List[Any]:
    if not values:
        return [None] * n_items
    if len(values) == 1 and n_items > 1:
        return values * n_items
    if len(values) != n_items:
        raise ValueError(
            f"{name} must provide either one value or exactly {n_items} values."
        )
    return values


def _build_target_specs(cli: Namespace) -> List[Dict[str, Any]]:
    fingerprints = _parse_csv_tokens(cli.target_chronic_fingerprints)
    if not fingerprints:
        return []

    names = _parse_csv_tokens(cli.target_chronic_names)
    if cli.target_chronic_name:
        if names:
            raise ValueError(
                "Use either --target-chronic-name or --target-chronic-names, not both."
            )
        names = [str(cli.target_chronic_name)]
    paths = _parse_csv_tokens(cli.target_chronic_paths)
    env_indices = _parse_int_tokens(cli.target_env_indices)
    if not env_indices:
        env_indices = [0] * len(fingerprints)

    names = _repeat_or_validate(names, len(fingerprints), "--target-chronic-names")
    paths = _repeat_or_validate(paths, len(fingerprints), "--target-chronic-paths")
    env_indices = _repeat_or_validate(
        env_indices, len(fingerprints), "--target-env-indices"
    )

    if any(path is not None for path in paths) and any(name is not None for name in names):
        raise ValueError(
            "Use either target chronic names or target chronic paths, not both."
        )

    return [
        {
            "fingerprint": fingerprint,
            "chronic_name": names[idx],
            "chronic_path": paths[idx],
            "env_idx": int(env_indices[idx]),
        }
        for idx, fingerprint in enumerate(fingerprints)
    ]


def _close_eval_resources(
    sim_pool: Optional[SimulationWorkerPool],
    greedy_env: Optional[MAEnvWrapper],
    do_nothing_env: Optional[MAEnvWrapper],
) -> None:
    if sim_pool is not None:
        sim_pool.close()
    for env in (greedy_env, do_nothing_env):
        if env is None:
            continue
        with _suppress_grid2op_cleanup_stderr():
            try:
                env.close()
            except Exception:
                pass


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
    episode_actions: List[Dict[str, int]],
    episode_start_max_rho: float,
    episode_chronic_id: Optional[int],
    episode_chronic_path: Optional[str],
    sim_worker_sync_mode: str,
    time_step: int,
    improvement_tolerance: float,
    require_improvement: bool,
) -> Tuple[Dict[str, int], Dict[str, Any], List[Dict[str, int]]]:
    do_nothing_request = [{"agent_id": agent_ids[0], "action_id": 0}]

    worker_fallback = False
    worker_error = ""
    do_nothing_outcome: Dict[str, Any]
    outcomes: List[Dict[str, Any]]

    try:
        if sim_pool is None:
            raise RuntimeError("no simulation worker pool")

        current_max_rho = float(env.get_current_max_rho())
        if str(sim_worker_sync_mode) == "replay":
            sim_pool.reset(
                float(episode_start_max_rho),
                chronic_id=episode_chronic_id,
                target_chronic_path=episode_chronic_path,
            )
            if episode_actions:
                sim_pool.sync(
                    episode_actions,
                    reference_max_rho=current_max_rho,
                )
            pending_worker_actions = []
        else:
            sim_pool.sync(
                pending_worker_actions,
                reference_max_rho=current_max_rho,
            )
            pending_worker_actions = []

        combined_outcomes = sim_pool.simulate(
            do_nothing_request + requests,
            time_step=time_step,
        )
        do_nothing_outcome = combined_outcomes[0]
        outcomes = combined_outcomes[1:]
    except Exception as exc:
        # WCCI opponent / maintenance environments are not always exactly
        # reproducible by replaying actions inside long-lived worker envs. When
        # the guard catches drift, keep the evaluation correct by simulating
        # from the live main observation for this decision instead of crashing.
        worker_fallback = sim_pool is not None
        worker_error = f"{type(exc).__name__}: {exc}"[:500]
        do_nothing_outcome = env.simulate_action_outcomes(
            do_nothing_request,
            time_step=time_step,
            num_workers=1,
        )[0]
        outcomes = env.simulate_action_outcomes(
            requests,
            time_step=time_step,
            num_workers=1,
        )
        pending_worker_actions = []
    rho_after_do_nothing = float(do_nothing_outcome["rho_after"])

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
            "worker_fallback": bool(worker_fallback),
            "worker_error": worker_error,
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
            "worker_fallback": bool(worker_fallback),
            "worker_error": worker_error,
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
        "worker_fallback": bool(worker_fallback),
        "worker_error": worker_error,
    }, pending_worker_actions


def _run_do_nothing_episode(
    *,
    env: MAEnvWrapper,
    agent_ids: List[str],
    max_env_steps: Optional[int],
    chronic_id: Optional[int],
    target_chronic_path: Optional[str],
    expected_fingerprint: Optional[str],
    strict_fingerprint: bool,
) -> Dict[str, Any]:
    if chronic_id is not None:
        env.set_chronic_id(int(chronic_id))
    elif target_chronic_path is not None:
        _set_env_chronic_sequence(env, [str(target_chronic_path)])
    _, _ = env.reset()
    chronic_info = _current_chronic_info(env)
    _validate_target_fingerprint(
        chronic_info,
        expected_fingerprint,
        strict=strict_fingerprint,
        context="do-nothing",
    )
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
    target_chronic_path: Optional[str],
    expected_fingerprint: Optional[str],
    strict_fingerprint: bool,
    sim_worker_sync_mode: str,
) -> Dict[str, Any]:
    if chronic_id is not None:
        env.set_chronic_id(int(chronic_id))
    elif target_chronic_path is not None:
        _set_env_chronic_sequence(env, [str(target_chronic_path)])
    _, _ = env.reset()
    chronic_info = _current_chronic_info(env)
    _validate_target_fingerprint(
        chronic_info,
        expected_fingerprint,
        strict=strict_fingerprint,
        context="greedy",
    )
    episode_start_max_rho = float(env.get_current_max_rho())
    episode_chronic_path = target_chronic_path
    if not episode_chronic_path:
        info_path = str(chronic_info.get("chronic_path", ""))
        if info_path and info_path != "unknown":
            episode_chronic_path = info_path
    episode_chronic_id = chronic_id
    if sim_pool is not None:
        sim_pool.reset(
            episode_start_max_rho,
            chronic_id=episode_chronic_id,
            target_chronic_path=(
                None if episode_chronic_id is not None else episode_chronic_path
            ),
        )

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
    episode_actions: List[Dict[str, int]] = []
    worker_fallback_decisions = 0
    last_worker_error = ""

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
                episode_actions=episode_actions,
                episode_start_max_rho=episode_start_max_rho,
                episode_chronic_id=episode_chronic_id,
                episode_chronic_path=episode_chronic_path,
                sim_worker_sync_mode=sim_worker_sync_mode,
                time_step=time_step,
                improvement_tolerance=improvement_tolerance,
                require_improvement=require_improvement,
            )
            simulated_actions += len(requests) + 1
            if bool(decision.get("worker_fallback", False)):
                worker_fallback_decisions += 1
                last_worker_error = str(decision.get("worker_error", ""))
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
        episode_actions.append(dict(actions))
        if sim_pool is not None and str(sim_worker_sync_mode) == "incremental":
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
        "worker_fallback_decisions": int(worker_fallback_decisions),
        "last_worker_error": last_worker_error,
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
        "requested_env_idx",
        "requested_chronic_name",
        "requested_chronic_path",
        "requested_chronic_fingerprint",
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
        "greedy_worker_fallback_decisions",
        "greedy_mean_selected_delta_vs_do_nothing",
        "greedy_last_worker_error",
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
    parser.add_argument(
        "--target-chronic-fingerprints",
        type=str,
        default="",
        help=(
            "Comma- or space-separated initial-state fingerprints to evaluate "
            "exactly. When set, these define the episode list."
        ),
    )
    parser.add_argument(
        "--target-chronic-name",
        type=str,
        default="",
        help="Single chronic basename to use for all target fingerprints.",
    )
    parser.add_argument(
        "--target-chronic-names",
        type=str,
        default="",
        help="Comma- or space-separated chronic basenames, one per target fingerprint.",
    )
    parser.add_argument(
        "--target-chronic-paths",
        type=str,
        default="",
        help=(
            "Comma- or space-separated exact chronic paths, one per target "
            "fingerprint. Use this if names are ambiguous."
        ),
    )
    parser.add_argument(
        "--target-env-indices",
        type=str,
        default="",
        help=(
            "Comma- or space-separated MAEnvWrapper idx values for target "
            "fingerprints. Defaults to 0 for all target fingerprints."
        ),
    )
    parser.add_argument(
        "--target-resolve-fingerprints",
        type=str2bool,
        default=True,
        help=(
            "Resolve target chronic paths by scanning the active split for the "
            "requested fingerprints. Chronic names are treated as hints."
        ),
    )
    parser.add_argument(
        "--target-fingerprint-strict",
        type=str2bool,
        default=True,
        help="Abort when a targeted reset does not match its requested fingerprint.",
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
        "--sim-worker-sync-mode",
        type=str,
        default="replay",
        choices=["replay", "incremental"],
        help=(
            "How process simulation workers are synchronized before each greedy "
            "decision. replay resets workers to the episode start and replays "
            "the realized action history, which is safer for opponent/maintenance "
            "environments. incremental keeps the old faster behavior and only "
            "replays actions since the previous decision."
        ),
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
    target_specs = _build_target_specs(cli)
    target_mode = bool(target_specs)
    if (
        target_mode
        and cli.max_episodes is not None
        and int(cli.max_episodes) != len(target_specs)
    ):
        raise ValueError(
            "--max-episodes is ignored in target fingerprint mode; either omit it "
            f"or set it to {len(target_specs)}."
        )
    current_env_idx = int(target_specs[0]["env_idx"]) if target_mode else 0

    greedy_env = MAEnvWrapper(
        env_args,
        idx=current_env_idx,
        eval_env=True,
        chronic_split=chronic_split,
    )
    do_nothing_env = (
        MAEnvWrapper(
            env_args,
            idx=current_env_idx,
            eval_env=True,
            chronic_split=chronic_split,
        )
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
    target_episodes = len(target_specs) if target_mode else cli.max_episodes
    split_size = getattr(greedy_env, "chronic_split_size", None)
    if target_episodes is None:
        target_episodes = int(split_size) if split_size is not None else 1
    chronic_ids: List[Optional[int]] = [None] * int(target_episodes)
    chronic_sample_seed = (
        int(cli.chronic_sample_seed)
        if cli.chronic_sample_seed is not None
        else int(env_args.seed)
    )
    if target_mode:
        chronic_ids = [None] * int(target_episodes)
    elif cli.chronic_sample_mode == "random":
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
            env_idx=current_env_idx,
        )

    print("========== Greedy Reduced Action Evaluation ==========")
    print(f"Env id: {env_args.env_id}")
    print(f"Split: {cli.split}")
    print(f"Reduced action space: {_safe_path(Path(env_args.reduced_action_space))}")
    print(f"Output dir: {_safe_path(output_dir)}")
    print(f"Episodes: {target_episodes}")
    print(f"Chronic sample mode: {cli.chronic_sample_mode}")
    if target_mode:
        print("Target fingerprint mode: true")
        print(f"Target resolve fingerprints: {cli.target_resolve_fingerprints}")
        print(
            "Target env indices: "
            + ", ".join(str(spec["env_idx"]) for spec in target_specs)
        )
        print(
            "Target fingerprints: "
            + ", ".join(str(spec["fingerprint"]) for spec in target_specs)
        )
        chronic_labels = [
            spec["chronic_path"] or spec["chronic_name"] or "unknown"
            for spec in target_specs
        ]
        print("Target chronics: " + ", ".join(str(value) for value in chronic_labels))
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
    print(f"Simulator worker sync mode: {cli.sim_worker_sync_mode}")
    print(
        "Reduced action sizes: "
        + ", ".join(f"{agent}={reduced_action_sizes[agent]}" for agent in agent_ids)
    )
    print(f"Candidate unilateral actions total: {len(all_requests)}")
    print(f"Candidate unilateral actions per decision: {len(requests)}")
    print("======================================================")

    rows: List[Dict[str, Any]] = []
    resolved_targets_by_env_idx: Dict[int, Dict[str, Dict[str, Any]]] = {}
    start_time = time.perf_counter()
    try:
        for episode in range(int(target_episodes)):
            target_spec = target_specs[episode] if target_mode else None
            if target_spec is not None and int(target_spec["env_idx"]) != current_env_idx:
                _close_eval_resources(sim_pool, greedy_env, do_nothing_env)
                sim_pool = None
                gc.collect()
                current_env_idx = int(target_spec["env_idx"])
                greedy_env = MAEnvWrapper(
                    env_args,
                    idx=current_env_idx,
                    eval_env=True,
                    chronic_split=chronic_split,
                )
                do_nothing_env = (
                    MAEnvWrapper(
                        env_args,
                        idx=current_env_idx,
                        eval_env=True,
                        chronic_split=chronic_split,
                    )
                    if bool(cli.compare_do_nothing)
                    else None
                )
                if cli.sim_workers > 1:
                    sim_pool = SimulationWorkerPool(
                        env_args=env_args,
                        chronic_split=chronic_split,
                        num_workers=cli.sim_workers,
                        start_method=cli.sim_start_method,
                        env_idx=current_env_idx,
                    )

            requested_chronic_id = chronic_ids[episode]
            expected_fingerprint = (
                str(target_spec["fingerprint"]) if target_spec is not None else None
            )
            target_chronic_path = None
            target_chronic_id = requested_chronic_id
            if target_spec is not None:
                explicit_path = target_spec.get("chronic_path")
                if explicit_path:
                    target_chronic_path = str(explicit_path)
                elif expected_fingerprint and bool(cli.target_resolve_fingerprints):
                    if current_env_idx not in resolved_targets_by_env_idx:
                        specs_for_env = [
                            spec
                            for spec in target_specs
                            if int(spec["env_idx"]) == current_env_idx
                        ]
                        resolved_targets_by_env_idx[current_env_idx] = (
                            _resolve_target_paths_by_fingerprint(
                                greedy_env,
                                fingerprints=[
                                    str(spec["fingerprint"]) for spec in specs_for_env
                                ],
                                chronic_name_hints=[
                                    spec.get("chronic_name") for spec in specs_for_env
                                ],
                                env_idx=current_env_idx,
                                progress=bool(cli.progress),
                            )
                        )
                    resolved_target = resolved_targets_by_env_idx[current_env_idx][
                        expected_fingerprint
                    ]
                    target_chronic_path = str(resolved_target["path"])
                    target_chronic_id = int(resolved_target["chronic_id"])
                else:
                    target_chronic_path = _resolve_target_chronic_path(
                        greedy_env,
                        chronic_name=target_spec.get("chronic_name"),
                        chronic_path=None,
                    )
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
                chronic_id=target_chronic_id,
                target_chronic_path=target_chronic_path,
                expected_fingerprint=expected_fingerprint,
                strict_fingerprint=bool(cli.target_fingerprint_strict),
                sim_worker_sync_mode=str(cli.sim_worker_sync_mode),
            )
            do_nothing_result: Optional[Dict[str, Any]] = None
            if do_nothing_env is not None:
                do_nothing_result = _run_do_nothing_episode(
                    env=do_nothing_env,
                    agent_ids=agent_ids,
                    max_env_steps=cli.max_env_steps,
                    chronic_id=target_chronic_id,
                    target_chronic_path=target_chronic_path,
                    expected_fingerprint=expected_fingerprint,
                    strict_fingerprint=bool(cli.target_fingerprint_strict),
                )
            row = {
                "episode": episode,
                "requested_chronic_id": (
                    "" if target_chronic_id is None else int(target_chronic_id)
                ),
                "requested_env_idx": int(current_env_idx),
                "requested_chronic_name": (
                    target_spec.get("chronic_name") if target_spec is not None else ""
                ),
                "requested_chronic_path": target_chronic_path or "",
                "requested_chronic_fingerprint": expected_fingerprint or "",
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
                "greedy_worker_fallback_decisions": greedy_result[
                    "worker_fallback_decisions"
                ],
                "greedy_mean_selected_delta_vs_do_nothing": greedy_result[
                    "mean_selected_delta_vs_do_nothing"
                ],
                "greedy_last_worker_error": greedy_result["last_worker_error"],
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
                    f"worker_fallbacks={row['greedy_worker_fallback_decisions']} "
                    f"requested_id={row['requested_chronic_id']} "
                    f"requested_env_idx={row['requested_env_idx']} "
                    f"requested_fp={str(row['requested_chronic_fingerprint'])[:8]} "
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
        _close_eval_resources(sim_pool, greedy_env, do_nothing_env)
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
    worker_fallbacks = np.asarray(
        [int(row["greedy_worker_fallback_decisions"]) for row in rows], dtype=np.int64
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
        "target_fingerprint_mode": bool(target_mode),
        "target_resolve_fingerprints": bool(cli.target_resolve_fingerprints),
        "target_fingerprint_strict": bool(cli.target_fingerprint_strict),
        "target_chronic_fingerprints": [
            str(spec["fingerprint"]) for spec in target_specs
        ],
        "target_env_indices": [int(spec["env_idx"]) for spec in target_specs],
        "requested_chronic_ids": [
            (
                None
                if row["requested_chronic_id"] == ""
                else int(row["requested_chronic_id"])
            )
            for row in rows
        ],
        "decision_rho_threshold": float(cli.decision_rho_threshold),
        "require_improvement": bool(cli.require_improvement),
        "improvement_tolerance": float(cli.improvement_tolerance),
        "compare_do_nothing": bool(cli.compare_do_nothing),
        "sim_worker_sync_mode": str(cli.sim_worker_sync_mode),
        "greedy_worker_fallback_decisions": (
            int(worker_fallbacks.sum()) if rows else 0
        ),
        "greedy_worker_fallback_episodes": (
            int((worker_fallbacks > 0).sum()) if rows else 0
        ),
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
    print(
        "Worker fallback decisions: "
        f"{summary['greedy_worker_fallback_decisions']} "
        f"across {summary['greedy_worker_fallback_episodes']} episode(s)"
    )
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
