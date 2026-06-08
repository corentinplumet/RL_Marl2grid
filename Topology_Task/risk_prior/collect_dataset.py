#!/usr/bin/env python3
"""Collect one-step action-risk labels for the learned Gibbs prior.

The collector samples hazardous Grid2Op states, evaluates candidate local
topology actions with all other agents set to no-op, and stores labels of the
form:

    state_graph, agent_id, local_action_id -> max(next_obs.rho)

Illegal, ambiguous, failed, or terminal simulations are kept as data and mapped
to a configurable high-risk target. This keeps phase 1 fully data-driven rather
than relying on hand-authored action masks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TASK_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = TASK_DIR.parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))


DEFAULT_RHO_THRESHOLD = 0.90


@dataclass
class SimulationResult:
    target_risk: float
    reward: float
    done: bool
    is_illegal: bool
    is_ambiguous: bool
    has_error: bool
    simulation_error: bool
    exception: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output .npz path. Defaults to "
            "outputs/risk_prior/<env_id>_seed<seed>_risk_dataset.npz."
        ),
    )
    parser.add_argument("--env-id", default="bus14", choices=["bus14", "bus36", "bus118"])
    parser.add_argument("--env-config-path", default="scenario.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--decentralized", type=parse_bool, default=True)
    parser.add_argument("--split-chronics", type=parse_bool, default=False)
    parser.add_argument("--test-chronics-pct", type=float, default=0.2)
    parser.add_argument("--chronic-split-seed", type=int, default=None)
    parser.add_argument("--optimize-mem", type=parse_bool, default=True)
    parser.add_argument("--gnn-include-neighbors", type=parse_bool, default=False)

    parser.add_argument(
        "--rho-threshold",
        type=float,
        default=DEFAULT_RHO_THRESHOLD,
        help="Only states with max rho at least this value are labelled.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20000,
        help="Stop once this many action-labelled examples have been collected.",
    )
    parser.add_argument(
        "--max-hazard-states",
        type=int,
        default=1000,
        help="Stop once this many hazardous states have been evaluated.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=200,
        help="Maximum number of environment episodes to scan.",
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=200000,
        help="Maximum number of real environment steps used to find hazards.",
    )
    parser.add_argument(
        "--hazard-stride",
        type=int,
        default=1,
        help="Evaluate only every Nth hazardous state to reduce correlation.",
    )

    parser.add_argument(
        "--actions-per-agent",
        type=int,
        default=32,
        help=(
            "Number of local actions sampled per agent at each hazard. "
            "Use <=0 or --all-actions true to evaluate full local action spaces."
        ),
    )
    parser.add_argument(
        "--all-actions",
        type=parse_bool,
        default=False,
        help="Evaluate every local action for every agent at each hazardous state.",
    )
    parser.add_argument(
        "--include-action0",
        type=parse_bool,
        default=True,
        help="Always include local action 0 in each sampled candidate set.",
    )
    parser.add_argument(
        "--risk-penalty",
        type=float,
        default=2.0,
        help="Target risk assigned to illegal, ambiguous, failed, or terminal simulations.",
    )
    parser.add_argument(
        "--terminal-is-high-risk",
        type=parse_bool,
        default=True,
        help="Treat one-step terminal simulations as high-risk labels.",
    )
    parser.add_argument(
        "--default-other-agent-action",
        type=int,
        default=0,
        help="Local action id used for agents other than the evaluated one.",
    )
    parser.add_argument(
        "--rollout-nonidle-prob",
        type=float,
        default=0.0,
        help=(
            "Probability that each agent takes a random non-idle action while "
            "walking the environment between labelled states. Default keeps the "
            "rollout policy at no-op."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress after this many labelled hazardous states.",
    )

    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}.")


def make_env_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the subset of training args required by MAEnvWrapper."""
    return argparse.Namespace(
        env_id=args.env_id,
        n_envs=1,
        action_type="topology",
        difficulty=args.difficulty,
        decentralized=args.decentralized,
        n1_reward=False,
        env_config_path=args.env_config_path,
        norm_obs=False,
        use_heuristic=False,
        heuristic_type="idle",
        line_margin_reward_weight=0.0,
        topology_reward_weight=0.0,
        optimize_mem=args.optimize_mem,
        constraints_type=0,
        split_chronics=args.split_chronics,
        test_chronics_pct=args.test_chronics_pct,
        chronic_split_seed=args.chronic_split_seed,
        actor_encoder="gnn",
        critic_encoder="gnn",
        gnn_include_neighbors=args.gnn_include_neighbors,
        seed=args.seed,
    )


def build_env(args: argparse.Namespace):
    from env.utils import MAEnvWrapper

    return MAEnvWrapper(make_env_args(args))


def max_rho(obs: Any, fallback: float = np.nan) -> float:
    rho = getattr(obs, "rho", None)
    if rho is None:
        return float(fallback)
    rho = np.asarray(rho, dtype=np.float32)
    finite = rho[np.isfinite(rho)]
    if finite.size == 0:
        return float(fallback)
    return float(finite.max())


def candidate_actions(
    rng: np.random.Generator,
    n_actions: int,
    actions_per_agent: int,
    all_actions: bool,
    include_action0: bool,
) -> np.ndarray:
    if all_actions or actions_per_agent <= 0 or actions_per_agent >= n_actions:
        return np.arange(n_actions, dtype=np.int64)

    selected: list[int] = []
    if include_action0 and n_actions > 0:
        selected.append(0)

    remaining = np.setdiff1d(
        np.arange(n_actions, dtype=np.int64),
        np.asarray(selected, dtype=np.int64),
        assume_unique=True,
    )
    n_to_sample = min(max(actions_per_agent - len(selected), 0), len(remaining))
    if n_to_sample > 0:
        sampled = rng.choice(remaining, size=n_to_sample, replace=False)
        selected.extend(int(action_id) for action_id in sampled)
    return np.asarray(selected, dtype=np.int64)


def make_joint_action_ids(
    agent_ids: Iterable[str],
    controlled_agent: str,
    action_id: int,
    default_action_id: int,
) -> dict[str, int]:
    return {
        agent_id: int(action_id) if agent_id == controlled_agent else int(default_action_id)
        for agent_id in agent_ids
    }


def central_action_from_joint_ids(env: Any, joint_action_ids: dict[str, int]) -> Any:
    """Convert local discrete ids to one central Grid2Op action.

    MAEnvWrapper normally passes a dict of local actions to Grid2Op's
    MultiAgentEnv. For offline one-step simulation we need a single central
    action, so we combine local Grid2Op actions with the central no-op action.
    """
    local_actions = env._get_grid2op_act(joint_action_ids)
    central_action = env.g2op_env.action_space({})
    for local_action in local_actions.values():
        central_action += local_action
    return central_action


def info_flag(info: Any, key: str) -> bool:
    if not isinstance(info, dict):
        return False
    value = info.get(key, False)
    if isinstance(value, np.ndarray):
        return bool(value.any())
    if isinstance(value, (list, tuple, set)):
        return any(bool(item) for item in value)
    return bool(value)


def info_exception(info: Any) -> str:
    if not isinstance(info, dict):
        return ""
    exc = info.get("exception", "")
    if isinstance(exc, (list, tuple)):
        exc = "; ".join(str(item) for item in exc if item)
    return "" if exc is None else str(exc)


def simulate_one_step(
    env: Any,
    agent_ids: list[str],
    controlled_agent: str,
    action_id: int,
    default_action_id: int,
    risk_penalty: float,
    terminal_is_high_risk: bool,
) -> SimulationResult:
    joint_action_ids = make_joint_action_ids(
        agent_ids,
        controlled_agent,
        int(action_id),
        default_action_id,
    )

    try:
        central_action = central_action_from_joint_ids(env, joint_action_ids)
        sim_obs, reward, done, info = env._obs.simulate(central_action)
    except Exception as exc:  # noqa: BLE001 - the exception text is useful metadata
        return SimulationResult(
            target_risk=float(risk_penalty),
            reward=0.0,
            done=True,
            is_illegal=False,
            is_ambiguous=False,
            has_error=True,
            simulation_error=True,
            exception=repr(exc),
        )

    is_illegal = info_flag(info, "is_illegal")
    is_ambiguous = info_flag(info, "is_ambiguous")
    has_error = info_flag(info, "has_error") or info_exception(info) != ""
    terminal_penalty = terminal_is_high_risk and bool(done)
    failed = is_illegal or is_ambiguous or has_error or terminal_penalty
    target = float(risk_penalty) if failed else max_rho(sim_obs, fallback=risk_penalty)

    return SimulationResult(
        target_risk=target,
        reward=float(reward),
        done=bool(done),
        is_illegal=is_illegal,
        is_ambiguous=is_ambiguous,
        has_error=has_error,
        simulation_error=False,
        exception=info_exception(info),
    )


def sample_rollout_action(
    env: Any,
    agent_ids: list[str],
    rng: np.random.Generator,
    nonidle_prob: float,
) -> dict[str, int]:
    actions: dict[str, int] = {}
    for agent_id in agent_ids:
        n_actions = int(env.action_space[agent_id].n)
        if n_actions <= 1 or rng.random() >= nonidle_prob:
            actions[agent_id] = 0
        else:
            actions[agent_id] = int(rng.integers(1, n_actions))
    return actions


def graph_snapshot(env: Any) -> dict[str, np.ndarray]:
    if env.graph_builder is None:
        raise RuntimeError("Risk-prior collection requires graph observations.")
    graph = env.graph_builder.build(env._obs)["state"]
    return {
        "node_features": graph["node_features"].astype(np.float32, copy=True),
        "edge_features": graph["edge_features"].astype(np.float32, copy=True),
        "node_mask": graph["node_mask"].astype(np.float32, copy=True),
        "edge_mask": graph["edge_mask"].astype(np.float32, copy=True),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output = args.output
    if output is None:
        output = (
            PROJECT_DIR
            / "outputs"
            / "risk_prior"
            / f"{args.env_id}_seed{args.seed}_risk_dataset.npz"
        )
    if output.suffix != ".npz":
        output = output.with_suffix(".npz")
    metadata = output.with_suffix(".meta.json")
    return output, metadata


def save_dataset(
    output: Path,
    metadata_path: Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "node_features": np.stack([record["node_features"] for record in records]),
        "edge_features": np.stack([record["edge_features"] for record in records]),
        "node_mask": np.stack([record["node_mask"] for record in records]),
        "edge_mask": np.stack([record["edge_mask"] for record in records]),
        "agent_index": np.asarray([record["agent_index"] for record in records], dtype=np.int64),
        "action_id": np.asarray([record["action_id"] for record in records], dtype=np.int64),
        "target_risk": np.asarray([record["target_risk"] for record in records], dtype=np.float32),
        "pre_risk": np.asarray([record["pre_risk"] for record in records], dtype=np.float32),
        "reward": np.asarray([record["reward"] for record in records], dtype=np.float32),
        "done": np.asarray([record["done"] for record in records], dtype=np.bool_),
        "is_illegal": np.asarray([record["is_illegal"] for record in records], dtype=np.bool_),
        "is_ambiguous": np.asarray([record["is_ambiguous"] for record in records], dtype=np.bool_),
        "has_error": np.asarray([record["has_error"] for record in records], dtype=np.bool_),
        "simulation_error": np.asarray(
            [record["simulation_error"] for record in records], dtype=np.bool_
        ),
        "episode": np.asarray([record["episode"] for record in records], dtype=np.int64),
        "env_step": np.asarray([record["env_step"] for record in records], dtype=np.int64),
        "hazard_index": np.asarray([record["hazard_index"] for record in records], dtype=np.int64),
    }
    np.savez_compressed(output, **arrays)
    metadata_path.write_text(json.dumps(jsonable(metadata), indent=2, sort_keys=True))


def collect(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    env = build_env(args)
    agent_ids = list(env.g2op_ma_env.agents)
    agent_to_index = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}

    records: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    hazard_states = 0
    seen_hazards = 0
    env_steps = 0
    episodes = 0

    try:
        while (
            len(records) < args.max_examples
            and hazard_states < args.max_hazard_states
            and episodes < args.max_episodes
            and env_steps < args.max_env_steps
        ):
            env.reset(seed=args.seed + episodes)
            done = False
            episode_step = 0

            while (
                not done
                and len(records) < args.max_examples
                and hazard_states < args.max_hazard_states
                and env_steps < args.max_env_steps
            ):
                pre_risk = max_rho(env._obs)
                if pre_risk >= args.rho_threshold:
                    seen_hazards += 1
                    if seen_hazards % max(args.hazard_stride, 1) == 0:
                        hazard_states += 1
                        snapshot = graph_snapshot(env)
                        for agent_id in agent_ids:
                            n_actions = int(env.action_space[agent_id].n)
                            actions = candidate_actions(
                                rng,
                                n_actions=n_actions,
                                actions_per_agent=args.actions_per_agent,
                                all_actions=args.all_actions,
                                include_action0=args.include_action0,
                            )
                            for action_id in actions:
                                result = simulate_one_step(
                                    env,
                                    agent_ids,
                                    controlled_agent=agent_id,
                                    action_id=int(action_id),
                                    default_action_id=args.default_other_agent_action,
                                    risk_penalty=args.risk_penalty,
                                    terminal_is_high_risk=args.terminal_is_high_risk,
                                )
                                records.append(
                                    {
                                        **snapshot,
                                        "agent_index": agent_to_index[agent_id],
                                        "action_id": int(action_id),
                                        "target_risk": result.target_risk,
                                        "pre_risk": pre_risk,
                                        "reward": result.reward,
                                        "done": result.done,
                                        "is_illegal": result.is_illegal,
                                        "is_ambiguous": result.is_ambiguous,
                                        "has_error": result.has_error,
                                        "simulation_error": result.simulation_error,
                                        "episode": episodes,
                                        "env_step": env_steps,
                                        "hazard_index": hazard_states - 1,
                                    }
                                )
                                if result.exception and len(exceptions) < 100:
                                    exceptions.append(
                                        {
                                            "agent_id": agent_id,
                                            "action_id": int(action_id),
                                            "env_step": env_steps,
                                            "exception": result.exception,
                                        }
                                    )
                                if len(records) >= args.max_examples:
                                    break
                            if len(records) >= args.max_examples:
                                break

                        if (
                            args.progress_every > 0
                            and hazard_states % args.progress_every == 0
                        ):
                            print(
                                f"hazards={hazard_states} examples={len(records)} "
                                f"episodes={episodes + 1} env_steps={env_steps}"
                            )

                rollout_action = sample_rollout_action(
                    env,
                    agent_ids,
                    rng,
                    nonidle_prob=args.rollout_nonidle_prob,
                )
                _, _, terminations, truncations, _ = env.step(rollout_action)
                done = bool(terminations["agent_0"] or truncations["agent_0"])
                env_steps += 1
                episode_step += 1

            episodes += 1

    finally:
        env.close()

    if not records:
        raise SystemExit(
            "No risk-prior records were collected. Try lowering --rho-threshold, "
            "raising --max-env-steps, or increasing --rollout-nonidle-prob."
        )

    metadata = {
        "collector": "risk_prior.collect_dataset",
        "phase": 1,
        "env_id": args.env_id,
        "seed": args.seed,
        "rho_threshold": args.rho_threshold,
        "risk_penalty": args.risk_penalty,
        "terminal_is_high_risk": args.terminal_is_high_risk,
        "actions_per_agent": args.actions_per_agent,
        "all_actions": args.all_actions,
        "include_action0": args.include_action0,
        "default_other_agent_action": args.default_other_agent_action,
        "rollout_nonidle_prob": args.rollout_nonidle_prob,
        "n_examples": len(records),
        "n_hazard_states": hazard_states,
        "n_seen_hazards": seen_hazards,
        "n_env_steps": env_steps,
        "n_episodes": episodes,
        "agent_ids": agent_ids,
        "agent_to_index": agent_to_index,
        "n_actions_by_agent": {
            agent_id: int(env_n_actions)
            for agent_id, env_n_actions in {
                agent_id: env.action_space[agent_id].n for agent_id in agent_ids
            }.items()
        },
        "action_domains": getattr(env, "action_domains", {}),
        "observation_domains": getattr(env, "observation_domains", {}),
        "state_graph_spec": env.graph_specs["state"],
        "sampled_exceptions": exceptions,
    }
    return records, metadata


def main() -> int:
    args = parse_args()
    output, metadata_path = output_paths(args)
    records, metadata = collect(args)
    save_dataset(output, metadata_path, records, metadata)
    print(f"Saved {len(records)} examples to {output}")
    print(f"Saved metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

