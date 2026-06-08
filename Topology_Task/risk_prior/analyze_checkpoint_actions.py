#!/usr/bin/env python3
"""Measure non-idle joint-action statistics for a saved MAPPO checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch as th


TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help=(
            "Path to a MAPPO .tar checkpoint, or a checkpoint name inside "
            "Topology_Task/checkpoint without the .tar suffix."
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Number of episodes to evaluate. Defaults to the checkpoint args value.",
    )
    parser.add_argument(
        "--deterministic",
        type=parse_bool,
        default=None,
        help="Override deterministic vs stochastic action selection.",
    )
    parser.add_argument(
        "--chronic-split",
        choices=["none", "train", "test"],
        default="test",
        help="Eval chronic split to use when the checkpoint was trained with split chronics.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path where the computed summary is written as JSON.",
    )
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        return path
    if path.suffix != ".tar":
        candidate = TASK_DIR / "checkpoint" / f"{path.name}.tar"
        if candidate.exists():
            return candidate
    candidate = TASK_DIR / "checkpoint" / path.name
    if candidate.exists():
        return candidate
    raise SystemExit(f"Checkpoint not found: {path}")


def choose_device(device_name: str) -> th.device:
    if device_name == "cpu":
        return th.device("cpu")
    if device_name == "cuda":
        if not th.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        return th.device("cuda")
    if device_name == "mps":
        if not th.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available.")
        return th.device("mps")
    if th.cuda.is_available():
        return th.device("cuda")
    if th.backends.mps.is_available():
        return th.device("mps")
    return th.device("cpu")


def load_checkpoint(path: Path) -> dict[str, Any]:
    return th.load(path, map_location="cpu", weights_only=False)


def make_actors(ckpt: dict[str, Any], env: Any, device: th.device) -> dict[str, Any]:
    from alg.mappo.agent import Actor
    from alg.mappo.core import _build_shared_actor_graph_encoder

    args = ckpt["args"]
    agent_ids = [f"agent_{idx}" for idx in range(len(env.observation_space.keys()))]
    shared_encoder = _build_shared_actor_graph_encoder(env, args, agent_ids)
    actors = {
        agent_id: Actor(
            idx,
            env,
            args,
            continuous_actions=False,
            shared_graph_encoder=shared_encoder,
        ).to(device)
        for idx, agent_id in enumerate(agent_ids)
    }
    for agent_id, actor in actors.items():
        actor.load_state_dict(ckpt[agent_id])
        actor.eval()
    return actors


def tensorize_obs(obj: Any, device: th.device) -> Any:
    from common.utils import cast_np_to_tensors

    return cast_np_to_tensors(obj, device)


def action_to_int(action: Any) -> int:
    if isinstance(action, th.Tensor):
        return int(action.detach().cpu().reshape(-1)[0].item())
    if isinstance(action, np.ndarray):
        return int(action.reshape(-1)[0].item())
    return int(action)


def evaluate_counts(
    ckpt: dict[str, Any],
    eval_episodes: int | None,
    deterministic: bool | None,
    chronic_split: str,
    device: th.device,
) -> dict[str, Any]:
    from env.utils import MAEnvWrapper

    args = ckpt["args"]
    if eval_episodes is not None:
        args.eval_episodes = int(eval_episodes)
    if deterministic is not None:
        args.deterministic_eval = bool(deterministic)

    split = None if chronic_split == "none" else chronic_split
    env = MAEnvWrapper(args, eval_env=True, chronic_split=split)
    actors = make_actors(ckpt, env, device)
    agent_ids = list(actors.keys())
    n_agents = len(agent_ids)

    counts: list[int] = []
    episode_lengths: list[int] = []
    episode_survival: list[float] = []
    max_steps = env.g2op_env.chronics_handler.max_episode_duration()
    target_episodes = int(getattr(args, "eval_episodes", 10))
    deterministic_eval = bool(getattr(args, "deterministic_eval", True))

    try:
        obs, _ = env.reset()
        obs = tensorize_obs(obs, device)
        current_episode_decisions = 0

        while len(episode_lengths) < target_episodes:
            action = {}
            with th.no_grad():
                for agent_id, actor in actors.items():
                    action[agent_id] = actor.get_eval_action(
                        obs[agent_id],
                        deterministic=deterministic_eval,
                    )

            n_non_idle = sum(action_to_int(action[agent_id]) != 0 for agent_id in agent_ids)
            counts.append(int(n_non_idle))
            current_episode_decisions += 1

            next_obs, _, terminations, truncations, _ = env.step(action)
            done = bool(terminations["agent_0"] or truncations["agent_0"])
            obs = tensorize_obs(next_obs, device)

            if done:
                episode_lengths.append(current_episode_decisions)
                episode_survival.append(float(env.g2op_ma_env._cent_env.nb_time_step / max_steps))
                obs, _ = env.reset()
                obs = tensorize_obs(obs, device)
                current_episode_decisions = 0
    finally:
        env.close()

    count_array = np.asarray(counts, dtype=np.int64)
    hist = np.bincount(count_array, minlength=n_agents + 1)
    frac = hist / max(len(count_array), 1)
    summary = {
        "global_step": int(ckpt.get("global_step", -1)),
        "eval_episodes": target_episodes,
        "deterministic_eval": deterministic_eval,
        "chronic_split": chronic_split,
        "n_agents": n_agents,
        "n_decision_steps": int(len(count_array)),
        "non_idle_agents_hist": hist.astype(int).tolist(),
        "non_idle_agents_frac": frac.astype(float).tolist(),
        "count_0_frac": float(frac[0]),
        "count_1_frac": float(frac[1]) if len(frac) > 1 else 0.0,
        "frac_any_non_idle": float(np.mean(count_array > 0)) if len(count_array) else 0.0,
        "frac_multi_agent_non_idle": float(np.mean(count_array > 1)) if len(count_array) else 0.0,
        "non_idle_agents_mean": float(np.mean(count_array)) if len(count_array) else 0.0,
        "non_idle_agents_max": int(np.max(count_array)) if len(count_array) else 0,
        "episode_decision_lengths": episode_lengths,
        "episode_survival": episode_survival,
        "avg_episode_survival": float(np.mean(episode_survival)) if episode_survival else 0.0,
    }
    for count, value in enumerate(frac):
        summary[f"count_{count}_frac"] = float(value)
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("========== Non-idle action diagnostic ==========")
    print(f"Checkpoint global_step: {summary['global_step']}")
    print(f"Eval episodes: {summary['eval_episodes']}")
    print(f"Deterministic eval: {summary['deterministic_eval']}")
    print(f"Chronic split: {summary['chronic_split']}")
    print(f"Decision steps: {summary['n_decision_steps']}")
    print(f"Average survival: {summary['avg_episode_survival'] * 100:.3f}%")
    print("")
    for count, frac in enumerate(summary["non_idle_agents_frac"]):
        print(f"count_{count}_frac: {frac:.6f}")
    print("")
    print(f"frac_any_non_idle: {summary['frac_any_non_idle']:.6f}")
    print(f"frac_multi_agent_non_idle: {summary['frac_multi_agent_non_idle']:.6f}")
    print(f"non_idle_agents_mean: {summary['non_idle_agents_mean']:.6f}")
    print("================================================")


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    ckpt = load_checkpoint(checkpoint_path)
    device = choose_device(args.device)
    summary = evaluate_counts(
        ckpt,
        eval_episodes=args.eval_episodes,
        deterministic=args.deterministic,
        chronic_split=args.chronic_split,
        device=device,
    )
    print_summary(summary)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote JSON summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
