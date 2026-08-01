#!/usr/bin/env python3
"""Count actor and critic parameters for a MAPPO TOML config.

This script builds the same actor/critic modules as training, but it does not
start a rollout, create optimizers, write checkpoints, or initialize WandB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 guard
    import tomli as tomllib


TASK_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = TASK_DIR.parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "marl2grid_mplconfig")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "marl2grid_cache")
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from alg.mappo.agent import Actor, Critic  # noqa: E402
from alg.mappo.config import get_alg_args  # noqa: E402
from alg.mappo.core import _build_shared_actor_graph_encoder  # noqa: E402
from common.imports import ap, th  # noqa: E402
from common.utils import set_random_seed, str2bool  # noqa: E402
from env.config import get_env_args  # noqa: E402
from env.utils import MAEnvWrapper  # noqa: E402


def _cli_value(value: Any) -> list[str]:
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_cli_value(item))
        return result
    return [str(value)]


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        TASK_DIR / path,
        PROJECT_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return TASK_DIR / path


def _load_config_args(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    return dict(config.get("args", {})), list(config.get("run", {}).get("extra_args", []))


def _build_config_argv(config_args: dict[str, Any], extra_args: Iterable[str]) -> list[str]:
    argv = [str(Path(__file__).name)]
    for key, value in config_args.items():
        if value == "":
            continue
        argv.append("--" + key.replace("_", "-"))
        argv.extend(_cli_value(value))
    argv.extend(extra_args)
    return argv


def _parse_main_args() -> argparse.Namespace:
    parser = ap.ArgumentParser(add_help=False)
    parser.add_argument("--alg", type=str, default="MAPPO")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=str2bool, default=False)
    parser.add_argument("--th-deterministic", type=str2bool, default=False)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--track", type=str2bool, default=False)
    parser.add_argument("--checkpoint", type=str2bool, default=False)
    parser.add_argument("--verbose", type=str2bool, default=False)
    return parser.parse_known_args()[0]


def _args_from_config(config_args: dict[str, Any], extra_args: list[str]) -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = _build_config_argv(config_args, extra_args)
        main_args = _parse_main_args()
        env_args = get_env_args()
        alg_args = get_alg_args()
        return argparse.Namespace(
            **vars(main_args),
            **vars(env_args),
            **vars(alg_args),
        )
    finally:
        sys.argv = old_argv


def _param_count(parameters: Iterable[th.nn.Parameter], trainable_only: bool = False) -> int:
    return sum(
        int(param.numel())
        for param in parameters
        if not trainable_only or param.requires_grad
    )


def _unique_parameters(modules: Iterable[th.nn.Module]) -> list[th.nn.Parameter]:
    params: list[th.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for param in module.parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(param)
    return params


def _count_named_modules(modules: dict[str, th.nn.Module]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "total": _param_count(module.parameters()),
            "trainable": _param_count(module.parameters(), trainable_only=True),
        }
        for name, module in modules.items()
    }


def _build_models(args: argparse.Namespace) -> tuple[MAEnvWrapper, dict[str, Actor], Critic]:
    if args.alg.upper() != "MAPPO":
        raise ValueError(f"Only MAPPO is supported by this script, got {args.alg!r}.")

    # Avoid accidental WandB/checkpoint side effects if a config has them enabled.
    args.track = False
    args.checkpoint = False

    set_random_seed(args.seed)
    env = MAEnvWrapper(args, idx=0)
    agent_ids = [f"agent_{idx}" for idx in range(len(env.observation_space.keys()))]
    continuous_actions = args.action_type == "redispatch"
    shared_actor_graph_encoder = _build_shared_actor_graph_encoder(env, args, agent_ids)
    actors = {
        agent_id: Actor(
            idx,
            env,
            args,
            continuous_actions,
            shared_graph_encoder=shared_actor_graph_encoder,
        )
        for idx, agent_id in enumerate(agent_ids)
    }
    critic = Critic(env, args)
    return env, actors, critic


def _summarize(args: argparse.Namespace, actors: dict[str, Actor], critic: Critic) -> dict[str, Any]:
    actor_modules = list(actors.values())
    actor_unique = _unique_parameters(actor_modules)
    all_unique = _unique_parameters([*actor_modules, critic])
    return {
        "env_id": args.env_id,
        "alg": args.alg,
        "actor_encoder": args.actor_encoder,
        "actor_action_head": getattr(args, "actor_action_head", "mlp"),
        "critic_encoder": args.critic_encoder,
        "intervention_gate": bool(getattr(args, "intervention_gate", False)),
        "share_actor_gnn": bool(getattr(args, "share_actor_gnn", False)),
        "agent_ids": list(actors.keys()),
        "actors_by_agent": _count_named_modules(actors),
        "actors_unique": {
            "total": _param_count(actor_unique),
            "trainable": _param_count(actor_unique, trainable_only=True),
        },
        "critic": {
            "total": _param_count(critic.parameters()),
            "trainable": _param_count(critic.parameters(), trainable_only=True),
        },
        "model_unique": {
            "total": _param_count(all_unique),
            "trainable": _param_count(all_unique, trainable_only=True),
        },
    }


def _print_human(summary: dict[str, Any], config_path: Path) -> None:
    print(f"Config: {config_path}")
    print(
        "Model: "
        f"{summary['alg']} | actor={summary['actor_encoder']} | "
        f"action_head={summary['actor_action_head']} | "
        f"critic={summary['critic_encoder']} | "
        f"intervention_gate={summary['intervention_gate']} | "
        f"share_actor_gnn={summary['share_actor_gnn']}"
    )
    print("")
    print("Per-agent actors:")
    for agent_id, counts in summary["actors_by_agent"].items():
        print(
            f"  {agent_id}: {counts['total']:,} params "
            f"({counts['trainable']:,} trainable)"
        )
    print("")
    print(
        "Actors unique: "
        f"{summary['actors_unique']['total']:,} params "
        f"({summary['actors_unique']['trainable']:,} trainable)"
    )
    print(
        "Critic:        "
        f"{summary['critic']['total']:,} params "
        f"({summary['critic']['trainable']:,} trainable)"
    )
    print(
        "Model unique:  "
        f"{summary['model_unique']['total']:,} params "
        f"({summary['model_unique']['trainable']:,} trainable)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a human-readable summary.",
    )
    parser.add_argument(
        "config",
        help="TOML config path, e.g. configs/foo.toml",
    )
    cli, overrides = parser.parse_known_args()

    config_path = _resolve_config_path(cli.config)
    config_args, extra_args = _load_config_args(config_path)
    args = _args_from_config(config_args, [*extra_args, *overrides])

    old_cwd = Path.cwd()
    os.chdir(TASK_DIR)
    env = None
    try:
        try:
            env, actors, critic = _build_models(args)
        except ImportError as exc:
            raise SystemExit(
                f"Could not build the model for this config: {exc}\n"
                "If this is a GNN/GINE config, run the script in the same "
                "environment used for GNN training, with PyTorch Geometric installed."
            ) from exc
        summary = _summarize(args, actors, critic)
        if cli.json:
            print(json.dumps({"config": str(config_path), **summary}, indent=2))
        else:
            _print_human(summary, config_path)
    finally:
        if env is not None:
            env.close()
        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
