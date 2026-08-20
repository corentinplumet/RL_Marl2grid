"""Fresh actor construction and clean checkpoint bases for supervised controls."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict

import torch as th


def apply_scratch_architecture_overrides(
    args: Namespace,
    *,
    initialization: str,
    gnn_layers: int | None,
) -> Dict[str, Dict[str, int]]:
    """Apply explicitly requested scratch-only architecture changes."""
    if gnn_layers is None:
        return {}
    if initialization != "scratch":
        raise ValueError(
            "--scratch-gnn-layers is valid only with --initialization scratch. "
            "Changing the depth of a warm-start actor would make its checkpoint "
            "weights incompatible."
        )
    if int(gnn_layers) <= 0:
        raise ValueError("--scratch-gnn-layers must be a positive integer.")

    template_layers = int(getattr(args, "gnn_layers", 0))
    args.gnn_layers = int(gnn_layers)
    return {
        "gnn_layers": {
            "template": template_layers,
            "target": int(gnn_layers),
        }
    }


def build_scratch_actors(
    args: Namespace,
    evaluator: Any,
    device: th.device,
) -> Dict[str, Any]:
    """Build target-environment actors without loading learned parameters.

    ``args`` can originate from a checkpoint used as an architecture template,
    but this construction path deliberately never receives that checkpoint's
    state dict. Shared graph/scorer modules are therefore initialized exactly
    once by ``build_actor_modules`` and are fresh for every scratch run.
    """
    from alg.mappo.core import build_actor_modules

    actor_env = evaluator.env.env
    agent_ids = [
        f"agent_{idx}" for idx in range(len(actor_env.observation_space.keys()))
    ]
    continuous_actions = getattr(args, "action_type", "topology") == "redispatch"
    actors = build_actor_modules(
        actor_env,
        args,
        agent_ids,
        continuous_actions,
        device=device,
    )
    for actor in actors.values():
        actor.train()
    return actors


def scratch_checkpoint_base(args: Namespace) -> Dict[str, Any]:
    """Return a clean, evaluation-compatible base for a scratch BC record.

    In particular, do not copy a template critic, MAPPO optimizer, rollout
    state, observation statistics, or learned actor weights into the output.
    The supervised actor optimizer is added later by the BC checkpoint writer.
    """
    return {
        "args": args,
        "global_step": 0,
        "last_rollout": 0,
        "training_state": {},
        "resume_state": None,
        "wb_run_name": "",
    }
