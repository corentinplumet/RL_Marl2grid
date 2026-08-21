"""Fresh actor construction and clean checkpoint bases for supervised controls."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict

import torch as th


def apply_scratch_architecture_overrides(
    args: Namespace,
    *,
    initialization: str,
    gnn_layers: int | None = None,
    candidate_action_pool: str | None = None,
    action_delta_encoder: bool | None = None,
    gnn_residual: bool | None = None,
    gnn_jumping_knowledge: str | None = None,
    gnn_readout_aggr: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Apply explicitly requested scratch-only architecture changes."""
    requested = {
        "gnn_layers": gnn_layers,
        "candidate_action_pool": candidate_action_pool,
        "candidate_action_delta_encoder": action_delta_encoder,
        "gnn_residual": gnn_residual,
        "gnn_jumping_knowledge": gnn_jumping_knowledge,
        "gnn_readout_aggr": gnn_readout_aggr,
    }
    requested = {name: value for name, value in requested.items() if value is not None}
    if not requested:
        return {}
    if initialization != "scratch":
        raise ValueError(
            "Scratch architecture overrides are valid only with "
            "--initialization scratch. Changing a warm-start architecture would "
            "make its checkpoint weights incompatible."
        )
    if gnn_layers is not None and int(gnn_layers) <= 0:
        raise ValueError("--scratch-gnn-layers must be a positive integer.")
    if candidate_action_pool is not None and str(candidate_action_pool) not in {
        "mean",
        "max",
        "typed_mean",
        "typed_max",
        "typed_attention",
    }:
        raise ValueError(
            "--scratch-candidate-action-pool must be mean, max, typed_mean, "
            "typed_max, or typed_attention."
        )
    if gnn_jumping_knowledge is not None and str(gnn_jumping_knowledge) not in {
        "none",
        "concat",
    }:
        raise ValueError("--scratch-gnn-jumping-knowledge must be 'none' or 'concat'.")

    overrides: Dict[str, Dict[str, Any]] = {}
    for name, target in requested.items():
        if name == "gnn_layers":
            target = int(target)
        elif name in {"candidate_action_delta_encoder", "gnn_residual"}:
            target = bool(target)
        elif name in {
            "candidate_action_pool",
            "gnn_jumping_knowledge",
            "gnn_readout_aggr",
        }:
            target = str(target)
        template = getattr(args, name, None)
        setattr(args, name, target)
        overrides[name] = {"template": template, "target": target}

    return overrides


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
