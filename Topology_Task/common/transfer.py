"""Cross-environment transfer of a trained actor graph encoder.

The GNN actor encoder is grid-size independent: its parameters are shaped by
node/edge feature widths and hidden sizes, never by the number of substations.
That makes it the one component of a bus14 policy that can be reused on a
larger grid, where the per-agent action heads (and the agent partition itself)
no longer line up.

This module lifts `encoder.graph_encoder.*` out of a saved actor checkpoint and
loads it into freshly built actors, optionally freezing it so only the heads
train.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch as th
import torch.nn as nn

# Persistent buffers that describe the *graph* rather than the learned mapping.
# They are re-registered from the target environment's own spec at construction
# time, and are overridden per agent at call time by GraphAndFlatEncoder, so a
# source grid's copies are both useless and the wrong shape here.
TOPOLOGY_BUFFER_NAMES = (
    "edge_index",
    "node_ids",
    "edge_type",
    "controlled_node_mask",
)

_ENCODER_PREFIX = "encoder.graph_encoder."
_CHECKPOINT_DIR = "checkpoint"


def resolve_checkpoint_path(name: str) -> str:
    """Resolve a checkpoint stem or path the same way CheckpointSaver does."""
    path = name if name.endswith(".tar") else name + ".tar"
    if os.path.isabs(path) or os.path.dirname(path):
        return path
    return os.path.join(_CHECKPOINT_DIR, path)


def _distinct_graph_encoders(actors: Dict[str, nn.Module]) -> List[Tuple[str, nn.Module]]:
    """Return each distinct graph encoder once, keyed by a representative agent.

    With share_actor_gnn=true every actor points at the same module, so this
    collapses to a single entry; with per-actor encoders it returns one entry
    per agent.
    """
    found: List[Tuple[str, nn.Module]] = []
    seen = set()
    for agent_id in sorted(actors.keys()):
        encoder = getattr(actors[agent_id], "encoder", None)
        graph_encoder = getattr(encoder, "graph_encoder", None)
        if graph_encoder is None:
            continue
        if id(graph_encoder) in seen:
            continue
        seen.add(id(graph_encoder))
        found.append((agent_id, graph_encoder))
    return found


def _source_encoder_state(record: Dict[str, Any], path: str) -> Dict[str, th.Tensor]:
    """Pull the graph-encoder weights out of a saved MAPPO checkpoint."""
    agent_keys = sorted(
        key
        for key, value in record.items()
        if key.startswith("agent_") and isinstance(value, dict)
    )
    if not agent_keys:
        raise ValueError(
            f"Checkpoint '{path}' holds no actor state dicts (looked for 'agent_*' "
            f"keys, found {sorted(record.keys())})."
        )

    # Every actor shares the encoder under share_actor_gnn, and when they do not
    # the first agent's copy is still the only sensible single source.
    source = record[agent_keys[0]]
    state = {
        key[len(_ENCODER_PREFIX) :]: value
        for key, value in source.items()
        if key.startswith(_ENCODER_PREFIX)
    }
    if not state:
        raise ValueError(
            f"Checkpoint '{path}' has no '{_ENCODER_PREFIX}*' weights under "
            f"'{agent_keys[0]}'. It was most likely trained with a non-GNN actor "
            "encoder."
        )
    return state


def freeze_graph_encoders(actors: Dict[str, nn.Module]) -> Dict[str, Any]:
    """Stop the actor graph encoders from requiring gradients.

    Used on its own when a frozen-encoder run resumes from its own checkpoint:
    the weights are already in that checkpoint, only the freeze needs reapplying.
    """
    targets = _distinct_graph_encoders(actors)
    if not targets:
        raise ValueError(
            "No actor exposes encoder.graph_encoder; freezing requires "
            "actor_encoder='gnn'."
        )
    n_frozen = 0
    for _, graph_encoder in targets:
        for param in graph_encoder.parameters():
            param.requires_grad_(False)
            n_frozen += param.numel()
    return {
        "checkpoint": None,
        "encoders": len(targets),
        "frozen": True,
        "frozen_parameter_count": int(n_frozen),
    }


def load_encoder_from_checkpoint(
    actors: Dict[str, nn.Module],
    checkpoint_name: str,
    freeze: bool = False,
    device: Optional[th.device] = None,
) -> Dict[str, Any]:
    """Load a pretrained graph encoder into `actors` and optionally freeze it.

    Args:
        actors: The freshly built actors, keyed by agent id.
        checkpoint_name: Checkpoint stem (resolved under Topology_Task/checkpoint)
            or an explicit .tar path.
        freeze: When True, the encoder parameters stop requiring gradients.
        device: Device the actors live on.

    Returns:
        A summary dict for logging.

    Raises:
        FileNotFoundError: The checkpoint does not exist.
        ValueError: The checkpoint holds no usable encoder, or the source and
            target encoder architectures disagree.
    """
    path = resolve_checkpoint_path(checkpoint_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find transfer checkpoint '{path}'. Pass either a stem from "
            "Topology_Task/checkpoint or a path to a .tar checkpoint."
        )

    targets = _distinct_graph_encoders(actors)
    if not targets:
        raise ValueError(
            "No actor exposes encoder.graph_encoder; transfer requires "
            "actor_encoder='gnn'."
        )

    record = th.load(path, map_location=device or "cpu", weights_only=False)
    source_state = _source_encoder_state(record, path)

    loaded_params: List[str] = []
    skipped_buffers: List[str] = []
    for agent_id, graph_encoder in targets:
        target_state = graph_encoder.state_dict()
        param_names = {name for name, _ in graph_encoder.named_parameters()}

        transferable: Dict[str, th.Tensor] = {}
        shape_mismatch: List[str] = []
        for name, target_tensor in target_state.items():
            if name not in source_state:
                continue
            source_tensor = source_state[name]
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                shape_mismatch.append(name)
                continue
            transferable[name] = source_tensor

        # Grid-shaped buffers are expected to differ between the source and
        # target grids; anything else that differs means the two encoders were
        # not built with the same architecture, and silently training a
        # half-initialized encoder would be worse than failing here.
        unexpected = [
            name
            for name in shape_mismatch
            if name.rsplit(".", 1)[-1] not in TOPOLOGY_BUFFER_NAMES
        ]
        if unexpected:
            details = ", ".join(
                f"{name}: checkpoint {tuple(source_state[name].shape)} vs model "
                f"{tuple(target_state[name].shape)}"
                for name in sorted(unexpected)
            )
            raise ValueError(
                f"Encoder architecture mismatch between '{path}' and {agent_id}: "
                f"{details}. The transfer target must use the same GNN "
                "hyperparameters and observation features as the source run."
            )

        missing_params = sorted(param_names.difference(transferable))
        if missing_params:
            raise ValueError(
                f"Transfer checkpoint '{path}' is missing {len(missing_params)} "
                f"encoder parameters needed by {agent_id}: "
                f"{', '.join(missing_params[:8])}"
                + (" ..." if len(missing_params) > 8 else "")
                + ". The transfer target must use the same GNN hyperparameters "
                "and observation features as the source run."
            )

        graph_encoder.load_state_dict(transferable, strict=False)
        loaded_params = sorted(param_names)
        skipped_buffers = sorted(shape_mismatch)

        if freeze:
            for param in graph_encoder.parameters():
                param.requires_grad_(False)

    n_frozen = sum(
        param.numel()
        for _, graph_encoder in targets
        for param in graph_encoder.parameters()
        if not param.requires_grad
    )
    return {
        "checkpoint": path,
        "encoders": len(targets),
        "loaded_parameters": len(loaded_params),
        "skipped_topology_buffers": skipped_buffers,
        "frozen": bool(freeze),
        "frozen_parameter_count": int(n_frozen),
        "source_global_step": int(record.get("global_step", 0) or 0),
    }
