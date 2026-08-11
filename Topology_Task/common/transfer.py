"""Cross-environment transfer of trained graph actors.

The GNN actor encoder is grid-size independent: its parameters are shaped by
node/edge feature widths and hidden sizes, never by the number of substations.
That makes it the one component of a bus14 policy that can be reused on a
larger grid, where the per-agent action heads (and the agent partition itself)
no longer line up.

The graph encoder is always transferable when its feature schema matches.  A
candidate-action scorer is grid-size independent too: its learned scalar scorer
does not depend on the number of target actions.  Its action metadata buffers
do depend on the target grid, so full-actor transfer copies learned head
parameters while retaining the freshly built target metadata.
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
_ACTION_HEAD_PREFIX = "actor."
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


def _load_candidate_action_heads(
    actors: Dict[str, nn.Module],
    record: Dict[str, Any],
    path: str,
    source_agent: str,
    freeze: bool,
) -> Dict[str, Any]:
    """Broadcast one learned candidate scorer to all target actors.

    Only named parameters are copied.  Candidate indices, masks, action
    descriptors, and other registered buffers remain those constructed from
    each target agent's WCCI action space.
    """
    if source_agent not in record or not isinstance(record[source_agent], dict):
        available = sorted(
            key
            for key, value in record.items()
            if key.startswith("agent_") and isinstance(value, dict)
        )
        raise ValueError(
            f"Transfer checkpoint '{path}' has no source actor {source_agent!r}; "
            f"available actors are {available}."
        )

    source_actor_state = record[source_agent]
    loaded_parameter_names: List[str] = []
    for target_agent, target_actor in sorted(actors.items()):
        if str(getattr(target_actor, "actor_action_head", "")) != "candidate_pool":
            raise ValueError(
                "Action-head transfer requires actor_action_head='candidate_pool'; "
                f"{target_agent} uses {getattr(target_actor, 'actor_action_head', None)!r}."
            )
        target_head = getattr(target_actor, "actor", None)
        if target_head is None:
            raise ValueError(f"{target_agent} exposes no candidate action head.")

        target_parameters = dict(target_head.named_parameters())
        transferable: Dict[str, th.Tensor] = {}
        missing: List[str] = []
        mismatched: List[str] = []
        for name, target_parameter in target_parameters.items():
            source_name = _ACTION_HEAD_PREFIX + name
            source_parameter = source_actor_state.get(source_name)
            if source_parameter is None:
                missing.append(name)
                continue
            if tuple(source_parameter.shape) != tuple(target_parameter.shape):
                mismatched.append(
                    f"{name}: checkpoint {tuple(source_parameter.shape)} vs model "
                    f"{tuple(target_parameter.shape)}"
                )
                continue
            transferable[name] = source_parameter

        if missing or mismatched:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing[:8]))
            if mismatched:
                details.append("shape mismatch " + "; ".join(mismatched[:8]))
            raise ValueError(
                f"Candidate action-head mismatch between '{path}' "
                f"({source_agent}) and {target_agent}: {'; '.join(details)}. "
                "The source and target must use the same candidate pooling, "
                "action-feature, do-nothing-head, and MLP settings."
            )

        target_head.load_state_dict(transferable, strict=False)
        if freeze:
            for parameter in target_head.parameters():
                parameter.requires_grad_(False)
        loaded_parameter_names = sorted(transferable)

    frozen_count = sum(
        parameter.numel()
        for actor in actors.values()
        for parameter in getattr(actor, "actor").parameters()
        if not parameter.requires_grad
    )
    return {
        "source_agent": source_agent,
        "target_heads": len(actors),
        "loaded_parameters": len(loaded_parameter_names),
        "frozen": bool(freeze),
        "frozen_parameter_count": int(frozen_count),
    }


def _drop_legacy_relation_self_column(
    graph_encoder: nn.Module,
    source_tensor: th.Tensor,
    target_tensor: th.Tensor,
) -> Optional[th.Tensor]:
    """Migrate an old edge-input weight whose removed input was always zero."""
    feature_names = list(getattr(graph_encoder, "edge_feature_names", []))
    if "relation_self" in feature_names:
        return None
    relation_columns = [
        idx for idx, name in enumerate(feature_names) if name.startswith("relation_")
    ]
    if not relation_columns or source_tensor.ndim != 2 or target_tensor.ndim != 2:
        return None
    removed_column = relation_columns[0]
    edge_dim = int(getattr(graph_encoder, "edge_dim", -1))
    if (
        target_tensor.shape[1] != edge_dim
        or source_tensor.shape[0] != target_tensor.shape[0]
        or source_tensor.shape[1] != target_tensor.shape[1] + 1
        or removed_column >= source_tensor.shape[1]
    ):
        return None
    return th.cat(
        [
            source_tensor[:, :removed_column],
            source_tensor[:, removed_column + 1 :],
        ],
        dim=1,
    )


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
    load_action_head: bool = False,
    action_head_source_agent: str = "agent_0",
    freeze_action_head: bool = False,
) -> Dict[str, Any]:
    """Load a pretrained graph encoder and optional candidate action head.

    Args:
        actors: The freshly built actors, keyed by agent id.
        checkpoint_name: Checkpoint stem (resolved under Topology_Task/checkpoint)
            or an explicit .tar path.
        freeze: When True, the encoder parameters stop requiring gradients.
        device: Device the actors live on.
        load_action_head: Also load learned candidate-scorer parameters.  Target
            action metadata buffers are deliberately retained.
        action_head_source_agent: Source scorer broadcast to every target actor.
            This explicit rule is needed because bus14 and WCCI have different
            numbers of independently parameterized actors.
        freeze_action_head: Stop the transferred candidate heads from requiring
            gradients.

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
    migrated_legacy_self_columns = 0
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
                migrated_tensor = _drop_legacy_relation_self_column(
                    graph_encoder, source_tensor, target_tensor
                )
                if migrated_tensor is None:
                    shape_mismatch.append(name)
                    continue
                source_tensor = migrated_tensor
                migrated_legacy_self_columns += 1
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
    head_report = None
    if load_action_head:
        head_report = _load_candidate_action_heads(
            actors,
            record,
            path,
            action_head_source_agent,
            freeze_action_head,
        )

    return {
        "checkpoint": path,
        "encoders": len(targets),
        "loaded_parameters": len(loaded_params),
        "migrated_legacy_relation_self_weights": int(
            migrated_legacy_self_columns
        ),
        "skipped_topology_buffers": skipped_buffers,
        "frozen": bool(freeze),
        "frozen_parameter_count": int(n_frozen),
        "source_global_step": int(record.get("global_step", 0) or 0),
        "action_head": head_report,
    }
