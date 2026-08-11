"""Architecture compatibility helpers for loading older checkpoints."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import torch as th


TASK_DIR = Path(__file__).resolve().parents[1]


def _saved_node_pre_encoder_width(record: Dict[str, Any]) -> Optional[int]:
    widths = set()
    for value in record.values():
        if not isinstance(value, dict):
            continue
        for key, tensor in value.items():
            if key.endswith("node_pre_encoder.0.weight") and (
                isinstance(tensor, th.Tensor) and tensor.ndim == 2
            ):
                widths.add(int(tensor.shape[1]))
    return next(iter(widths)) if len(widths) == 1 else None


def _scenario_has_maintenance(args: Namespace) -> bool:
    config_path = TASK_DIR / "env" / str(
        getattr(args, "env_config_path", "scenario.json")
    )
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        return bool(
            config["environments"][str(getattr(args, "env_id", "bus14"))]
            .get("maintenance", False)
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def configure_legacy_connected_feature(
    args: Namespace, record: Dict[str, Any]
) -> Namespace:
    """Restore the obsolete ``connected`` input when saved weights require it."""
    graph_type = str(getattr(args, "gnn_graph_type", "bus")).lower()
    if graph_type not in {
        "heterogeneous",
        "hetero",
        "heterogeneous_line",
        "heterogeneous_line_nodes",
        "hetero_line",
    }:
        return args

    saved_width = _saved_node_pre_encoder_width(record)
    if saved_width is None:
        return args

    if graph_type in {
        "heterogeneous_line",
        "heterogeneous_line_nodes",
        "hetero_line",
    }:
        current_raw_width = 12 + (2 if _scenario_has_maintenance(args) else 0)
    else:
        current_raw_width = 7
        if str(getattr(args, "gnn_angle_representation", "node")) == "edge_diff":
            current_raw_width -= 1

    id_width = 0
    if bool(getattr(args, "gnn_node_id_embeddings", False)):
        id_width = 2 * int(getattr(args, "gnn_node_id_emb_dim", 8))
    current_encoder_width = current_raw_width + id_width

    if saved_width == current_encoder_width + 1:
        args.gnn_include_legacy_connected_feature = True
        print(
            "Detected legacy graph node schema: restoring the saved "
            f"'connected' column ({saved_width} inputs instead of "
            f"{current_encoder_width}).",
            flush=True,
        )
    return args
