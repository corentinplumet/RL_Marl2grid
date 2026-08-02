#!/usr/bin/env python3
"""Inspect thesis-style graph specs for each Grid2Op multi-agent region."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "marl2grid_mplconfig")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "marl2grid_cache")
)

from alg.mappo.config import get_alg_args  # noqa: E402
from common.imports import Namespace, np  # noqa: E402
from env.config import get_env_args  # noqa: E402
from env.utils import MAEnvWrapper  # noqa: E402


def _edge_type_counts(spec: dict) -> dict[str, int]:
    edge_type = np.asarray(spec.get("edge_type", []), dtype=np.int64)
    names = {
        int(type_id): name
        for name, type_id in dict(spec.get("edge_type_names", {})).items()
    }
    counts: dict[str, int] = {}
    for type_id in sorted(set(edge_type.tolist())):
        counts[names.get(type_id, str(type_id))] = int(np.sum(edge_type == type_id))
    return counts


def _node_type_counts(spec: dict) -> dict[str, int]:
    node_type = np.asarray(spec.get("node_type", []), dtype=np.int64)
    names = {
        int(type_id): name
        for name, type_id in dict(spec.get("node_type_names", {})).items()
    }
    counts: dict[str, int] = {}
    for type_id in sorted(set(node_type.tolist())):
        counts[names.get(type_id, str(type_id))] = int(np.sum(node_type == type_id))
    return counts


def _print_spec(name: str, spec: dict) -> None:
    edge_type_counts = _edge_type_counts(spec)
    node_type_counts = _node_type_counts(spec)
    controlled_mask = np.asarray(spec.get("controlled_node_mask", []), dtype=np.float32)
    print(f"{name}:")
    print(f"  graph_type: {spec.get('graph_type', 'bus')}")
    print(f"  nodes: {len(spec.get('node_ids', []))}")
    if node_type_counts:
        print(f"  node_types: {node_type_counts}")
    print(f"  controlled_nodes: {int(controlled_mask.sum())}")
    print(f"  lines: {len(spec.get('line_ids', []))}")
    print(f"  edges: {int(np.asarray(spec.get('edge_index')).shape[1])}")
    print(f"  edge_types: {edge_type_counts}")
    print(f"  node_dim: {spec.get('node_dim')}")
    print(f"  edge_dim: {spec.get('edge_dim')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--help", action="help")
    # --seed is owned by main.py's parser, which this tool does not use, but
    # MAEnvWrapper needs it to seed the environment.
    parser.add_argument("--seed", type=int, default=0)
    cli, _ = parser.parse_known_args()

    env_args = get_env_args()
    alg_args = get_alg_args()
    args = Namespace(**vars(env_args), **vars(alg_args))
    args.seed = cli.seed
    args.actor_encoder = "gnn"
    if getattr(args, "critic_encoder", "mlp") != "gnn":
        args.critic_encoder = "mlp"
    args.track = False
    args.checkpoint = False

    env = MAEnvWrapper(args, idx=0)
    try:
        print("========== Graph Spec Inspection ==========")
        print(f"env_id: {args.env_id}")
        print(f"gnn_type: {args.gnn_type}")
        print(f"gnn_graph_type: {args.gnn_graph_type}")
        print(f"gnn_include_neighbors: {args.gnn_include_neighbors}")
        print(f"sparse_gt_add_self_edges: {getattr(args, 'sparse_gt_add_self_edges', None)}")
        print(
            "sparse_gt_add_substation_edges: "
            f"{getattr(args, 'sparse_gt_add_substation_edges', None)}"
        )
        print("===========================================")
        for name, spec in env.graph_specs.items():
            _print_spec(name, spec)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
