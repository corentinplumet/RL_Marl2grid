"""Dependency-free helpers for composing command-line configuration groups."""

from argparse import Namespace


def merge_runtime_args(*groups: Namespace) -> Namespace:
    """Merge parsed argument groups, with later parsers taking precedence."""
    merged = {}
    for group in groups:
        merged.update(vars(group))
    return Namespace(**merged)


def sparse_transformer_self_edges_enabled(gnn_type: str, configured) -> bool:
    """Resolve the sparse-transformer-only explicit self-edge option."""
    is_sparse_transformer = str(gnn_type).lower() == "sparse_transformer"
    if configured is None:
        return is_sparse_transformer
    return is_sparse_transformer and bool(configured)
