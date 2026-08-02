#!/usr/bin/env python3
"""Compare actor-graph feature distributions across two Grid2Op environments.

The GNN actor encoder consumes graph node/edge features directly. When
`gnn_physical_scaling` and `gnn_running_norm` are both off those features are
raw physical quantities, so an encoder trained on one grid is fed a different
input distribution on another. This tool quantifies that shift and plots it.

Example:

    python tools/compare_graph_features.py \
        --envs bus14 bus36_wcci_nomaint \
        --steps 3000 \
        --actor-encoder gnn --gnn-graph-type bus --gnn-include-neighbors true \
        --output-dir outputs/graph_feature_comparison/bus
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "marl2grid_mplconfig")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "marl2grid_cache")
)

import numpy as np  # noqa: E402

from alg.mappo.config import get_alg_args  # noqa: E402
from common.feature_stats import (  # noqa: E402
    compare_samples,
    markdown_table,
    plot_distributions,
)
from common.imports import Namespace  # noqa: E402
from env.config import get_env_args  # noqa: E402
from env.utils import MAEnvWrapper  # noqa: E402

def _build_env_args(cli: Namespace, env_id: str) -> Namespace:
    env_args = get_env_args()
    alg_args = get_alg_args()
    args = Namespace(**vars(env_args), **vars(alg_args))
    args.env_id = env_id
    args.seed = cli.seed
    args.actor_encoder = "gnn"
    if getattr(args, "critic_encoder", "mlp") != "gnn":
        args.critic_encoder = "mlp"
    args.norm_obs = False
    args.use_heuristic = False
    args.track = False
    args.checkpoint = False
    args.n_envs = 1
    # --split only has an effect when chronic splitting is enabled.
    args.split_chronics = bool(cli.split)
    return args


def _node_type_labels(spec: Dict[str, Any]) -> Tuple[np.ndarray, Dict[int, str]]:
    node_type = spec.get("node_type")
    if node_type is None:
        n_nodes = len(spec.get("node_ids", []))
        return np.zeros((n_nodes,), dtype=np.int64), {0: "node"}
    names = {
        int(type_id): str(name)
        for name, type_id in dict(spec.get("node_type_names", {})).items()
    }
    node_type = np.asarray(node_type, dtype=np.int64)
    for type_id in np.unique(node_type):
        names.setdefault(int(type_id), f"type_{int(type_id)}")
    return node_type, names


def collect_env_samples(
    env_id: str, cli: Namespace
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Roll out do-nothing and gather the features the shared encoder sees."""
    args = _build_env_args(cli, env_id)
    env = MAEnvWrapper(args, idx=0, eval_env=True, chronic_split=cli.split or None)
    buckets: Dict[str, List[np.ndarray]] = {}
    meta: Dict[str, Any] = {"env_id": env_id, "agents": [], "episodes": 0}

    try:
        agents = sorted(env.g2op_ma_env.agents)
        meta["agents"] = agents
        specs = {agent: env.graph_specs[agent] for agent in agents}
        node_names = list(specs[agents[0]].get("node_feature_names", []))
        edge_names = list(specs[agents[0]].get("edge_feature_names", []))
        meta["node_feature_names"] = node_names
        meta["edge_feature_names"] = edge_names

        steps = 0
        episodes = 0
        while steps < cli.steps:
            env.reset()
            episodes += 1
            done = False
            episode_steps = 0
            while not done and steps < cli.steps:
                if cli.max_episode_steps and episode_steps >= cli.max_episode_steps:
                    break
                graphs = env.graph_builder.build(env._obs)
                # Measure what the encoder is actually fed: with both scaling
                # flags off this is a no-op and the values stay raw, but with
                # them on the tool reports the scaled distribution instead.
                if env.graph_feature_processor is not None:
                    graphs = env.graph_feature_processor.process(graphs, update=True)
                for agent in agents:
                    graph = graphs[agent]
                    spec = specs[agent]

                    node_type, type_names = _node_type_labels(spec)
                    nodes = np.asarray(graph["node_features"], dtype=np.float32)
                    node_mask = np.asarray(graph["node_mask"], dtype=np.float32) > 0
                    for type_id in np.unique(node_type):
                        rows = node_mask & (node_type == type_id)
                        if not rows.any():
                            continue
                        label = type_names[int(type_id)]
                        for col, name in enumerate(node_names):
                            key = f"node/{label}/{name}"
                            buckets.setdefault(key, []).append(nodes[rows, col])

                    # Structural relations carry no physical value; only the
                    # active physical edges describe the grid state.
                    edges = np.asarray(graph["edge_features"], dtype=np.float32)
                    edge_mask = np.asarray(graph["edge_mask"], dtype=np.float32) > 0
                    if edge_mask.any():
                        for col, name in enumerate(edge_names):
                            key = f"edge/physical/{name}"
                            buckets.setdefault(key, []).append(edges[edge_mask, col])

                actions = {agent: 0 for agent in agents}
                _, _, terminated, truncated, _ = env.step(actions)
                # Both come back keyed by agent, so a plain bool() on the dict
                # would end every episode after one step.
                done = any(terminated.values()) or any(truncated.values())
                steps += 1
                episode_steps += 1

        meta["episodes"] = episodes
        meta["steps"] = steps
    finally:
        try:
            env.close()
        except Exception:
            pass

    rng = np.random.default_rng(cli.seed)
    samples: Dict[str, np.ndarray] = {}
    for key, chunks in buckets.items():
        values = np.concatenate(chunks)
        values = values[np.isfinite(values)]
        if values.size > cli.max_samples:
            values = values[rng.choice(values.size, cli.max_samples, replace=False)]
        samples[key] = values
    return samples, meta







def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--help", action="help")
    parser.add_argument(
        "--envs",
        nargs=2,
        default=["bus14", "bus36_wcci_nomaint"],
        help="The two scenario ids to compare.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--steps", type=int, default=3000, help="Env steps sampled per environment."
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=300,
        help=(
            "Move to the next chronic after this many steps. Keeps the sample "
            "spread over many scenarios instead of one long episode. 0 disables."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        choices=["", "train", "test"],
        help="Optional chronic split to sample from. Empty uses every chronic.",
    )
    parser.add_argument("--max-samples", type=int, default=400000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/graph_feature_comparison")
    )
    cli, _ = parser.parse_known_args()
    return cli


def main() -> int:
    cli = parse_args()
    label_a, label_b = cli.envs
    output_dir = cli.output_dir
    if not output_dir.is_absolute():
        output_dir = TASK_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    collected = {}
    metas = {}
    for env_id in cli.envs:
        print(f"=== sampling {env_id} for {cli.steps} steps ===", flush=True)
        samples, meta = collect_env_samples(env_id, cli)
        collected[env_id] = samples
        metas[env_id] = meta
        print(
            f"{env_id}: {meta['episodes']} episodes, {meta['steps']} steps, "
            f"{len(samples)} feature streams",
            flush=True,
        )

    rows = compare_samples(collected[label_a], collected[label_b], label_a, label_b)

    (output_dir / "meta.json").write_text(
        json.dumps(metas, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text(
        markdown_table(rows, label_a, label_b), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / f"samples_{label_a}.npz", **collected[label_a]
    )
    np.savez_compressed(
        output_dir / f"samples_{label_b}.npz", **collected[label_b]
    )
    try:
        plot_distributions(
            collected[label_a],
            collected[label_b],
            label_a,
            label_b,
            output_dir / "distributions.png",
        )
    except ImportError:
        # The metrics are the point; plotting is optional and matplotlib is not
        # installed everywhere. The sample archives are already on disk.
        print(
            "matplotlib is unavailable here, so distributions.png was skipped. "
            "Render it anywhere with numpy+matplotlib:\n"
            f"  python tools/plot_feature_samples.py {output_dir}"
        )

    print("")
    print("========== Largest distribution shifts ==========")
    ranked = sorted(
        (row for row in rows if row["present_in_both"]),
        key=lambda r: -(r["ks"] if np.isfinite(r["ks"]) else 0),
    )
    for row in ranked[:12]:
        print(
            f"{row['feature']:<34} ks={row['ks']:.3f} "
            f"shift={row['std_mean_shift']:+.2f}sd "
            f"std_ratio={row['std_ratio']:.2f} "
            f"w1={row['w1_over_pooled_std']:.2f}sd"
        )
    print("")
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
