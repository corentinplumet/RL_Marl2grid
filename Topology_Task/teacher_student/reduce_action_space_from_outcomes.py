#!/usr/bin/env python3
"""Build a reduced discrete action set from action-outcome shards."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from teacher_student.dataset import (
    load_metadata,
    metadata_dir,
    list_shards,
    resolve_dataset_dir,
    task_relative,
)


def _finite_metric(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.isfinite(values)


def _best_per_state_counts(
    *,
    action_ids: np.ndarray,
    metric: np.ndarray,
    valid: np.ndarray,
    episode_ids: np.ndarray,
    dataset_steps: np.ndarray,
    require_improvement: bool,
    improvement_tolerance: float,
) -> Tuple[Counter, Dict[str, Any]]:
    action_ids = np.asarray(action_ids, dtype=np.int64)
    metric = np.asarray(metric, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    dataset_steps = np.asarray(dataset_steps, dtype=np.int64)

    keep = valid & _finite_metric(metric)
    if require_improvement:
        keep &= metric < -float(improvement_tolerance)
    if not np.any(keep):
        return Counter(), {
            "candidate_rows": int(action_ids.size),
            "eligible_rows": 0,
            "states_with_eligible_action": 0,
        }

    kept_action_ids = action_ids[keep]
    kept_metric = metric[keep]
    kept_episode_ids = episode_ids[keep]
    kept_dataset_steps = dataset_steps[keep]

    order = np.lexsort((kept_metric, kept_dataset_steps, kept_episode_ids))
    kept_action_ids = kept_action_ids[order]
    kept_metric = kept_metric[order]
    kept_episode_ids = kept_episode_ids[order]
    kept_dataset_steps = kept_dataset_steps[order]

    counts: Counter = Counter()
    states = 0
    previous_state: Optional[Tuple[int, int]] = None
    for action_id, episode_id, dataset_step in zip(
        kept_action_ids,
        kept_episode_ids,
        kept_dataset_steps,
    ):
        state = (int(episode_id), int(dataset_step))
        if state == previous_state:
            continue
        counts[int(action_id)] += 1
        states += 1
        previous_state = state

    return counts, {
        "candidate_rows": int(action_ids.size),
        "eligible_rows": int(keep.sum()),
        "states_with_eligible_action": int(states),
    }


def _all_improving_counts(
    *,
    action_ids: np.ndarray,
    metric: np.ndarray,
    valid: np.ndarray,
    require_improvement: bool,
    improvement_tolerance: float,
) -> Tuple[Counter, Dict[str, Any]]:
    action_ids = np.asarray(action_ids, dtype=np.int64)
    metric = np.asarray(metric, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    keep = valid & _finite_metric(metric)
    if require_improvement:
        keep &= metric < -float(improvement_tolerance)
    counts = Counter(int(action_id) for action_id in action_ids[keep])
    return counts, {
        "candidate_rows": int(action_ids.size),
        "eligible_rows": int(keep.sum()),
        "states_with_eligible_action": None,
    }


def _merge_counter(dst: Counter, src: Counter) -> None:
    for key, value in src.items():
        dst[int(key)] += int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce each agent's discrete action space from action-outcome "
            "dataset shards by keeping frequent useful actions."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON. Defaults to "
            "<dataset>/metadata/reduced_action_space.json."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="delta_vs_do_nothing",
        choices=["delta_vs_do_nothing", "delta_vs_now", "rho_after_action"],
        help=(
            "Metric minimized when selecting actions. Delta metrics should be "
            "negative for improvement."
        ),
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default="best_per_state",
        choices=["best_per_state", "all_improving"],
        help=(
            "best_per_state mimics the paper's teacher: one winning action per "
            "state. all_improving counts every action that improves the metric."
        ),
    )
    parser.add_argument("--top-k", type=int, default=208)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--include-action-zero", type=str, default="true")
    parser.add_argument("--require-improvement", type=str, default="true")
    parser.add_argument("--improvement-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-shards", type=int, default=None)
    return parser.parse_args()


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    cli = parse_args()
    if cli.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if cli.min_count <= 0:
        raise ValueError("--min-count must be positive.")

    dataset_dir = resolve_dataset_dir(cli.dataset)
    metadata = load_metadata(dataset_dir)
    if metadata.get("dataset_mode") != "action_outcomes":
        raise ValueError(
            f"{task_relative(dataset_dir)} does not look like an action_outcomes "
            "dataset. Collect with --dataset-mode action_outcomes first."
        )

    output = cli.output
    if output is None:
        output = metadata_dir(dataset_dir) / "reduced_action_space.json"
    output = output.expanduser()
    if not output.is_absolute():
        output = (TASK_DIR / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    agent_ids: List[str] = list(metadata["agent_ids"])
    include_action_zero = _str_to_bool(cli.include_action_zero)
    require_improvement = _str_to_bool(cli.require_improvement)

    counts_by_agent: Dict[str, Counter] = {agent: Counter() for agent in agent_ids}
    stats_by_agent: Dict[str, Dict[str, Any]] = {
        agent: defaultdict(int) for agent in agent_ids
    }

    for shard in list_shards(dataset_dir, cli.max_shards):
        with np.load(shard) as data:
            for agent in agent_ids:
                prefix = f"outcome_{agent}"
                action_key = f"{prefix}_action_id"
                metric_key = f"{prefix}_{cli.metric}"
                valid_key = f"{prefix}_action_is_valid"
                if action_key not in data.files:
                    continue
                if metric_key not in data.files:
                    raise KeyError(f"Missing {metric_key} in {task_relative(shard)}")

                kwargs = {
                    "action_ids": data[action_key],
                    "metric": data[metric_key],
                    "valid": data[valid_key],
                    "require_improvement": require_improvement,
                    "improvement_tolerance": cli.improvement_tolerance,
                }
                if cli.selection_method == "best_per_state":
                    counts, stats = _best_per_state_counts(
                        **kwargs,
                        episode_ids=data[f"{prefix}_episode_id"],
                        dataset_steps=data[f"{prefix}_dataset_step"],
                    )
                else:
                    counts, stats = _all_improving_counts(**kwargs)
                _merge_counter(counts_by_agent[agent], counts)
                for key, value in stats.items():
                    if value is not None:
                        stats_by_agent[agent][key] += int(value)

    action_sizes = {
        agent: int(size) for agent, size in metadata.get("action_sizes", {}).items()
    }
    reduced: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": task_relative(dataset_dir),
        "source_metadata": {
            "env_id": metadata.get("env_id"),
            "collection_rho_threshold": metadata.get("collection_rho_threshold"),
            "n_candidate_states": metadata.get("n_candidate_states"),
            "n_outcome_examples": metadata.get("n_outcome_examples"),
        },
        "metric": cli.metric,
        "selection_method": cli.selection_method,
        "top_k": int(cli.top_k),
        "min_count": int(cli.min_count),
        "include_action_zero": bool(include_action_zero),
        "require_improvement": bool(require_improvement),
        "improvement_tolerance": float(cli.improvement_tolerance),
        "agents": {},
    }

    total_original = 0
    total_reduced = 0
    for agent in agent_ids:
        total_original += int(action_sizes.get(agent, 0))
        ranked = [
            {"action_id": int(action_id), "count": int(count)}
            for action_id, count in counts_by_agent[agent].most_common()
            if int(count) >= cli.min_count
        ]
        selected = [item["action_id"] for item in ranked[: cli.top_k]]
        if include_action_zero and 0 not in selected:
            selected = [0] + selected
        selected = selected[: cli.top_k] if len(selected) > cli.top_k else selected
        total_reduced += len(selected)
        original_size = int(action_sizes.get(agent, 0))
        reduced_frac = (
            len(selected) / original_size if original_size > 0 else float("nan")
        )
        reduced["agents"][agent] = {
            "original_action_size": original_size,
            "selected_action_size": len(selected),
            "selected_fraction": reduced_frac,
            "selected_action_ids": selected,
            "ranked_actions": ranked[: max(cli.top_k, 50)],
            "stats": dict(stats_by_agent[agent]),
        }

    reduced["total_original_action_size"] = int(total_original)
    reduced["total_selected_action_size"] = int(total_reduced)
    reduced["total_selected_fraction"] = (
        total_reduced / total_original if total_original > 0 else float("nan")
    )

    with output.open("w", encoding="utf-8") as f:
        json.dump(reduced, f, indent=2, sort_keys=True)
        f.write("\n")

    print("========== Reduced action space ==========")
    print(f"Dataset: {task_relative(dataset_dir)}")
    print(f"Output: {task_relative(output)}")
    print(
        f"Total selected: {total_reduced}/{total_original} "
        f"({reduced['total_selected_fraction']:.4f})"
    )
    for agent, payload in reduced["agents"].items():
        print(
            f"{agent}: selected={payload['selected_action_size']}/"
            f"{payload['original_action_size']} "
            f"({payload['selected_fraction']:.4f})"
        )


if __name__ == "__main__":
    main()
