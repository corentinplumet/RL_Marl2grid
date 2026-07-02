#!/usr/bin/env python3
"""Build a reduced discrete action set from action-outcome shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


def _improvement_rate_counts(
    *,
    action_ids: np.ndarray,
    metric: np.ndarray,
    valid: np.ndarray,
    improvement_tolerance: float,
) -> Tuple[Dict[int, Dict[str, int]], Dict[str, Any]]:
    action_ids = np.asarray(action_ids, dtype=np.int64)
    metric = np.asarray(metric, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)

    seen = Counter(int(action_id) for action_id in action_ids)
    improved_mask = valid & _finite_metric(metric) & (
        metric < -float(improvement_tolerance)
    )
    improved = Counter(int(action_id) for action_id in action_ids[improved_mask])
    valid_finite = Counter(
        int(action_id) for action_id in action_ids[valid & _finite_metric(metric)]
    )

    stats_by_action: Dict[int, Dict[str, int]] = {}
    for action_id, seen_count in seen.items():
        stats_by_action[int(action_id)] = {
            "seen_count": int(seen_count),
            "improved_count": int(improved.get(action_id, 0)),
            "valid_finite_count": int(valid_finite.get(action_id, 0)),
        }

    return stats_by_action, {
        "candidate_rows": int(action_ids.size),
        "seen_action_ids": int(len(seen)),
        "improved_rows": int(improved_mask.sum()),
        "valid_finite_rows": int(sum(valid_finite.values())),
    }


def _merge_action_stats(
    dst: Dict[int, Dict[str, int]],
    src: Dict[int, Dict[str, int]],
) -> None:
    for action_id, values in src.items():
        target = dst.setdefault(
            int(action_id),
            {"seen_count": 0, "improved_count": 0, "valid_finite_count": 0},
        )
        target["seen_count"] += int(values.get("seen_count", 0))
        target["improved_count"] += int(values.get("improved_count", 0))
        target["valid_finite_count"] += int(values.get("valid_finite_count", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce each agent's discrete action space from action-outcome "
            "dataset shards by keeping frequent useful actions."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more action_outcomes dataset directories. For sharded "
            "SLURM array runs, pass every parts/part_* directory."
        ),
    )
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
        choices=["best_per_state", "all_improving", "improvement_rate"],
        help=(
            "best_per_state mimics the paper's teacher: one winning action per "
            "state. all_improving counts every action that improves the metric. "
            "improvement_rate ranks actions by improved_count / seen_count."
        ),
    )
    parser.add_argument("--top-k", type=int, default=208)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--include-action-zero", type=str, default="true")
    parser.add_argument("--require-improvement", type=str, default="true")
    parser.add_argument("--improvement-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument(
        "--progress-every-shards",
        type=int,
        default=25,
        help=(
            "Print reducer progress every N shards. Set to 0 to print only one "
            "line per dataset."
        ),
    )
    return parser.parse_args()


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_output_for_datasets(dataset_dirs: List[Path]) -> Path:
    if len(dataset_dirs) == 1:
        return metadata_dir(dataset_dirs[0]) / "reduced_action_space.json"

    parents = {dataset.parent.resolve() for dataset in dataset_dirs}
    if len(parents) == 1:
        parent = next(iter(parents))
        if parent.name == "parts":
            return parent.parent / "metadata" / "reduced_action_space.json"
        return parent / "metadata" / "reduced_action_space.json"

    common = Path(os.path.commonpath([str(path.resolve()) for path in dataset_dirs]))
    return common / "metadata" / "reduced_action_space.json"


def _list_shards_or_empty(dataset_dir: Path, max_shards: Optional[int]) -> List[Path]:
    try:
        return list_shards(dataset_dir, max_shards)
    except FileNotFoundError:
        print(f"warning: no shards found in {task_relative(dataset_dir)}; skipping")
        return []


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "unknown"
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:04.1f}s"
    if minutes:
        return f"{minutes}m{sec:04.1f}s"
    return f"{sec:.1f}s"


def main() -> None:
    cli = parse_args()
    if cli.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if cli.min_count <= 0:
        raise ValueError("--min-count must be positive.")
    if cli.progress_every_shards < 0:
        raise ValueError("--progress-every-shards must be non-negative.")

    run_start = time.perf_counter()
    dataset_dirs = [resolve_dataset_dir(path) for path in cli.dataset]
    metadatas = []
    for dataset_dir in dataset_dirs:
        metadata = load_metadata(dataset_dir)
        if metadata.get("dataset_mode") != "action_outcomes":
            raise ValueError(
                f"{task_relative(dataset_dir)} does not look like an "
                "action_outcomes dataset."
            )
        metadatas.append(metadata)
    metadata = metadatas[0]
    agent_ids: List[str] = list(metadata["agent_ids"])
    action_sizes = {
        agent: int(size) for agent, size in metadata.get("action_sizes", {}).items()
    }
    for dataset_dir, other_metadata in zip(dataset_dirs[1:], metadatas[1:]):
        if list(other_metadata["agent_ids"]) != agent_ids:
            raise ValueError(
                f"Agent ids in {task_relative(dataset_dir)} do not match the "
                "first dataset."
            )
        other_action_sizes = {
            agent: int(size)
            for agent, size in other_metadata.get("action_sizes", {}).items()
        }
        if other_action_sizes != action_sizes:
            raise ValueError(
                f"Action sizes in {task_relative(dataset_dir)} do not match the "
                "first dataset."
            )

    output = cli.output
    if output is None:
        output = _default_output_for_datasets(dataset_dirs)
    output = output.expanduser()
    if not output.is_absolute():
        output = (TASK_DIR / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    include_action_zero = _str_to_bool(cli.include_action_zero)
    require_improvement = _str_to_bool(cli.require_improvement)
    print("========== Action-space reduction ==========", flush=True)
    print(f"Datasets: {len(dataset_dirs)}", flush=True)
    print(f"Method: {cli.selection_method}", flush=True)
    print(f"Metric: {cli.metric}", flush=True)
    print(f"Top-k: {cli.top_k}", flush=True)
    print(f"Min count: {cli.min_count}", flush=True)
    print(f"Require improvement: {require_improvement}", flush=True)
    print(f"Improvement tolerance: {cli.improvement_tolerance}", flush=True)
    print(f"Output: {task_relative(output)}", flush=True)
    print("Agents: " + ", ".join(agent_ids), flush=True)
    print("===========================================", flush=True)

    counts_by_agent: Dict[str, Counter] = {agent: Counter() for agent in agent_ids}
    rate_stats_by_agent: Dict[str, Dict[int, Dict[str, int]]] = {
        agent: {} for agent in agent_ids
    }
    stats_by_agent: Dict[str, Dict[str, Any]] = {
        agent: defaultdict(int) for agent in agent_ids
    }

    datasets_with_shards = 0
    total_shards_processed = 0
    total_agent_rows_processed = 0
    for dataset_index, dataset_dir in enumerate(dataset_dirs, start=1):
        shards = _list_shards_or_empty(dataset_dir, cli.max_shards)
        if shards:
            datasets_with_shards += 1
        print(
            "dataset "
            f"{dataset_index}/{len(dataset_dirs)}: "
            f"{task_relative(dataset_dir)} shards={len(shards)}",
            flush=True,
        )
        dataset_start = time.perf_counter()
        dataset_agent_rows = 0
        for shard_index, shard in enumerate(shards, start=1):
            shard_start = time.perf_counter()
            shard_agent_rows = 0
            with np.load(shard) as data:
                for agent in agent_ids:
                    prefix = f"outcome_{agent}"
                    action_key = f"{prefix}_action_id"
                    metric_key = f"{prefix}_{cli.metric}"
                    valid_key = f"{prefix}_action_is_valid"
                    if action_key not in data.files:
                        continue
                    shard_agent_rows += int(data[action_key].shape[0])
                    if metric_key not in data.files:
                        raise KeyError(
                            f"Missing {metric_key} in {task_relative(shard)}"
                        )

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
                        _merge_counter(counts_by_agent[agent], counts)
                    elif cli.selection_method == "improvement_rate":
                        action_stats, stats = _improvement_rate_counts(
                            action_ids=data[action_key],
                            metric=data[metric_key],
                            valid=data[valid_key],
                            improvement_tolerance=cli.improvement_tolerance,
                        )
                        _merge_action_stats(rate_stats_by_agent[agent], action_stats)
                    else:
                        counts, stats = _all_improving_counts(**kwargs)
                        _merge_counter(counts_by_agent[agent], counts)
                    for key, value in stats.items():
                        if value is not None:
                            stats_by_agent[agent][key] += int(value)
            total_shards_processed += 1
            dataset_agent_rows += shard_agent_rows
            total_agent_rows_processed += shard_agent_rows
            if (
                cli.progress_every_shards > 0
                and (
                    shard_index == 1
                    or shard_index == len(shards)
                    or shard_index % cli.progress_every_shards == 0
                )
            ):
                elapsed = time.perf_counter() - run_start
                dataset_elapsed = time.perf_counter() - dataset_start
                avg_dataset_shard = dataset_elapsed / max(shard_index, 1)
                eta_dataset = avg_dataset_shard * max(len(shards) - shard_index, 0)
                print(
                    "progress: "
                    f"dataset={dataset_index}/{len(dataset_dirs)} "
                    f"shard={shard_index}/{len(shards)} "
                    f"total_shards={total_shards_processed} "
                    f"shard_rows={shard_agent_rows} "
                    f"dataset_rows={dataset_agent_rows} "
                    f"total_rows={total_agent_rows_processed} "
                    f"last_shard={_format_duration(time.perf_counter() - shard_start)} "
                    f"elapsed={_format_duration(elapsed)} "
                    f"dataset_eta={_format_duration(eta_dataset)}",
                    flush=True,
                )
        print(
            "dataset complete: "
            f"{dataset_index}/{len(dataset_dirs)} "
            f"rows={dataset_agent_rows} "
            f"elapsed={_format_duration(time.perf_counter() - dataset_start)}",
            flush=True,
        )

    print(
        "ranking actions after reading "
        f"{total_shards_processed} shards and {total_agent_rows_processed} rows...",
        flush=True,
    )

    total_candidate_states = sum(
        int(metadata.get("n_candidate_states") or 0) for metadata in metadatas
    )
    total_outcome_examples = sum(
        int(metadata.get("n_outcome_examples") or 0) for metadata in metadatas
    )
    reduced: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": task_relative(dataset_dirs[0]),
        "source_datasets": [task_relative(dataset_dir) for dataset_dir in dataset_dirs],
        "source_metadata": {
            "env_id": metadata.get("env_id"),
            "collection_rho_threshold": metadata.get("collection_rho_threshold"),
            "n_candidate_states": total_candidate_states,
            "n_outcome_examples": total_outcome_examples,
            "n_parts": len(dataset_dirs),
            "n_parts_with_shards": datasets_with_shards,
        },
        "metric": cli.metric,
        "selection_method": cli.selection_method,
        "top_k": int(cli.top_k),
        "min_count": int(cli.min_count),
        "min_count_semantics": (
            "minimum_seen_count"
            if cli.selection_method == "improvement_rate"
            else "minimum_selected_count"
        ),
        "include_action_zero": bool(include_action_zero),
        "require_improvement": bool(require_improvement),
        "improvement_tolerance": float(cli.improvement_tolerance),
        "agents": {},
    }

    total_original = 0
    total_reduced = 0
    for agent in agent_ids:
        print(f"ranking {agent}...", flush=True)
        total_original += int(action_sizes.get(agent, 0))
        if cli.selection_method == "improvement_rate":
            ranked = []
            for action_id, values in rate_stats_by_agent[agent].items():
                seen_count = int(values.get("seen_count", 0))
                if seen_count < cli.min_count:
                    continue
                improved_count = int(values.get("improved_count", 0))
                valid_finite_count = int(values.get("valid_finite_count", 0))
                ranked.append(
                    {
                        "action_id": int(action_id),
                        "score": (
                            improved_count / seen_count
                            if seen_count > 0
                            else float("nan")
                        ),
                        "improved_count": improved_count,
                        "seen_count": seen_count,
                        "valid_finite_count": valid_finite_count,
                    }
                )
            ranked.sort(
                key=lambda item: (
                    -float(item["score"]),
                    -int(item["improved_count"]),
                    -int(item["seen_count"]),
                    int(item["action_id"]),
                )
            )
        else:
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
        print(
            f"{agent}: ranked={len(ranked)} selected={len(selected)}/"
            f"{original_size}",
            flush=True,
        )

    reduced["total_original_action_size"] = int(total_original)
    reduced["total_selected_action_size"] = int(total_reduced)
    reduced["total_selected_fraction"] = (
        total_reduced / total_original if total_original > 0 else float("nan")
    )

    with output.open("w", encoding="utf-8") as f:
        json.dump(reduced, f, indent=2, sort_keys=True)
        f.write("\n")

    print("========== Reduced action space ==========")
    print(f"Datasets: {len(dataset_dirs)}")
    print(f"First dataset: {task_relative(dataset_dirs[0])}")
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
    print(f"Elapsed: {_format_duration(time.perf_counter() - run_start)}")


if __name__ == "__main__":
    main()
